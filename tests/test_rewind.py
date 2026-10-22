import asyncio
import time

import pytest

from rewind.providers import Completion, ProviderError
from rewind.store import Store
from rewind.worker import RetryPolicy, WorkerPool, drain

MSG = [{"role": "user", "content": "hello"}]
FAST = RetryPolicy(base_seconds=0.01, cap_seconds=0.05)


class Scripted:
    """Provider that follows a script of outcomes, one per call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    async def complete(self, model, messages, **params):
        self.calls += 1
        outcome = self.script.pop(0) if self.script else "ok"
        await asyncio.sleep(0.001)
        if outcome == "ok":
            return Completion("done", 10, 5)
        if outcome == "429":
            raise ProviderError("HTTP 429", retryable=True, status=429)
        if outcome == "400":
            raise ProviderError("HTTP 400", retryable=False, status=400)
        raise ProviderError("HTTP 503", retryable=True, status=503)


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "t.db"))


def run(store, provider, **kw):
    pool = WorkerPool(store, provider, concurrency=kw.pop("concurrency", 1),
                      retry_policy=FAST, poll_interval=0.005, **kw)
    asyncio.run(drain(store, pool, timeout=10))


def test_duplicate_submit_returns_original_job(store):
    a, created_a = store.enqueue("m", MSG, idempotency_key="order-1")
    b, created_b = store.enqueue("m", MSG, idempotency_key="order-1")
    assert created_a and not created_b
    assert a.id == b.id
    assert store.metrics()["total_jobs"] == 1


def test_transient_failures_retry_then_succeed(store):
    job, _ = store.enqueue("m", MSG)
    provider = Scripted(["429", "503", "ok"])
    run(store, provider)
    done = store.get(job.id)
    assert done.status == "succeeded"
    assert done.attempts == 3
    assert provider.calls == 3


def test_bad_request_goes_straight_to_dead_letter(store):
    job, _ = store.enqueue("m", MSG)
    provider = Scripted(["400"])
    run(store, provider)
    dead = store.get(job.id)
    assert dead.status == "dead"
    assert dead.attempts == 1  # no wasted retries on a request that can never work


def test_retries_are_capped(store):
    job, _ = store.enqueue("m", MSG, max_attempts=3)
    run(store, Scripted(["503"] * 10))
    dead = store.get(job.id)
    assert dead.status == "dead" and dead.attempts == 3


def test_replay_gives_dead_jobs_a_fresh_start(store):
    job, _ = store.enqueue("m", MSG, max_attempts=2)
    run(store, Scripted(["503", "503"]))
    assert store.get(job.id).status == "dead"

    assert store.replay() == 1
    run(store, Scripted(["ok"]))
    revived = store.get(job.id)
    assert revived.status == "succeeded"
    assert revived.replay_count == 1


def test_crashed_worker_lease_expires_and_stale_write_is_fenced(store):
    job, _ = store.enqueue("m", MSG)

    # Worker A claims the job, then "crashes" (never reports back).
    a = store.claim("worker-a", lease_seconds=0.05)
    assert a.id == job.id
    assert store.claim("worker-b", lease_seconds=5) is None  # still leased

    time.sleep(0.08)
    b = store.claim("worker-b", lease_seconds=5)  # lease expired, B takes over
    assert b.id == job.id and b.attempts == 2

    # A wakes up and tries to write a result. It must be rejected.
    assert store.complete(job.id, "worker-a", "stale", 1.0) is False
    assert store.complete(job.id, "worker-b", "fresh", 1.0) is True
    final = store.get(job.id)
    assert final.result == "fresh"
    m = store.metrics()
    assert m["leases_recovered"] == 1 and m["stale_writes_rejected"] == 1


def test_many_workers_never_double_process(store):
    ids = [store.enqueue("m", MSG)[0].id for _ in range(200)]
    provider = Scripted([])
    run(store, provider, concurrency=16)
    assert provider.calls == 200  # every job ran exactly once
    assert all(store.get(i).status == "succeeded" for i in ids)


def test_metrics_shape(store):
    store.enqueue("m", MSG)
    run(store, Scripted(["429", "ok"]))
    m = store.metrics()
    assert m["counts"]["succeeded"] == 1
    assert m["success_rate"] == 1.0
    assert m["retries_scheduled"] == 1
    assert m["latency_ms_p50"] is not None
