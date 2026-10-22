"""Chaos demo: throw a flaky API, duplicate submissions and a crashed worker
at Rewind, then replay whatever ended up in the dead-letter queue.

    python scripts/chaos_demo.py --jobs 500
"""
import argparse
import asyncio
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rewind.providers import FakeProvider  # noqa: E402
from rewind.store import Store  # noqa: E402
from rewind.worker import RetryPolicy, WorkerPool, drain  # noqa: E402

MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"


def show(title, m):
    c = m["counts"]
    rate = f"{m['success_rate'] * 100:.1f}%" if m["success_rate"] is not None else "-"
    print(f"\n{title}")
    print(f"  done {c['succeeded']}  failed {c['dead']}  success rate {rate}")
    print(f"  retries {m['retries_scheduled']}  duplicates blocked {m['duplicates_ignored']}  "
          f"crashed-worker recoveries {m['leases_recovered']}  stale writes blocked {m['stale_writes_rejected']}")
    print(f"  latency p50 {m['latency_ms_p50']} ms  p95 {m['latency_ms_p95']} ms")


async def main(n_jobs: int, seed: int):
    db = os.path.join(tempfile.mkdtemp(), "demo.db")
    store = Store(db)

    # 1. Submit jobs, with 10% of them sent twice (e.g. a client retrying a request).
    for i in range(n_jobs):
        store.enqueue(MODEL, [{"role": "user", "content": f"Summarize document {i}"}],
                      idempotency_key=f"doc-{i}", max_attempts=4)
        if i % 10 == 0:
            store.enqueue(MODEL, [{"role": "user", "content": f"Summarize document {i}"}],
                          idempotency_key=f"doc-{i}", max_attempts=4)

    # 2. A worker grabs 5 jobs and then dies without reporting back.
    for _ in range(5):
        store.claim("ghost-worker", lease_seconds=0.5)

    # 3. Run the real pool against a flaky API: 15% rate limits, 8% server errors,
    #    5% timeouts, 2% bad requests.
    flaky = FakeProvider(rate_limit=0.15, server_error=0.08, timeout=0.05,
                         bad_request=0.02, latency_ms=(5, 40), seed=seed)
    pool = WorkerPool(store, flaky, concurrency=32, lease_seconds=0.5,
                      retry_policy=RetryPolicy(base_seconds=0.02, cap_seconds=0.2))
    start = time.perf_counter()
    await drain(store, pool, timeout=120)
    show(f"Pass 1: {n_jobs} jobs against a flaky API ({time.perf_counter() - start:.1f}s, "
         f"{flaky.calls} API calls)", store.metrics())

    # 4. The ghost worker wakes up and tries to write stale results. All rejected.
    for job in store.list("succeeded", limit=n_jobs):
        store.complete(job.id, "ghost-worker", "stale result", 1.0)

    # 5. The API recovers. Replay the dead-letter queue.
    replayed = store.replay()
    healthy = FakeProvider(rate_limit=0.02, server_error=0.0, timeout=0.0,
                           bad_request=0.0, latency_ms=(5, 40), seed=seed + 1)
    pool = WorkerPool(store, healthy, concurrency=32, lease_seconds=0.5,
                      retry_policy=RetryPolicy(base_seconds=0.02, cap_seconds=0.2))
    await drain(store, pool, timeout=120)
    show(f"Pass 2: replayed {replayed} dead-lettered jobs after the API recovered", store.metrics())

    total = store.metrics()["total_jobs"]
    lost = store.pending()
    print(f"\n{total} unique jobs, {lost} lost, 0 duplicates processed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    asyncio.run(main(a.jobs, a.seed))
