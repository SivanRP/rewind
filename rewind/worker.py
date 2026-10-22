"""Async worker pool that drains the job store.

Each worker loops: claim a job -> call the provider -> record success or
failure. Concurrency, rate limiting and backoff are all configurable.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from .providers import Provider, ProviderError
from .store import Job, Store


@dataclass
class RetryPolicy:
    base_seconds: float = 0.5
    cap_seconds: float = 30.0

    def delay(self, attempt: int, retry_after: Optional[float] = None) -> float:
        """Exponential backoff with full jitter.

        Jitter spreads retries out so a burst of failures doesn't turn into a
        synchronized retry storm that hits the API all at once again.
        If the server sent Retry-After, respect it instead of guessing."""
        if retry_after is not None:
            return min(self.cap_seconds, retry_after)
        ceiling = min(self.cap_seconds, self.base_seconds * (2 ** (attempt - 1)))
        return random.uniform(0, ceiling)


class TokenBucket:
    """Caps request rate across all workers (requests per second)."""

    def __init__(self, rate: float, capacity: Optional[float] = None):
        self.rate = rate
        self.capacity = capacity or max(1.0, rate)
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                await asyncio.sleep((1 - self.tokens) / self.rate)


def load_prices() -> dict[str, dict[str, float]]:
    """Per-model prices in USD per 1M tokens, from PRICING_JSON if set.
    Prices change, so they live in a config file instead of the code."""
    path = os.getenv("PRICING_JSON")
    if path and os.path.exists(path):
        with open(path) as f:
            return {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    return {}


def cost_for(prices: dict, model: str, pt: Optional[int], ct: Optional[int]) -> Optional[float]:
    p = prices.get(model)
    if not p or pt is None or ct is None:
        return None
    return (pt * p.get("input", 0) + ct * p.get("output", 0)) / 1_000_000


class WorkerPool:
    def __init__(self, store: Store, provider: Provider, concurrency: int = 8,
                 lease_seconds: float = 60.0, rate_per_sec: Optional[float] = None,
                 poll_interval: float = 0.05, retry_policy: Optional[RetryPolicy] = None,
                 request_timeout: Optional[float] = None):
        self.store = store
        self.provider = provider
        self.concurrency = concurrency
        self.lease_seconds = lease_seconds
        self.bucket = TokenBucket(rate_per_sec) if rate_per_sec else None
        self.poll_interval = poll_interval
        self.retry = retry_policy or RetryPolicy()
        # The provider call must finish well before the lease expires, otherwise
        # another worker could reclaim the job while this one is still running it.
        self.request_timeout = request_timeout or lease_seconds * 0.8
        self.prices = load_prices()
        self._stop = asyncio.Event()

    async def run(self) -> None:
        self._stop.clear()
        workers = [asyncio.create_task(self._loop(f"w{i}-{uuid.uuid4().hex[:6]}"))
                   for i in range(self.concurrency)]
        await self._stop.wait()
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    def stop(self) -> None:
        self._stop.set()

    async def _loop(self, worker_id: str) -> None:
        while not self._stop.is_set():
            job = await asyncio.to_thread(self.store.claim, worker_id, self.lease_seconds)
            if job is None:
                await asyncio.sleep(self.poll_interval)
                continue
            await self._process(job, worker_id)

    async def _process(self, job: Job, worker_id: str) -> None:
        if self.bucket:
            await self.bucket.acquire()
        start = time.perf_counter()
        try:
            comp = await asyncio.wait_for(
                self.provider.complete(job.model, job.messages, **job.params),
                timeout=self.request_timeout,
            )
        except asyncio.TimeoutError:
            err = ProviderError("timed out waiting for provider", retryable=True)
            await self._record_failure(job, worker_id, err)
        except ProviderError as err:
            await self._record_failure(job, worker_id, err)
        except Exception as err:  # unexpected bug: don't retry forever
            await self._record_failure(
                job, worker_id, ProviderError(f"unexpected: {err!r}", retryable=False))
        else:
            latency_ms = (time.perf_counter() - start) * 1000
            await asyncio.to_thread(
                self.store.complete, job.id, worker_id, comp.text, latency_ms,
                comp.prompt_tokens, comp.completion_tokens,
                cost_for(self.prices, job.model, comp.prompt_tokens, comp.completion_tokens),
            )

    async def _record_failure(self, job: Job, worker_id: str, err: ProviderError) -> None:
        delay = self.retry.delay(job.attempts, err.retry_after)
        await asyncio.to_thread(
            self.store.fail, job.id, worker_id, str(err), err.retryable, delay)


async def drain(store: Store, pool: WorkerPool, timeout: float = 60.0) -> None:
    """Run the pool until every job is finished (succeeded or dead)."""
    task = asyncio.create_task(pool.run())
    deadline = time.monotonic() + timeout
    try:
        while store.pending() > 0:
            if time.monotonic() > deadline:
                raise TimeoutError("jobs did not finish in time")
            await asyncio.sleep(0.05)
    finally:
        pool.stop()
        await task
