"""Inference providers.

TogetherProvider talks to Together AI's OpenAI-compatible chat completions API.
FakeProvider simulates a flaky API (rate limits, 5xx, timeouts, bad requests)
so the reliability logic can be tested without spending money.
"""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Optional, Protocol

import httpx


class ProviderError(Exception):
    def __init__(self, message: str, retryable: bool,
                 status: Optional[int] = None, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status
        self.retry_after = retry_after


@dataclass
class Completion:
    text: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class Provider(Protocol):
    async def complete(self, model: str, messages: list, **params) -> Completion: ...


def classify_status(status: int) -> bool:
    """True if an HTTP status is worth retrying.
    429 = rate limited, 408 = request timeout, 5xx = server side.
    Everything else in 4xx means the request itself is wrong, so retrying
    would just fail again and waste money."""
    return status in (408, 409, 429) or status >= 500


def parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


class TogetherProvider:
    BASE_URL = "https://api.together.xyz/v1"

    def __init__(self, api_key: str, timeout: float = 60.0):
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    async def complete(self, model: str, messages: list, **params) -> Completion:
        try:
            resp = await self._client.post(
                "/chat/completions", json={"model": model, "messages": messages, **params}
            )
        except (httpx.TimeoutException, httpx.TransportError) as e:
            raise ProviderError(f"network: {type(e).__name__}", retryable=True) from e

        if resp.status_code >= 400:
            raise ProviderError(
                f"HTTP {resp.status_code}: {resp.text[:200]}",
                retryable=classify_status(resp.status_code),
                status=resp.status_code,
                retry_after=parse_retry_after(resp.headers.get("retry-after")),
            )

        data = resp.json()
        usage = data.get("usage") or {}
        return Completion(
            text=data["choices"][0]["message"]["content"],
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )

    async def aclose(self) -> None:
        await self._client.aclose()


class FakeProvider:
    """Simulates a real inference API, including the ways it breaks."""

    def __init__(self, rate_limit: float = 0.10, server_error: float = 0.05,
                 timeout: float = 0.03, bad_request: float = 0.01,
                 latency_ms: tuple[float, float] = (40, 250), seed: Optional[int] = None):
        self.p_429, self.p_5xx, self.p_timeout, self.p_400 = rate_limit, server_error, timeout, bad_request
        self.latency_ms = latency_ms
        self.rng = random.Random(seed)
        self.calls = 0

    async def complete(self, model: str, messages: list, **params) -> Completion:
        self.calls += 1
        await asyncio.sleep(self.rng.uniform(*self.latency_ms) / 1000)
        r = self.rng.random()
        if r < self.p_400:
            raise ProviderError("HTTP 400: invalid request", retryable=False, status=400)
        r -= self.p_400
        if r < self.p_429:
            raise ProviderError("HTTP 429: rate limited", retryable=True, status=429,
                                retry_after=self.rng.choice([None, 0.2]))
        r -= self.p_429
        if r < self.p_5xx:
            raise ProviderError("HTTP 503: upstream overloaded", retryable=True, status=503)
        r -= self.p_5xx
        if r < self.p_timeout:
            raise ProviderError("network: ReadTimeout", retryable=True)
        prompt = " ".join(m.get("content", "") for m in messages)
        pt = max(1, len(prompt) // 4)
        ct = self.rng.randint(20, 200)
        return Completion(text=f"[{model}] ok", prompt_tokens=pt, completion_tokens=ct)
