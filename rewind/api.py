"""HTTP API.

Run with:   uvicorn rewind.api:app --reload
Env vars:   PROVIDER=fake|together   TOGETHER_API_KEY=...   REWIND_DB=rewind.db
            CONCURRENCY=8   RATE_PER_SEC=5   PRICING_JSON=pricing.json
"""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, model_validator

from .providers import FakeProvider, TogetherProvider
from .store import Store
from .worker import WorkerPool

DASHBOARD = Path(__file__).with_name("dashboard.html")


class JobIn(BaseModel):
    model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    prompt: Optional[str] = None
    messages: Optional[list[dict]] = None
    params: dict = Field(default_factory=dict)
    idempotency_key: Optional[str] = None
    max_attempts: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def _need_input(self):
        if not self.messages and not self.prompt:
            raise ValueError("send either 'prompt' or 'messages'")
        if not self.messages:
            self.messages = [{"role": "user", "content": self.prompt}]
        return self


class ReplayIn(BaseModel):
    job_ids: Optional[list[str]] = None


def build_provider():
    if os.getenv("PROVIDER", "fake") == "together":
        key = os.getenv("TOGETHER_API_KEY")
        if not key:
            raise RuntimeError("PROVIDER=together needs TOGETHER_API_KEY")
        return TogetherProvider(key)
    return FakeProvider()


def create_app(store: Optional[Store] = None, pool: Optional[WorkerPool] = None) -> FastAPI:
    store = store or Store(os.getenv("REWIND_DB", "rewind.db"))
    rate = os.getenv("RATE_PER_SEC")
    pool = pool or WorkerPool(
        store, build_provider(),
        concurrency=int(os.getenv("CONCURRENCY", "8")),
        rate_per_sec=float(rate) if rate else None,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        task = asyncio.create_task(pool.run())
        yield
        pool.stop()
        await task

    app = FastAPI(title="Rewind", lifespan=lifespan)

    @app.post("/jobs", status_code=201)
    def submit(job: JobIn):
        j, created = store.enqueue(job.model, job.messages, job.params,
                                   job.idempotency_key, job.max_attempts)
        return {"job": j.to_dict(), "created": created}

    @app.post("/jobs/batch", status_code=201)
    def submit_batch(jobs: list[JobIn]):
        out = [store.enqueue(j.model, j.messages, j.params, j.idempotency_key, j.max_attempts)
               for j in jobs]
        return {"created": sum(c for _, c in out),
                "duplicates": sum(not c for _, c in out),
                "job_ids": [j.id for j, _ in out]}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str):
        j = store.get(job_id)
        if not j:
            raise HTTPException(404, f"No job with id {job_id}")
        return {"job": j.to_dict(), "events": store.events(job_id)}

    @app.get("/jobs")
    def list_jobs(status: Optional[str] = None, limit: int = 50):
        return [j.to_dict() for j in store.list(status, limit)]

    @app.post("/dlq/replay")
    def replay(body: ReplayIn):
        return {"replayed": store.replay(body.job_ids)}

    @app.get("/metrics")
    def metrics():
        return store.metrics()

    @app.get("/events")
    def events(limit: int = 30):
        return store.events(limit=limit)

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return DASHBOARD.read_text()

    return app


app = create_app()
