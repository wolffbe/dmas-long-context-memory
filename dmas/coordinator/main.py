"""Coordinator entrypoint.

One container that owns:
- LoCoMo + LongMemEval datasets (loaded at startup, no HTTP egress to dataset svcs)
- Toxic syncing on the three toxiproxy admins
- Per-question Prometheus snapshotting (CPU/RAM/disk/network split by edge/cloud)
- Round-robin loading + the per-experiment QA loop
- LLM-as-judge invocation (verbatim mem0 + zep prompts)

The driver POSTs `/load` and `/experiment` and writes whatever comes back.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
import httpx

from app.datasets import datasets
from app.routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [coord] %(levelname)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(timeout=httpx.Timeout(300.0))
    datasets.ensure_loaded()
    yield
    await app.state.http.aclose()


app = FastAPI(title="dmas-coordinator", lifespan=lifespan)
app.include_router(router)
