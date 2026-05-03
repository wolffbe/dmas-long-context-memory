"""Slim coordinator HTTP API: /ask + /memorize + /reset.

Each endpoint handles a single request from the benchmark service.
/memorize forwards one message-sized payload to memory:/memorize via
toxiproxy so load-phase traffic experiences the same network conditions
as ask-phase traffic. /ask runs the Ollama tool-calling loop.
"""
from __future__ import annotations

import os
from typing import Any

from shared import otel_init

otel_init.init("coordinator")
otel_init.instrument_requests()
otel_init.instrument_httpx()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.coordinator_service import CoordinatorService
from shared.models import ResetRequest, WarmupRequest

app = FastAPI(title="coordinator", version="2.0")
otel_init.instrument_fastapi(app)

coordinator = CoordinatorService(
    memory_url=os.getenv("MEMORY_URL", "http://toxiproxy:18005"),
    responder_url=os.getenv("RESPONDER_URL", "http://toxiproxy:18006"),
    ollama_model=os.getenv("OLLAMA_MODEL"),
)


class AskRequest(BaseModel):
    question: str
    backend: str
    # LoCoMo session date string for the question's evidence
    # ("8 May 2023 at 4:42 pm"). The responder uses this as the anchor
    # when resolving relative time references ("yesterday", "last week"),
    # since gpt-4o-mini otherwise falls back to its training cutoff.
    session_date: str = ""
    # Forwarded to the responder so its detached `responder.respond`
    # root span lands in the same langfuse session and carries the same
    # tags as the bench's `ask.question` and `load.message` traces.
    session_id: str | None = None
    conv_index: int | None = None
    mode: str | None = None


class MemorizeRequest(BaseModel):
    backend: str
    conv_index: int
    data: dict[str, Any]
    # Optional caller-supplied langfuse trace ID. When set, every
    # memory-side LLM call this /memorize produces is pinned to it; when
    # absent the coordinator mints one and returns it on the response.
    trace_id: str | None = None
    # Forwarded to memory so the per-framework save span lands in the
    # same langfuse session as the bench's load.message trace.
    session_id: str | None = None
    mode: str | None = None


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/ask")
async def ask(req: AskRequest):
    result = coordinator.ask(question=req.question, backend=req.backend,
                             session_date=req.session_date,
                             session_id=req.session_id,
                             conv_index=req.conv_index, mode=req.mode)
    if result.get("status") == "error":
        raise HTTPException(500, result.get("error"))
    return result


@app.post("/memorize")
async def memorize(req: MemorizeRequest):
    """Forward one /memorize call to memory and return its confirmation.

    Bench drives granularity: one call typically carries a single message,
    so the response confirms exactly that message was persisted.
    """
    result = coordinator.memorize(backend=req.backend, conv_index=req.conv_index,
                                  data=req.data, trace_id=req.trace_id,
                                  session_id=req.session_id, mode=req.mode)
    if result.get("status") == "error":
        raise HTTPException(502, result.get("error"))
    return result


@app.post("/reset")
async def reset(req: ResetRequest):
    result = coordinator.reset(backend=req.backend)
    if result.get("status") == "error":
        raise HTTPException(502, result.get("error"))
    return result


@app.post("/warmup")
async def warmup(req: WarmupRequest):
    result = coordinator.warmup(backend=req.backend, conv_index=req.conv_index)
    if result.get("status") == "error":
        raise HTTPException(502, result.get("error"))
    return result
