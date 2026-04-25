"""Slim coordinator HTTP API: /ask + /memorize + /reset.

Each endpoint handles a single request from the benchmark service.
/memorize forwards one message-sized payload to memory:/memorize via
toxiproxy so load-phase traffic experiences the same network conditions
as ask-phase traffic. /ask runs the Ollama tool-calling loop.
"""
from __future__ import annotations

import os
from typing import Any

# Import BEFORE the service — patches openai chat/embeddings/responses
# `create` to inject `metadata.tags` so litellm tags the trace.
from app import langfuse_tags  # noqa: F401

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.coordinator_service import CoordinatorService

app = FastAPI(title="coordinator", version="2.0")

coordinator = CoordinatorService(
    memory_url=os.getenv("MEMORY_URL", "http://toxiproxy:18005"),
    responder_url=os.getenv("RESPONDER_URL", "http://toxiproxy:18006"),
    ollama_model=os.getenv("OLLAMA_MODEL"),
)


class AskRequest(BaseModel):
    question: str
    backend: str


class MemorizeRequest(BaseModel):
    backend: str
    conv_index: int
    data: dict[str, Any]


class ResetRequest(BaseModel):
    backend: str


class WarmupRequest(BaseModel):
    backend: str
    conv_index: int


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/ask")
async def ask(req: AskRequest):
    result = coordinator.ask(question=req.question, backend=req.backend)
    if result.get("status") == "error":
        raise HTTPException(500, result.get("error"))
    return result


@app.post("/memorize")
async def memorize(req: MemorizeRequest):
    """Forward one /memorize call to memory and return its confirmation.

    Bench drives granularity: one call typically carries a single message,
    so the response confirms exactly that message was persisted.
    """
    result = coordinator.memorize(backend=req.backend, conv_index=req.conv_index, data=req.data)
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
