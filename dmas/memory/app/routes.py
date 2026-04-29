"""Memory service. Instantiates both backends at startup; the request
selects which one to use via a `backend` field. No MEMORY_BACKEND env."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

# Import BEFORE the services — patches openai chat/embeddings/responses
# `create` to inject `metadata.tags` so litellm tags the trace.
from app import langfuse_tags  # noqa: F401
from app.langfuse_tags import active_backend

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.services.graphiti_service import GraphitiService
from app.services.mem0_service import Mem0Service
from app.services.full_context_service import FullContextService
from app.services.rag_service import RagService
from app.services.cognee_service import CogneeService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="memory", version="2.0")

_backends: dict[str, Any] = {
    "mem0": Mem0Service(),
    "graphiti": GraphitiService(),
    "rag": RagService(),
    "cognee": CogneeService(),
    "full_context": FullContextService(),
}


def _resolve(backend: str) -> Any:
    if backend not in _backends:
        raise HTTPException(400, f"unknown backend {backend!r}; expected one of {sorted(_backends)}")
    return _backends[backend]


async def _dispatch(backend: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    """Call `backend.<method_name>` async-first.

    If the backend exposes `<method_name>_async`, await it on the running
    event loop — this keeps the neo4j async driver in graphiti/cognee bound
    to a single FastAPI loop, avoiding the "Future attached to a different
    loop" bug that the old per-request `new_event_loop()` wrappers caused.
    Otherwise run the sync method in a worker thread so it doesn't block
    the loop. ContextVars (langfuse tags) propagate across `to_thread`.
    """
    async_fn = getattr(backend, f"{method_name}_async", None)
    if callable(async_fn):
        return await async_fn(*args, **kwargs)
    return await asyncio.to_thread(getattr(backend, method_name), *args, **kwargs)


class MemorizeRequest(BaseModel):
    backend: str
    conv_index: int
    data: dict[str, Any]


class RememberRequest(BaseModel):
    backend: str
    question: str


class ResetRequest(BaseModel):
    backend: str


class WarmupRequest(BaseModel):
    backend: str
    conv_index: int


@app.get("/health")
def health():
    return {"status": "ok", "backends": sorted(_backends)}


@app.post("/memorize")
async def memorize(req: MemorizeRequest):
    """Persist whatever sessions/turns are in `data`. The bench drives the
    granularity: a single-turn payload commits one message, a full-session
    payload commits the whole session in one call. Returns the per-add
    `memories` list and a status summary."""
    backend = _resolve(req.backend)
    with active_backend(req.backend):
        return await _dispatch(backend, "memorize_conversation", req.conv_index, req.data)


@app.post("/remember")
async def remember(req: RememberRequest):
    if not req.question:
        raise HTTPException(400, "missing question")
    backend = _resolve(req.backend)
    with active_backend(req.backend):
        memories = await _dispatch(backend, "remember", req.question) or []
    # `count` is the number of memory items handed to the responder LLM
    # (i.e. the length of the joined list, regardless of how the backend
    # formats them internally). For graphiti, which packs facts+entities
    # into a single rendered block, count == 1; for mem0/rag it's the
    # number of distilled memories or turns retrieved; for full_context
    # it's 1. `top_k` is the configured ceiling (None for full_context).
    return {
        "status": "success",
        "memory": "\n\n".join(memories),
        "count": len(memories),
        "top_k": getattr(backend, "TOP_K", None),
    }


@app.post("/warmup")
async def warmup(req: WarmupRequest):
    """Pay any one-time backend init cost (graphiti index build, qdrant
    collection creation) up front so it shows up in its own row instead
    of inflating the first /memorize call."""
    backend = _resolve(req.backend)
    if not (callable(getattr(backend, "warmup_async", None))
            or callable(getattr(backend, "warmup", None))):
        raise HTTPException(501, f"backend {req.backend!r} does not implement warmup")
    with active_backend(req.backend):
        result = await _dispatch(backend, "warmup", req.conv_index)
    return {"status": "success", **(result or {})}


@app.post("/reset")
async def reset(req: ResetRequest):
    """Wipe the named backend's persisted state so the next /memorize
    starts on empty storage. The benchmark calls this between every
    (backend, conv, mode) leg so memory state never leaks across runs."""
    backend = _resolve(req.backend)
    if not (callable(getattr(backend, "reset_async", None))
            or callable(getattr(backend, "reset", None))):
        raise HTTPException(501, f"backend {req.backend!r} does not implement reset")
    with active_backend(req.backend):
        result = await _dispatch(backend, "reset")
    return {"status": "success", **(result or {})}
