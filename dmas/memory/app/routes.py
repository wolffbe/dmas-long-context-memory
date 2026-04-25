"""Memory service. Instantiates both backends at startup; the request
selects which one to use via a `backend` field. No MEMORY_BACKEND env."""
from __future__ import annotations

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="memory", version="2.0")

_backends: dict[str, Any] = {
    "mem0": Mem0Service(),
    "graphiti": GraphitiService(),
    "rag": RagService(),
    "full_context": FullContextService(),
}


def _resolve(backend: str) -> Any:
    if backend not in _backends:
        raise HTTPException(400, f"unknown backend {backend!r}; expected one of {sorted(_backends)}")
    return _backends[backend]


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
def memorize(req: MemorizeRequest):
    """Persist whatever sessions/turns are in `data`. The bench drives the
    granularity: a single-turn payload commits one message, a full-session
    payload commits the whole session in one call. Returns the per-add
    `memories` list and a status summary."""
    backend = _resolve(req.backend)
    with active_backend(req.backend):
        return backend.memorize_conversation(req.conv_index, req.data)


@app.post("/remember")
def remember(req: RememberRequest):
    if not req.question:
        raise HTTPException(400, "missing question")
    backend = _resolve(req.backend)
    with active_backend(req.backend):
        memories = backend.remember(req.question) or []
    # `count` is items returned (after backend's own filtering); `top_k`
    # is the configured ceiling (None for full_context which dumps the
    # whole convo). Both are surfaced so the benchmark can verify
    # retrieval breadth matches the configured limit.
    # Backends that pre-format their results into a single block (e.g.
    # graphiti emits Zep's FACTS+ENTITIES template as one string) expose
    # the underlying item count via `_last_search_count`; everything
    # else falls back to the list length.
    top_k = getattr(backend, "TOP_K", None)
    count = getattr(backend, "_last_search_count", None)
    if count is None:
        count = len(memories)
    return {
        "status": "success",
        "memory": "\n\n".join(memories),
        "count": count,
        "top_k": top_k,
    }


@app.post("/warmup")
def warmup(req: WarmupRequest):
    """Pay any one-time backend init cost (graphiti index build, qdrant
    collection creation) up front so it shows up in its own row instead
    of inflating the first /memorize call."""
    backend = _resolve(req.backend)
    fn = getattr(backend, "warmup", None)
    if not callable(fn):
        raise HTTPException(501, f"backend {req.backend!r} does not implement warmup")
    with active_backend(req.backend):
        return {"status": "success", **(fn(req.conv_index) or {})}


@app.post("/reset")
def reset(req: ResetRequest):
    """Wipe the named backend's persisted state so the next /memorize
    starts on empty storage. The benchmark calls this between every
    (backend, conv, mode) leg so memory state never leaks across runs."""
    backend = _resolve(req.backend)
    fn = getattr(backend, "reset", None)
    if not callable(fn):
        raise HTTPException(501, f"backend {req.backend!r} does not implement reset")
    with active_backend(req.backend):
        return {"status": "success", **(fn() or {})}
