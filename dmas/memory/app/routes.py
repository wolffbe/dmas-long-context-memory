"""Memory service. Instantiates both backends at startup; the request
selects which one to use via a `backend` field. No MEMORY_BACKEND env."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from shared import otel_init

otel_init.init("memory")
otel_init.instrument_httpx()
otel_init.instrument_requests()

import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from opentelemetry import trace as otel_trace

from shared.models import ResetRequest, WarmupRequest

from app.services.graphiti_service import GraphitiService
from app.services.mem0_service import Mem0Service
from app.services.full_context_service import FullContextService
from app.services.rag_service import RagService
from app.services.cognee_service import CogneeService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="memory", version="2.0")
otel_init.instrument_fastapi(app)
_tracer = otel_trace.get_tracer(__name__)

# Per-backend span names that mirror each framework's own terminology
# (mem0 calls it `add`, graphiti `add_episode`, cognee `cognify`, etc.).
# Lets the operator filter the langfuse trace list by `name=mem0.add` or
# `name=graphiti.search` instead of the generic `memory.persist`.
_SAVE_NAME = {
    "mem0": "mem0.add",
    "graphiti": "graphiti.add_episode",
    "cognee": "cognee.cognify",
    "rag": "rag.upsert",
    "full_context": "full_context.append",
}
_SEARCH_NAME = {
    "mem0": "mem0.search",
    "graphiti": "graphiti.search",
    "cognee": "cognee.search",
    "rag": "rag.search",
    "full_context": "full_context.dump",
}

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
    # Forwarded by the coordinator (which got them from the bench) so
    # the manual save span lands in the same langfuse session and
    # carries consistent tags.
    session_id: str | None = None
    mode: str | None = None


class RememberRequest(BaseModel):
    backend: str
    question: str
    # Forwarded by the responder (which got them from the coordinator
    # which got them from the bench) so the manual search span lands in
    # the same langfuse session as the ask.question / responder.respond
    # traces.
    session_id: str | None = None
    conv_index: int | None = None
    mode: str | None = None


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
    # Span name reflects the framework's own API verb (e.g. `mem0.add`,
    # `graphiti.add_episode`, `cognee.cognify`) so the operator can
    # filter the langfuse trace list by `name=mem0.add`.
    span_name = _SAVE_NAME.get(req.backend, f"{req.backend}.persist")
    attrs: dict[str, Any] = {
        "dmas.backend": req.backend,
        "dmas.conv_index": req.conv_index,
        "input": json.dumps(req.data)[:8000],
    }
    if req.session_id:
        attrs["langfuse.session.id"] = req.session_id
        tag_parts = ["service:memory", f"memory:{req.backend}", f"conv:{req.conv_index}"]
        if req.mode:
            tag_parts.append(f"mode:{req.mode}")
        attrs["langfuse.tags"] = ",".join(tag_parts)
    # Nested under the bench's `load.message` trace via httpx context
    # propagation: load.message → coordinator.memorize → memory's
    # POST /memorize → this span. Lets the operator click into a single
    # `load.message` trace and see the full chain including the
    # framework's persist call.
    with _tracer.start_as_current_span(
        span_name,
        attributes=attrs,
    ) as sp:
        result = await _dispatch(backend, "memorize_conversation", req.conv_index, req.data)
        try:
            sp.set_attribute("dmas.memories_added", len(result.get("memories") or []))
            sp.set_attribute("output", json.dumps(result)[:8000])
        except Exception:
            pass
        return result


@app.post("/remember")
async def remember(req: RememberRequest):
    if not req.question:
        raise HTTPException(400, "missing question")
    backend = _resolve(req.backend)
    # Span name uses each framework's own retrieval verb. input = the
    # query the responder issued; output = the memories the framework
    # returned (joined just like the responder receives them).
    span_name = _SEARCH_NAME.get(req.backend, f"{req.backend}.search")
    attrs: dict[str, Any] = {
        "dmas.backend": req.backend,
        "input": req.question[:8000],
    }
    if req.session_id:
        attrs["langfuse.session.id"] = req.session_id
        tag_parts = ["service:memory", f"memory:{req.backend}"]
        if req.conv_index is not None:
            tag_parts.append(f"conv:{req.conv_index}")
        if req.mode:
            tag_parts.append(f"mode:{req.mode}")
        attrs["langfuse.tags"] = ",".join(tag_parts)
    # Nested under `responder.respond` via httpx context propagation:
    # responder.respond → responder.search → POST /remember → this span.
    # The operator drills into the responder trace to see top_k + the
    # raw memories returned for this query inline; no separate
    # session-level entry needed.
    with _tracer.start_as_current_span(
        span_name,
        attributes=attrs,
    ) as sp:
        memories = await _dispatch(backend, "remember", req.question) or []
        top_k = getattr(backend, "TOP_K", None)
        try:
            joined = "\n\n".join(memories)
            sp.set_attribute("dmas.memories_returned", len(memories))
            if top_k is not None:
                sp.set_attribute("dmas.top_k", int(top_k))
            # Structured output: top_k, count, and the raw memories text
            # the framework returned to the responder. Lets the operator
            # see, in the same panel, both the configured retrieval ceiling
            # AND what came back for this query.
            sp.set_attribute("output", json.dumps({
                "top_k": top_k,
                "count": len(memories),
                "memories": joined[:30000],
            })[:32000])
        except Exception:
            pass
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
        "top_k": top_k,
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
    result = await _dispatch(backend, "reset")
    return {"status": "success", **(result or {})}
