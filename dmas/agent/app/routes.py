from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.config import CFG
from app.jitter import measured_peer_latency_ms
from app.llm_accounting import bind as _bind_acc
from app.memory import BACKEND_NAMES

router = APIRouter()


_ALLOWED_OLLAMA_MODEL = "gemma4:e4b"


def _normalize_llm_model(name: str) -> str:
    """Enforce the two-model contract.

    Allowed: 'gemma4:e4b' (ollama) or any OpenAI name (gpt-*, o*, text-*,
    openai/*). OpenAI bare names get an 'openai/' prefix so they hit the
    wildcard route in litellm. Everything else raises 400.
    """
    n = (name or "").strip()
    if not n:
        raise HTTPException(status_code=400, detail="llm_model required")
    if n == _ALLOWED_OLLAMA_MODEL:
        return n
    bare = n[len("openai/"):] if n.startswith("openai/") else n
    lower = bare.lower()
    if (
        lower.startswith(("gpt-", "o1", "o3", "o4", "text-"))
        or lower in {"gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-5"}
    ):
        return f"openai/{bare}"
    raise HTTPException(
        status_code=400,
        detail=(
            f"llm_model {n!r} not allowed; use 'gemma4:e4b' or an OpenAI model"
        ),
    )


class AskRequest(BaseModel):
    question: str
    conv_idx: int | None = None
    question_id: str | None = None
    backend: str  # which memory backend to recall from (and propagate to peers)
    llm_model: str  # which LLM to use — required; agents have no default
    # Per-agent LLM map carried through from the toxiproxy-proxy. Agent-1 uses
    # `models[CFG.agent_id]` (== llm_model) itself and forwards `models` on
    # peer fan-outs so the contract is preserved end-to-end.
    models: dict[str, str] | None = None
    # Per-request peer-latency policy: the agent service measures outbound RTT
    # before invoking the LLM and OMITS the ask_peers tool from the LLM's
    # toolset when measured > threshold. 0 (or None) = always allow ask_peers.
    peer_latency_threshold_ms: float | None = None


class PeerAskRequest(BaseModel):
    question: str
    conv_idx: int | None = None
    question_id: str | None = None
    backend: str  # propagated from the originating /ask
    models: dict[str, str] | None = None  # full per-agent LLM map


class LoadRequest(BaseModel):
    items: list[dict[str, Any]]
    backend: str  # which memory backend to ingest into


def _check_backend(name: str) -> None:
    if name not in BACKEND_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"unknown backend {name!r}; expected one of {list(BACKEND_NAMES)}",
        )


@router.get("/health")
async def health(request: Request) -> dict:
    backends = sorted(getattr(request.app.state, "agent", None)._backends.keys()) \
        if getattr(request.app.state, "agent", None) else []
    return {"status": "ok", "agent": CFG.agent_id, "backends": backends}


@router.get("/admin/jitter")
async def admin_jitter() -> dict:
    return {
        "agent": CFG.agent_id,
        "measured_latency_ms": await measured_peer_latency_ms(),
        "threshold_ms": CFG.help_jitter_threshold_ms,
    }


@router.post("/admin/load")
async def admin_load(req: LoadRequest, request: Request) -> dict:
    _check_backend(req.backend)
    agent = request.app.state.agent
    return await agent.ingest(req.items, backend=req.backend)


def _lookup_key(req: AskRequest | PeerAskRequest) -> str | int | None:
    if req.question_id is not None:
        return req.question_id
    return req.conv_idx


def _accounting(acc) -> dict[str, dict[str, float | int]]:
    """Serialise the per-request accumulator for inclusion in the metrics
    payload. Two buckets: `agent` (model name without `memory/` prefix —
    the agent's chat-tool loop) and `memory` (mem0/graphiti internals,
    which call `memory/...` aliases on the litellm proxy)."""
    return {
        "agent": {
            "input_tokens": acc.agent.input_tokens,
            "output_tokens": acc.agent.output_tokens,
            "total_tokens": acc.agent.total_tokens,
            "cost_usd": acc.agent.cost_usd,
        },
        "memory": {
            "input_tokens": acc.memory.input_tokens,
            "output_tokens": acc.memory.output_tokens,
            "total_tokens": acc.memory.total_tokens,
            "cost_usd": acc.memory.cost_usd,
        },
    }


@router.post("/peer/ask")
async def peer_ask(req: PeerAskRequest, request: Request) -> dict:
    _check_backend(req.backend)
    agent = request.app.state.agent
    acc = _bind_acc()
    snippets = await agent.peer_recall(req.question, _lookup_key(req), backend=req.backend)
    return {
        "memory": snippets,
        "backend": req.backend,
        "agent_id": CFG.agent_id,
        "accounting": _accounting(acc),
    }


@router.post("/ask")
async def ask(req: AskRequest, request: Request) -> dict:
    _check_backend(req.backend)
    agent = request.app.state.agent
    acc = _bind_acc()
    llm_model = _normalize_llm_model(req.llm_model)
    outcome = await agent.ask(
        req.question, _lookup_key(req), backend=req.backend, llm_model=llm_model,
        models=req.models,
        peer_latency_threshold_ms=req.peer_latency_threshold_ms,
    )
    return {
        "answer": outcome.answer,
        "metrics": {
            "agent_id": CFG.agent_id,
            "tokens_prompt": outcome.metrics.tokens_prompt,
            "tokens_completion": outcome.metrics.tokens_completion,
            "cost_usd": outcome.metrics.cost_usd,
            "latency_ms": outcome.metrics.latency_ms,
            "measured_jitter_ms": outcome.metrics.measured_jitter_ms,
            "threshold_ms": outcome.metrics.threshold_ms,
            "peer_help_allowed": outcome.metrics.peer_help_allowed,
            "peers_asked": outcome.metrics.peers_asked,
            "peer_memories": outcome.metrics.peer_memories,
            "own_memories": outcome.metrics.own_memories,
            "decision_reason": outcome.metrics.decision_reason,
            "backend": req.backend,
            "model": llm_model,
            "accounting": _accounting(acc),
            "peer_accounting": getattr(outcome.metrics, "peer_accounting", {}),
        },
    }
