"""Per-request token + cost accumulator.

Every LLM/embedding call from this agent goes through the local litellm
proxy via the openai SDK — the agent's own chat-tool loop (`llm.py`), mem0's
fact extraction & embedder, and graphiti's entity/edge extraction & embedder
all share the same `openai.OpenAI` / `openai.AsyncOpenAI` classes.

We monkey-patch those classes' `chat.completions.create`,
`completions.create`, and `embeddings.create` to:
  1. After a successful response, pull `usage.prompt_tokens` /
     `usage.completion_tokens` and `_hidden_params.response_cost` (which
     litellm computes per-call from its pricing tables — local/ollama
     models are priced at $0).
  2. Record the call into a contextvar-scoped bucket. Calls whose model
     name starts with `memory/` (mem0/graphiti use that prefix) go into
     the `memory` bucket; everything else goes into `agent`.

The /ask route binds a fresh accumulator at the start of the request and
reads totals back at the end. Concurrent /ask calls run in independent
asyncio tasks and contextvars cleanly inherit per task, so per-request
isolation is guaranteed without explicit thread-locals.
"""
from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Any

import openai


@dataclass
class Bucket:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class Accumulator:
    agent: Bucket = field(default_factory=Bucket)
    memory: Bucket = field(default_factory=Bucket)

    def record(self, model: str, prompt_tokens: int, completion_tokens: int, cost: float) -> None:
        bucket = self.memory if (model or "").startswith("memory/") else self.agent
        bucket.input_tokens += int(prompt_tokens or 0)
        bucket.output_tokens += int(completion_tokens or 0)
        bucket.cost_usd += float(cost or 0.0)


_current: contextvars.ContextVar[Accumulator | None] = contextvars.ContextVar(
    "llm_accumulator", default=None,
)

# Re-entry guard: `with_raw_response.create()` internally calls `Completions.create`,
# which is now our wrapper — without this flag we'd recurse. The flag is per-task
# (contextvar), so concurrent /ask calls don't interfere.
_in_patch: contextvars.ContextVar[bool] = contextvars.ContextVar("in_patch", default=False)


def bind() -> Accumulator:
    """Install a fresh accumulator on the current context. Returns it so the
    caller can read totals after the request finishes."""
    acc = Accumulator()
    _current.set(acc)
    return acc


def current() -> Accumulator | None:
    return _current.get()


def _extract(resp: Any, headers: Any) -> tuple[str, int, int, float]:
    """Pull (model, prompt_tokens, completion_tokens, cost_usd) from a parsed
    response + raw HTTP headers. Cost comes from the `x-litellm-response-cost`
    header — LiteLLM strips `_hidden_params` from the JSON body in current
    versions, but always sets the header. Local/ollama models report 0 there
    (matches the yaml's `input_cost_per_token: 0`)."""
    model = getattr(resp, "model", "") or ""
    usage = getattr(resp, "usage", None)
    pt = getattr(usage, "prompt_tokens", 0) if usage is not None else 0
    ct = getattr(usage, "completion_tokens", 0) if usage is not None else 0
    cost = 0.0
    if headers is not None:
        try:
            cost = float(headers.get("x-litellm-response-cost") or 0.0)
        except (TypeError, ValueError):
            cost = 0.0
    return model, pt or 0, ct or 0, cost


def _wrap_method(owner_cls):
    """Patch `owner_cls.create` to call through `.with_raw_response.create`
    so we can read the LiteLLM cost header, then return the parsed response
    (matching the original signature)."""
    original_create = owner_cls.create

    def wrapped(self, *args, **kwargs):
        # The raw-response path calls back into `create`, which is now this
        # wrapper. The guard short-circuits the inner call to the original.
        if _in_patch.get():
            return original_create(self, *args, **kwargs)
        token = _in_patch.set(True)
        try:
            raw = self.with_raw_response.create(*args, **kwargs)
            resp = raw.parse()
        finally:
            _in_patch.reset(token)
        acc = _current.get()
        if acc is not None:
            try:
                acc.record(*_extract(resp, raw.http_response.headers))
            except Exception:
                pass
        return resp

    owner_cls.create = wrapped


def _wrap_async_method(owner_cls):
    original_create = owner_cls.create

    async def wrapped(self, *args, **kwargs):
        if _in_patch.get():
            return await original_create(self, *args, **kwargs)
        token = _in_patch.set(True)
        try:
            raw = await self.with_raw_response.create(*args, **kwargs)
            resp = raw.parse()
        finally:
            _in_patch.reset(token)
        acc = _current.get()
        if acc is not None:
            try:
                acc.record(*_extract(resp, raw.http_response.headers))
            except Exception:
                pass
        return resp

    owner_cls.create = wrapped


def install() -> None:
    """Patch the OpenAI SDK classes used by every caller in this process so
    every chat/embedding call records into the current accumulator. Idempotent
    — safe to call multiple times. Must be called BEFORE mem0/graphiti
    construct their internal clients."""
    if getattr(install, "_done", False):
        return
    # Sync client (mem0 uses this for fact extraction)
    _wrap_method(openai.resources.chat.completions.Completions)
    _wrap_method(openai.resources.embeddings.Embeddings)
    # Async client (the agent's own chat-tool loop, graphiti async paths)
    _wrap_async_method(openai.resources.chat.completions.AsyncCompletions)
    _wrap_async_method(openai.resources.embeddings.AsyncEmbeddings)
    install._done = True  # type: ignore[attr-defined]
