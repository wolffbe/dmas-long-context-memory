"""Process-wide patch: every `openai.OpenAI`/`AsyncOpenAI` chat /
completion / embedding / responses call gets
`extra_body={"metadata": {"tags": [...]}}` injected so litellm forwards
those tags into the langfuse trace.

LiteLLM doesn't honour the `langfuse-tags` HTTP header (it reads tags
from the request body's `metadata.tags`), so a default header doesn't
work; the body merge does.

Tag composition:
  - base tag: `LANGFUSE_SERVICE_TAG` env (default `memory`)
  - sub tag : if `set_active_backend("mem0"|"graphiti"|"rag"|"cognee")` was called
              for the current async context, also append `<base>:<backend>`

Why a contextvar: a single uvicorn process serves all four backends and
their LLM calls happen deep inside library code (mem0, graphiti) we
don't control. A contextvar set at the request entry point flows into
every nested coroutine / sync call without threading it through args.

Imported once at the top of `routes.py` *before* the services so library
imports happen after the patch is in place.
"""
from __future__ import annotations

import contextvars
import logging
import os
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

_BASE_TAG = os.getenv("LANGFUSE_SERVICE_TAG", "memory")
_active_backend: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "langfuse_active_backend", default=None
)


def set_active_backend(backend: str | None) -> contextvars.Token:
    """Set the current backend tag for the active async context. Returns a
    token the caller passes to `_active_backend.reset(token)` if needed."""
    return _active_backend.set(backend)


@contextmanager
def active_backend(backend: str | None):
    """Context manager wrapper around `set_active_backend`. The contextvar
    is reset on exit so concurrent requests don't bleed tags."""
    tok = _active_backend.set(backend)
    try:
        yield
    finally:
        _active_backend.reset(tok)


def _current_tags() -> list[str]:
    """Tags emitted on every traced openai call. Composed from:
      - `_BASE_TAG`        — which service is making the call
                             (coordinator|responder|memory)
      - `memory:<backend>` — which memory framework the request is
                             working with (mem0|graphiti|rag|full_context).
                             Fixed `memory:` prefix (not `{_BASE_TAG}:`)
                             so the same tag groups every trace involved
                             with a backend across all three services.
    """
    tags: list[str] = [_BASE_TAG]
    sub = _active_backend.get()
    if sub:
        tags.append(f"memory:{sub}")
    return tags


def _inject(extra_body: dict[str, Any] | None) -> dict[str, Any]:
    eb = dict(extra_body) if extra_body else {}
    md = dict(eb.get("metadata") or {})
    existing = list(md.get("tags") or [])
    for t in _current_tags():
        if t not in existing:
            existing.append(t)
    md["tags"] = existing
    eb["metadata"] = md
    return eb


def _wrap(method):
    def wrapper(self, *args, extra_body: dict[str, Any] | None = None, **kwargs):
        return method(self, *args, extra_body=_inject(extra_body), **kwargs)
    wrapper.__wrapped__ = method
    return wrapper


def _wrap_async(method):
    async def wrapper(self, *args, extra_body: dict[str, Any] | None = None, **kwargs):
        return await method(self, *args, extra_body=_inject(extra_body), **kwargs)
    wrapper.__wrapped__ = method
    return wrapper


def _patch():
    from openai.resources.chat.completions import Completions, AsyncCompletions
    from openai.resources.embeddings import Embeddings, AsyncEmbeddings
    from openai.resources.responses import Responses, AsyncResponses

    Completions.create = _wrap(Completions.create)
    AsyncCompletions.create = _wrap_async(AsyncCompletions.create)
    Embeddings.create = _wrap(Embeddings.create)
    AsyncEmbeddings.create = _wrap_async(AsyncEmbeddings.create)
    Responses.create = _wrap(Responses.create)
    AsyncResponses.create = _wrap_async(AsyncResponses.create)


def _patch_threadpool():
    """Make `ThreadPoolExecutor.submit` propagate contextvars into the
    worker thread. Stdlib's submit doesn't do this — vanilla OS threads
    don't inherit a parent's context — so any contextvar set by the
    request handler is invisible to code mem0 runs via its internal
    executor. We wrap the callable in `copy_context().run(...)` so the
    worker sees the snapshot the caller had at submit time.

    asyncio's `to_thread` and `loop.run_until_complete` already do this,
    which is why graphiti's tags landed before this patch and mem0's
    didn't."""
    import concurrent.futures as _cf
    orig = _cf.ThreadPoolExecutor.submit

    def submit(self, fn, /, *args, **kwargs):
        ctx = contextvars.copy_context()
        return orig(self, ctx.run, fn, *args, **kwargs)

    _cf.ThreadPoolExecutor.submit = submit


_patch()
_patch_threadpool()
logger.info("openai chat/embeddings/responses patched with base tag=%s", _BASE_TAG)
