"""Compact failure formatting for ingestion handlers.

Each backend's load loop catches `Exception`, stores `str(exc)`, and
truncates to ~300 chars in the CSV. Bare `str(exc)` loses the traceback,
which makes errors raised deep inside a third-party library (cognee,
graphiti, mem0) untraceable from the CSV alone. `exc_trace` preserves
the deepest frame's file:line and function name so the failing site
survives the CSV cell width — and for OpenAI SDK errors (which LiteLLM
proxy surfaces on any backend failure) additionally inlines the
upstream `status_code` and JSON `body`, so LiteLLM-side errors are
visible in the CSV without having to grep `docker logs litellm`.
"""
from __future__ import annotations

import traceback


def exc_trace(exc: BaseException) -> str:
    """`ExcType: msg @ file:line in func` plus either:
       - `| status=<code> body=<...>` when exc looks like an OpenAI API
         error (.status_code or .body present — covers RateLimitError,
         APIStatusError, APIConnectionError, BadRequestError, etc.); or
       - `| tb_tail: <last frames>` otherwise.
    """
    tb = exc.__traceback__
    while tb and tb.tb_next:
        tb = tb.tb_next
    summary = f"{type(exc).__name__}: {exc}"
    if tb is not None:
        co = tb.tb_frame.f_code
        summary += f" @ {co.co_filename}:{tb.tb_lineno} in {co.co_name}"

    # Duck-typed check so we don't import openai here. Any exception
    # carrying these attributes (the entire openai.APIError tree does)
    # gets the upstream context inlined instead of the local tb_tail —
    # the local tb_tail for an SDK error is just the SDK's plumbing,
    # whereas .body carries the litellm proxy's actual error JSON.
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    if status is not None or body is not None:
        body_repr = str(body)[:160] if body is not None else ""
        return f"{summary} | status={status} body={body_repr}"

    formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return f"{summary} | tb_tail: {formatted[-180:]}"
