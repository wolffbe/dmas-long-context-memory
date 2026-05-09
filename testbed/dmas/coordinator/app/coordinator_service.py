"""Coordinator: receives /ask from the benchmark and runs a small SLM
tool loop (qwen2.5:3b on Ollama) whose only job is to call the
`ask_responder` tool with the user's question. The responder owns the
real work — searching memory and composing the answer — so this loop is
essentially a routing decision that keeps the SLM in the ask path
(visible as a `coordinator`-tagged trace in langfuse).

If the SLM ignores `tool_choice="required"` and emits a direct content
reply (qwen2.5:3b's known failure mode), we fall back to calling the
responder anyway. The contract with the benchmark is "always reach the
responder"; the SLM gets the chance to route, but never the chance to
short-circuit it.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

import requests
from openai import OpenAI
from opentelemetry import trace as otel_trace

logger = logging.getLogger(__name__)
_tracer = otel_trace.get_tracer(__name__)


def _tagstr(service: str, backend: str, conv_index: int | None, mode: str | None) -> str:
    """Comma-separated `langfuse.tags` value for a root span. Lets the
    operator filter the trace list by service / framework / conv / mode
    in the langfuse UI."""
    parts = [f"service:{service}", f"memory:{backend}"]
    if conv_index is not None:
        parts.append(f"conv:{conv_index}")
    if mode:
        parts.append(f"mode:{mode}")
    return ",".join(parts)


class CoordinatorService:
    ASK_RESPONDER_TOOL = {
        "type": "function",
        "function": {
            "name": "ask_responder",
            "description": (
                "Forward the user's question to the responder, which retrieves "
                "relevant memories from the active backend and returns a "
                "grounded answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    }

    def __init__(self, memory_url: str, responder_url: str, ollama_model: str):
        self.memory_url = memory_url
        self.responder_url = responder_url
        # Pass the bare ollama model tag straight through. LiteLLM's
        # `model_list` is templated from OLLAMA_MODEL at boot, so the
        # alias matches the tag exactly — langfuse generations render as
        # e.g. `qwen2.5:3b-instruct-q4_K_M` rather than an opaque alias.
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "")
        if not self.ollama_model:
            raise RuntimeError(
                "OLLAMA_MODEL must be set (or coordinator must be constructed with "
                "ollama_model=...); empty model names cause every /ask to fail at "
                "the LiteLLM call site."
            )
        self.client = OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY", "sk-litellm-master"),
        )

    # ---- pass-through forwards (memorize / reset / warmup) -------------

    def _forward(self, path: str, body: dict[str, Any], timeout: int,
                 op: str) -> dict[str, Any]:
        try:
            r = requests.post(f"{self.memory_url}{path}", json=body, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.exception("%s forward failed", op)
            return {"status": "error", "error": str(exc)}

    def memorize(self, backend: str, conv_index: int, data: dict[str, Any],
                 trace_id: str | None = None,
                 session_id: str | None = None,
                 mode: str | None = None) -> dict[str, Any]:
        tid = trace_id or uuid.uuid4().hex
        resp = self._forward(
            "/memorize",
            {"backend": backend, "conv_index": conv_index, "data": data,
             "session_id": session_id, "mode": mode},
            timeout=3600,
            op="memorize",
        )
        resp["trace_id"] = tid
        return resp

    def reset(self, backend: str) -> dict[str, Any]:
        return self._forward("/reset", {"backend": backend}, timeout=600, op="reset")

    def warmup(self, backend: str, conv_index: int) -> dict[str, Any]:
        return self._forward(
            "/warmup",
            {"backend": backend, "conv_index": conv_index},
            timeout=600,
            op="warmup",
        )

    # ---- ask: SLM tool loop -------------------------------------------

    def _call_responder(self, question: str, backend: str,
                        session_date: str, trace_id: str,
                        session_id: str | None = None,
                        conv_index: int | None = None,
                        mode: str | None = None) -> dict[str, Any]:
        """Single HTTP forward to responder/respond. Returns the raw
        responder envelope (answer + retrieval stats) so the caller can
        propagate them up to the benchmark CSV row. `trace_id` is the
        langfuse trace ID we want every responder-side openai call to
        share with this coordinator request; `session_id` groups the
        responder's detached `responder.respond` trace under the same
        langfuse session as the bench's `ask.question`."""
        try:
            r = requests.post(
                f"{self.responder_url}/respond",
                json={"question": question, "backend": backend,
                      "session_date": session_date,
                      "trace_id": trace_id,
                      "session_id": session_id,
                      "conv_index": conv_index,
                      "mode": mode},
                timeout=600,
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.exception("responder forward failed")
            return {"status": "error", "error": str(exc), "answer": ""}

    @staticmethod
    def _wrap(resp: dict[str, Any], trace_id: str) -> dict[str, Any]:
        return {
            "status": resp.get("status", "success"),
            "answer": resp.get("answer", ""),
            "error": resp.get("error"),
            "memories_returned": resp.get("memories_returned", 0),
            "top_k": resp.get("top_k"),
            "search_calls": resp.get("search_calls", 0),
            "responder_context_window_tokens": resp.get("responder_context_window_tokens"),
            "responder_context_window_cost_usd": resp.get("responder_context_window_cost_usd"),
            "trace_id": trace_id,
        }

    def ask(self, question: str, backend: str, session_date: str = "",
            max_iterations: int = 3,
            session_id: str | None = None,
            conv_index: int | None = None,
            mode: str | None = None) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You route questions to the responder. ALWAYS call "
                    "the `ask_responder` tool with the user's question. "
                    "Do not answer directly."
                ),
            },
            {"role": "user", "content": question},
        ]
        attrs: dict[str, Any] = {
            "dmas.backend": backend,
            "dmas.session_date": session_date or "",
            "input": question[:8000],
        }
        if session_id:
            attrs["langfuse.session.id"] = session_id
        tags = _tagstr("coordinator", backend, conv_index, mode)
        if tags:
            attrs["langfuse.tags"] = tags
        with _tracer.start_as_current_span(
            "coordinator.ask",
            attributes=attrs,
        ) as sp:
            trace_id = format(sp.get_span_context().trace_id, "032x")
            result = self._ask_loop(messages, question, backend, session_date,
                                    max_iterations, trace_id,
                                    session_id, conv_index, mode)
            try:
                sp.set_attribute(
                    "output",
                    (result.get("answer") or "")[:8000],
                )
            except Exception:
                pass
            return result

    def _ask_loop(self, messages: list[dict[str, Any]], question: str,
                  backend: str, session_date: str,
                  max_iterations: int, trace_id: str,
                  session_id: str | None = None,
                  conv_index: int | None = None,
                  mode: str | None = None) -> dict[str, Any]:
        try:
            for _ in range(max_iterations):
                resp = self.client.chat.completions.create(
                    model=self.ollama_model,
                    messages=messages,
                    tools=[self.ASK_RESPONDER_TOOL],
                    tool_choice="required",
                    temperature=0,
                )
                msg = resp.choices[0].message
                if msg.tool_calls:
                    tc = msg.tool_calls[0]
                    args = tc.function.arguments
                    if isinstance(args, str):
                        args = json.loads(args or "{}")
                    forwarded = args.get("question") or question
                    return self._wrap(
                        self._call_responder(forwarded, backend, session_date, trace_id,
                                             session_id, conv_index, mode),
                        trace_id,
                    )
                # SLM ignored tool_choice="required" and emitted plain
                # content (qwen2.5:3b is non-deterministic under that
                # constraint). Push it back with a stronger nudge; if
                # that still fails, fall back below.
                messages.append({"role": "assistant", "content": msg.content or ""})
                messages.append({
                    "role": "user",
                    "content": (
                        "You must call the `ask_responder` tool now with "
                        "the original question. Do not write a direct answer."
                    ),
                })

            # Safety net: SLM never produced a tool call. The benchmark's
            # contract is that the responder is always reached.
            logger.warning("SLM did not emit ask_responder tool call after %d iterations; "
                           "falling back to direct call", max_iterations)
            return self._wrap(
                self._call_responder(question, backend, session_date, trace_id,
                                     session_id, conv_index, mode),
                trace_id,
            )

        except Exception as exc:
            logger.exception("ask failed")
            return {"status": "error", "error": str(exc), "answer": "",
                    "memories_returned": 0, "top_k": None, "search_calls": 0,
                    "responder_context_window_tokens": None,
                    "responder_context_window_cost_usd": None,
                    "trace_id": trace_id}
