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
from typing import Any

import requests
from openai import OpenAI

from app.langfuse_tags import active_backend

logger = logging.getLogger(__name__)


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
        # Stable LiteLLM alias — see litellm/config.yaml. Don't read the raw
        # OLLAMA_MODEL tag here; LiteLLM doesn't expand env in model_name.
        self.ollama_model = "local-slm"
        self.client = OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY", "sk-litellm-master"),
        )

    # ---- pass-through forwards (memorize / reset / warmup) -------------

    def memorize(self, backend: str, conv_index: int, data: dict[str, Any]) -> dict[str, Any]:
        try:
            r = requests.post(
                f"{self.memory_url}/memorize",
                json={"backend": backend, "conv_index": conv_index, "data": data},
                timeout=3600,
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.exception("memorize forward failed")
            return {"status": "error", "error": str(exc)}

    def reset(self, backend: str) -> dict[str, Any]:
        try:
            r = requests.post(
                f"{self.memory_url}/reset",
                json={"backend": backend},
                timeout=600,
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.exception("reset forward failed")
            return {"status": "error", "error": str(exc)}

    def warmup(self, backend: str, conv_index: int) -> dict[str, Any]:
        try:
            r = requests.post(
                f"{self.memory_url}/warmup",
                json={"backend": backend, "conv_index": conv_index},
                timeout=600,
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.exception("warmup forward failed")
            return {"status": "error", "error": str(exc)}

    # ---- ask: SLM tool loop -------------------------------------------

    def _call_responder(self, question: str, backend: str) -> dict[str, Any]:
        """Single HTTP forward to responder/respond. Returns the raw
        responder envelope (answer + retrieval stats) so the caller can
        propagate them up to the benchmark CSV row."""
        try:
            r = requests.post(
                f"{self.responder_url}/respond",
                json={"question": question, "backend": backend},
                timeout=600,
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.exception("responder forward failed")
            return {"status": "error", "error": str(exc), "answer": ""}

    @staticmethod
    def _wrap(resp: dict[str, Any], asked: bool) -> dict[str, Any]:
        """Augment a responder envelope with the coordinator-level flag
        of whether the SLM emitted the ask_responder tool call (vs the
        fallback path)."""
        return {
            "status": resp.get("status", "success"),
            "answer": resp.get("answer", ""),
            "error": resp.get("error"),
            "coordinator_asked_responder": asked,
            "memories_returned": resp.get("memories_returned", 0),
            "top_k": resp.get("top_k"),
            "search_calls": resp.get("search_calls", 0),
            "responder_context_tokens": resp.get("responder_context_tokens"),
        }

    def ask(self, question: str, backend: str, max_iterations: int = 3) -> dict[str, Any]:
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
        # Tag every coordinator-side LLM call with the active backend so
        # langfuse traces from all three services share the same
        # `memory:<backend>` label.
        with active_backend(backend):
            return self._ask_loop(messages, question, backend, max_iterations)

    def _ask_loop(self, messages: list[dict[str, Any]], question: str,
                  backend: str, max_iterations: int) -> dict[str, Any]:
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
                    return self._wrap(self._call_responder(forwarded, backend), asked=True)
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
            return self._wrap(self._call_responder(question, backend), asked=False)

        except Exception as exc:
            logger.exception("ask failed")
            return {"status": "error", "error": str(exc), "answer": "",
                    "coordinator_asked_responder": False,
                    "memories_returned": 0, "top_k": None, "search_calls": 0,
                    "responder_context_tokens": None}
