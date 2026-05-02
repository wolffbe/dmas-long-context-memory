"""Responder: owns the LLM tool loop. Coordinator forwards a question +
backend; responder uses gpt-4o-mini with a `search_memories` tool that
hits memory:/remember, then composes the final answer.

Why the loop lives here (not in coordinator):
  - gpt-4o-mini is reliable under tool_choice; qwen2.5:3b is not, so
    keeping the loop on the SLM caused the responder to be bypassed
    whenever the SLM ignored the tool requirement.
  - Every /respond call now produces a langfuse trace tagged `responder`,
    so the operator can see the prompt + completion path.
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


class ResponderService:
    SEARCH_TOOL = {
        "type": "function",
        "function": {
            "name": "search_memories",
            "description": (
                "Retrieve relevant context from the active memory backend. "
                "Always call this before answering."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }

    def __init__(self, model: str, memory_url: str, max_iterations: int = 4):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.memory_url = memory_url
        self.max_iterations = max_iterations

    # ---- tool implementation ------------------------------------------

    def _search_memories(self, backend: str, query: str) -> tuple[str, dict[str, Any]]:
        """Return (text_for_LLM, meta) where meta carries `count` and
        `top_k` so the responder can surface them up the chain."""
        try:
            r = requests.post(
                f"{self.memory_url}/remember",
                json={"backend": backend, "question": query},
                timeout=600,
            )
            r.raise_for_status()
            data = r.json()
            if data.get("status") == "error":
                return "Error searching memories.", {"count": 0, "top_k": None}
            mem = data.get("memory") or ""
            meta = {"count": data.get("count", 0), "top_k": data.get("top_k")}
            return (mem or "No relevant memories found."), meta
        except Exception as exc:
            return f"Memory search error: {exc}", {"count": 0, "top_k": None}

    # ---- main entrypoint ----------------------------------------------

    def respond(self, question: str, backend: str) -> dict[str, Any]:
        logger.info("respond backend=%s q=%r", backend, question[:80])
        # Tag every responder-side LLM call with the active backend so
        # langfuse traces share the `memory:<backend>` label across the
        # coordinator/responder/memory chain.
        with active_backend(backend):
            return self._respond_inner(question, backend)

    def _respond_inner(self, question: str, backend: str) -> dict[str, Any]:
        # Accumulate retrieval stats across every search_memories call in
        # this loop. `count` sums items returned; `top_k` records the
        # configured ceiling (same on every call for a given backend).
        memories_returned = 0
        observed_top_k: int | None = None
        search_calls = 0
        # `responder_context_tokens` records the prompt_tokens of the
        # completion that produced the final answer — the actual context
        # length the answering LLM consumed for this question. Reported
        # back to the benchmark so efficiency analyses can normalise
        # accuracy by how much retrieved context the responder had to
        # process. Updated on every completion; the value at return time
        # is the one tied to the answer.
        last_prompt_tokens: int | None = None
        ctx_label = "RAW CONVERSATION JSON" if backend == "full_context" else "CONTEXT"

        system_msg = (
            "You are a helpful assistant answering questions about a long conversation.\n"
            f"Use the `search_memories` tool to retrieve {ctx_label.lower()}, then answer "
            "using ONLY what it returns.\n"
            "You may do simple reasoning and convert relative time expressions to dates "
            "when timestamps are present. Do not invent facts not implied by the context. "
            'If the context is insufficient, reply exactly: "I don\'t know based on the given context."'
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question},
        ]

        def _bundle(answer: str, status: str = "success", error: str | None = None) -> dict[str, Any]:
            return {
                "status": status,
                "answer": answer,
                "error": error,
                "memories_returned": memories_returned,
                "top_k": observed_top_k,
                "search_calls": search_calls,
                "responder_context_tokens": last_prompt_tokens,
            }

        try:
            for i in range(self.max_iterations):
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=[self.SEARCH_TOOL],
                    tool_choice="auto",
                    temperature=0,
                )
                if resp.usage is not None:
                    last_prompt_tokens = resp.usage.prompt_tokens
                msg = resp.choices[0].message
                if not msg.tool_calls:
                    return _bundle((msg.content or "").strip())
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in msg.tool_calls
                    ],
                })
                for tc in msg.tool_calls:
                    args = tc.function.arguments
                    if isinstance(args, str):
                        args = json.loads(args or "{}")
                    if tc.function.name == "search_memories":
                        result, meta = self._search_memories(backend, args.get("query", question))
                        search_calls += 1
                        memories_returned += int(meta.get("count") or 0)
                        if observed_top_k is None and meta.get("top_k") is not None:
                            observed_top_k = meta["top_k"]
                    else:
                        result = f"Unknown tool: {tc.function.name}"
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

            # Loop exhausted without a final answer — force one last
            # completion with no tools so we always return content.
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages + [{
                    "role": "user",
                    "content": "Provide your final answer now using only the context already gathered.",
                }],
                temperature=0,
            )
            if resp.usage is not None:
                last_prompt_tokens = resp.usage.prompt_tokens
            return _bundle((resp.choices[0].message.content or "").strip())

        except Exception as exc:
            logger.exception("respond failed")
            return _bundle("", status="error", error=str(exc))
