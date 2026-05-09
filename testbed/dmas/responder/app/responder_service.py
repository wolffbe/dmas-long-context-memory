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
from litellm import cost_per_token
from openai import OpenAI
from opentelemetry import trace as otel_trace
from opentelemetry.context import Context as OTelContext

logger = logging.getLogger(__name__)
_tracer = otel_trace.get_tracer(__name__)


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

    def _search_memories(self, backend: str, query: str,
                         session_id: str | None = None,
                         conv_index: int | None = None,
                         mode: str | None = None) -> tuple[str, dict[str, Any]]:
        """Return (text_for_LLM, meta) where meta carries `count` and
        `top_k` so the responder can surface them up the chain.

        Wrapped in a `responder.search` span so the operator can see, in
        Langfuse, exactly which query the responder issued and what raw
        memories came back from the backend BEFORE the LLM saw them."""
        attrs: dict[str, Any] = {
            "dmas.backend": backend,
            "input": query[:8000],
        }
        if session_id:
            attrs["langfuse.session.id"] = session_id
        tag_parts = ["service:responder", f"memory:{backend}"]
        if conv_index is not None:
            tag_parts.append(f"conv:{conv_index}")
        if mode:
            tag_parts.append(f"mode:{mode}")
        attrs["langfuse.tags"] = ",".join(tag_parts)
        with _tracer.start_as_current_span(
            "responder.search",
            attributes=attrs,
        ) as sp:
            try:
                r = requests.post(
                    f"{self.memory_url}/remember",
                    json={"backend": backend, "question": query,
                          "session_id": session_id, "conv_index": conv_index,
                          "mode": mode},
                    timeout=600,
                )
                r.raise_for_status()
                data = r.json()
                if data.get("status") == "error":
                    sp.set_attribute("output", "Error searching memories.")
                    return "Error searching memories.", {"count": 0, "top_k": None}
                mem = data.get("memory") or ""
                meta = {"count": data.get("count", 0), "top_k": data.get("top_k")}
                # `tool_message` is the exact string the responder hands
                # back to the LLM as the tool result — i.e. the raw
                # memories with the empty-result / error fallbacks the
                # responder substitutes in. Surfacing both `memories`
                # (raw from the backend) and `tool_message` (what the
                # LLM actually sees) lets the operator audit what the
                # responder added to the backend's output.
                tool_message = mem or "No relevant memories found."
                try:
                    sp.set_attribute("output", json.dumps({
                        "top_k": meta.get("top_k"),
                        "count": int(meta.get("count") or 0),
                        "memories": (mem or "")[:6000],
                        "tool_message": tool_message[:6000],
                    })[:16000])
                    sp.set_attribute("dmas.memories_returned", int(meta.get("count") or 0))
                    if meta.get("top_k") is not None:
                        sp.set_attribute("dmas.top_k", int(meta["top_k"]))
                except Exception:
                    pass
                return tool_message, meta
            except Exception as exc:
                try:
                    sp.set_attribute("dmas.error", str(exc)[:300])
                except Exception:
                    pass
                return f"Memory search error: {exc}", {"count": 0, "top_k": None}

    # ---- main entrypoint ----------------------------------------------

    def respond(self, question: str, backend: str,
                session_date: str = "",
                trace_id: str | None = None,
                session_id: str | None = None,
                conv_index: int | None = None,
                mode: str | None = None) -> dict[str, Any]:
        logger.info("respond backend=%s session_date=%r session_id=%s q=%r",
                    backend, session_date, session_id, question[:80])
        # Detached root span: each /respond is its own trace under the
        # same langfuse session as the bench's `ask.question`. Lets the
        # operator see the responder's memory retrieval + answer
        # composition as a separate trace in the session view (the
        # request that searched memory and produced the response to the
        # coordinator), instead of being buried inside the ask tree.
        attrs: dict[str, Any] = {
            "dmas.backend": backend,
            "dmas.session_date": session_date or "",
            "input": question[:8000],
        }
        if session_id:
            attrs["langfuse.session.id"] = session_id
        tag_parts = ["service:responder", f"memory:{backend}"]
        if conv_index is not None:
            tag_parts.append(f"conv:{conv_index}")
        if mode:
            tag_parts.append(f"mode:{mode}")
        attrs["langfuse.tags"] = ",".join(tag_parts)
        with _tracer.start_as_current_span(
            "responder.respond",
            context=OTelContext(),
            attributes=attrs,
        ) as sp:
            result, trace_extras = self._respond_inner(question, backend, session_date,
                                                       session_id=session_id,
                                                       conv_index=conv_index, mode=mode)
            try:
                # Surface every search_memories call (query + raw memories)
                # in the top-level span output so the Langfuse session view
                # shows the full responder picture — question, answer, and
                # every memory the tool returned — in one place.
                sp.set_attribute("output", json.dumps({
                    "answer": (result.get("answer") or "")[:6000],
                    "memories_returned": result.get("memories_returned", 0),
                    "search_calls": result.get("search_calls", 0),
                    "top_k": result.get("top_k"),
                    "responder_context_window_tokens": result.get("responder_context_window_tokens"),
                    "responder_context_window_cost_usd": result.get("responder_context_window_cost_usd"),
                    "retrieved_memories": trace_extras.get("retrieved_memories", []),
                    "response_prompt": trace_extras.get("response_prompt", []),
                })[:64000])
            except Exception:
                pass
            return result

    def _respond_inner(self, question: str, backend: str,
                       session_date: str,
                       session_id: str | None = None,
                       conv_index: int | None = None,
                       mode: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        # Accumulate retrieval stats across every search_memories call in
        # this loop. `count` sums items returned; `top_k` records the
        # configured ceiling (same on every call for a given backend).
        memories_returned = 0
        observed_top_k: int | None = None
        search_calls = 0
        # Capture each tool invocation's query and the raw memory text the
        # backend returned, so the top-level `responder.respond` span can
        # show the operator every memory the responder pulled in one view.
        retrieved_memories: list[dict[str, Any]] = []
        # `responder_context_window_tokens` records the prompt_tokens of
        # the completion that produced the final answer — the actual
        # context length the answering LLM consumed for this question.
        # Reported back to the benchmark so efficiency analyses can
        # normalise accuracy by how much retrieved context the responder
        # had to process. Updated on every completion; the value at
        # return time is the one tied to the answer.
        last_prompt_tokens: int | None = None
        ctx_label = "RAW CONVERSATION JSON" if backend == "full_context" else "CONTEXT"

        # The conversation took place years before gpt-4o-mini's training
        # cutoff. Without an explicit anchor, the model emits relative
        # references ("yesterday", "last week") resolved against its own
        # cutoff or even today's wallclock — yielding 2024/2026 dates for
        # a 2023 conversation. Pin the anchor here so the answer matches
        # the gold answer's date period.
        date_anchor = (
            f"The conversation took place on {session_date}. Use this as the anchor "
            "(\"today\" / \"now\") when resolving relative time references like "
            "\"yesterday\" or \"last week\".\n"
            if session_date else ""
        )
        system_msg = (
            "You are a helpful assistant answering questions about a long conversation.\n"
            f"{date_anchor}"
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

        def _bundle(answer: str, status: str = "success", error: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
            # Price the prompt slice through litellm's cost table so the
            # rate stays in lockstep with the proxy config — no separate
            # constant to drift out of sync. Returns (input_cost, output_cost);
            # we only need the input side.
            ctx_cost = None
            if last_prompt_tokens is not None:
                try:
                    ctx_cost, _ = cost_per_token(
                        model=self.model,
                        prompt_tokens=last_prompt_tokens,
                        completion_tokens=0,
                    )
                except Exception:
                    logger.exception("cost_per_token failed for model=%s", self.model)
            api_result = {
                "status": status,
                "answer": answer,
                "error": error,
                "memories_returned": memories_returned,
                "top_k": observed_top_k,
                "search_calls": search_calls,
                "responder_context_window_tokens": last_prompt_tokens,
                "responder_context_window_cost_usd": ctx_cost,
            }
            # Trace-only payload: the per-tool memory text and the full
            # prompt (system + user + tool messages) that produced the
            # answer. Surfaced on the responder.respond span for the
            # operator; not part of the API contract.
            trace_extras = {
                "retrieved_memories": retrieved_memories,
                "response_prompt": messages,
            }
            return api_result, trace_extras

        # Per-span tagging so children of `responder.respond` show up in
        # the same Langfuse session as the parent. Set on every child we
        # open inside the loop because the langfuse OTel processor reads
        # the attribute per-span rather than inheriting via context.
        child_attrs_base: dict[str, Any] = {"dmas.backend": backend}
        if session_id:
            child_attrs_base["langfuse.session.id"] = session_id
        _child_tags = ["service:responder", f"memory:{backend}"]
        if conv_index is not None:
            _child_tags.append(f"conv:{conv_index}")
        if mode:
            _child_tags.append(f"mode:{mode}")
        child_attrs_base["langfuse.tags"] = ",".join(_child_tags)

        try:
            for i in range(self.max_iterations):
                # `responder.llm` per iteration: input = the prompt the
                # LLM sees on this turn (full message list, including
                # any prior tool results); output = the LLM's reply
                # (content + any tool calls it requested). Lets the
                # operator step through "search → reply → search → reply"
                # in Langfuse.
                with _tracer.start_as_current_span(
                    "responder.llm",
                    attributes={
                        **child_attrs_base,
                        "dmas.iteration": i,
                        "input": json.dumps(messages)[:8000],
                    },
                ) as llm_sp:
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
                    try:
                        llm_sp.set_attribute("output", json.dumps({
                            "content": (msg.content or "")[:6000],
                            "tool_calls": [
                                {"name": tc.function.name, "arguments": tc.function.arguments}
                                for tc in (msg.tool_calls or [])
                            ],
                            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
                            "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
                        })[:8000])
                    except Exception:
                        pass
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
                    if tc.function.name == "search_memories":
                        raw = tc.function.arguments
                        if isinstance(raw, str):
                            try:
                                args = json.loads(raw or "{}")
                            except json.JSONDecodeError:
                                args = {}
                        else:
                            args = raw or {}
                        query = args.get("query", question)
                        result, meta = self._search_memories(
                            backend, query,
                            session_id=session_id, conv_index=conv_index, mode=mode,
                        )
                        search_calls += 1
                        memories_returned += int(meta.get("count") or 0)
                        if observed_top_k is None and meta.get("top_k") is not None:
                            observed_top_k = meta["top_k"]
                        retrieved_memories.append({
                            "query": query,
                            "count": int(meta.get("count") or 0),
                            "memories": result,
                        })
                    else:
                        result = f"Unknown tool: {tc.function.name}"
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

            # Loop exhausted without a final answer — force one last
            # completion with no tools so we always return content.
            forced_messages = messages + [{
                "role": "user",
                "content": "Provide your final answer now using only the context already gathered.",
            }]
            with _tracer.start_as_current_span(
                "responder.llm",
                attributes={
                    **child_attrs_base,
                    "dmas.iteration": self.max_iterations,
                    "dmas.forced_final": True,
                    "input": json.dumps(forced_messages)[:8000],
                },
            ) as llm_sp:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=forced_messages,
                    temperature=0,
                )
                if resp.usage is not None:
                    last_prompt_tokens = resp.usage.prompt_tokens
                final_content = (resp.choices[0].message.content or "").strip()
                try:
                    llm_sp.set_attribute("output", json.dumps({
                        "content": final_content[:6000],
                        "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
                        "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
                    })[:8000])
                except Exception:
                    pass
            return _bundle(final_content)

        except Exception as exc:
            logger.exception("respond failed")
            return _bundle("", status="error", error=str(exc))
