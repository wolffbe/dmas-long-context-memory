import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config import CFG
from app.jitter import measured_peer_latency_ms
from app.langfuse_tracer import trace
from app.llm import chat_tools
from app.memory.base import MemoryBackend
from app.metrics import (
    ask_total,
    help_decision_total,
    peer_help_gate_total,
    peer_latency_threshold_ms as _peer_threshold_gauge,
    peer_request_duration_seconds,
    peer_request_failures_total,
    src,
)


SYSTEM_PROMPT_TEMPLATE = (
    "You are agent {agent_id} in a 3-agent distributed memory system. Each /ask call "
    "hits one agent; agents may query the other two for additional memory snippets.\n\n"
    "Answer the user's question using ONLY memory snippets — your own and (optionally) "
    "those returned by peers. Do not use outside knowledge. If the memories do not "
    "suffice, reply exactly: 'I don't know based on the given memories.' Simple temporal "
    "or arithmetic reasoning over memory facts is allowed.\n\n"
    "Tools available this turn:\n"
    "{tools_clause}\n"
    "Procedure for every question:\n"
    "  1. If your local memories already answer the question, answer directly — call no tools.\n"
    "  2. Otherwise, you may call list_peers to see which other agents exist.\n"
    "  3. {ask_clause}\n"
    "Reply in plain text — no JSON, no commentary about tool use."
)


LIST_PEERS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "list_peers",
        "description": (
            "Return the IDs of the other agents in the system that you may ask for "
            "additional memory snippets."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


ASK_PEERS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "ask_peers",
        "description": (
            "Fan out the original question to the peer agents and return their "
            "memory snippets."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


MAX_TOOL_ITERS = 6


@dataclass
class AskMetrics:
    tokens_prompt: int = 0
    tokens_completion: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    measured_jitter_ms: float = 0.0
    threshold_ms: float = 0.0
    peer_help_allowed: bool = True
    peers_asked: bool = False
    peer_memories: int = 0
    own_memories: int = 0
    decision_reason: str = ""
    # Per-peer accounting picked up from /peer/ask responses during fan-out.
    # Shape: {"<peer-agent-id>": {"agent": {tokens/cost}, "memory": {tokens/cost}}}
    peer_accounting: dict = field(default_factory=dict)


@dataclass
class AskOutcome:
    answer: str
    metrics: AskMetrics = field(default_factory=AskMetrics)


class AgentService:
    def __init__(self, backends: dict[str, MemoryBackend]) -> None:
        self._backends = backends
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(30.0))

    def _backend(self, name: str) -> MemoryBackend:
        try:
            return self._backends[name]
        except KeyError:
            raise ValueError(f"unknown backend: {name!r}; have {sorted(self._backends)}")

    async def close(self) -> None:
        await self._http.aclose()
        for b in self._backends.values():
            try:
                await b.close()
            except Exception:
                pass

    async def ingest(self, items: list[dict], backend: str) -> dict:
        b = self._backend(backend)
        added, failed = 0, 0
        details: list[dict] = []
        for it in items:
            try:
                r = await b.ingest(it)
                added += 1
                details.append({"ok": True, "result": r})
            except Exception as exc:
                failed += 1
                details.append({"ok": False, "error": str(exc)[:200]})
        return {"added": added, "failed": failed, "details": details}

    async def peer_recall(
        self, question: str, lookup_key: str | int | None, backend: str,
    ) -> list[str]:
        return await self._backend(backend).recall(question, lookup_key)

    async def ask(
        self, question: str, lookup_key: str | int | None, backend: str,
        llm_model: str, models: dict[str, str] | None = None,
        peer_latency_threshold_ms: float | None = None,
    ) -> AskOutcome:
        ask_total.labels(agent=CFG.agent_id).inc()
        t0 = time.monotonic()
        threshold = (
            float(peer_latency_threshold_ms)
            if peer_latency_threshold_ms is not None
            else float(CFG.help_jitter_threshold_ms)
        )
        m = AskMetrics(threshold_ms=threshold)
        _peer_threshold_gauge.labels(src=CFG.agent_id).set(threshold)
        store = self._backend(backend)
        model = llm_model

        # Measure outbound latency BEFORE the LLM loop and decide whether peer
        # help is allowed for this request. The agent never sees the latency
        # number — it only sees the toolset that results from this gate.
        measured_latency_ms = await measured_peer_latency_ms() if CFG.peers else 0.0
        m.measured_jitter_ms = measured_latency_ms
        peer_help_allowed = (
            bool(CFG.peers)
            and (threshold <= 0 or measured_latency_ms <= threshold)
        )
        m.peer_help_allowed = peer_help_allowed
        peer_help_gate_total.labels(
            agent=CFG.agent_id,
            decision="allow" if peer_help_allowed else "block",
        ).inc()
        tools = self._build_tools(peer_help_allowed)

        with trace(
            name="ask",
            input=question,
            lookup_key=lookup_key,
            backend=backend,
            model=model,
        ) as t:
            own = await store.recall(question, lookup_key)
            m.own_memories = len(own)

            system = self._build_system_prompt(peer_help_allowed)
            user_msg = (
                f"Local memories:\n{self._format_block(own)}\n\n"
                f"Question: {question}"
            )
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ]

            answer = ""
            for _ in range(MAX_TOOL_ITERS):
                turn = await chat_tools(messages, tools=tools, model=model)
                m.tokens_prompt += turn.prompt_tokens
                m.tokens_completion += turn.completion_tokens
                m.cost_usd += turn.cost_usd

                asst: dict[str, Any] = {"role": "assistant", "content": turn.content}
                if turn.tool_calls:
                    asst["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments_raw},
                        }
                        for tc in turn.tool_calls
                    ]
                messages.append(asst)

                if not turn.tool_calls:
                    answer = turn.content or ""
                    break

                for tc in turn.tool_calls:
                    result = await self._dispatch_tool(tc.name, question, lookup_key, backend, m, models)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
            else:
                answer = answer or "I don't know based on the given memories."

            m.decision_reason = (
                f"latency={m.measured_jitter_ms:.0f}ms "
                f"threshold={m.threshold_ms:.0f}ms "
                f"peer_help_allowed={m.peer_help_allowed} "
                f"peers_asked={m.peers_asked}"
            )
            help_decision_total.labels(
                agent=CFG.agent_id,
                decision="ask" if m.peers_asked else "skip",
            ).inc()

            m.latency_ms = (time.monotonic() - t0) * 1000.0
            if t is not None:
                try:
                    t.update(
                        output=answer,
                        metadata={
                            "tokens_prompt": m.tokens_prompt,
                            "tokens_completion": m.tokens_completion,
                            "cost_usd": m.cost_usd,
                            "latency_ms": m.latency_ms,
                            "measured_latency_ms": m.measured_jitter_ms,
                            "peer_help_allowed": m.peer_help_allowed,
                            "peers_asked": m.peers_asked,
                            "peer_memories": m.peer_memories,
                            "own_memories": m.own_memories,
                        },
                    )
                except Exception:
                    pass
            return AskOutcome(answer=answer, metrics=m)

    @staticmethod
    def _build_tools(peer_help_allowed: bool) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = [LIST_PEERS_TOOL]
        if peer_help_allowed:
            tools.append(ASK_PEERS_TOOL)
        return tools

    @staticmethod
    def _build_system_prompt(peer_help_allowed: bool) -> str:
        if peer_help_allowed:
            tools_clause = (
                "  - list_peers(): returns the IDs of the other agents.\n"
                "  - ask_peers(): queries the peer agents for snippets relevant to the question."
            )
            ask_clause = (
                "If your local memories are thin, call ask_peers once and incorporate any "
                "returned snippets into your answer."
            )
        else:
            tools_clause = (
                "  - list_peers(): returns the IDs of the other agents.\n"
                "  (ask_peers is unavailable this turn — you cannot query other agents.)"
            )
            ask_clause = (
                "You cannot ask peers this turn — answer from local memory only, or reply "
                "'I don't know based on the given memories.' if local memory is insufficient."
            )
        return SYSTEM_PROMPT_TEMPLATE.format(
            agent_id=CFG.agent_id,
            tools_clause=tools_clause,
            ask_clause=ask_clause,
        )

    async def _dispatch_tool(
        self, name: str, question: str, lookup_key: str | int | None,
        backend: str, m: AskMetrics, models: dict[str, str] | None = None,
    ) -> str:
        if name == "list_peers":
            return json.dumps({"peers": [_peer_id(p) for p in CFG.peers]})
        if name == "ask_peers":
            if not CFG.peers:
                return json.dumps({"snippets": [], "error": "no peers configured"})
            snippets = await self._fan_out(question, lookup_key, backend, models, m=m)
            m.peers_asked = True
            m.peer_memories = len(snippets)
            return json.dumps({"snippets": snippets})
        return json.dumps({"error": f"unknown tool: {name}"})

    async def _fan_out(
        self, question: str, lookup_key: str | int | None, backend: str,
        models: dict[str, str] | None = None,
        m: AskMetrics | None = None,
    ) -> list[str]:
        async def _call(peer_url: str, dst: str) -> list[str]:
            url = f"{peer_url}/peer/ask"
            payload: dict[str, Any] = {"question": question, "backend": backend}
            if models is not None:
                payload["models"] = models
            if isinstance(lookup_key, str):
                payload["question_id"] = lookup_key
            else:
                payload["conv_idx"] = lookup_key
            with peer_request_duration_seconds.labels(src=src(), dst=dst).time():
                try:
                    r = await self._http.post(
                        url,
                        json=payload,
                        headers={"x-trace-id": str(uuid.uuid4())},
                    )
                    r.raise_for_status()
                    data = r.json()
                    if m is not None:
                        peer_id = data.get("agent_id") or dst
                        acc = data.get("accounting") or {}
                        if acc:
                            m.peer_accounting[str(peer_id)] = acc
                    return list(data.get("memory", []))
                except Exception:
                    peer_request_failures_total.labels(src=src(), dst=dst).inc()
                    return []

        tasks = []
        for peer in CFG.peers:
            dst = _peer_id(peer)
            tasks.append(_call(peer, dst))
        results = await asyncio.gather(*tasks)
        out: list[str] = []
        for r in results:
            out.extend(r)
        return out

    @staticmethod
    def _format_block(snippets: list[str]) -> str:
        cap = CFG.max_context_memories
        clipped = snippets[:cap]
        if not clipped:
            return "(no relevant memories)"
        return "\n".join(f"- {s}" for s in clipped)


def _peer_id(peer_url: str) -> str:
    # Peer URLs route through THIS agent's toxiproxy on a per-destination port:
    #   18001 → agent-1, 18002 → agent-2, 18003 → agent-3.
    # So the destination is the LAST digit of the proxy port, not the host.
    for token in peer_url.replace("/", " ").replace(":", " ").split():
        if token.isdigit() and len(token) >= 4 and token.startswith("180"):
            return token[-1]
    # Direct-host fallback (e.g. http://agent-2:8000)
    for token in peer_url.replace("/", " ").replace(":", " ").split():
        if token.startswith("agent-") and len(token) > 6:
            return token.split("-", 1)[1]
    return peer_url
