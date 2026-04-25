"""mem0 backend.

Ingestion mirrors `benchmarks/locomo/run.py` from mem0ai/memory-benchmarks:
  - per-turn message {"role": "user"|"assistant", "content": "{speaker}: {text}{photo_tag}"}
  - role assigned by speaker_a == "user"
  - mem0.add(messages=[...], user_id=f"locomo_{conv_idx}", timestamp=session_epoch)

Round-robin in our system passes one message at a time, so each ingest call submits a
single-element messages list — that's the upstream chunking degenerate case.
"""
import asyncio
from typing import Any
from urllib.parse import urlparse

from mem0 import Memory

from app.config import CFG
from app.memory._locomo import mem0_message, session_epoch


class Mem0Backend:
    def __init__(self) -> None:
        self._mem: Memory | None = None

    async def start(self) -> None:
        u = urlparse(CFG.qdrant_url)
        # mem0's default LLM (fact extraction) and embedder both call OpenAI.
        # Route them through this agent's litellm proxy instead, under the
        # `memory/...` model aliases. Langfuse will then attribute these calls
        # to model="memory/..." so the coordinator can separate memory-upkeep
        # tokens & cost from the agent's answer-loop tokens & cost.
        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": CFG.memory_llm_model,
                    "openai_base_url": CFG.litellm_url,
                    "api_key": CFG.litellm_api_key,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": CFG.memory_embed_model,
                    "openai_base_url": CFG.litellm_url,
                    "api_key": CFG.litellm_api_key,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": CFG.mem0_collection,
                    "host": u.hostname or "qdrant",
                    "port": u.port or 6333,
                },
            },
        }
        self._mem = await asyncio.to_thread(Memory.from_config, config)

    async def close(self) -> None:
        self._mem = None

    @staticmethod
    def _user_id(item_or_lookup: dict[str, Any]) -> str:
        ds = item_or_lookup.get("dataset", "locomo")
        if ds == "longmemeval":
            qid = item_or_lookup.get("question_id") or "unknown"
            return f"longmemeval_{qid}"
        return f"locomo_{item_or_lookup.get('conv_idx', 0)}"

    async def ingest(self, item: dict[str, Any]) -> dict[str, Any]:
        assert self._mem is not None
        msg = mem0_message(item)
        if msg is None:
            return {"skipped": True, "reason": "empty"}
        ts = session_epoch(item.get("session_datetime"), dataset=item.get("dataset", "locomo"))
        user_id = self._user_id(item)
        metadata: dict[str, Any] = {
            "agent_id": CFG.agent_id,
            "dataset": item.get("dataset", "locomo"),
            "conv_idx": item.get("conv_idx"),
            "question_id": item.get("question_id"),
            "session_id": item.get("session_id"),
            "session_datetime": item.get("session_datetime"),
            "speaker": item.get("speaker") or item.get("role"),
            "to": item.get("to"),
            "dia_id": item.get("dia_id"),
        }
        # mem0 2.x dropped the `timestamp` kwarg; carry it inside metadata so
        # recall() can still sort chronologically.
        if ts is not None:
            metadata["timestamp"] = ts
        result = await asyncio.to_thread(
            self._mem.add, [msg], user_id=user_id, metadata=metadata,
        )
        return {"stored": True, "result": result}

    async def recall(self, question: str, lookup_key: str | int | None) -> list[str]:
        assert self._mem is not None
        if isinstance(lookup_key, str):
            user_id = f"longmemeval_{lookup_key}"
        else:
            user_id = f"locomo_{lookup_key if lookup_key is not None else 0}"
        # Match memory-benchmarks/benchmarks/locomo/run.py: top_k=200, no score
        # threshold, prompts formatted as "(date) memory" sorted oldest-first.
        # mem0 2.x renamed `limit` → `top_k` and moved `user_id` into `filters`,
        # also added a default `threshold=0.1` we explicitly disable.
        hits = await asyncio.to_thread(
            self._mem.search,
            query=question,
            filters={"user_id": user_id},
            top_k=CFG.search_limit,
            threshold=0.0,
        )
        results = hits.get("results", []) if isinstance(hits, dict) else hits
        top = results[: CFG.max_context_memories]

        def _ts(h: dict[str, Any]) -> float:
            md = h.get("metadata") or {}
            ts = md.get("timestamp")
            if isinstance(ts, (int, float)):
                return float(ts)
            created = h.get("created_at") or md.get("created_at")
            if isinstance(created, str):
                try:
                    from datetime import datetime
                    return datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
                except Exception:
                    return 0.0
            return 0.0

        top.sort(key=_ts)
        out: list[str] = []
        for h in top:
            mem = h.get("memory") or h.get("text") or ""
            md = h.get("metadata") or {}
            date = md.get("session_datetime") or h.get("created_at") or ""
            out.append(f"({date}) {mem}" if date else mem)
        return out
