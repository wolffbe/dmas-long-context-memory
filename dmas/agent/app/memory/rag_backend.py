"""Plain RAG backend.

Stores each turn verbatim with full LOCOMO metadata and retrieves by cosine top-k —
no LLM extraction, no graph. Embeddings produced locally via sentence-transformers.
"""
import asyncio
import uuid
from typing import Any
from urllib.parse import urlparse

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

from app.config import CFG


class RagBackend:
    def __init__(self) -> None:
        self._qd: AsyncQdrantClient | None = None
        self._embed: SentenceTransformer | None = None
        self._dim: int = 0

    async def start(self) -> None:
        u = urlparse(CFG.qdrant_url)
        self._qd = AsyncQdrantClient(host=u.hostname or "qdrant", port=u.port or 6333)
        self._embed = await asyncio.to_thread(SentenceTransformer, CFG.rag_embed_model)
        self._dim = int(self._embed.get_sentence_embedding_dimension())
        existing = await self._qd.get_collections()
        names = {c.name for c in existing.collections}
        if CFG.rag_collection not in names:
            await self._qd.create_collection(
                collection_name=CFG.rag_collection,
                vectors_config=qm.VectorParams(size=self._dim, distance=qm.Distance.COSINE),
            )

    async def close(self) -> None:
        if self._qd is not None:
            await self._qd.close()

    def _vec(self, text: str) -> list[float]:
        assert self._embed is not None
        return self._embed.encode(text, normalize_embeddings=True).tolist()

    async def ingest(self, item: dict[str, Any]) -> dict[str, Any]:
        assert self._qd is not None
        ds = item.get("dataset", "locomo")
        if ds == "longmemeval":
            speaker = (item.get("role") or "").strip()
            text = (item.get("content") or "").strip()
        else:
            speaker = (item.get("speaker") or "").strip()
            text = (item.get("text") or "").strip()
        if not text:
            return {"skipped": True, "reason": "empty"}
        body = f"{speaker}: {text}"
        vec = await asyncio.to_thread(self._vec, body)
        payload = {
            "agent_id": CFG.agent_id,
            "dataset": ds,
            "conv_idx": item.get("conv_idx"),
            "question_id": item.get("question_id"),
            "session_id": item.get("session_id"),
            "session_datetime": item.get("session_datetime"),
            "speaker": speaker,
            "to": item.get("to"),
            "dia_id": item.get("dia_id"),
            "text": text,
            "blip_caption": item.get("blip_caption"),
        }
        await self._qd.upsert(
            collection_name=CFG.rag_collection,
            points=[qm.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload)],
        )
        return {"stored": True}

    async def recall(self, question: str, lookup_key: str | int | None) -> list[str]:
        assert self._qd is not None
        vec = await asyncio.to_thread(self._vec, question)
        flt = None
        if isinstance(lookup_key, str):
            flt = qm.Filter(must=[qm.FieldCondition(
                key="question_id", match=qm.MatchValue(value=lookup_key),
            )])
        elif lookup_key is not None:
            flt = qm.Filter(must=[qm.FieldCondition(
                key="conv_idx", match=qm.MatchValue(value=lookup_key),
            )])
        hits = await self._qd.search(
            collection_name=CFG.rag_collection,
            query_vector=vec,
            query_filter=flt,
            limit=CFG.max_context_memories,
            with_payload=True,
        )
        out: list[str] = []
        for h in hits:
            p = h.payload or {}
            date = p.get("session_datetime") or ""
            speaker = p.get("speaker") or ""
            text = p.get("text") or ""
            prefix = f"{date} {speaker}".strip()
            out.append(f"[{prefix}] {text}" if prefix else text)
        return out
