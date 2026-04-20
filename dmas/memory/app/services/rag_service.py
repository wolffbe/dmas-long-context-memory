import logging
import os
import uuid
from typing import Dict, Any, List

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

logger = logging.getLogger(__name__)

COLLECTION_NAME = "rag_chunks"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

CHUNK_CHARS = int(os.getenv("RAG_CHUNK_CHARS", "1024"))   # ≈ 256 tokens
OVERLAP_CHARS = int(os.getenv("RAG_OVERLAP_CHARS", "200"))
TOP_K = int(os.getenv("RAG_TOP_K", "2"))
SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.3"))


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


class RagService:
    """Standard RAG baseline: chunks raw conversation text and stores in Qdrant.

    Mirrors the RAG approach from the Mem0 paper: fixed-size text chunks (≈256 tokens)
    with overlap, retrieved via cosine similarity (Top-K=2).
    """

    def __init__(self):
        self.openai = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"))
        self.qdrant = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
        )
        self._ensure_collection()

    def _ensure_collection(self):
        existing = {c.name for c in self.qdrant.get_collections().collections}
        if COLLECTION_NAME not in existing:
            self.qdrant.create_collection(
                COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection '%s'", COLLECTION_NAME)

    def _embed(self, text: str) -> List[float]:
        resp = self.openai.embeddings.create(model=EMBED_MODEL, input=text)
        return resp.data[0].embedding

    def memorize_conversation(self, conv_index: int, data: Dict[str, Any]) -> Dict[str, Any]:
        sessions = data.get("sessions") or {}
        session_datetimes = data.get("session_datetimes") or {}

        if not isinstance(sessions, dict):
            return {"status": "error", "reason": "'sessions' must be a dict"}

        added = 0
        failed = 0

        for session_key in sorted(sessions.keys()):
            turns = sessions[session_key]
            if not isinstance(turns, list):
                continue

            ts = session_datetimes.get(f"{session_key}_date_time", "")
            session_lines: List[str] = []
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                text = str(turn.get("text", "")).strip()
                speaker = str(turn.get("speaker", "")).strip()
                if not text:
                    continue
                prefix = f"[{speaker}]" if speaker else "[Unknown]"
                session_lines.append(f"{prefix}: {text}")

            session_text = "\n".join(session_lines)
            chunks = _chunk_text(session_text, CHUNK_CHARS, OVERLAP_CHARS)

            points = []
            for chunk_idx, chunk in enumerate(chunks):
                try:
                    vector = self._embed(chunk)
                    points.append(PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "conversation_id": conv_index,
                            "session": session_key,
                            "chunk_idx": chunk_idx,
                            "text": chunk,
                            "timestamp": ts,
                        },
                    ))
                    added += 1
                except Exception as e:
                    logger.exception("RAG embed failed for chunk %d in %s: %s", chunk_idx, session_key, e)
                    failed += 1

            if points:
                self.qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

        logger.info("RAG: conv %d memorized — %d chunks added, %d failed", conv_index, added, failed)
        return {
            "status": "success" if failed == 0 else "partial_failure",
            "conversation_id": conv_index,
            "chunks_added": added,
            "failed": failed,
        }

    def remember(self, question: str) -> List[str]:
        try:
            query_vector = self._embed(question)
            results = self.qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=TOP_K,
                score_threshold=SCORE_THRESHOLD,
            )
            memories = [hit.payload.get("text", "").strip() for hit in results if hit.payload.get("text")]
            logger.info("RAG: retrieved %d chunks for question", len(memories))
            return memories
        except Exception as e:
            logger.exception("RAG search error: %s", e)
            return []
