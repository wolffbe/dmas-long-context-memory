"""zep/graphiti backend.

Ingestion mirrors `zep-papers/locomo_eval/zep_locomo_ingestion.py`:
  - data = "{speaker}: {text}{img_description}"
  - type = "message"
  - created_at = ISO datetime of session
  - group_id = f"locomo_experiment_user_{conv_idx}"

We use graphiti-core locally (Neo4j) to stand in for Zep Cloud — same graph model.
"""
import asyncio
from typing import Any

from graphiti_core import Graphiti
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client import LLMConfig, OpenAIClient
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_CROSS_ENCODER,
    NODE_HYBRID_SEARCH_RRF,
)
from graphiti_core.search.search_filters import SearchFilters

from app.config import CFG
from app.memory._locomo import iso_date, zep_message_data


class ZepBackend:
    def __init__(self) -> None:
        self._g: Graphiti | None = None

    async def start(self) -> None:
        # Route graphiti's entity/edge extraction LLM and embedder through
        # this agent's litellm proxy under the `memory/...` aliases, mirroring
        # mem0_backend. Langfuse then attributes these calls separately from
        # the agent's answer-loop spend.
        llm_client = OpenAIClient(
            config=LLMConfig(
                api_key=CFG.litellm_api_key,
                base_url=CFG.litellm_url,
                model=CFG.memory_llm_model,
                small_model=CFG.memory_llm_model,
            ),
        )
        embedder = OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key=CFG.litellm_api_key,
                base_url=CFG.litellm_url,
                embedding_model=CFG.memory_embed_model,
            ),
        )
        self._g = Graphiti(
            uri=CFG.neo4j_uri,
            user=CFG.neo4j_user,
            password=CFG.neo4j_password,
            llm_client=llm_client,
            embedder=embedder,
        )
        await self._g.build_indices_and_constraints()

    async def close(self) -> None:
        if self._g is not None:
            try:
                await self._g.close()
            except Exception:
                pass

    @staticmethod
    def _group_id_for_item(item: dict[str, Any]) -> str:
        if item.get("dataset") == "longmemeval":
            return f"longmemeval_{item.get('question_id', 'unknown')}_a{CFG.agent_id}"
        return f"locomo_experiment_user_{item.get('conv_idx', 0)}_a{CFG.agent_id}"

    @staticmethod
    def _group_id_for_lookup(lookup_key: str | int | None) -> str:
        if isinstance(lookup_key, str):
            return f"longmemeval_{lookup_key}_a{CFG.agent_id}"
        return f"locomo_experiment_user_{lookup_key}_a{CFG.agent_id}"

    async def ingest(self, item: dict[str, Any]) -> dict[str, Any]:
        assert self._g is not None
        data = zep_message_data(item)
        if data is None:
            return {"skipped": True, "reason": "empty"}
        ts = iso_date(item.get("session_datetime"), dataset=item.get("dataset", "locomo")) or None
        from datetime import datetime, timezone
        reference_time = datetime.fromisoformat(ts) if ts else datetime.now(tz=timezone.utc)
        group_id = self._group_id_for_item(item)
        name = (
            item.get("dia_id")
            or f"{item.get('question_id', 'turn')}-s{item.get('session_id', 0)}-t{item.get('turn_index', 0)}"
        )
        ep = await self._g.add_episode(
            name=name,
            episode_body=data,
            source=EpisodeType.message,
            source_description=(
                f"agent-{CFG.agent_id} {item.get('dataset','locomo')}-{item.get('conv_idx') or item.get('question_id')}"
            ),
            reference_time=reference_time,
            group_id=group_id,
        )
        return {"stored": True, "episode_uuid": getattr(ep, "uuid", None)}

    async def recall(self, question: str, lookup_key: str | int | None) -> list[str]:
        assert self._g is not None
        group_id = self._group_id_for_lookup(lookup_key)
        # Match zep-papers/locomo_eval/zep_locomo_search.py: two parallel
        # searches — edges with cross-encoder rerank, nodes with RRF — formatted
        # as "  - {fact} (event_time: {valid_at})" / "  - {name}: {summary}".
        edge_cfg = EDGE_HYBRID_SEARCH_CROSS_ENCODER.model_copy(deep=True)
        edge_cfg.limit = CFG.search_limit
        node_cfg = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        node_cfg.limit = CFG.search_limit
        flt = SearchFilters(group_ids=[group_id])

        edge_task = self._g._search(query=question, config=edge_cfg, search_filter=flt)
        node_task = self._g._search(query=question, config=node_cfg, search_filter=flt)
        edge_results, node_results = await asyncio.gather(edge_task, node_task)

        snippets: list[str] = []
        for e in (edge_results.edges or [])[: CFG.max_context_memories]:
            fact = getattr(e, "fact", None) or ""
            valid_at = getattr(e, "valid_at", None)
            if not fact:
                continue
            snippets.append(
                f"  - {fact} (event_time: {valid_at})" if valid_at else f"  - {fact}"
            )
        if len(snippets) < CFG.max_context_memories:
            for n in (node_results.nodes or []):
                if len(snippets) >= CFG.max_context_memories:
                    break
                name = getattr(n, "name", "") or ""
                summary = getattr(n, "summary", "") or ""
                if name or summary:
                    snippets.append(f"  - {name}: {summary}".strip())
        return snippets

    # Allow caller to await close
    def __del__(self) -> None:  # pragma: no cover
        try:
            loop = asyncio.get_event_loop()
            if self._g is not None and loop.is_running():
                loop.create_task(self.close())
        except Exception:
            pass
