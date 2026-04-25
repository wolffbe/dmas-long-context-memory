import os
from dataclasses import dataclass


def _peers() -> list[str]:
    raw = os.getenv("PEERS", "").strip()
    return [p.strip().rstrip("/") for p in raw.split(",") if p.strip()]


@dataclass(frozen=True)
class Config:
    agent_id: str = os.getenv("AGENT_ID", "1")
    peers: tuple[str, ...] = tuple(_peers())

    litellm_url: str = os.getenv("LITELLM_URL", "http://litellm:4000/v1")
    litellm_api_key: str = os.getenv("LITELLM_API_KEY", "sk-anything")

    toxiproxy_admin: str = os.getenv("TOXIPROXY_ADMIN", "")
    jitter_proxy_name: str = os.getenv("JITTER_PROXY_NAME", "peers")
    # Above this jitter (ms) the agent skips the peer fan-out and answers from
    # local memory only. Set to 0 to always allow peer help.
    help_jitter_threshold_ms: float = float(os.getenv("HELP_JITTER_THRESHOLD_MS", "0"))

    qdrant_url: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")

    rag_collection: str = os.getenv("RAG_COLLECTION", "rag_agent_default")
    mem0_collection: str = os.getenv("MEM0_COLLECTION", "mem0_agent_default")
    rag_embed_model: str = os.getenv("RAG_EMBED_MODEL", "all-MiniLM-L6-v2")

    # Models used by mem0/graphiti for fact extraction & embedding. They go
    # through this agent's litellm proxy under `memory/...` aliases so Langfuse
    # records them with a distinct model name we can split out from the agent's
    # answer-loop spend.
    memory_llm_model: str = os.getenv("MEMORY_LLM_MODEL", "memory/openai/gpt-4o-mini")
    memory_embed_model: str = os.getenv("MEMORY_EMBED_MODEL", "memory/text-embedding-3-small")

    max_context_memories: int = int(os.getenv("MAX_CONTEXT_MEMORIES", "40"))
    search_limit: int = int(os.getenv("SEARCH_LIMIT", "200"))

    langfuse_host: str = os.getenv("LANGFUSE_HOST", "")
    langfuse_public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret_key: str = os.getenv("LANGFUSE_SECRET_KEY", "")


CFG = Config()
