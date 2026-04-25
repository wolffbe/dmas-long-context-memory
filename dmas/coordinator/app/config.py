"""Coordinator configuration — read from env at import time."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _admins() -> tuple[str, ...]:
    raw = os.getenv("TOXIPROXY_ADMINS", "").strip()
    return tuple(s.strip() for s in raw.split(",") if s.strip())


def _agent_urls() -> tuple[str, ...]:
    raw = os.getenv("AGENT_URLS", "").strip()
    return tuple(s.strip() for s in raw.split(",") if s.strip())


@dataclass(frozen=True)
class Config:
    upstream_agent_url: str = os.getenv("AGENT_UPSTREAM", "http://agent-1:8000")
    upstream_agent_id: str = os.getenv("UPSTREAM_AGENT_ID", "1")
    toxiproxy_admins: tuple[str, ...] = field(default_factory=_admins)
    agent_urls: tuple[str, ...] = field(default_factory=_agent_urls)
    prometheus_url: str = os.getenv("PROMETHEUS_URL", "http://prometheus:9090").rstrip("/")
    locomo_url: str = os.getenv(
        "LOCOMO_DATA_URL",
        "https://raw.githubusercontent.com/snap-research/locomo/refs/heads/main/data/locomo10.json",
    )
    longmemeval_url: str = os.getenv(
        "LONGMEMEVAL_DATA_URL",
        "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s",
    )


CFG = Config()
