"""Driver-side reproducibility fingerprints — passed to coordinator in /experiment
body and stamped on every CSV row.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LITELLM_CONFIG = REPO_ROOT / "dmas" / "litellm" / "agent-1.yaml"
AGENT_SERVICE = REPO_ROOT / "dmas" / "agent" / "app" / "agent_service.py"
ENV_FILE = REPO_ROOT / ".env"


def _sha256_path(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


@lru_cache(maxsize=1)
def git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()[:12]
    except Exception:
        return ""


@lru_cache(maxsize=1)
def litellm_config_sha() -> str:
    return _sha256_path(LITELLM_CONFIG)


@lru_cache(maxsize=1)
def system_prompt_sha() -> str:
    """Hash everything that shapes the rendered system prompt.

    Captures the template, the tool schemas, and `_build_system_prompt`
    (which holds the conditional clauses chosen by the latency gate). If any
    of those change, the SHA changes — so reproducibility surveys can't
    silently merge runs whose prompts differed only in a conditional branch.
    """
    if not AGENT_SERVICE.exists():
        return ""
    src = AGENT_SERVICE.read_text(encoding="utf-8")
    m = re.search(
        r"SYSTEM_PROMPT_TEMPLATE\s*=.*?(?=\n    async def _dispatch_tool)",
        src, re.S,
    )
    if not m:
        return ""
    return hashlib.sha256(m.group(0).encode("utf-8")).hexdigest()[:12]


def _from_env_file() -> dict[str, str]:
    if not ENV_FILE.exists():
        return {}
    out: dict[str, str] = {}
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _knob(name: str, default: str = "") -> str:
    val = os.environ.get(name)
    if val is not None:
        return val
    return _from_env_file().get(name, default)


def context_fields() -> dict[str, str]:
    return {
        "git_sha": git_sha(),
        "max_context_memories": _knob("MAX_CONTEXT_MEMORIES", ""),
        "search_limit": _knob("SEARCH_LIMIT", ""),
        "litellm_config_sha": litellm_config_sha(),
        "system_prompt_sha": system_prompt_sha(),
    }
