"""Render per-agent LiteLLM config files.

Per-agent contract:
  * agent-1 (edge):  gemma4:e4b (ollama, local) + openai/* wildcard
  * agent-2 (cloud): openai/* wildcard ONLY — ollama is exclusive to agent-1
  * agent-3 (cloud): openai/* wildcard ONLY — ollama is exclusive to agent-1

Both AGENT and MEMORY traffic routes through the same proxy. To split them
in Langfuse / cost accounting, MEMORY callers (mem0, graphiti) use a parallel
set of model names prefixed with `memory/`. Same upstream models, different
labels — Langfuse sees distinct `model` strings, so the coordinator can sum
agent-vs-memory tokens & cost cleanly.

Outputs:
  dmas/litellm/agent-1.yaml
  dmas/litellm/agent-2.yaml
  dmas/litellm/agent-3.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

ALLOWED_OLLAMA_MODEL = "gemma4:e4b"

_OLLAMA_BLOCK = f"""  - model_name: {ALLOWED_OLLAMA_MODEL}
    litellm_params:
      model: ollama/{ALLOWED_OLLAMA_MODEL}
      api_base: http://ollama:11434
      input_cost_per_token: 0
      output_cost_per_token: 0
  - model_name: memory/{ALLOWED_OLLAMA_MODEL}
    litellm_params:
      model: ollama/{ALLOWED_OLLAMA_MODEL}
      api_base: http://ollama:11434
      input_cost_per_token: 0
      output_cost_per_token: 0
"""

_OPENAI_BLOCK = """  - model_name: "openai/*"
    litellm_params:
      model: openai/*
      api_key: os.environ/OPENAI_API_KEY
  - model_name: "memory/openai/*"
    litellm_params:
      model: openai/*
      api_key: os.environ/OPENAI_API_KEY
  - model_name: text-embedding-3-small
    litellm_params:
      model: openai/text-embedding-3-small
      api_key: os.environ/OPENAI_API_KEY
  - model_name: memory/text-embedding-3-small
    litellm_params:
      model: openai/text-embedding-3-small
      api_key: os.environ/OPENAI_API_KEY
"""

_SETTINGS = """
litellm_settings:
  drop_params: true
  success_callback: ["langfuse", "prometheus"]
  failure_callback: ["langfuse", "prometheus"]
  redact_user_api_key_info: true

general_settings:
  master_key: sk-anything
"""


def _yaml(agent_id: int) -> str:
    body = "model_list:\n"
    if agent_id == 1:
        body += _OLLAMA_BLOCK
    body += _OPENAI_BLOCK
    body += _SETTINGS
    return body


def main() -> int:
    for i in (1, 2, 3):
        out = HERE / f"agent-{i}.yaml"
        out.write_text(_yaml(i))
        models = "gemma4:e4b + openai/*" if i == 1 else "openai/* (no ollama)"
        print(f"[render] agent-{i} -> {out}  ({models})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
