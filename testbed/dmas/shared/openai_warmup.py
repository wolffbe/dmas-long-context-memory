"""Pre-warm the litellm-cloud LLM endpoint at service warmup time.

Each memory backend lazily opens its first connection to the litellm
proxy on its first real LLM call, which inflates row 1 of every load
loop. Firing a tiny request at warmup time pays the server-side route
compile + upstream keep-alive before the measured loop begins.

The client built here is local to the helper and is closed on exit,
so the *client-side* httpx pool warmed here is not the pool the
framework's own client later reuses. The win is server-side.
"""
from __future__ import annotations

import os
from typing import Optional

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

DEFAULT_LLM_MODEL = "gpt-4o-mini"


class _Ping(BaseModel):
    ok: bool


def _model(model: Optional[str]) -> str:
    return model or os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)


def _api_key() -> Optional[str]:
    return os.getenv("OPENAI_API_KEY")


def _base_url() -> Optional[str]:
    return os.getenv("OPENAI_BASE_URL")


async def warmup_chat_async(model: Optional[str] = None) -> None:
    async with AsyncOpenAI(api_key=_api_key(), base_url=_base_url()) as client:
        await client.chat.completions.create(
            model=_model(model),
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )


def warmup_chat_sync(model: Optional[str] = None) -> None:
    with OpenAI(api_key=_api_key(), base_url=_base_url()) as client:
        client.chat.completions.create(
            model=_model(model),
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )


async def warmup_responses_async(model: Optional[str] = None) -> None:
    # Graphiti's runtime path is responses.parse with structured output,
    # so the ping must exercise that endpoint rather than chat.completions.
    async with AsyncOpenAI(api_key=_api_key(), base_url=_base_url()) as client:
        await client.responses.parse(
            model=_model(model),
            input=[{"role": "user", "content": "Reply with ok=true."}],
            text_format=_Ping,
        )
