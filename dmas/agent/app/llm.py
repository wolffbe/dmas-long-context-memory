from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from app.config import CFG


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


@dataclass
class LLMToolCall:
    id: str
    name: str
    arguments_raw: str


@dataclass
class LLMTurn:
    """A single assistant turn: either a final message or a list of tool calls."""
    content: str | None
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


_client = AsyncOpenAI(base_url=CFG.litellm_url, api_key=CFG.litellm_api_key)


def _usage_cost(resp: Any) -> tuple[int, int, float]:
    usage = resp.usage
    raw = getattr(resp, "model_extra", None) or {}
    hidden = raw.get("_hidden_params") or {}
    cost = float(hidden.get("response_cost") or 0.0)
    return (
        getattr(usage, "prompt_tokens", 0) or 0,
        getattr(usage, "completion_tokens", 0) or 0,
        cost,
    )


async def chat(system: str, user: str, model: str | None = None,
               temperature: float = 0.0, response_format_json: bool = False) -> LLMResult:
    kwargs: dict = {
        "model": model or CFG.llm_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
    }
    if response_format_json:
        kwargs["response_format"] = {"type": "json_object"}
    resp = await _client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or ""
    pt, ct, cost = _usage_cost(resp)
    return LLMResult(text=text, prompt_tokens=pt, completion_tokens=ct, cost_usd=cost)


async def chat_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    model: str | None = None,
    temperature: float = 0.0,
) -> LLMTurn:
    resp = await _client.chat.completions.create(
        model=model or CFG.llm_model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=temperature,
    )
    msg = resp.choices[0].message
    calls: list[LLMToolCall] = []
    for tc in (msg.tool_calls or []):
        calls.append(LLMToolCall(
            id=tc.id,
            name=tc.function.name,
            arguments_raw=tc.function.arguments or "",
        ))
    pt, ct, cost = _usage_cost(resp)
    return LLMTurn(
        content=msg.content,
        tool_calls=calls,
        prompt_tokens=pt,
        completion_tokens=ct,
        cost_usd=cost,
    )
