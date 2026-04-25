"""LLM-as-a-judge grader (placeholder until prompts are filled in)."""
from __future__ import annotations

import json
from dataclasses import dataclass

from openai import OpenAI

from app.prompts import judge as judge_prompts


@dataclass
class JudgeResult:
    label: str  # "CORRECT" | "WRONG" | "PLACEHOLDER" | "ERROR"
    reasoning: str = ""


_client: OpenAI | None = None


def _openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _grade_label(raw: str) -> tuple[str, str]:
    text = (raw or "").strip()
    try:
        parsed = json.loads(text)
        label = (parsed.get("label") or parsed.get("is_correct") or "").strip().upper()
        reason = parsed.get("reasoning", "") if isinstance(parsed, dict) else ""
        if label in {"CORRECT", "WRONG"}:
            return label, reason
    except (json.JSONDecodeError, AttributeError):
        pass
    upper = text.upper()
    if "CORRECT" in upper and "WRONG" not in upper:
        return "CORRECT", text[:300]
    if "WRONG" in upper and "CORRECT" not in upper:
        return "WRONG", text[:300]
    return "ERROR", text[:300]


def _call(model: str, system: str, user: str) -> str:
    resp = _openai_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or ""


def judge(question: str, gold_answer: str, response: str, session_date: str = "") -> JudgeResult:
    if not judge_prompts.JUDGE_PROMPT_TEMPLATE:
        return JudgeResult(label="PLACEHOLDER", reasoning="judge not implemented")
    if not gold_answer or not response:
        return JudgeResult(label="WRONG")
    user = judge_prompts.get_judge_prompt(question=question, gold_answer=gold_answer, response=response, session_date=session_date)
    try:
        label, reason = _grade_label(_call(judge_prompts.JUDGE_MODEL, judge_prompts.JUDGE_SYSTEM_PROMPT, user))
        return JudgeResult(label=label, reasoning=reason)
    except Exception as exc:
        return JudgeResult(label="ERROR", reasoning=f"judge_error: {exc}"[:300])
