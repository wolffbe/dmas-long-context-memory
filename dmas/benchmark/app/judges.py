"""LLM-as-a-judge grader (placeholder until prompts are filled in)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from openai import OpenAI

from app.prompts import judge as judge_prompts


@dataclass
class JudgeResult:
    label: str  # "CORRECT" | "WRONG" | "PLACEHOLDER" | "ERROR"
    reasoning: str = ""


@dataclass
class JudgeAggregate:
    """Result of running the judge `n` independent times against the same
    (question, gold, response) triple and majority-voting the labels.

    `label` is the consensus verdict (CORRECT iff strictly more than half
    the judge calls returned CORRECT — default protocol "2/3 true = true").
    `votes` carries every individual label so analyses can recover
    inter-judge agreement; `reasonings` carries each non-empty reasoning
    string. `correct_votes` and `n` give the raw fraction the consensus
    is built from.
    """
    label: str
    votes: list[str] = field(default_factory=list)
    reasonings: list[str] = field(default_factory=list)
    correct_votes: int = 0
    n: int = 0


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


def judge(question: str, gold_answer: str, response: str) -> JudgeResult:
    if not judge_prompts.JUDGE_PROMPT_TEMPLATE:
        return JudgeResult(label="PLACEHOLDER", reasoning="judge not implemented")
    if not gold_answer or not response:
        return JudgeResult(label="WRONG")
    user = judge_prompts.get_judge_prompt(question=question, gold_answer=gold_answer, response=response)
    try:
        label, reason = _grade_label(_call(judge_prompts.JUDGE_MODEL, judge_prompts.JUDGE_SYSTEM_PROMPT, user))
        return JudgeResult(label=label, reasoning=reason)
    except Exception as exc:
        return JudgeResult(label="ERROR", reasoning=f"judge_error: {exc}"[:300])


def judge_majority(question: str, gold_answer: str, response: str,
                   n: int = 3) -> JudgeAggregate:
    """Run `judge` n independent times and majority-vote the verdict.

    Consensus protocol: CORRECT iff strictly more than half the calls
    returned CORRECT (so for n=3 the threshold is 2). When every vote
    agrees on a non-CORRECT/non-WRONG label (e.g. a unanimous PLACEHOLDER
    or ERROR run), that label is propagated as the verdict so an
    unimplemented or broken judge isn't masked by a default WRONG.
    Otherwise non-CORRECT votes count as WRONG for the tally.
    """
    if n < 1:
        n = 1
    votes: list[str] = []
    reasonings: list[str] = []
    for _ in range(n):
        r = judge(question, gold_answer, response)
        votes.append(r.label)
        if r.reasoning:
            reasonings.append(r.reasoning)

    # If every vote agrees on a non-CORRECT/WRONG label (PLACEHOLDER/ERROR),
    # propagate that — don't bury an unimplemented or broken judge under a
    # WRONG verdict.
    if votes and all(v == votes[0] and v not in ("CORRECT", "WRONG") for v in votes):
        return JudgeAggregate(label=votes[0], votes=votes, reasonings=reasonings,
                              correct_votes=0, n=n)

    correct_votes = sum(1 for v in votes if v == "CORRECT")
    label = "CORRECT" if correct_votes * 2 > n else "WRONG"
    return JudgeAggregate(label=label, votes=votes, reasonings=reasonings,
                          correct_votes=correct_votes, n=n)
