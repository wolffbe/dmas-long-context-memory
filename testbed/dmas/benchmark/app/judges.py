"""LLM-as-a-judge grader (placeholder until prompts are filled in)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from openai import OpenAI, RateLimitError

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


# Judge bypasses litellm so its tokens stay out of the SUT /metrics
# totals; this dual-client setup is the direct-OpenAI equivalent of
# litellm's pool fallback.
_clients: dict[str, OpenAI | None] = {"primary": None, "backup": None}
_PING_PONG_ATTEMPTS = 4


def _client_for(slot: str) -> OpenAI | None:
    if _clients[slot] is not None:
        return _clients[slot]
    if slot == "primary":
        if not os.getenv("OPENAI_API_KEY"):
            return None
        _clients[slot] = OpenAI()
    else:
        key = (os.getenv("OPENAI_API_KEY_BACKUP") or "").strip()
        # docker-compose defaults OPENAI_API_KEY_BACKUP to OPENAI_API_KEY
        # so the litellm pool stays healthy when no real backup is set;
        # for the judge that means "same key twice", which would just
        # double-hit the same rate limit. Treat dup as no backup.
        primary_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not key or key == primary_key:
            return None
        _clients[slot] = OpenAI(api_key=key)
    return _clients[slot]


def _client_chain() -> list[OpenAI]:
    chain: list[OpenAI] = []
    primary = _client_for("primary")
    if primary is not None:
        chain.append(primary)
    backup = _client_for("backup")
    if backup is not None:
        chain.append(backup)
    if not chain:
        raise RuntimeError(
            "OPENAI_API_KEY is not set; the judge cannot reach OpenAI."
        )
    return chain


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
    """Judge completion with primary/backup ping-pong on 429."""
    chain = _client_chain()
    last_err: RateLimitError | None = None
    for i in range(_PING_PONG_ATTEMPTS):
        client = chain[i % len(chain)]
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content or ""
        except RateLimitError as exc:
            last_err = exc
            if len(chain) == 1:
                break
    assert last_err is not None
    raise last_err


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
