"""LLM-as-a-judge grader (placeholder until prompts are filled in)."""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field

from openai import OpenAI, RateLimitError

from app.prompts import judge as judge_prompts

logger = logging.getLogger(__name__)

# Parses OpenAI's `x-ratelimit-reset-*` header values ("1s", "6m0s",
# "500ms", "1h2m3s"). Mirrors responder_service._parse_reset_duration —
# kept here so the judge doesn't depend on the responder package.
_DURATION_RE = re.compile(r"^(?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?$")


def _parse_reset_duration(s: str | None) -> float:
    if not s:
        return 0.0
    s = s.strip().lower()
    if s.endswith("ms"):
        try:
            return float(s[:-2]) / 1000.0
        except ValueError:
            return 0.0
    m = _DURATION_RE.fullmatch(s)
    if not m:
        return 0.0
    h, mn, sec = m.groups()
    return int(h or 0) * 3600 + int(mn or 0) * 60 + float(sec or 0)


def _retry_wait_seconds(err: RateLimitError) -> float:
    """Use OpenAI's `x-ratelimit-reset-{requests,tokens}` headers (the
    longer of the two, plus a 100 ms buffer). Falls back to
    `retry-after`, then 5 s, so we never spin."""
    headers = {}
    try:
        headers = dict(getattr(err.response, "headers", {}) or {})
    except Exception:
        pass
    waits = []
    for h in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        v = headers.get(h) or headers.get(h.title())
        if v:
            waits.append(_parse_reset_duration(v))
    ra = headers.get("retry-after") or headers.get("Retry-After")
    if ra:
        try:
            waits.append(float(ra))
        except ValueError:
            pass
    return (max(waits) + 0.1) if waits else 5.0


_MAX_RETRIES = int(os.getenv("JUDGE_MAX_RETRIES", "20"))


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
# totals. Single direct-OpenAI client; on 429 we sleep the exact
# x-ratelimit-reset-* window from the response headers and retry.
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set; the judge cannot reach OpenAI."
            )
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
    """Judge completion with 429-retry using OpenAI's reset headers.

    On RateLimitError we read `x-ratelimit-reset-{requests,tokens}` from
    the failing response and sleep `max(reset_requests, reset_tokens)
    + 100 ms`. Tracks total slept time only inside this call — the
    judge runs after `compute_ms` is recorded so its retry waits don't
    contaminate per-row wall_ms.
    """
    client = _get_client()
    last_err: RateLimitError | None = None
    for attempt in range(_MAX_RETRIES):
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
            wait_s = _retry_wait_seconds(exc)
            logger.warning(
                "judge: 429 from OpenAI; sleeping %.2fs (attempt %d/%d)",
                wait_s, attempt + 1, _MAX_RETRIES,
            )
            time.sleep(wait_s)
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
