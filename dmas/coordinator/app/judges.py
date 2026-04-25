"""LLM-as-a-judge graders, verbatim from upstream.

- All graded backends use the zep judge prompts (gpt-5-mini default, env-overridable).
  - LoCoMo:      `getzep/zep-papers` kg_architecture_agent_memory/locomo_eval/zep_locomo_eval.py
  - LongMemEval: `getzep/zep-papers` kg_architecture_agent_memory/zep_longmem_eval.ipynb

The prompts below are reproduced verbatim. If you change wording, document the
divergence — the eval is intentionally tied to the published methodology.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Models (env-overridable; defaults match upstream)
# ---------------------------------------------------------------------------

JUDGE_MODEL_ZEP = os.getenv("JUDGE_MODEL_ZEP", "gpt-5-mini")


def set_judge_model(model: str) -> None:
    """Override the judge model with a single CLI-supplied name.

    Used by `experiment.py --judge-model MODEL`. Reads at call time, so any
    judge invocation after this returns will use the new value.
    """
    global JUDGE_MODEL_ZEP
    JUDGE_MODEL_ZEP = model


_client: OpenAI | None = None


def _openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()  # uses OPENAI_API_KEY env
    return _client


def _temperature_kwargs(model: str, temperature: float) -> dict[str, Any]:
    """gpt-5 / o-series only accept the default temperature; omit the param."""
    m = model.lower()
    if m.startswith(("gpt-5", "o1", "o3", "o4")):
        return {}
    return {"temperature": temperature}


@dataclass
class JudgeResult:
    label: bool | None  # True=correct, False=wrong, None=judge errored
    reasoning: str


# ===========================================================================
# Zep — LoCoMo judge (verbatim from zep_locomo_eval.py:locomo_grader)
# ===========================================================================

class _ZepGrade(BaseModel):
    is_correct: str = Field(description="CORRECT or WRONG")
    reasoning: str = Field(description="Explain why the answer is correct or incorrect.")


_ZEP_LOCOMO_SYSTEM = """
        You are an expert grader that determines if answers to questions match a gold standard answer
        """


def _zep_locomo_prompt(question: str, gold_answer: str, response: str) -> str:
    return f"""
    Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
        (1) a question (posed by one user to another user),
        (2) a ’gold’ (ground truth) answer,
        (3) a generated answer
    which you will score as CORRECT/WRONG.

    The point of the question is to ask about something one user should know about the other user based on their prior conversations.
    The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
    Question: Do you remember what I got the last time I went to Hawaii?
    Gold answer: A shell necklace
    The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

    For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

    Now it’s time for the real question:
    Question: {question}
    Gold answer: {gold_answer}
    Generated answer: {response}

    First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
    Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

    Just return the label CORRECT or WRONG in a json format with the key as "label".
    """


def _zep_judge_locomo(question: str, gold_answer: str, response: str) -> JudgeResult:
    user = _zep_locomo_prompt(question, gold_answer, response)
    try:
        completion = _openai_client().beta.chat.completions.parse(
            model=JUDGE_MODEL_ZEP,
            messages=[
                {"role": "system", "content": _ZEP_LOCOMO_SYSTEM},
                {"role": "user", "content": user},
            ],
            response_format=_ZepGrade,
            **_temperature_kwargs(JUDGE_MODEL_ZEP, 0.0),
        )
        parsed = completion.choices[0].message.parsed
    except Exception as exc:
        return JudgeResult(label=None, reasoning=f"judge_error: {exc}")
    if parsed is None:
        return JudgeResult(label=None, reasoning="parse_failed")
    is_correct = (parsed.is_correct or "").strip().lower() == "correct"
    return JudgeResult(label=is_correct, reasoning=(parsed.reasoning or "")[:1000])


# ===========================================================================
# Zep — LongMemEval judge (verbatim from zep_longmem_eval.ipynb:lme_grader)
# ===========================================================================

_ZEP_LME_SYSTEM = """
        You are an expert grader that determines if answers to questions match a gold standard answer
        """


def _zep_lme_temporal(question: str, gold_answer: str, response: str) -> str:
    return f"""
    I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model’s response is still correct.

    <QUESTION>
    B: {question}
    </QUESTION>
    <CORRECT ANSWER>
    {gold_answer}
    </CORRECT ANSWER>
    <RESPONSE>
    A: {response}
    </RESPONSE>
    """


def _zep_lme_knowledge_update(question: str, gold_answer: str, response: str) -> str:
    return f"""
    I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.

    <QUESTION>
    B: {question}
    </QUESTION>
    <CORRECT ANSWER>
    {gold_answer}
    </CORRECT ANSWER>
    <RESPONSE>
    A: {response}
    </RESPONSE>
    """


def _zep_lme_preference(question: str, gold_answer: str, response: str) -> str:
    return f"""
    I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user’s personal information correctly.

    <QUESTION>
    B: {question}
    </QUESTION>
    <RUBRIC>
    {gold_answer}
    </RUBRIC>
    <RESPONSE>
    A: {response}
    </RESPONSE>
    """


def _zep_lme_default(question: str, gold_answer: str, response: str) -> str:
    return f"""
    I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

    <QUESTION>
    B: {question}
    </QUESTION>
    <CORRECT ANSWER>
    {gold_answer}
    </CORRECT ANSWER>
    <RESPONSE>
    A: {response}
    </RESPONSE>
    """


class _ZepLMEGrade(BaseModel):
    is_correct: str = Field(description="yes or no")


def _zep_judge_lme(question: str, gold_answer: str, response: str, question_type: str | None) -> JudgeResult:
    qt = (question_type or "").strip()
    if qt == "temporal-reasoning":
        prompt = _zep_lme_temporal(question, gold_answer, response)
    elif qt == "knowledge-update":
        prompt = _zep_lme_knowledge_update(question, gold_answer, response)
    elif qt == "single-session-preference":
        prompt = _zep_lme_preference(question, gold_answer, response)
    else:
        prompt = _zep_lme_default(question, gold_answer, response)

    try:
        completion = _openai_client().beta.chat.completions.parse(
            model=JUDGE_MODEL_ZEP,
            messages=[
                {"role": "system", "content": _ZEP_LME_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format=_ZepLMEGrade,
            **_temperature_kwargs(JUDGE_MODEL_ZEP, 0.0),
        )
        parsed = completion.choices[0].message.parsed
    except Exception as exc:
        return JudgeResult(label=None, reasoning=f"judge_error: {exc}")
    if parsed is None:
        return JudgeResult(label=None, reasoning="parse_failed")
    is_correct = (parsed.is_correct or "").strip().lower() == "yes"
    return JudgeResult(label=is_correct, reasoning="")


# ===========================================================================
# Public dispatchers
# ===========================================================================

def zep_judge(
    dataset: str, question: str, gold_answer: str, response: str,
    question_type: str | None = None,
) -> JudgeResult:
    """Run the upstream zep judge for `dataset`."""
    if dataset == "longmemeval":
        return _zep_judge_lme(question, gold_answer, response, question_type)
    return _zep_judge_locomo(question, gold_answer, response)


def judge_for_backend(backend: str):
    """Return the zep judge for any graded backend, or None for ungraded."""
    if backend in ("mem0", "zep"):
        return zep_judge
    return None
