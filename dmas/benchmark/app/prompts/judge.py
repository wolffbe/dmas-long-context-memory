"""LLM-as-a-judge prompt — verbatim from Zep's locomo_grader (getzep/zep-papers,
kg_architecture_agent_memory/locomo_eval/zep_locomo_eval.py)."""

JUDGE_MODEL = "gpt-4o-mini"

JUDGE_SYSTEM_PROMPT = """
        You are an expert grader that determines if answers to questions match a gold standard answer
        """

JUDGE_PROMPT_TEMPLATE = """
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

    For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer should also be a specific absolute date — the responder is anchored to the conversation date and should not emit relative phrases. As long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

    Now it’s time for the real question:
    Question: {question}
    Gold answer: {gold_answer}
    Generated answer: {response}

    Return a JSON object with two keys:
      - "reasoning": a short (one sentence) explanation of your judgment
      - "label": exactly one of "CORRECT" or "WRONG"
    Do NOT include both CORRECT and WRONG in the label, or it will break the evaluation script.
    """


def get_judge_prompt(question: str, gold_answer: str, response: str) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(
        question=question, gold_answer=gold_answer, response=response,
    )
