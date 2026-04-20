#!/usr/bin/env python3
"""Standalone LLM-as-a-Judge script for offline evaluation.

Run 10 times on collected CSVs and average the results to account for
judge stochasticity, as per the Mem0 paper methodology.
Generation agents use temperature=0 (deterministic), so only the judge
needs multiple runs.

Usage:
    python judge.py                        # all results/*_qa*.csv, 10 runs
    python judge.py --runs 5
    python judge.py --files results/mem0_conv_0_qa.csv results/graphiti_conv_0_qa.csv
    python judge.py --reset                # wipe existing judge columns and redo
"""

import argparse
import glob
import json
import os
import time

import pandas as pd
from openai import OpenAI

JUDGE_SYSTEM_PROMPT = (
    "Your task is to label an answer to a question as \"CORRECT\" or \"WRONG\". You will be given "
    "the following data: (1) a question (posed by one user to another user), (2) a 'gold' "
    "(ground truth) answer, (3) a generated answer which you will score as CORRECT/WRONG.\n"
    "The point of the question is to ask about something one user should know about the other "
    "user based on their prior conversations. The gold answer will usually be a concise and "
    "short answer that includes the referenced topic, for example:\n"
    "Question: Do you remember what I got the last time I went to Hawaii?\n"
    "Gold answer: A shell necklace\n"
    "The generated answer might be much longer, but you should be generous with your grading "
    "- as long as it touches on the same topic as the gold answer, it should be counted as "
    "CORRECT.\n"
    "For time related questions, the gold answer will be a specific date, month, year, etc. The "
    "generated answer might be much longer or use relative time references (like 'last Tuesday' "
    "or 'next month'), but you should be generous with your grading - as long as it refers to "
    "the same date or time period as the gold answer, it should be counted as CORRECT. Even if "
    "the format differs (e.g., 'May 7th' vs '7 May'), consider it CORRECT if it's the same date.\n"
    "First, provide a short (one sentence) explanation of your reasoning, then finish with "
    "CORRECT or WRONG. Do NOT include both CORRECT and WRONG in your response, or it will break "
    "the evaluation script.\n"
    "Just return the label CORRECT or WRONG in a json format with the key as \"label\"."
)


def _is_idk(answer: str) -> bool:
    a = str(answer).lower()
    return "i don't know" in a or "i dont know" in a


def judge_single(client: OpenAI, question: str, answer_actual: str,
                 answer_received: str, is_adversarial: bool = False) -> bool:
    """Judge one QA pair.

    For adversarial questions IDK is the correct response (avoids hallucination).
    For non-adversarial questions IDK is always wrong (no API call needed).
    """
    if _is_idk(answer_received):
        return is_adversarial

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Question: {question}\n"
                    f"Gold answer: {answer_actual}\n"
                    f"Generated answer: {answer_received}"
                )},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        label = json.loads(resp.choices[0].message.content).get("label", "WRONG")
        return label == "CORRECT"
    except Exception as e:
        print(f"  [judge error] {e}")
        return False


def run_judge_on_csv(client: OpenAI, path: str, n_runs: int, reset: bool) -> None:
    df = pd.read_csv(path)

    has_adversarial = "is_adversarial" in df.columns

    for run in range(1, n_runs + 1):
        col = f"llm_judge_run_{run}"
        if reset and col in df.columns:
            df[col] = None
        if col not in df.columns:
            df[col] = None

        mask = df[col].isna()
        n_todo = int(mask.sum())
        if n_todo == 0:
            print(f"  Run {run}/{n_runs}: already complete, skipping.")
            continue

        done_count = int((~mask).sum())
        print(f"  Run {run}/{n_runs}: judging {n_todo} rows...")
        for i, (idx, row) in enumerate(df[mask].iterrows(), start=1):
            is_adv = bool(row["is_adversarial"]) if has_adversarial else False
            result = judge_single(
                client,
                str(row["question"]),
                str(row["answer_actual"]),
                str(row["answer_received"]),
                is_adversarial=is_adv,
            )
            df.at[idx, col] = result
            df.to_csv(path, index=False)
            label = "✓" if result else "✗"
            q_short = str(row["question"])[:60]
            print(f"    [{done_count + i}/{len(df)}] {label} {q_short}")
            time.sleep(0.3)

    # Aggregate across all completed run columns
    run_cols = [f"llm_judge_run_{r}" for r in range(1, n_runs + 1)
                if f"llm_judge_run_{r}" in df.columns]
    if run_cols:
        df["llm_judge_avg"] = df[run_cols].astype(float).mean(axis=1)
        df["llm_judge_correct"] = df["llm_judge_avg"] >= 0.5

    df.to_csv(path, index=False)

    acc = df["llm_judge_correct"].mean() * 100 if "llm_judge_correct" in df.columns else float("nan")
    avg_std = df["llm_judge_avg"].std() if "llm_judge_avg" in df.columns else float("nan")
    n_adv_idk = int((df["is_adversarial"] & df["answer_received"].apply(_is_idk)).sum()) \
        if has_adversarial else 0

    print(f"  → Saved {path}")
    print(f"     accuracy (non-adversarial majority vote): {acc:.1f}%")
    print(f"     judge avg std (stability): {avg_std:.3f}")
    if has_adversarial:
        print(f"     adversarial IDK (correct refusals): {n_adv_idk}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge — multi-run offline evaluation")
    parser.add_argument("--files", nargs="*", help="CSV paths (default: results/*_qa*.csv)")
    parser.add_argument("--runs", type=int, default=10,
                        help="Number of judge runs to average (default: 10, as per Mem0 paper)")
    parser.add_argument("--reset", action="store_true",
                        help="Wipe existing llm_judge_run_* columns and redo from scratch")
    args = parser.parse_args()

    if not args.files:
        print("Specify files explicitly: python judge.py --files results/mem0_qwen3-8b_conv_0_qa.csv")
        print("To run on all CSVs: python judge.py --files results/*_qa*.csv")
        return
    files = args.files
    if not files:
        print("No CSV files found.")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        env_file = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_file):
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("OPENAI_API_KEY=") and not line.startswith("#"):
                        api_key = line.split("=", 1)[1].strip()
                        break
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    print(f"Judge: {len(files)} file(s), {args.runs} run(s) each\n")
    for path in files:
        print(f"\n=== {path} ===")
        run_judge_on_csv(client, path, args.runs, args.reset)

    print("\nAll done.")


if __name__ == "__main__":
    main()
