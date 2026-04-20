# Experiment Matrix

Parameters to set in the **Parameters cell** of `analysis.ipynb` before each run.

**Run control cheatsheet:**
- First run of a backend → `RESET=True, REBUILD=True`
- Switch backend or model → `RESET=True, REBUILD=False`
- Re-run Q&A only (memories already loaded) → `RESET=False, REBUILD=False`

---

## Core experiments

| # | MEMORY_BACKEND | OLLAMA_MODEL | NETWORK_PROFILE | RESET | REBUILD | Output CSV | Done |
|---|----------------|--------------|-----------------|-------|---------|------------|------|
| 1 | `mem0` | `qwen3:8b` | `unconstrained` | True | **True** | `mem0_qwen3-8b_conv_0_qa.csv` | ⬜ |
| 2 | `mem0` | `qwen3:8b` | `constrained` | False | False | `mem0_qwen3-8b_conv_0_qa_constrained.csv` | ⬜ |
| 3 | `graphiti` | `qwen3:8b` | `unconstrained` | True | False | `graphiti_qwen3-8b_conv_0_qa.csv` | ⬜ |
| 4 | `graphiti` | `qwen3:8b` | `constrained` | False | False | `graphiti_qwen3-8b_conv_0_qa_constrained.csv` | ⬜ |

## Baselines (required by reviewers)

| # | MEMORY_BACKEND | OLLAMA_MODEL | NETWORK_PROFILE | RESET | REBUILD | Output CSV | Done |
|---|----------------|--------------|-----------------|-------|---------|------------|------|
| 5 | `rag` | `qwen3:8b` | `unconstrained` | True | False | `rag_qwen3-8b_conv_0_qa.csv` | ⬜ |
| 6 | `rag` | `qwen3:8b` | `constrained` | False | False | `rag_qwen3-8b_conv_0_qa_constrained.csv` | ⬜ |
| 7 | `full_context` | `qwen3:8b` | `unconstrained` | True | False | `full_context_qwen3-8b_conv_0_qa.csv` | ⬜ |

## Model comparison (optional)

| # | MEMORY_BACKEND | OLLAMA_MODEL | NETWORK_PROFILE | RESET | REBUILD | Output CSV | Done |
|---|----------------|--------------|-----------------|-------|---------|------------|------|
| 8 | `mem0` | `ministral:8b` | `unconstrained` | True | False | `mem0_ministral-8b_conv_0_qa.csv` | ⬜ |
| 9 | `mem0` | `ministral:8b` | `constrained` | False | False | `mem0_ministral-8b_conv_0_qa_constrained.csv` | ⬜ |
| 10 | `graphiti` | `ministral:8b` | `unconstrained` | True | False | `graphiti_ministral-8b_conv_0_qa.csv` | ⬜ |
| 11 | `graphiti` | `ministral:8b` | `constrained` | False | False | `graphiti_ministral-8b_conv_0_qa_constrained.csv` | ⬜ |

---

## Per-run notes

**#2 (mem0 constrained):** do NOT reset — memories from run #1 are still in the DB.
Only change `NETWORK_PROFILE = "constrained"`.

**#3 (graphiti unconstrained):** the Warmup cell will block until Neo4j background
graph-building completes. Elapsed time is a paper metric.

**#5–6 (rag):** `MEMORY_WINDOW_SIZE` is auto-set to 0 (chunk-based, no windowing needed).

**#7 (full_context):** expect very high latency and OpenAI cost — this is the "naive" baseline.
Consider testing with `NO_OF_SESSIONS = 3` first to estimate cost before the full run.

---

## After each run: LLM Judge

```bash
cd analysis
python judge.py --runs 10
```

Adds `llm_judge_run_1..10`, `llm_judge_avg`, `llm_judge_correct` to all CSVs in `results/`.
Resume-safe: if interrupted, re-run and it picks up where it left off.
