"""Thin client for the coordinator's /load endpoint.

The coordinator owns dataset access, flattening, and round-robin to the three
agents — driver just POSTs the load config and prints the result. Per-turn
progress is logged by the coordinator (visible via `make logs`).
"""
from __future__ import annotations

import argparse
import os

import requests

COORDINATOR_URL = os.getenv("COORDINATOR_URL", "http://localhost:8010").rstrip("/")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["mem0", "zep", "rag", "none"])
    ap.add_argument("--dataset", required=True, choices=["locomo", "longmemeval"])
    ap.add_argument("--conv", type=int, default=None)
    ap.add_argument("--qid", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="Smoke-test: load only the first N turns.")
    args = ap.parse_args()

    body = {
        "backend": args.backend, "dataset": args.dataset,
        "conv": args.conv, "qid": args.qid, "all": args.all,
        "limit": args.limit,
    }
    print(f"[load] POST {COORDINATOR_URL}/load  body={body}")
    print(f"[load] coordinator stdout: docker logs -f dmas-coordinator")

    r = requests.post(f"{COORDINATOR_URL}/load", json=body, timeout=2 * 3600)
    r.raise_for_status()
    payload = r.json()
    print(f"[load] done: {payload}")
    return 1 if payload.get("failed", 0) > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
