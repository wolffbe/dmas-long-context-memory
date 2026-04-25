"""Dataset loading + per-turn flattening.

Pulled into the coordinator from the deleted locomo/longmemeval services and
the deleted experiments/lib/drivers.py — same logic, served in-process.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from app.config import CFG

log = logging.getLogger(__name__)

DATA_DIR = Path("/data")


# ---------- shared shape ----------

@dataclass
class FlatTurn:
    dataset: str
    session_id: str
    session_datetime: str
    turn_index: int
    conv_idx: int | None = None
    speaker_a: str | None = None
    speaker_b: str | None = None
    speaker: str | None = None
    to: str | None = None
    text: str | None = None
    dia_id: str | None = None
    blip_caption: str | None = None
    query: str | None = None
    question_id: str | None = None
    role: str | None = None
    content: str | None = None
    question_type: str | None = None

    def to_item(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "dataset": self.dataset, "turn_index": self.turn_index,
            "session_id": self.session_id, "session_datetime": self.session_datetime,
        }
        if self.dataset == "longmemeval":
            d.update({
                "question_id": self.question_id, "role": self.role,
                "content": self.content, "question_type": self.question_type,
            })
        else:
            d.update({
                "conv_idx": self.conv_idx, "speaker_a": self.speaker_a,
                "speaker_b": self.speaker_b, "speaker": self.speaker, "to": self.to,
                "text": self.text, "dia_id": self.dia_id,
                "blip_caption": self.blip_caption, "query": self.query,
            })
        return d


# ---------- locomo flatten ----------

def flatten_locomo(conv: dict[str, Any]) -> list[FlatTurn]:
    conv_idx = 0
    sample_id = conv.get("sample_id", "")
    if sample_id and "_" in sample_id:
        try:
            conv_idx = int(sample_id.split("_")[-1])
        except ValueError:
            conv_idx = 0
    speaker_a = conv.get("speaker_a", "")
    speaker_b = conv.get("speaker_b", "")
    sessions = conv.get("conversation") or conv.get("sessions") or {}
    sessions_list: list[tuple[str, str, list[dict[str, Any]]]] = []
    if isinstance(sessions, dict):
        items: list[tuple[int, str, str, list[dict[str, Any]]]] = []
        for k, v in sessions.items():
            if k.startswith("session_") and not k.endswith("_date_time") and isinstance(v, list):
                try:
                    num = int(k.split("_")[1])
                except ValueError:
                    continue
                date = sessions.get(f"session_{num}_date_time", "")
                items.append((num, k, date, v))
        items.sort(key=lambda x: x[0])
        sessions_list = [(k, d, turns) for _, k, d, turns in items]

    out: list[FlatTurn] = []
    idx = 0
    for sid, sdate, turns in sessions_list:
        for t in turns:
            speaker = t.get("speaker", "")
            counterpart = speaker_b if speaker == speaker_a else speaker_a
            out.append(FlatTurn(
                dataset="locomo", session_id=sid, session_datetime=sdate,
                turn_index=idx, conv_idx=conv_idx,
                speaker_a=speaker_a, speaker_b=speaker_b, speaker=speaker, to=counterpart,
                text=t.get("text", ""), dia_id=t.get("dia_id", ""),
                blip_caption=t.get("blip_caption"), query=t.get("query"),
            ))
            idx += 1
    return out


_LME_RE_PAREN = re.compile(r"\s*\([A-Za-z]+\)\s*")


def flatten_longmemeval(question: dict[str, Any]) -> list[FlatTurn]:
    qid = question.get("question_id", "unknown")
    qtype = question.get("question_type")
    sessions = question.get("haystack_sessions") or []
    dates = question.get("haystack_dates") or [""] * len(sessions)
    session_ids = (
        question.get("haystack_session_ids")
        or [f"session_{i}" for i in range(len(sessions))]
    )

    paired = list(zip(session_ids, dates, sessions))

    def _key(t: tuple) -> tuple:
        try:
            cleaned = _LME_RE_PAREN.sub(" ", t[1]).strip()
            return (0, datetime.strptime(cleaned, "%Y/%m/%d %H:%M"))
        except Exception:
            return (1, datetime.min)

    paired.sort(key=_key)

    out: list[FlatTurn] = []
    idx = 0
    for sid, sdate, session in paired:
        for t in session:
            out.append(FlatTurn(
                dataset="longmemeval", session_id=str(sid),
                session_datetime=sdate or "", turn_index=idx,
                question_id=qid, role=t.get("role"), content=t.get("content"),
                question_type=qtype,
            ))
            idx += 1
    return out


# ---------- in-memory store ----------

class Datasets:
    def __init__(self) -> None:
        self.locomo: list[dict[str, Any]] = []
        self.locomo_by_sample: dict[str, dict[str, Any]] = {}
        self.lme: list[dict[str, Any]] = []
        self.lme_by_id: dict[str, dict[str, Any]] = {}

    def ensure_loaded(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._load_locomo()
        self._load_longmemeval()

    def _download(self, url: str, dest: Path) -> None:
        if dest.exists() and dest.stat().st_size > 0:
            return
        log.info(f"downloading {url} -> {dest}")
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        dest.write_bytes(r.content)

    def _load_locomo(self) -> None:
        path = DATA_DIR / "locomo10.json"
        try:
            self._download(CFG.locomo_url, path)
            self.locomo = json.loads(path.read_text())
        except Exception as exc:
            log.error(f"locomo load failed: {exc}")
            return
        if isinstance(self.locomo, dict):
            self.locomo = self.locomo.get("data", [])
        self.locomo_by_sample = {c.get("sample_id", ""): c for c in self.locomo}
        log.info(f"locomo loaded: {len(self.locomo)} conversations")

    def _load_longmemeval(self) -> None:
        path = DATA_DIR / "longmemeval_s.json"
        try:
            self._download(CFG.longmemeval_url, path)
            self.lme = json.loads(path.read_text())
        except Exception as exc:
            log.error(f"longmemeval load failed: {exc}")
            return
        if isinstance(self.lme, dict):
            self.lme = self.lme.get("data", [])
        self.lme_by_id = {q["question_id"]: q for q in self.lme if "question_id" in q}
        log.info(f"longmemeval loaded: {len(self.lme)} questions")

    # ---- locomo accessors ----
    def locomo_conversation(self, conv_idx: int) -> dict[str, Any]:
        if not (0 <= conv_idx < len(self.locomo)):
            raise IndexError(f"locomo conv_idx out of range: {conv_idx}")
        return self.locomo[conv_idx]

    def locomo_questions(self, conv_idx: int) -> list[dict[str, Any]]:
        sample_id = self.locomo_conversation(conv_idx).get("sample_id", "")
        sample = self.locomo_by_sample.get(sample_id, {})
        # locomo10.json puts QA under conv["qa"]
        return list(sample.get("qa") or [])

    # ---- longmemeval accessors ----
    def lme_question(self, qid: str) -> dict[str, Any]:
        q = self.lme_by_id.get(qid)
        if q is None:
            raise KeyError(f"unknown longmemeval question_id: {qid}")
        return q

    def lme_index(self) -> list[dict[str, Any]]:
        return [{
            "question_id": q.get("question_id"),
            "question_type": q.get("question_type"),
            "question": q.get("question"),
            "answer": q.get("answer"),
        } for q in self.lme]


datasets = Datasets()
