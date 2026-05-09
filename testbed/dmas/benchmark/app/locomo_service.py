"""Loads the LOCOMO benchmark dataset from disk (or downloads it once).

Only the surface used by `routes.py` is exposed:
  - `load_from_json()` populates `self.conversations` + `self.all_questions`
  - `get_conversation_by_index(i)` / `get_conversation_questions_by_index(i)`
"""
from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from app.models import Conversation, DialogTurn, Question

logger = logging.getLogger(__name__)


class LocomoService:
    def __init__(self, data_url: str):
        self.data_url = data_url
        self.data_path = Path("/data/locomo10.json")
        self.conversations: List[Conversation] = []
        self.all_questions: List[Dict] = []
        self.is_loaded = False

    def download_data(self) -> bool:
        if self.data_path.exists():
            logger.info("dataset already at %s", self.data_path)
            return True
        try:
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("downloading %s", self.data_url)
            urllib.request.urlretrieve(self.data_url, self.data_path)
            return True
        except Exception as exc:
            logger.error("download failed: %s", exc)
            return False

    def load_from_json(self) -> bool:
        if not self.data_path.exists() and not self.download_data():
            return False

        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.error("read/parse failed: %s", exc)
            return False

        if not isinstance(data, list):
            logger.error("expected JSON array, got %s", type(data).__name__)
            return False

        for conv_data in data:
            try:
                cd = conv_data.get("conversation", {})
                speaker_a = conv_data.get("speaker_a") or cd.get("speaker_a", "")
                speaker_b = conv_data.get("speaker_b") or cd.get("speaker_b", "")

                sessions: Dict[str, List[DialogTurn]] = {}
                session_datetimes: Dict[str, str] = {}
                for key, value in cd.items():
                    if not key.startswith("session_"):
                        continue
                    if key.endswith("_date_time"):
                        session_datetimes[key] = value
                    elif isinstance(value, list):
                        sessions[key] = [DialogTurn(**turn) for turn in value]

                qa = [Question(**q) for q in conv_data.get("qa") or []] or None

                conv = Conversation(
                    sample_id=conv_data.get("sample_id", ""),
                    speaker_a=speaker_a,
                    speaker_b=speaker_b,
                    sessions=sessions,
                    session_datetimes=session_datetimes,
                    qa=qa,
                )
                self.conversations.append(conv)

                if qa:
                    for idx, question in enumerate(qa):
                        self.all_questions.append({
                            "sample_id": conv.sample_id,
                            "question_id": f"{conv.sample_id}_q_{idx}",
                            "question": question.question,
                            "answer": question.get_answer(),
                            "category": question.category,
                            "evidence": question.evidence,
                        })
            except Exception as exc:
                logger.warning("skipping conversation: %s", exc)

        self.is_loaded = True
        logger.info(
            "loaded %d conversations / %d questions",
            len(self.conversations), len(self.all_questions),
        )
        return True

    def get_conversation_by_index(self, index: int) -> Optional[Conversation]:
        if 0 <= index < len(self.conversations):
            return self.conversations[index]
        return None

    def get_conversation_questions_by_index(self, conv_index: int) -> List[Dict]:
        conv = self.get_conversation_by_index(conv_index)
        if conv is None:
            return []
        return [q for q in self.all_questions if q["sample_id"] == conv.sample_id]
