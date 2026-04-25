"""Per-dataset upstream message-shape helpers.

LOCOMO mirrors mem0/memory-benchmarks `benchmarks/locomo/run.py:165` (session_to_chunks)
and zep/zep-papers `locomo_eval/zep_locomo_ingestion.py`.

LongMemEval mirrors mem0/memory-benchmarks `benchmarks/longmemeval/run.py` (pair_turns +
parse_longmemeval_date) and the same Zep ingest pattern adapted to question_id/role.
"""
import re
from datetime import datetime, timezone
from typing import Any


# ---------------- LOCOMO ----------------

def photo_tag(blip: str | None, query: str | None) -> str:
    if query and blip:
        return f"[Sharing image - query: {query}. The image shows: {blip}]"
    if query:
        return f"[Sharing image - query for: {query}]"
    if blip:
        return f"[Sharing image that shows: {blip}]"
    return ""


def mem0_message_locomo(item: dict[str, Any]) -> dict[str, str] | None:
    speaker = (item.get("speaker") or "").strip()
    text = (item.get("text") or "").strip()
    tag = photo_tag(item.get("blip_caption"), item.get("query"))
    if tag:
        text = f"{text} {tag}" if text else tag
    if not text:
        return None
    speaker_a = item.get("speaker_a") or speaker
    role = "user" if speaker == speaker_a else "assistant"
    return {"role": role, "content": f"{speaker}: {text}"}


def zep_message_data_locomo(item: dict[str, Any]) -> str | None:
    speaker = (item.get("speaker") or "").strip()
    text = (item.get("text") or "").strip()
    if not text and not item.get("blip_caption"):
        return None
    blip = item.get("blip_caption")
    img_desc = f"(description of attached image: {blip})" if blip else ""
    return f"{speaker}: {text}{img_desc}"


def session_epoch_locomo(date_str: str | None) -> int | None:
    if not date_str:
        return None
    try:
        dt = datetime.strptime(date_str, "%I:%M %p on %d %B, %Y")
        return int(dt.replace(tzinfo=timezone.utc).timestamp())
    except ValueError:
        return None


def iso_date_locomo(date_str: str | None) -> str | None:
    if not date_str:
        return None
    try:
        dt = datetime.strptime(date_str + " UTC", "%I:%M %p on %d %B, %Y UTC")
        return dt.replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        return None


# ---------------- LongMemEval ----------------

_LME_DATE_RE = re.compile(r"\s*\([A-Za-z]+\)\s*")


def _parse_lme_date(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    cleaned = _LME_DATE_RE.sub(" ", date_str).strip()
    try:
        return datetime.strptime(cleaned, "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def session_epoch_longmemeval(date_str: str | None) -> int | None:
    dt = _parse_lme_date(date_str)
    return int(dt.timestamp()) if dt else None


def iso_date_longmemeval(date_str: str | None) -> str | None:
    dt = _parse_lme_date(date_str)
    return dt.isoformat() if dt else None


def mem0_message_longmemeval(item: dict[str, Any]) -> dict[str, str] | None:
    role = (item.get("role") or "").strip()
    content = (item.get("content") or "").strip()
    if not content or role not in ("user", "assistant"):
        return None
    return {"role": role, "content": content}


def zep_message_data_longmemeval(item: dict[str, Any]) -> str | None:
    role = (item.get("role") or "").strip()
    content = (item.get("content") or "").strip()
    if not content:
        return None
    return f"{role}: {content}"


# ---------------- dispatch ----------------

def mem0_message(item: dict[str, Any]) -> dict[str, str] | None:
    if item.get("dataset") == "longmemeval":
        return mem0_message_longmemeval(item)
    return mem0_message_locomo(item)


def zep_message_data(item: dict[str, Any]) -> str | None:
    if item.get("dataset") == "longmemeval":
        return zep_message_data_longmemeval(item)
    return zep_message_data_locomo(item)


def session_epoch(date_str: str | None, dataset: str = "locomo") -> int | None:
    if dataset == "longmemeval":
        return session_epoch_longmemeval(date_str)
    return session_epoch_locomo(date_str)


def iso_date(date_str: str | None, dataset: str = "locomo") -> str | None:
    if dataset == "longmemeval":
        return iso_date_longmemeval(date_str)
    return iso_date_locomo(date_str)
