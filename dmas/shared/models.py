"""Pydantic request models shared across coordinator + memory.

Both services accept the same `(backend)` and `(backend, conv_index)`
shapes for /reset and /warmup; defining them once here keeps the two
sides in sync and avoids drift if a field is added later.
"""
from __future__ import annotations

from pydantic import BaseModel


class ResetRequest(BaseModel):
    backend: str


class WarmupRequest(BaseModel):
    backend: str
    conv_index: int
