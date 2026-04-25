from contextlib import contextmanager
from typing import Any, Iterator

from app.config import CFG


_lf: Any = None


def init_langfuse() -> None:
    global _lf
    if not (CFG.langfuse_host and CFG.langfuse_public_key and CFG.langfuse_secret_key):
        return
    try:
        from langfuse import Langfuse
    except Exception:
        return
    _lf = Langfuse(
        host=CFG.langfuse_host,
        public_key=CFG.langfuse_public_key,
        secret_key=CFG.langfuse_secret_key,
    )


def get_client() -> Any:
    return _lf


@contextmanager
def trace(name: str, *, input: Any = None, **metadata: Any) -> Iterator[Any]:
    """Open a Langfuse trace. Pass `input=...` to populate the Input panel.
    Set `output` later via the yielded handle: `t.update(output=...)`."""
    if _lf is None:
        yield None
        return
    kwargs: dict[str, Any] = {
        "name": name,
        "metadata": {"agent_id": CFG.agent_id, **metadata},
    }
    if input is not None:
        kwargs["input"] = input
    t = _lf.trace(**kwargs)
    try:
        yield t
    finally:
        try:
            _lf.flush()
        except Exception:
            pass
