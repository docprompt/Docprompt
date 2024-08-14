import base64
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, List

_init_context_var = ContextVar("_init_context_var", default=None)


@contextmanager
def init_context(value: Dict[str, Any]) -> Iterator[None]:
    token = _init_context_var.set(value)
    try:
        yield
    finally:
        _init_context_var.reset(token)


def bytes_to_base64_uris(img: bytes) -> List[str]:
    """Convert image bytes to base64-encoded image URIs."""
    return f"data:image/png;base64,{base64.b64encode(img).decode('utf-8')}"
