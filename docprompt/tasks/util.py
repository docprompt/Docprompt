import base64
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, Union

_init_context_var = ContextVar("_init_context_var", default=None)


@contextmanager
def init_context(value: Dict[str, Any]) -> Iterator[None]:
    token = _init_context_var.set(value)
    try:
        yield
    finally:
        _init_context_var.reset(token)


def bytes_to_base64_uris(img: Union[bytes, str]) -> str:
    """
    Convert image bytes to base64-encoded image URIs or return the string if it's already in the correct format.

    :param img: Either bytes of an image or a string potentially already in base64 URI format
    :return: A base64-encoded image URI string
    """
    # Check if input is already a string
    if isinstance(img, str):
        # Check if it's already in the correct format
        if img.startswith("data:image/"):
            return img
        else:
            raise ValueError("Input string is not in the expected base64 URI format")

    # If it's bytes, encode it
    elif isinstance(img, bytes):
        return f"data:image/png;base64,{base64.b64encode(img).decode('utf-8')}"

    else:
        raise TypeError("Input must be either bytes or a string")
