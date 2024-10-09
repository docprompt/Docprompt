from typing import Any, Dict, List

try:
    import litellm
except ImportError:
    print(
        "litellm is required for this function. Install with `pip install docprompt[litellm]`"
    )
    raise


def get_sync_litellm_callable(model: str, /, **kwargs):
    if "messages" in kwargs:
        raise ValueError("messages should only be passed at runtime")

    def wrapper(messages: List[Dict[str, Any]]):
        response = litellm.completion(model=model, messages=messages, **kwargs)

        return response.to_dict()

    return wrapper


def get_async_litellm_callable(model: str, /, **kwargs):
    if "messages" in kwargs:
        raise ValueError("messages should only be passed at runtime")

    async def wrapper(messages: List[Dict[str, Any]]):
        response = await litellm.acompletion(model=model, messages=messages, **kwargs)

        return response.to_dict()

    return wrapper
