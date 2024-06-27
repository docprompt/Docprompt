"""A utility file for running inference with various LLM providers."""

import asyncio
import os
from typing import TYPE_CHECKING, List, TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm.asyncio import tqdm

if TYPE_CHECKING:
    from docprompt.tasks.message import OpenAIMessage

OpenAIMessage = TypeVar("OpenAIMessage")


def get_anthropic_retry_decorator():
    import anthropic

    return retry(
        wait=wait_random_exponential(multiplier=0.5, max=60),
        stop=stop_after_attempt(14),
        retry=retry_if_exception_type(anthropic.RateLimitError)
        | retry_if_exception_type(anthropic.InternalServerError)
        | retry_if_exception_type(anthropic.APITimeoutError),
        reraise=True,
    )


async def run_inference_anthropic(
    model_name: str, messages: List[OpenAIMessage], **kwargs
):
    """Run inference using an Anthropic model asynchronously."""
    from anthropic import AsyncAnthropic

    api_key = kwargs.get("api_key", os.environ.get("ANTHROPIC_API_KEY"))
    client = AsyncAnthropic(api_key=api_key)

    system = None
    if messages and messages[0].role == "system":
        system = messages[0].content
        messages = messages[1:]

    processed_messages = []
    for msg in messages:
        if isinstance(msg.content, list):
            processed_content = []
            for content in msg.content:
                if getattr(content, "type", None) == "image_url":
                    url = content.image_url.url
                    base64 = url.split("image/png;base64,")[1]
                    image_content_block = {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64,
                        },
                    }
                    processed_content.append(image_content_block)
                else:
                    processed_content.append(content)
            msg.content = processed_content
        processed_messages.append(msg)

    response = await client.messages.create(
        model=model_name, max_tokens=512, system=system, messages=processed_messages
    )

    content = response.content[0].text

    return content


async def run_batch_inference_anthropic(
    model_name: str, messages: List[List[OpenAIMessage]], **kwargs
):
    """Run batch inference using an Anthropic model asynchronously."""
    retry_decorator = get_anthropic_retry_decorator()

    @retry_decorator
    async def process_message_set(msg_set):
        return await run_inference_anthropic(model_name, msg_set, **kwargs)

    tasks = [process_message_set(msg_set) for msg_set in messages]

    responses = []
    for f in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Processing messages"
    ):
        response = await f
        responses.append(response)

    return responses
