"""A utility file for running inference with various LLM providers."""

import asyncio
import os
from typing import List

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm.asyncio import tqdm

from docprompt.tasks.message import OpenAIComplexContent, OpenAIMessage


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


def get_openai_retry_decorator():
    import openai

    return retry(
        wait=wait_random_exponential(multiplier=0.5, max=60),
        stop=stop_after_attempt(14),
        retry=retry_if_exception_type(openai.RateLimitError)
        | retry_if_exception_type(openai.InternalServerError)
        | retry_if_exception_type(openai.APITimeoutError),
        reraise=True,
    )


async def run_inference_anthropic(
    model_name: str, messages: List[OpenAIMessage], **kwargs
) -> str:
    """Run inference using an Anthropic model asynchronously."""
    from anthropic import AsyncAnthropic

    api_key = kwargs.pop("api_key", os.environ.get("ANTHROPIC_API_KEY"))
    base_url = kwargs.pop("base_url", os.environ.get("ANTHROPIC_BASE_URL"))
    client = AsyncAnthropic(api_key=api_key, base_url=base_url)

    system = None
    if messages and messages[0].role == "system":
        system = messages[0].content
        messages = messages[1:]

    processed_messages = []
    for msg in messages:
        if isinstance(msg.content, list):
            processed_content = []
            for content in msg.content:
                if isinstance(content, OpenAIComplexContent):
                    content = content.to_anthropic_message()
                    processed_content.append(content)
                else:
                    pass
                    # raise ValueError(f"Invalid content type: {type(content)} Expected OpenAIComplexContent")

            dumped = msg.model_dump()
            dumped["content"] = processed_content
            processed_messages.append(dumped)
        else:
            processed_messages.append(msg)

    client_kwargs = {
        "model": model_name,
        "max_tokens": 2048,
        "messages": processed_messages,
        **kwargs,
    }

    if system:
        client_kwargs["system"] = system

    response = await client.messages.create(**client_kwargs)

    content = response.content[0].text

    return content


async def run_batch_inference_anthropic(
    model_name: str, messages: List[List[OpenAIMessage]], **kwargs
) -> List[str]:
    """Run batch inference using an Anthropic model asynchronously."""
    retry_decorator = get_anthropic_retry_decorator()

    @retry_decorator
    async def process_message_set(msg_set, index: int):
        return await run_inference_anthropic(model_name, msg_set, **kwargs), index

    tasks = [process_message_set(msg_set, i) for i, msg_set in enumerate(messages)]

    # TODO: Need cleaner implementation to ensure message ordering is perserved
    responses: List[str] = []
    for f in tqdm(asyncio.as_completed(tasks), desc="Processing messages"):
        response, index = await f
        responses.append((response, index))

    # Sort and extract the responses
    responses.sort(key=lambda x: x[1])
    responses = [r[0] for r in responses]

    return responses
