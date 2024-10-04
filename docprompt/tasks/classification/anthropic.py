"""The anthropic implementation of page level classification."""

import re
from typing import Iterable, List

from pydantic import Field

from docprompt.tasks.message import OpenAIComplexContent, OpenAIImageURL, OpenAIMessage
from docprompt.utils import inference

from .base import (
    BaseClassificationProvider,
    BasePageClassificationOutputParser,
    ClassificationConfig,
    ClassificationOutput,
)


def get_classification_system_prompt(input: ClassificationConfig) -> str:
    prompt_parts = [
        "You are a classification expert. You are given a single page to perform a classification task on.\n"
    ]

    if input.instructions:
        prompt_parts.append(f"Task Instructions:\n{input.instructions}\n\n")

    if input.type == "binary":
        prompt_parts.append(
            'You must classify the page with a binary label:\n"YES"/"NO"\n'
        )
    else:
        classification_task = (
            "all labels that apply"
            if input.type == "multi_label"
            else "one of the following"
        )
        prompt_parts.append(f"Classify the page as {classification_task}:\n")
        for label in input.formatted_labels:
            prompt_parts.append(f"- {label}\n")
        prompt_parts.append(
            "\nThese are the only label values you may use when providing your classifications!\n"
        )

    prompt_parts.append(
        "\nIt is crucial that your response is accurate and provides a valid answer using "
    )
    if input.type == "multi_label":
        prompt_parts.append("the labels ")
    else:
        prompt_parts.append("one of the labels ")
    prompt_parts.append(
        "above. There are consequences for providing INVALID or INACCURATE labels.\n\n"
    )

    prompt_parts.append(
        "Answer in the following format:\n\nReasoning: { your reasoning and analysis }\n"
    )

    if input.type == "binary":
        prompt_parts.append('Answer: { "YES" or "NO" }\n')
    elif input.type == "single_label":
        prompt_parts.append('Answer: { "label-value" }\n')
    else:
        prompt_parts.append('Answer: { "label-value", "label-value", ... }\n')

    if input.confidence:
        prompt_parts.append("Confidence: { low, medium, high }\n")

    prompt_parts.append(
        "\nYou MUST ONLY use the labels provided and described above. Do not use ANY additional labels.\n"
    )

    return "".join(prompt_parts).strip()


class AnthropicPageClassificationOutputParser(BasePageClassificationOutputParser):
    """The output parser for the page classification system."""

    def parse(self, text: str) -> ClassificationOutput:
        """Parse the results of the classification task."""
        pattern = re.compile(r"Answer:\s*(?:['\"`]?)(.+?)(?:['\"`]?)\s*$", re.MULTILINE)
        match = pattern.search(text)

        result = self.resolve_match(match)

        if self.confidence:
            conf_pattern = re.compile(r"Confidence: (.+)")
            conf_match = conf_pattern.search(text)
            conf_result = self.resolve_confidence(conf_match)

            return ClassificationOutput(
                type=self.type,
                labels=result,
                score=conf_result,
                provider_name=self.name,
            )

        return ClassificationOutput(
            type=self.type, labels=result, provider_name=self.name
        )


def _prepare_messages(
    document_images: Iterable[bytes],
    config: ClassificationConfig,
):
    messages = []

    for image_bytes in document_images:
        messages.append(
            [
                OpenAIMessage(
                    role="user",
                    content=[
                        OpenAIComplexContent(
                            type="image_url",
                            image_url=OpenAIImageURL(url=image_bytes),
                        ),
                        OpenAIComplexContent(
                            type="text",
                            text=get_classification_system_prompt(config),
                        ),
                    ],
                ),
            ]
        )

    return messages


class AnthropicClassificationProvider(BaseClassificationProvider):
    """The Anthropic implementation of unscored page classification."""

    name = "anthropic"

    anthropic_model_name: str = Field("claude-3-haiku-20240307")

    async def _ainvoke(
        self, input: Iterable[bytes], config: ClassificationConfig = None, **kwargs
    ) -> List[ClassificationOutput]:
        messages = _prepare_messages(input, config)

        parser = AnthropicPageClassificationOutputParser.from_task_input(
            config, provider_name=self.name
        )

        model_name = kwargs.pop("model_name", self.anthropic_model_name)
        completions = await inference.run_batch_inference_anthropic(
            model_name, messages, **kwargs
        )
        return [parser.parse(res) for res in completions]
