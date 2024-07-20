"""The antrhopic implementation of page level calssification."""

import re
from typing import Iterable, List

from jinja2 import Template
from pydantic import Field

from docprompt.tasks.message import OpenAIComplexContent, OpenAIImageURL, OpenAIMessage
from docprompt.utils import inference

from .base import (
    BaseClassificationProvider,
    BasePageClassificationOutputParser,
    ClassificationConfig,
    ClassificationOutput,
)

PAGE_CLASSIFICATION_SYSTEM_PROMPT = Template(
    """
You are a classification expert. You are given a single page to perform a classification task on.

{% if input.instructions %}\
Task Instructions:
{{ input.instructions }}

{% endif %}\
{% if input.type == "binary" %}\
You must classify the page with a binary label:
"YES"/"NO"
{% else %}\
Classify the page as {% if input.type == 'multi_label' %}all labels that apply{% else %}one of the following{% endif %}:
{% for label in input.formatted_labels %}
- {{ label }}
{% endfor %}\

These are the only label values you may use when providing your classifications!
{% endif %}\

It is crucial that your response is accurate and provides a valid answer using \
{% if input.type == 'multi_label' %}\
the labels \
{% else %}\
one of the labels \
{% endif %}\
above. There are consequences for providing INVALID or INACCURATE labels.

Answer in the following format:

Reasoning: { your reasoning and analysis }
{% if input.type == "binary" %}\
Answer: { "YES" or "NO" }
{% elif input.type == "single_label" %}\
Answer: { "label-value" }
{% else %}\
Answer: { "label-value", "label-value", ... }
{% endif %}\
{% if input.confidence %}\
Confidence: { low, medium, high }
{% endif %}\

You MUST ONLY use the labels provided and described above. Do not use ANY additional labels.
""".strip()
)


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
                            text=PAGE_CLASSIFICATION_SYSTEM_PROMPT.render(input=config),
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
