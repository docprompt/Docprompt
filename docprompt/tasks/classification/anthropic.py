"""The antrhopic implementation of page level calssification."""

from typing import Dict

from jinja2 import Template
from pydantic import BaseModel, Field

from docprompt.schema.pipeline import DocumentNode
from docprompt.tasks.message import OpenAIMessage

from .base import (
    BaseClassificationProvider,
    ClassificationInput,
    ClassificationOutput,
    ClassificationTypes,
    LabelType,
)

PAGE_CLASSIFICATION_SYSTEM_PROMPT = Template(
    """
{% if input.type == "binary" %}\
...binary prompt


{% else %}\
You are given a page from a deposition document. Classify the page into one of the following:

{% for label in input.formatted_labels %}\
- {{ label }}
{% endfor %}

For each page provided to you, response
{% endif %}\

Answer in the following format:

Reasoning: { your reasoning and analysis }
Answer: { the page type of the deposition page }
""".strip()
)


class PageClassificationOutputParser(BaseModel):
    """The output parser for the page classification system."""

    type: ClassificationTypes = Field(...)
    labels: LabelType = Field(...)

    def parse(self, text: str) -> ClassificationOutput:
        pass


class AnthropicClassificationProvider(BaseClassificationProvider):
    """The Anthropic implementation of unscored page classification."""

    def process_document_node(
        self,
        document_node: DocumentNode,
        task_input: ClassificationInput,
        start: int | None = None,
        stop: int | None = None,
        contribute_to_document: bool = True,
        **kwargs,
    ) -> Dict[int, ClassificationOutput]:
        start = start or 0
        stop = stop or len(document_node.page_nodes)

        assert (
            0 <= start < stop <= len(document_node.page_nodes)
        ), f"Invalid start and stop values: {start}, {stop}"

        messages = []
        for i in range(start, stop):
            page = document_node.page_nodes[i]

            image_uri = page.rasterizer.rasterize_to_data_uri("default")
            image_message = OpenAIMessage.from_image_uri(image_uri)

            messages.append(
                [
                    PAGE_CLASSIFICATION_SYSTEM_PROMPT.render(input=task_input),
                    image_message,
                ]
            )

        return messages

    def process_document_pages(self, *args, **kwargs):
        pass

    def contribute_to_document_node(self, *args, **kwargs):
        pass
