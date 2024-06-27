"""The antrhopic implementation of page level calssification."""

import re
from typing import Dict, Union

from jinja2 import Template
from pydantic import Field

from docprompt.schema.pipeline import DocumentNode
from docprompt.tasks.message import OpenAIMessage
from docprompt.tasks.parser import BaseOutputParser

from .base import (
    BaseClassificationProvider,
    ClassificationInput,
    ClassificationOutput,
    ClassificationTypes,
    ConfidenceLevel,
    LabelType,
)

PAGE_CLASSIFICATION_SYSTEM_PROMPT = Template(
    """
You are given a page of a PDF document. Classifiy the page by following the instructions below:
{% if input.type == "binary" %}\
{{ input.instructions }}\
\
Classify the page with a binary label:
YES/NO
{% else %}\

{% if input.instructions %}\
{{ input.instructions }}
{% endif %}\

{% if input.type == 'single_label' %}\
Classify the page as one of the following:
{% elif input.type == 'multi_label' %}\
Classify the page with all labels that apply:
{% endif %}\

{% for label in input.formatted_labels %}\
- {{ label }}
{% endfor %}\
{% endif %}\

Answer in the following format:

Reasoning: { your reasoning and analysis }
{% if input.type == "binary" %}\
Answer: { YES/NO (only the label and NOTHING ELSE)}
{% elif input.type == "single_label" %}\
Answer: { label (only the label and NOTHING ELSE)}
{% else %}\
Answer: { label1, label2, ... (comma separated list of labels and NOTHING ELSE)}
{% endif %}\
{% if input.confidence %}\
Confidence: { low, medium, high }
{% endif %}\
""".strip()
)


class PageClassificationOutputParser(
    BaseOutputParser[ClassificationInput, ClassificationOutput]
):
    """The output parser for the page classification system."""

    type: ClassificationTypes = Field(...)
    labels: LabelType = Field(...)
    confidence: bool = Field(False)

    @classmethod
    def from_task_input(cls, task_input: ClassificationInput):
        return cls(
            type=task_input.type,
            labels=task_input.labels,
            confidence=task_input.confidence,
        )

    def resolve_match(self, _match: Union[re.Match, None]) -> LabelType:
        """Get the regex pattern for the output parser."""

        if not _match:
            raise ValueError("Could not find the answer in the text.")

        val = _match.group(1)
        match self.type:
            case ClassificationTypes.BINARY:
                if val not in self.labels:
                    raise ValueError(f"Invalid label: {val}")
                return val

            case ClassificationTypes.SINGLE_LABEL:
                if val not in self.labels:
                    raise ValueError(f"Invalid label: {val}")
                return val

            case ClassificationTypes.MULTI_LABEL:
                labels = val.split(", ")
                for label in labels:
                    if label not in self.labels:
                        raise ValueError(f"Invalid label: {label}")
                return labels

            case _:
                raise ValueError(f"Invalid classification type: {self.type}")

    def resolve_confidence(self, _match: Union[re.Match, None]) -> ConfidenceLevel:
        """Get the confidence level from the text."""

        if not _match:
            return None

        val = _match.group(1).lower()

        return ConfidenceLevel(val)

    def parse(self, text: str) -> ClassificationOutput:
        """Parse the results of the classification task."""
        pattern = re.compile(r"Answer: (.+)")
        match = pattern.search(text)

        result = self.resolve_match(match)

        if self.confidence:
            conf_pattern = re.compile(r"Confidence: (.+)")
            conf_match = conf_pattern.search(text)
            conf_result = self.resolve_confidence(conf_match)

            return ClassificationOutput(
                type=self.type, labels=result, score=conf_result
            )

        return ClassificationOutput(type=self.type, labels=result)


class AnthropicClassificationProvider(BaseClassificationProvider):
    """The Anthropic implementation of unscored page classification."""

    def process_document_pages(
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

            # TODO: Optimize rastrization with concurrency??
            image_uri = page.rasterizer.rasterize_to_data_uri("default")
            _image_message = OpenAIMessage.from_image_uri(image_uri)

            messages.append(
                [
                    PAGE_CLASSIFICATION_SYSTEM_PROMPT.render(input=task_input),
                    # image_message,
                ]
            )

        # Process all messages concurrently here
        assert (
            len(messages) == stop - start
        ), f"Invalid number of messages: {len(messages)}"

        _parser = PageClassificationOutputParser.from_task_input(task_input)

        # NOTE: A mock implementation of a mini-chain for running inference in parallel
        # async def process_page(messages: List[OpenAIMessage]):
        #     result = await run_inference(messages)
        #     return parser.parse(result)

        # NOTE: Executing the tasks in a gather loop to run in parallel
        # tasks = asyncio.gather(*[process_page(msgs) for msgs in messages])
        # results = asyncio.run(tasks)
        # return results

        # NOTE: For now return document messages array
        return messages

    def contribute_to_document_node(self, *args, **kwargs):
        """Eventually this will define how the results are added to the document node.

        NOTE: How should this be configurable for the user? Theoretically, they should be
        able to declare how each page classification result is added to the Page/Document
        metadata.

        Potentially could this be done through a config argument that is passed as a kwarg? For
        example, letting the user declare the field of the metadata model that the result should
        be stored in. Maybe a lambda function?

        We need sometheing to make storage of result configurable.
        """
        pass
