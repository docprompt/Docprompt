import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Optional, Union

from pydantic import BaseModel, Field, model_validator

from docprompt import DocumentNode
from docprompt.tasks.base import AbstractPageTaskProvider
from docprompt.tasks.parser import BaseOutputParser
from docprompt.tasks.result import BasePageResult

from ..capabilities import PageLevelCapabilities

LabelType = Union[List[str], Enum, str]


class ConfidenceLevel(str, Enum):
    """The confidence level of the classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ClassificationTypes(str, Enum):
    SINGLE_LABEL = "single_label"
    MULTI_LABEL = "multi_label"
    BINARY = "binary"


class ClassificationConfig(BaseModel):
    type: ClassificationTypes
    labels: LabelType
    descriptions: Optional[List[str]] = Field(
        None, description="The descriptions for each label (if any)."
    )

    instructions: Optional[str] = Field(
        None,
        description="Additional instructions to pass to the LLM for the task. Required for Binary Classification.",
    )

    confidence: bool = Field(False)

    def resolve_labels(self):
        if isinstance(self.labels, Enum):
            return [self.labels.value]
        return self.labels

    @model_validator(mode="before")
    def validate_label_bindings(cls, data: Any) -> Any:
        """Validate the the label/description bindings based on the type."""

        classification_type = data.get("type", None)
        match classification_type:
            # Here, we just want to assert that we have a lables value
            case ClassificationTypes.SINGLE_LABEL:
                labels = data.get("labels", None)
                if not labels:
                    raise ValueError(
                        "labels must be provided for single_label classification"
                    )
                return data

            # Here, the labels must be hardcoded to YES/NO. Instead we just want
            # a single instruction value
            case ClassificationTypes.BINARY:
                instructions = data.get("instructions", None)
                if not instructions:
                    raise ValueError(
                        "instructions must be provided for binary classification"
                    )
                data["labels"] = ["YES", "NO"]
                return data

            case ClassificationTypes.MULTI_LABEL:
                labels = data.get("labels", None)
                if not labels:
                    raise ValueError(
                        "labels must be provided for multi_label classification"
                    )
                return data

        # Anything that get's to here will hit a validation error anyways. We just return
        # to avoid redundant error raising.
        return data

    @model_validator(mode="after")
    def validate_descriptions_length(self):
        if self.descriptions is not None:
            labels = self.labels
            if labels is not None:
                if isinstance(labels, Enum):
                    if len(self.descriptions) != len(labels.__members__):
                        raise ValueError(
                            "descriptions must have the same length as labels"
                        )
                elif len(self.descriptions) != len(labels):
                    raise ValueError("descriptions must have the same length as labels")
        return self

    @property
    def formatted_labels(self):
        """Produce the formatted labels for the prompt template."""
        raw_labels = self.resolve_labels()
        if self.descriptions:
            for label, description in zip(raw_labels, self.descriptions):
                yield f"{label}: {description}"
        else:
            yield from raw_labels


class ClassificationOutput(BasePageResult):
    type: ClassificationTypes
    labels: LabelType
    score: Optional[ConfidenceLevel] = Field(None)
    task_name: str = "classification"


class BasePageClassificationOutputParser(
    ABC, BaseOutputParser[ClassificationConfig, ClassificationOutput]
):
    """The output parser for the page classification system."""

    name: str = Field(...)
    type: ClassificationTypes = Field(...)
    labels: LabelType = Field(...)
    confidence: bool = Field(False)

    @classmethod
    def from_task_input(cls, task_input: ClassificationConfig, provider_name: str):
        return cls(
            type=task_input.type,
            name=provider_name,
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

    @abstractmethod
    def parse(self, text: str) -> ClassificationOutput:
        pass


class BaseClassificationProvider(
    AbstractPageTaskProvider[bytes, ClassificationConfig, ClassificationOutput]
):
    """
    The base classification provider.
    """

    capabilities = [PageLevelCapabilities.PAGE_CLASSIFICATION]

    class Meta:
        abstract = True

    def process_document_node(
        self,
        document_node: "DocumentNode",
        task_config: ClassificationConfig = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        contribute_to_document: bool = True,
        **kwargs,
    ):
        raster_bytes = []
        for page_number in range(start or 1, (stop or len(document_node)) + 1):
            image_bytes = document_node.page_nodes[
                page_number - 1
            ].rasterizer.rasterize("default")
            raster_bytes.append(image_bytes)

        results = self._invoke(raster_bytes, config=task_config, **kwargs)

        return {
            i: res
            for i, res in zip(
                range(start or 1, (stop or len(document_node)) + 1), results
            )
        }
