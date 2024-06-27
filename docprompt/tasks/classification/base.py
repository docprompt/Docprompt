from abc import abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, model_validator
from typing_extensions import override

from docprompt.schema.pipeline import DocumentNode
from docprompt.tasks.base import AbstractPageTaskProvider

if TYPE_CHECKING:
    from docprompt.schema.pipeline import DocumentNode

LabelType = Union[List[str], Enum]
ConfidenceLevels = Literal["low", "medium", "high"]


class ClassificationTypes(str, Enum):
    SINGLE_LABEL = "single_label"
    MULTI_LABEL = "multi_label"
    BINARY = "binary"


class ClassificationInput(BaseModel):
    type: ClassificationTypes
    labels: LabelType
    descriptions: Optional[List[str]] = None

    def resolve_labels(self):
        if isinstance(self.labels, Enum):
            return [self.labels.value]
        return self.labels

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


class ClassificationOutput(BaseModel):
    type: ClassificationTypes
    labels: List[str]


class ScoredClassificationOutput(ClassificationOutput):
    scores: List[ConfidenceLevels]


class BaseClassificationProvider(
    AbstractPageTaskProvider[ClassificationInput, ClassificationOutput]
):
    input: ClassificationInput
    output: ClassificationOutput

    @override
    @abstractmethod
    def process_document_node(
        self,
        document_node: DocumentNode,
        task_input: ClassificationInput,
        start: int | None = None,
        stop: int | None = None,
        contribute_to_document: bool = True,
        **kwargs,
    ) -> Dict[int, ClassificationOutput]: ...


class BaseScoredClassificationProvider(
    AbstractPageTaskProvider[ClassificationInput, ScoredClassificationOutput]
):
    input: ClassificationInput
    output: ScoredClassificationOutput

    @override
    @abstractmethod
    def process_document_node(
        self,
        document_node: DocumentNode,
        task_input: ClassificationInput,
        start: int | None = None,
        stop: int | None = None,
        contribute_to_document: bool = True,
        **kwargs,
    ) -> Dict[int, ScoredClassificationOutput]: ...
