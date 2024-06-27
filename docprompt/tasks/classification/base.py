from abc import abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Literal, Union

from pydantic import BaseModel
from typing_extensions import override

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

    def resolve_labels(self):
        if isinstance(self.labels, Enum):
            return [self.labels.value]
        return self.labels


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