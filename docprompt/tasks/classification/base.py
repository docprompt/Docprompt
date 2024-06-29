from abc import abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, model_validator
from typing_extensions import override

from docprompt.schema.pipeline import DocumentNode
from docprompt.tasks.base import AbstractPageTaskProvider

if TYPE_CHECKING:
    from docprompt.schema.pipeline import DocumentNode

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


class ClassificationInput(BaseModel):
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


class ClassificationOutput(BaseModel):
    type: ClassificationTypes
    labels: LabelType
    score: Optional[ConfidenceLevel] = Field(None)


class BaseClassificationProvider(
    AbstractPageTaskProvider[ClassificationInput, ClassificationOutput]
):
    # NOTE: We override the method here for more accurate type-hinting
    @override
    async def aprocess_document_node(
        self,
        document_node: DocumentNode,
        task_input: ClassificationInput,
        start: int | None = None,
        stop: int | None = None,
        contribute_to_document: bool = True,
        **kwargs,
    ) -> Dict[int, ClassificationOutput]:
        # NOTE: We need to pass the document node to `process_document_pages` instead of `document_node.document`
        # because the `DocumentNode` object contains an easy iterable of `PageNode` objects.
        # The defualt implementation of `process_document_nodes` passes, the `document_node.document` object to
        # `process_document_pages` which is not iterable.

        # Essentially an identical implementation as the `AbstractPageTaskProvider` class, but with
        # `DocumentNode` instead of `Document` as the first argument to `process_document_pages`
        kwargs = {**(self.provider_kwargs or {}), **kwargs}
        results = await self.aprocess_document_pages(
            document_node,
            task_input,
            start=start,
            stop=stop,
            **kwargs,
        )

        if contribute_to_document:
            self.contribute_to_document_node(document_node, results)

        return results

    @abstractmethod
    @override
    def process_document_pages(
        self,
        document_node: DocumentNode,  # NOTE: We override here to DocumentNode, instead of Document
        task_input: ClassificationInput,
        start: int | None = None,
        stop: int | None = None,
        contribute_to_document: bool = True,
        **kwargs,
    ) -> Dict[int, ClassificationOutput]: ...
