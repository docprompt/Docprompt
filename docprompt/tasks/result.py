from abc import abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar, Optional, TypeVar

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from docprompt.schema.pipeline import DocumentNode


class BaseResult(BaseModel):
    provider_name: str = Field(
        description="The name of the provider which produced the result"
    )
    when: datetime = Field(
        default_factory=datetime.now, description="The time the result was produced"
    )

    task_name: ClassVar[str]

    @property
    def task_key(self):
        return f"{self.provider_name}_{self.task_name}"

    @abstractmethod
    def contribute_to_document_node(
        self, document_node: "DocumentNode", **kwargs
    ) -> None:
        """
        Contribute this task result to the document node or a specific page node.

        :param document_node: The DocumentNode to contribute to
        :param page_number: If provided, contribute to a specific page. If None, contribute to the document.
        """


class BaseDocumentResult(BaseResult):
    def contribute_to_document_node(
        self, document_node: "DocumentNode", **kwargs
    ) -> None:
        document_node.metadata.task_results[self.task_key] = self


class BasePageResult(BaseResult):
    def contribute_to_document_node(
        self, document_node: "DocumentNode", page_number: Optional[int] = None, **kwargs
    ) -> None:
        assert (
            page_number is not None
        ), "Page number must be provided for page level results"
        assert (
            0 < page_number <= len(document_node)
        ), "Page number must be less than or equal to the number of pages in the document"

        page_node = document_node.page_nodes[page_number - 1]
        page_node.metadata.task_results[self.task_key] = self


TTaskInput = TypeVar("TTaskInput")  # What invoke requires
TTaskConfig = TypeVar("TTaskConfig")  # Task specific config like classification labels
PageTaskResult = TypeVar("PageTaskResult", bound=BasePageResult)
DocumentTaskResult = TypeVar("DocumentTaskResult", bound=BaseDocumentResult)
PageOrDocumentTaskResult = TypeVar("PageOrDocumentTaskResult", bound=BaseResult)
