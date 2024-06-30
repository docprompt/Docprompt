from abc import abstractmethod
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Dict,
    Generic,
    TypeVar,
)

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
    def contribute_to_document_node(self, document_node: "DocumentNode") -> None:
        """
        Contribute this task result to the document node or a specific page node.

        :param document_node: The DocumentNode to contribute to
        :param page_number: If provided, contribute to a specific page. If None, contribute to the document.
        """


class BaseDocumentResult(BaseResult):
    document_name: str = Field(description="The name of the document")
    file_hash: str = Field(description="The hash of the document")

    def contribute_to_document_node(self, document_node: "DocumentNode") -> None:
        document_node.metadata.task_results[self.task_key] = self


class BasePageResult(BaseResult):
    page_number: int = Field(description="The page number")

    def contribute_to_document_node(self, document_node: "DocumentNode") -> None:
        page_node = document_node.page_nodes[self.page_number - 1]
        print(page_node.metadata)
        page_node.metadata.task_results[self.task_key] = self


TTaskInput = TypeVar("TTaskInput")  # What invoke requires
TTaskConfig = TypeVar("TTaskConfig")  # Task specific config like classification labels
PageTaskResult = TypeVar("PageTaskResult", bound=BasePageResult)
DocumentTaskResult = TypeVar("DocumentTaskResult", bound=BaseDocumentResult)
PageOrDocumentTaskResult = TypeVar("PageOrDocumentTaskResult", bound=BaseResult)


# TODO: Will likely depricate this now in favor of the Task Result Descriptor on the BaseMetadata
class ResultContainer(BaseModel, Generic[PageOrDocumentTaskResult]):
    """
    Represents a container for results of a task
    """

    results: Dict[str, PageOrDocumentTaskResult] = Field(
        description="The results of the task, keyed by provider", default_factory=dict
    )

    @property
    def result(self):
        return next(iter(self.results.values()), None)
