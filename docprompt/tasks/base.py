from abc import ABC, abstractmethod
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
)

from pydantic import BaseModel, Field

from docprompt.schema.document import Document

from .capabilities import DocumentLevelCapabilities, PageLevelCapabilities

if TYPE_CHECKING:
    from docprompt.schema.pipeline import DocumentNode


class BaseResult(BaseModel):
    provider_name: str = Field(
        description="The name of the provider which produced the result"
    )
    when: datetime = Field(
        default_factory=datetime.now, description="The time the result was produced"
    )


class BaseDocumentResult(BaseResult):
    document_name: str = Field(description="The name of the document")
    file_hash: str = Field(description="The hash of the document")


class BasePageResult(BaseDocumentResult):
    page_number: int = Field(description="The page number")


TTaskInput = TypeVar("TTaskInput")
PageTaskResult = TypeVar("PageTaskResult", bound=BasePageResult)
DocumentTaskResult = TypeVar("DocumentTaskResult", bound=BaseDocumentResult)
PageOrDocumentTaskResult = TypeVar("PageOrDocumentTaskResult", bound=BaseResult)


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


class AbstractPageTaskProvider(ABC, Generic[TTaskInput, PageTaskResult]):
    """
    A task provider performs a specific, repeatable task on a document or its pages
    """

    name: str
    capabilities: List[PageLevelCapabilities]
    requires_input: bool

    @abstractmethod
    def process_document_pages(
        self,
        document: Document,
        task_input: Optional[TTaskInput] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        **kwargs,
    ) -> Dict[int, PageTaskResult]:
        raise NotImplementedError

    @abstractmethod
    def contribute_to_document_node(
        self,
        document_node: "DocumentNode",
        results: Dict[int, PageTaskResult],
    ) -> None:
        """
        Adds the results of this task to the document node and/or its page nodes
        """
        pass

    def process_document_node(
        self,
        document_node: "DocumentNode",
        task_input: Optional[TTaskInput] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        contribute_to_document: bool = True,
        **kwargs,
    ) -> Dict[int, PageTaskResult]:
        results = self.process_document_pages(
            document_node.document,
            task_input=task_input,
            start=start,
            stop=stop,
            **kwargs,
        )

        if contribute_to_document:
            self.contribute_to_document_node(document_node, results)

        return results


class AbstractDocumentTaskProvider(ABC, Generic[TTaskInput, DocumentTaskResult]):
    """
    A task provider performs a specific, repeatable task on a document
    """

    name: str
    capabilities: List[DocumentLevelCapabilities]

    @abstractmethod
    def process_document(
        self, document: Document, task_input: Optional[TTaskInput] = None, **kwargs
    ) -> DocumentTaskResult:
        raise NotImplementedError

    @abstractmethod
    def contribute_to_document_node(
        self,
        document_node: "DocumentNode",
        result: DocumentTaskResult,
    ) -> None:
        """
        Adds the results of this task to the document node
        """
        pass

    def process_document_node(
        self,
        document_node: "DocumentNode",
        task_input: Optional[TTaskInput] = None,
        contribute_to_document: bool = True,
        **kwargs,
    ) -> DocumentTaskResult:
        result = self.process_document(
            document_node.document, task_input=task_input, **kwargs
        )

        if contribute_to_document:
            self.contribute_to_document_node(document_node, result)

        return result
