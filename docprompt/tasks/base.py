from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field

from docprompt._decorators import flexible_methods
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

    task_name: ClassVar[str]

    @property
    def task_key(self):
        return f"{self.provider_name}_{self.task_name}"

    @abstractmethod
    def contribute_to_document_node(
        self, document_node: "DocumentNode", page_number: int = None
    ) -> None:
        """
        Contribute this task result to the document node or a specific page node.

        :param document_node: The DocumentNode to contribute to
        :param page_number: If provided, contribute to a specific page. If None, contribute to the document.
        """
        pass


class BaseDocumentResult(BaseResult):
    document_name: str = Field(description="The name of the document")
    file_hash: str = Field(description="The hash of the document")

    def contribute_to_document_node(
        self, document_node: "DocumentNode", page_number: int = None
    ) -> None:
        document_node.metadata.task_results[self.task_key] = self


class BasePageResult(BaseResult):
    page_number: int = Field(description="The page number")

    def contribute_to_document_node(
        self, document_node: "DocumentNode", page_number: int = None
    ) -> None:
        assert page_number is not None, "Page number must be provided for a page result"
        assert page_number > 0, "Page number must be greater than 0"

        page_node = document_node.page_nodes[page_number - 1]
        page_node.metadata.task_results[self.task_key] = self


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


@flexible_methods(
    ("process_document_pages", "aprocess_document_pages"),
    ("process_document_node", "aprocess_document_node"),
)
class AbstractPageTaskProvider(ABC, Generic[TTaskInput, PageTaskResult]):
    """
    A task provider performs a specific, repeatable task on a document or its pages
    """

    name: str
    capabilities: List[PageLevelCapabilities]
    requires_input: bool

    provider_kwargs: Dict[str, Any]

    @classmethod
    def with_kwargs(cls, **kwargs):
        """Create the provider with kwargs."""
        obj = cls()
        obj.provider_kwargs = kwargs
        return obj

    async def aprocess_document_pages(
        self,
        document: Document,
        task_input: Optional[TTaskInput] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        **kwargs,
    ):
        raise NotImplementedError

    def process_document_pages(
        self,
        document: Document,
        task_input: Optional[TTaskInput] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        **kwargs,
    ) -> Dict[int, PageTaskResult]:
        raise NotImplementedError

    def process_document_node(
        self,
        document_node: "DocumentNode",
        task_input: Optional[TTaskInput] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        contribute_to_document: bool = True,
        **kwargs,
    ) -> Dict[int, PageTaskResult]:
        kwargs = {**(self.provider_kwargs or {}), **kwargs}
        results = self.process_document_pages(
            document_node,
            task_input=task_input,
            start=start,
            stop=stop,
            **kwargs,
        )

        # If we want to contribute to the node, we can here by setting the kwarg
        if contribute_to_document:
            for page_number, page_result in results.items():
                page_result.contribute_to_document_node(document_node, page_number)

        return results


class AbstractDocumentTaskProvider(ABC, Generic[TTaskInput, DocumentTaskResult]):
    """
    A task provider performs a specific, repeatable task on a document
    """

    name: str
    capabilities: List[DocumentLevelCapabilities]

    # NOTE: Temporary solution to allo kwargs from the factory to providers who
    # don't take arbitrary kwargs in there __init__ method
    _provider_kwargs: Dict[str, Any]

    @classmethod
    def with_kwargs(cls, **kwargs):
        """Create the provider with kwargs."""
        obj = cls()
        obj.provider_kwargs = kwargs
        return obj

    @abstractmethod
    def process_document(
        self, document: Document, task_input: Optional[TTaskInput] = None, **kwargs
    ) -> DocumentTaskResult:
        raise NotImplementedError

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
            result.contribute_to_document_node(document_node)

        return result
