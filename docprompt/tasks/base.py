from typing import Generic, TypeVar, TYPE_CHECKING
from pydantic import BaseModel, Field

from docprompt.schema.document import Document
from typing import Optional
from enum import Enum
from datetime import datetime


if TYPE_CHECKING:
    from docprompt.schema.pipeline import DocumentNode


class CAPABILITIES(Enum):
    """
    Represents a capability that a provider can fulfill
    """

    PAGE_RASTERIZATION = "page-rasterization"
    PAGE_LAYOUT_OCR = "page-layout-ocr"
    PAGE_TEXT_OCR = "page-text-ocr"
    PAGE_CLASSIFICATION = "page-classification"
    PAGE_SEGMENTATION = "page-segmentation"
    PAGE_VQA = "page-vqa"
    PAGE_TABLE_IDENTIFICATION = "page-table-identification"
    PAGE_TABLE_EXTRACTION = "page-table-extraction"


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


PageTaskResult = TypeVar("PageTaskResult", bound=BasePageResult)
DocumentTaskResult = TypeVar("DocumentTaskResult", bound=BaseDocumentResult)
PageOrDocumentTaskResult = TypeVar("PageOrDocumentTaskResult", bound=BaseResult)


class ResultContainer(BaseModel, Generic[PageOrDocumentTaskResult]):
    """
    Represents a container for results of a task
    """

    results: dict[str, PageOrDocumentTaskResult] = Field(
        description="The results of the task, keyed by provider", default_factory=dict
    )

    @property
    def result(self):
        return next(iter(self.results.values()), None)


class AbstractTaskProvider(Generic[PageTaskResult]):
    name: str
    capabilities: list[str]

    def process_document_pages(
        self,
        document: Document,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        **kwargs,
    ) -> dict[int, PageTaskResult]:
        raise NotImplementedError

    def contribute_to_document_node(
        self,
        document_node: "DocumentNode",
        results: dict[int, PageTaskResult],
    ) -> None:
        """
        Adds the results of this task to the document node and/or its page nodes
        """
        pass

    def process_document_node(
        self,
        document_node: "DocumentNode",
        start: Optional[int] = None,
        stop: Optional[int] = None,
        contribute_to_document: bool = True,
        **kwargs,
    ) -> dict[int, PageTaskResult]:
        results = self.process_document_pages(
            document_node.document, start=start, stop=stop, **kwargs
        )

        if contribute_to_document:
            self.contribute_to_document_node(document_node, results)

        return results
