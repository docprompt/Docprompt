from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Optional

from attrs import frozen

from docprompt.service_providers.types import (
    OPERATIONS,
    ImageProcessResult,
    LayoutAnalysisOutput,
    PageTextExtractionOutput,
)

if TYPE_CHECKING:
    from docprompt.schema.document import Document


@frozen
class PageResult:
    """
    Represents the result of processing a single page
    (page-wise)
    """

    provider_name: str

    page_number: int
    ocr_result: Optional[PageTextExtractionOutput] = None
    image_process_result: Optional[ImageProcessResult] = None
    layout_analysis_result: Optional[LayoutAnalysisOutput] = None


@frozen
class DocumentResult:
    """
    Represents the result of operations that consider the
    document as a whole (document-wise)

    Examples:

    * Multi-page DocVQA
    """

    provider_name: str


@frozen
class ProviderResult:
    """
    Represents the result of processing a document
    """

    provider_name: str

    page_results: Optional[list[PageResult]] = None
    document_result: Optional[DocumentResult] = None


class BaseProvider(metaclass=ABCMeta):
    name: str

    @abstractmethod
    def _call(self, document: "Document", pages=list[int]) -> ProviderResult:
        raise NotImplementedError

    @property
    @abstractmethod
    def capabilities(self) -> list[OPERATIONS]:
        raise NotImplementedError

    def process_document(self, document: "Document", pages: Optional[list[int]] = None) -> ProviderResult:
        """
        Should return a ProviderResult object
        """
        pages = pages or list(range(1, document.num_pages + 1))
        return self._call(document, pages)
