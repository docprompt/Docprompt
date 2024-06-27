"""Define the base factory for creating task providers."""

from abc import ABC, abstractmethod
from typing import ClassVar, List, TypeVar

from pydantic import BaseModel, model_validator
from typing_extensions import Generic, Self

from docprompt.tasks.capabilities import (
    DocumentLevelCapabilities,
    PageLevelCapabilities,
)

from .base import AbstractTaskProvider

TTaskProvider = TypeVar("TTaskProvider", bound=AbstractTaskProvider)


class AbstractTaskMixin(BaseModel, ABC):
    """Base class for all task mixins."""

    tags: ClassVar[List[PageLevelCapabilities | DocumentLevelCapabilities]]


class PageRasterizationMixin(AbstractTaskMixin, Generic[TTaskProvider]):
    """Mixin for page rasterization task."""

    tags = [PageLevelCapabilities.PAGE_RASTERIZATION]

    @abstractmethod
    def get_rasterize_page_provider(self, **kwargs) -> TTaskProvider:
        """Perform page rasterization."""


class PageOCRMixin(AbstractTaskMixin, Generic[TTaskProvider]):
    """Mixin for page OCR task."""

    tags = [PageLevelCapabilities.PAGE_LAYOUT_OCR, PageLevelCapabilities.PAGE_TEXT_OCR]

    @abstractmethod
    def perform_ocr(self, *args, **kwargs) -> TTaskProvider:
        """Perform OCR on a page."""


class PageClassificationMixin(AbstractTaskMixin, Generic[TTaskProvider]):
    """Mixin for page classification task."""

    tags = [PageLevelCapabilities.PAGE_CLASSIFICATION]

    @abstractmethod
    def classify_page(self, *args, **kwargs) -> TTaskProvider:
        """Perform page classification."""


class PageSegmentationMixin(AbstractTaskMixin, Generic[TTaskProvider]):
    """Mixin for page segmentation task."""

    tags = [PageLevelCapabilities.PAGE_SEGMENTATION]

    @abstractmethod
    def segment_page(self, *args, **kwargs) -> TTaskProvider:
        """Perform page segmentation."""


class PageVQAMixin(AbstractTaskMixin, Generic[TTaskProvider]):
    """Mixin for page VQA task."""

    tags = [PageLevelCapabilities.PAGE_VQA]

    @abstractmethod
    def perform_page_vqa(self, *args, **kwargs) -> TTaskProvider:
        """Perform page VQA."""


class PageTableIdentificationMixin(AbstractTaskMixin, Generic[TTaskProvider]):
    """Mixin for page table identification task."""

    tags = [PageLevelCapabilities.PAGE_TABLE_IDENTIFICATION]

    @abstractmethod
    def identify_tables(self, *args, **kwargs) -> TTaskProvider:
        """Perform page table identification."""


class PageTableExtractionMixin(AbstractTaskMixin, Generic[TTaskProvider]):
    """Mixin for page table extraction task."""

    tags = [PageLevelCapabilities.PAGE_TABLE_EXTRACTION]

    @abstractmethod
    def extract_tables(self, *args, **kwargs) -> TTaskProvider:
        """Extract tables from a page."""


class DocumentVQAMixin(AbstractTaskMixin, Generic[TTaskProvider]):
    """Mixin for multi-page document VQA task."""

    tags = [DocumentLevelCapabilities.DOCUMENT_VQA]

    @abstractmethod
    def perform_document_vqa(self, *args, **kwargs) -> TTaskProvider:
        """Perform multi-page document VQA."""


class AbstractTaskProviderFactory(ABC, BaseModel):
    """The abstract interface for a provider task factory.

    We need to define the basic interface for how we can create task providers. The task provider factory
    is responsible for allowing the creation of task providers for specific backends, (i.e. Anthropic, OpenAI, etc.)
    """

    @model_validator(mode="after")
    def _validate_provider(self) -> Self:
        """Validate the provider before returning it."""
        raise NotImplementedError("You must provide custom provider validation!")
