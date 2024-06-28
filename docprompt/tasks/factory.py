"""Define the base factory for creating task providers."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, ClassVar, Dict, Iterator, List, TypeVar

from pydantic import BaseModel, PrivateAttr, ValidationInfo, model_validator
from typing_extensions import Generic, Self

from docprompt.tasks.capabilities import (
    DocumentLevelCapabilities,
    PageLevelCapabilities,
)

from .base import AbstractPageTaskProvider
from .credentials import APIKeyCredential, AWSCredentials, GCPServiceFileCredentials

TTaskProvider = TypeVar("TTaskProvider", bound=AbstractPageTaskProvider)


class AbstractTaskMixin(ABC):
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
    def get_page_ocr_provider(self, *args, **kwargs) -> TTaskProvider:
        """Perform OCR on a page."""


class PageClassificationMixin(AbstractTaskMixin, Generic[TTaskProvider]):
    """Mixin for page classification task."""

    tags = [PageLevelCapabilities.PAGE_CLASSIFICATION]

    @abstractmethod
    def get_page_classification_provider(self, *args, **kwargs) -> TTaskProvider:
        """Perform page classification."""


class PageMarkerizationMixin(AbstractTaskMixin, Generic[TTaskProvider]):
    """Mixin for page markerization task."""

    tags = [PageLevelCapabilities.PAGE_MARKERIZATION]

    @abstractmethod
    def get_page_markerization_provider(self, *args, **kwargs) -> TTaskProvider:
        """Perform page markerization."""


class PageSegmentationMixin(AbstractTaskMixin, Generic[TTaskProvider]):
    """Mixin for page segmentation task."""

    tags = [PageLevelCapabilities.PAGE_SEGMENTATION]

    @abstractmethod
    def get_page_segmentation_provider(self, *args, **kwargs) -> TTaskProvider:
        """Perform page segmentation."""


class PageVQAMixin(AbstractTaskMixin, Generic[TTaskProvider]):
    """Mixin for page VQA task."""

    tags = [PageLevelCapabilities.PAGE_VQA]

    @abstractmethod
    def get_page_vqa_provider(self, *args, **kwargs) -> TTaskProvider:
        """Perform page VQA."""


class PageTableIdentificationMixin(AbstractTaskMixin, Generic[TTaskProvider]):
    """Mixin for page table identification task."""

    tags = [PageLevelCapabilities.PAGE_TABLE_IDENTIFICATION]

    @abstractmethod
    def get_page_table_identification_provider(self, *args, **kwargs) -> TTaskProvider:
        """Perform page table identification."""


class PageTableExtractionMixin(AbstractTaskMixin, Generic[TTaskProvider]):
    """Mixin for page table extraction task."""

    tags = [PageLevelCapabilities.PAGE_TABLE_EXTRACTION]

    @abstractmethod
    def get_page_table_extraction_provider(self, *args, **kwargs) -> TTaskProvider:
        """Extract tables from a page."""


class DocumentVQAMixin(AbstractTaskMixin, Generic[TTaskProvider]):
    """Mixin for multi-page document VQA task."""

    tags = [DocumentLevelCapabilities.DOCUMENT_VQA]

    @abstractmethod
    def get_document_vqa_provider(self, *args, **kwargs) -> TTaskProvider:
        """Perform multi-page document VQA."""

        _init_context_var = ContextVar("_init_context_var", default=None)


_init_context_var = ContextVar("_init_context_var", default=None)


@contextmanager
def init_context(value: Dict[str, Any]) -> Iterator[None]:
    token = _init_context_var.set(value)
    try:
        yield
    finally:
        _init_context_var.reset(token)


class AbstractTaskProviderFactory(ABC, BaseModel):
    """The abstract interface for a provider task factory.

    We need to define the basic interface for how we can create task providers. The task provider factory
    is responsible for allowing the creation of task providers for specific backends, (i.e. Anthropic, OpenAI, etc.)
    """

    def __init__(self, **data):
        with init_context({"payload": data}):
            self.__pydantic_validator__.validate_python(
                data,
                self_instance=self,
                context=_init_context_var.get(),
            )

    @model_validator(mode="after")
    def _validate_provider(self) -> Self:
        """Validate the provider before returning it.

        This method needs to handle credential validation, to ensure that the provider is properly
        configured and can be utilized for the tasks it can be used to provide.
        """
        raise NotImplementedError("You must provide custom provider validation!")


class AnthropicTaskProviderFactory(
    AbstractTaskProviderFactory,
    PageClassificationMixin,
    PageMarkerizationMixin,
    PageTableExtractionMixin,
):
    """The task provider factory for Anthropic.

    NOTE: We can either utilize the standard Anthropic API or we can utilize AWS Bedrock. In the event
    that a user wants to utilize the standard Anthropic API.
    """

    _credentials: APIKeyCredential = PrivateAttr()

    @model_validator(mode="after")
    def _validate_provider(self, info: ValidationInfo) -> Self:
        """Validate the provider before returning it."""
        _payload = info.context["payload"]
        self._credentials = APIKeyCredential(
            environ_path="ANTHROPIC_API_KEY", **_payload
        )
        return self

    def get_page_classification_provider(self, **kwargs) -> TTaskProvider:
        """Get the page classification provider."""
        from docprompt.tasks.classification.anthropic import (
            AnthropicClassificationProvider,
        )

        kwargs = {**self._credentials.kwargs, **kwargs}
        return AnthropicClassificationProvider.with_kwargs(**kwargs)

    def get_page_table_extraction_provider(self, **kwargs) -> TTaskProvider:
        """Get the page table extraction provider."""
        from docprompt.tasks.table_extraction.anthropic import (
            AnthropicTableExtractionProvider,
        )

        kwargs = {**self._credentials.kwargs, **kwargs}
        return AnthropicTableExtractionProvider.with_kwargs(**kwargs)

    def get_page_markerization_provider(self, **kwargs) -> TTaskProvider:
        """Get the page markerization provider."""
        from docprompt.tasks.markerization.anthropic import (
            AnthropicMarkerizationProvider,
        )

        kwargs = {**self._credentials.kwargs, **kwargs}
        return AnthropicMarkerizationProvider.with_kwargs(**kwargs)


class AmazonTaskProviderFactory(AbstractTaskProviderFactory, PageOCRMixin):
    """The task provider factory for Amazon."""

    @model_validator(mode="after")
    def _validate_provider(self, info: ValidationInfo) -> Self:
        """Validate the provider before returning it."""
        _payload = info.context["payload"]
        self._credentials = AWSCredentials(**_payload)

    def get_page_ocr_provider(self, **kwargs) -> TTaskProvider:
        """Get the page OCR provider."""
        from docprompt.tasks.ocr.amazon import AmazonOCRProvider

        kwargs = {**self._credentials.kwargs, **kwargs}
        return AmazonOCRProvider(**kwargs)


class GCPTaskProviderFactory(
    AbstractTaskProviderFactory,
    PageOCRMixin,
):
    """The task provider factory for GCP."""

    @model_validator(mode="after")
    def _validate_provider(self, info: ValidationInfo) -> Self:
        """Validate the provider before returning it."""
        _payload = info.context["payload"]
        self._credentials = GCPServiceFileCredentials

    def get_page_ocr_provider(
        self, project_id: str, processor_id: str, **kwargs
    ) -> TTaskProvider:
        """Get the page OCR provider."""
        from docprompt.tasks.ocr.gcp import GoogleOcrProvider

        kwargs = {**self._credentials.kwargs, **kwargs}
        return GoogleOcrProvider(project_id, processor_id, **kwargs)
