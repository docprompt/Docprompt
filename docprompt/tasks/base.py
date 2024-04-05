from typing import Any, Dict, Generic, List, Literal, TypeVar, TYPE_CHECKING, Union
from pydantic import BaseModel, Field

from docprompt.schema.document import Document
from typing import Optional
from enum import Enum
from datetime import datetime
import importlib


if TYPE_CHECKING:
    from docprompt.schema.pipeline import DocumentNode
    from langchain.schema import SystemMessage, HumanMessage


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

    results: Dict[str, PageOrDocumentTaskResult] = Field(
        description="The results of the task, keyed by provider", default_factory=dict
    )

    @property
    def result(self):
        return next(iter(self.results.values()), None)


class AbstractTaskProvider(Generic[PageTaskResult]):
    """
    A task provider performs a specific, repeatable task on a document or its pages
    """

    name: str
    capabilities: List[str]

    def process_document_pages(
        self,
        document: Document,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        **kwargs,
    ) -> Dict[int, PageTaskResult]:
        raise NotImplementedError

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
        start: Optional[int] = None,
        stop: Optional[int] = None,
        contribute_to_document: bool = True,
        **kwargs,
    ) -> Dict[int, PageTaskResult]:
        results = self.process_document_pages(
            document_node.document, start=start, stop=stop, **kwargs
        )

        if contribute_to_document:
            self.contribute_to_document_node(document_node, results)

        return results


def attempt_import(name: str):
    """
    Attempts to import a module or class by name
    """
    package, obj = name.rsplit(".", 1)

    try:
        module = importlib.import_module(package)
    except ImportError:
        return None

    return getattr(module, obj, None)


SupportedModels = Literal["openai", "openai_async", "langchain"]


def validate_language_model(model: Any):
    langchain_chat_klass = attempt_import(
        "langchain_core.language_models.chat_models.BaseChatModel"
    )

    if langchain_chat_klass and isinstance(model, langchain_chat_klass):
        return "langchain"

    openai_klass = attempt_import("openai.OpenAI")

    if openai_klass and isinstance(model, openai_klass):
        return "openai"

    openai_async_klass = attempt_import("openai.OpenAIAsync")

    if openai_async_klass and isinstance(model, openai_async_klass):
        return "openai_async"

    raise ValueError(
        f"Model must be one of langchain_core.language_models.chat_models.BaseChatModel or openai.OpenAI. Got {type(model)}"
    )


SystemMessageLike = Union["SystemMessage", Dict[str, str], str]
HumanMessageLike = Union["HumanMessage", Dict[str, Union[str, Dict[str, str]]], str]


class AbstractLanguageModelTaskProvider(AbstractTaskProvider):
    """
    Provides additional methods for language model specific tasks
    """

    def __init__(self, language_model: Any, *, model_name: Optional[str] = None):
        self.language_model = language_model
        self.model_type = validate_language_model(language_model)

        self.model_name = model_name

        self._validate_kwargs()

    def _validate_kwargs(self):
        """
        Validates the kwargs for the language model
        """
        if self.model_type == "openai" and self.model_name is None:
            raise ValueError("model_name must be provided for OpenAI language models")
