from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Coroutine,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Type,
    TypedDict,
    TypeVar,
    Union,
)

from pydantic import (
    BaseModel,
    Field,
    GetCoreSchemaHandler,
    PrivateAttr,
    ValidationInfo,
    model_validator,
)
from pydantic_core import core_schema
from typing_extensions import Annotated, Self

from docprompt._decorators import flexible_methods

from .capabilities import DocumentLevelCapabilities, PageLevelCapabilities
from .result import BaseDocumentResult, BasePageResult
from .util import _init_context_var, init_context

if TYPE_CHECKING:
    from docprompt.schema.pipeline import DocumentNode, PageNode


TTaskInput = TypeVar("TTaskInput")  # What invoke requires
TTaskConfig = TypeVar("TTaskConfig")  # Task specific config like classification labels
TPageResult = TypeVar("TPageResult", bound=BasePageResult)
TDocumentResult = TypeVar("TDocumentResult", bound=BaseDocumentResult)
TTaskResult = TypeVar("TTaskResult", bound=Union[BasePageResult, BaseDocumentResult])

Capabilites = TypeVar(
    "Capabilities", bound=Union[DocumentLevelCapabilities, PageLevelCapabilities]
)


class NullSchema:
    def __get_pydantic_core_schema__(
        self, source: Type[Any], handler: GetCoreSchemaHandler
    ):
        def noop_validate(value: Any) -> Any:
            return value

        return core_schema.no_info_plain_validator_function(noop_validate)


@flexible_methods(
    ("process_document_node", "aprocess_document_node"),
    ("_invoke", "_ainvoke"),
)
class AbstractTaskProvider(BaseModel, Generic[TTaskInput, TTaskConfig, TTaskResult]):
    """
    A task provider performs a specific, repeatable task on a document or its pages.

    NOTE: Either the `process_document_pages` or `aprocess_document_pages` method must be implemented in
    a valid subclass. The `process_document_pages` method is explicitly defined, while the `aprocess_document_pages`
    method is an async version of the same method.

    If you wish to provide seperate implementations for sync and async, you can define both methods individually, and
    they will each use their own custom implementation when called. Otherwise, if you only implement one or the other of
    a flexible method pair, the other will automatically be generated and provided for you at runtime.
    """

    name: ClassVar[str]
    capabilities: ClassVar[List[Capabilites]]

    # TODO: Potentially utilize context here during instantiation from Factory??
    _default_invoke_kwargs: Dict[str, str] = PrivateAttr()

    class Meta:
        """The meta class is utilized by the flexible methods decorator.

        For all classes that are not concrete implementations, we should set the
        abstract attribute to True, which will prevent the check from failing when
        the flexible methods decorator is looking for the implementation of the
        methods.
        """

        abstract = True

    def __init__(self, invoke_kwargs: Dict[str, str] = None, **data):
        with init_context({"invoke_kwargs": invoke_kwargs or {}}):
            self.__pydantic_validator__.validate_python(
                data,
                self_instance=self,
                context=_init_context_var.get(),
            )

    @model_validator(mode="before")
    @classmethod
    def validate_class_vars(cls, data: Any) -> Any:
        """
        Ensure that the class has a name and capabilities defined.
        """

        if not hasattr(cls, "name"):
            raise ValueError("Task providers must have a name defined")

        if not hasattr(cls, "capabilities"):
            raise ValueError("Task providers must have capabilities defined")

        if not cls.capabilities:
            raise ValueError("Task providers must have at least one capability defined")

        return data

    @model_validator(mode="after")
    def set_invoke_kwargs(self, info: ValidationInfo) -> Self:
        """
        Set the default invoke kwargs for the task provider.
        """
        self._default_invoke_kwargs = info.context["invoke_kwargs"]
        return self

    async def _ainvoke(
        self,
        input: Iterable[TTaskInput],
        config: Optional[TTaskConfig] = None,
        **kwargs,
    ) -> List[TTaskResult]:
        raise NotImplementedError

    async def ainvoke(
        self,
        input: Iterable[TTaskInput],
        config: Optional[TTaskConfig] = None,
        **kwargs,
    ) -> List[TTaskResult]:
        invoke_kwargs = {
            **self._default_invoke_kwargs,
            **kwargs,
        }

        return await self._ainvoke(input, config, **invoke_kwargs)

    def _invoke(
        self,
        input: Iterable[TTaskInput],
        config: Optional[TTaskConfig] = None,
        **kwargs,
    ) -> List[TTaskResult]:
        raise NotImplementedError

    def invoke(
        self,
        input: Iterable[TTaskInput],
        config: Optional[TTaskConfig] = None,
        **kwargs,
    ) -> List[TTaskResult]:
        invoke_kwargs = {
            **self._default_invoke_kwargs,
            **kwargs,
        }

        return self._invoke(input, config, **invoke_kwargs)

    def process_document_node(
        self,
        document_node: "DocumentNode",
        task_config: Optional[TTaskConfig] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        contribute_to_document: bool = True,
        **kwargs,
    ) -> Dict[int, TTaskResult]:
        raise NotImplementedError

    async def aprocess_document_node(
        self,
        document_node: "DocumentNode",
        task_config: Optional[TTaskConfig] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        contribute_to_document: bool = True,
        **kwargs,
    ) -> Dict[int, TTaskResult]:
        raise NotImplementedError


class SupportsOpenAIMessages(BaseModel, Generic[TTaskInput]):
    """
    Mixin for task providers that support OpenAI.
    """

    def get_openai_messages(self, input: TTaskInput, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError

    async def aget_openai_messages(
        self, input: TTaskInput, **kwargs
    ) -> Coroutine[None, None, Dict[str, Any]]:
        raise NotImplementedError


class SupportsParsing(BaseModel, Generic[TTaskResult]):
    """
    Mixin for task providers that support parsing.
    """

    def parse(self, response: str, **kwargs) -> TTaskResult:
        raise NotImplementedError

    async def aparse(self, response: str, **kwargs) -> TTaskResult:
        raise NotImplementedError


class SupportsPageNode(BaseModel, Generic[TTaskConfig, TPageResult]):
    """
    Mixin for task providers that support page processing.
    """

    def process_page_node(
        self,
        page_node: "PageNode",
        task_config: Optional[TTaskConfig] = None,
        **kwargs,
    ) -> TPageResult:
        raise NotImplementedError

    async def aprocess_page_node(
        self,
        page_node: "PageNode",
        task_config: Optional[TTaskConfig] = None,
        **kwargs,
    ) -> TPageResult:
        raise NotImplementedError


class SupportsDirectInvocation(
    BaseModel, Generic[TTaskInput, TTaskConfig, TTaskResult]
):
    """
    Mixin for task providers that support direct invocation on
    non-node based items.s
    """

    def invoke(
        self, input: TTaskInput, config: Optional[TTaskConfig] = None, **kwargs
    ) -> TTaskResult:
        raise NotImplementedError

    async def ainvoke(
        self, input: TTaskInput, config: Optional[TTaskConfig] = None, **kwargs
    ) -> TTaskResult:
        raise NotImplementedError


class SupportsDocumentNode(BaseModel, Generic[TTaskInput, TDocumentResult]):
    """
    Mixin for task providers that support document processing.
    """

    def process_document_node(
        self,
        document_node: "DocumentNode",
        task_config: Optional[TTaskConfig] = None,
        **kwargs,
    ) -> TDocumentResult:
        raise NotImplementedError

    async def aprocess_document_node(
        self,
        document_node: "DocumentNode",
        task_config: Optional[TTaskConfig] = None,
        **kwargs,
    ) -> TDocumentResult:
        raise NotImplementedError


@flexible_methods(
    ("process_image", "aprocess_image"),
)
class SupportsImage(BaseModel, Generic[TTaskInput, TTaskResult]):
    """
    Mixin for task providers that support image processing.
    """

    def process_image(self, input: TTaskInput, **kwargs) -> TTaskResult:
        raise NotImplementedError

    async def aprocess_image(self, input: TTaskInput, **kwargs) -> TTaskResult:
        raise NotImplementedError


class OpenAIMessageItem(TypedDict):
    content: str


class OpenAiChoiceItem(TypedDict):
    finish_reason: str
    index: int
    message: OpenAIMessageItem


class OpenAICompletionResponse(TypedDict):
    choices: List[OpenAiChoiceItem]


SyncOAICallable = Callable[[List[Dict[str, Any]]], OpenAICompletionResponse]
AsyncOAICallable = Coroutine[List[Dict[str, Any]], None, OpenAICompletionResponse]


class ProviderAgnosticOAI(BaseModel):
    sync_callable: Annotated[SyncOAICallable, NullSchema()] = Field(
        default=None, exclude=True
    )
    async_callable: Annotated[AsyncOAICallable, NullSchema()] = Field(
        default=None, exclude=True
    )

    @model_validator(mode="after")
    def validate_callable(self):
        if not self.sync_callable and not self.async_callable:
            raise ValueError(
                f"{self.__class__.__name__} must be initialized with either `sync_callable` and/or `async_callable`"
            )

        return self


@flexible_methods(
    ("process_webpage", "aprocess_webpage"),
)
class SupportsWebPage(BaseModel, Generic[TTaskInput, TPageResult]):
    """
    Mixin for task providers that support webpage processing.
    """

    def process_webpage(self, input: TTaskInput, **kwargs) -> TPageResult:
        raise NotImplementedError

    async def aprocess_webpage(self, input: TTaskInput, **kwargs) -> TPageResult:
        raise NotImplementedError


class SupportsTaskConfig(BaseModel, Generic[TTaskConfig]):
    task_config: TTaskConfig = None

    def get_config(self) -> TTaskConfig:
        return self.task_config


class AbstractPageTaskProvider(AbstractTaskProvider):
    """
    A page task provider performs a specific, repeatable task on a page.
    """

    capabilities: ClassVar[List[PageLevelCapabilities]]

    # NOTE: We need the stubs defined here for the flexible decorators to work
    # for now

    class Meta:
        abstract = True


class AbstractDocumentTaskProvider(AbstractTaskProvider):
    """
    A task provider performs a specific, repeatable task on a document.
    """

    capabilities: ClassVar[List[DocumentLevelCapabilities]]

    # NOTE: We need the stubs defined here for the flexible decorators to work
    # for now

    class Meta:
        abstract = True
