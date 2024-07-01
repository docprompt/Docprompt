from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    TypeVar,
    Union,
)

from pydantic import BaseModel, PrivateAttr, ValidationInfo, model_validator
from typing_extensions import Self

from docprompt._decorators import flexible_methods

from .capabilities import DocumentLevelCapabilities, PageLevelCapabilities
from .result import BaseDocumentResult, BasePageResult
from .util import _init_context_var, init_context

if TYPE_CHECKING:
    from docprompt.schema.pipeline import DocumentNode


TTaskInput = TypeVar("TTaskInput")  # What invoke requires
TTaskConfig = TypeVar("TTaskConfig")  # Task specific config like classification labels
TPageResult = TypeVar("TPageResult", bound=BasePageResult)
TDocumentResult = TypeVar("TDocumentResult", bound=BaseDocumentResult)
TTaskResult = TypeVar("TTaskResult", bound=Union[BasePageResult, BaseDocumentResult])

Capabilites = TypeVar(
    "Capabilities", bound=Union[DocumentLevelCapabilities, PageLevelCapabilities]
)


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
