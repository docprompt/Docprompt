from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, ContextManager, Optional

from docprompt.schema.operations import ProviderResult
from docprompt.service_providers.types import OPERATIONS

if TYPE_CHECKING:
    from docprompt.schema.document import Document


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
