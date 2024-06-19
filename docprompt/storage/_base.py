"""Define the schema for storage providers and provide concrete implementation of
basic S3 and local storage providers.
"""

from abc import abstractmethod
from typing import TypeVar, Type, Any, Generic
from typing_extensions import Annotated

from pydantic import BaseModel, Field, AfterValidator

from docprompt.schema.pipeline import DocumentNode as BaseDocumentNode

FALLBACK_LOCAL_STORAGE_PATH = "/tmp/.docprompt"


DocumentNode = TypeVar("DocumentNode", bound=BaseDocumentNode)
DocumentNodeMeatadata = TypeVar("DocumentNodeMeatadata", bound=BaseModel)
StoragePathModel = TypeVar("StoragePathModel", bound=BaseModel)


def validate_document_node_class(value: Type[Any]) -> Type[DocumentNode]:
    """Validate that the document node class is a subclass of DocumentNode from DocPrompt.

    Args:
        value: The document node class to validate

    Returns:
        The document node class if it is valid

    Raises:
        ValueError: If the document node class is not a subclass of DocumentNode
    """

    if not issubclass(value, BaseDocumentNode):
        raise ValueError("The document node class must be a subclass of Document")
    return value


def validate_document_metadata_class(value: Type[Any]) -> Type[DocumentNodeMeatadata]:
    """Validate that the document metadata class is a subclass of BaseModel from Pydantic.

    Args:
        value: The document metadata class to validate

    Returns:
        The document metadata class if it is valid

    Raises:
        ValueError: If the document metadata class is not a subclass of Pydantic BaseModel
    """

    if not issubclass(value, BaseModel):
        raise ValueError(
            "The document metadata class must be a subclass of Pydantic BaseModel"
        )
    return value


class AbstractStorageProvider(BaseModel, Generic[StoragePathModel]):
    """The abstract class interface for a storage provider."""

    document_node_class: Annotated[
        Type[DocumentNode], AfterValidator(validate_document_node_class)
    ] = Field(..., repr=False)
    document_metadata_class: Annotated[
        Type[DocumentNodeMeatadata], AfterValidator(validate_document_metadata_class)
    ] = Field(..., repr=False)

    @abstractmethod
    def store(self, document_node: DocumentNode, **kwargs) -> StoragePathModel:
        """Store the document node in the storage provider.

        Args:
            document_node: The document node to store
        """

    @abstractmethod
    def retrieve(self, file_hash: str, **kwargs) -> DocumentNode:
        """Retrieve the document node from the storage provider.

        Args:
            file_hash: The hash of the document to retrieve

        Returns:
            The document node

        Raises:
            FileNotFoundError: If the document node is not found
        """
