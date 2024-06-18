"""Define the schema for storage providers and provide concrete implementation of
basic S3 and local storage providers.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, ClassVar

FALLBACK_LOCAL_STORAGE_PATH = "/tmp/.docprompt"


class AbstractStorageProvider(ABC):
    """The abstract class interface for a storage provider."""

    @abstractmethod
    def store(self, document_node: Any) -> None:
        """Store the document node in the storage provider.

        Args:
            document_node: The document node to store
        """

    @abstractmethod
    def retrieve(self, file_hash: str) -> Any:
        """Retrieve the document node from the storage provider.

        Args:
            file_hash: The hash of the document to retrieve

        Returns:
            The document node

        Raises:
            FileNotFoundError: If the document node is not found
        """


class LocalFileSystemStorageProvider(AbstractStorageProvider):
    """The concrete implementation of a local file system storage provider."""

    base_storage_path: ClassVar[str] = os.environ.get(
        "DOCPROMPT_LOCAL_STORAGE_PATH", FALLBACK_LOCAL_STORAGE_PATH
    )

    def store(self, document_node: Any) -> None:
        pass

    def retrieve(self, file_hash: str) -> Any:
        pass
