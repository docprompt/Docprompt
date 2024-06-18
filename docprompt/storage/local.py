"""The local storage provider implements a local storage solution for DocPrompt.

It allows developers to easily save PDFs and metadata to the local file system.
"""

import os
import json
from typing import ClassVar
from typing_extensions import Annotated

from pydantic import BaseModel, Field, AfterValidator

from docprompt.utils import load_document

from ._base import AbstractStorageProvider, DocumentNode

FALLBACK_LOCAL_STORAGE_PATH = "/tmp/.docprompt"


def validate_file_path(value: str) -> str:
    """Validate that a file path is valid and exits."""

    try:
        # Extract everything except for the last directory (which is
        # the file hash, but that will the base name of each sidecar)
        base_path = os.path.dirname(value)

        if not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)

        return value
    except Exception as e:
        raise ValueError(f"Invalid file path: {value}") from e


class FilePathSidecars(BaseModel):
    """The pair of file names for a document node.

    Attributes:
        base_file_path: The base file path for the document node
        pdf: The file path for the pdf bytes
        metadata: The file path for the metadata
    """

    base_file_path: Annotated[str, AfterValidator(validate_file_path)] = Field(...)

    @property
    def pdf(self) -> str:
        """Get the file path for the pdf bytes."""
        return f"{self.base_file_path}.pdf"

    @property
    def metadata(self) -> str:
        """Get the file path for the metadata."""
        return f"{self.base_file_path}.json"


class LocalFileSystemStorageProvider(AbstractStorageProvider):
    """The concrete implementation of a local file system storage provider."""

    base_storage_path: ClassVar[str] = os.environ.get(
        "DOCPROMPT_LOCAL_STORAGE_PATH", FALLBACK_LOCAL_STORAGE_PATH
    )

    def _file_path(self, file_hash: str) -> FilePathSidecars:
        """Get the base file path for a document node.

        Will return a file path in the form of: {base_storage_path}/{file_hash}
        """
        # This will run validation to ensure the fp exists and is valid
        return FilePathSidecars(
            base_file_path=os.path.join(self.base_storage_path, file_hash)
        )

    def store(self, document_node: DocumentNode) -> None:
        """Store the document node in the local file system.

        Args:
            document_node: The document node to store
        """

        file_paths = self._file_path(document_node.file_hash)

        with open(file_paths.pdf, "wb") as f:
            f.write(document_node.document.get_bytes())

        if document_node.metadata is not None:
            with open(file_paths.metadata, "w", encoding="utf-8") as f:
                json.dump(document_node.metadata.model_dump(mode="json"), f)

    def retrieve(self, file_hash: str) -> DocumentNode:
        """Retrieve the document node from the local file system.

        Args:
            file_hash: The hash of the document to retrieve

        Returns:
            DocumentNode: The document node

        Raises:
            FileNotFoundError: If the document node is not found
        """

        file_paths = self._file_path(file_hash)

        # Assert that the PDF file exists
        if not os.path.exists(file_paths.pdf):
            raise FileNotFoundError(f"PDF file not found for hash: {file_hash}")

        # Load the Document Node Metadata, if the file exists
        if os.path.exists(file_paths.metadata):
            with open(file_paths.metadata, "r", encoding="utf-8") as f:
                metadata = self.document_metadata_class.parse_obj(json.load(f))
        # Otherwise, set the metadata to None
        else:
            metadata = None

        # Using the metadata, load the document node from the file path
        document = load_document(file_paths.pdf)
        document_node = self.document_node_class.from_document(  # pylint: disable=no-member
            document=document, document_metadata=metadata
        )

        return document_node
