"""The local storage provider implements a local storage solution for DocPrompt.

It allows developers to easily save PDFs and metadata to the local file system. To configure
docprompt to use a particular local storage path, set the `DOCPROMPT_LOCAL_STORAGE_PATH`
environment variable. If this environment variable is not set, the default path will be
used.
"""

import os
import json
from typing import Any, ClassVar, Optional
from typing_extensions import Annotated

from pydantic import BaseModel, Field, AfterValidator, computed_field

from docprompt.schema.document import Document

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

    base_file_path: Annotated[str, AfterValidator(validate_file_path)] = Field(
        ..., repr=False
    )

    @computed_field
    @property
    def pdf(self) -> str:
        """Get the file path for the pdf bytes."""
        return f"{self.base_file_path}.pdf"

    @computed_field
    @property
    def metadata(self) -> str:
        """Get the file path for the metadata."""
        return f"{self.base_file_path}.json"


class LocalFileSystemStorageProvider(AbstractStorageProvider[FilePathSidecars]):
    """The concrete implementation of a local file system storage provider.

    Attributes:
        document_node_class: The document node class to store and retrieve
        document_metadata_class: The document metadata class to store and retrieve
        base_storage_path: The base storage path for the local file system
    """

    base_storage_path: ClassVar[str] = os.environ.get(
        "DOCPROMPT_LOCAL_STORAGE_PATH", FALLBACK_LOCAL_STORAGE_PATH
    )

    def paths(self, file_hash: str) -> FilePathSidecars:
        """Get the base file path for a document node.

        Args:
            file_hash: The hash of the document node

        Returns:
            FilePathSidecars: The file paths for the document node

        Raises:
            pydantic_core.ValidationError: If the file path is invalid
        """
        # This will run validation to ensure the fp exists and is valid
        return FilePathSidecars(
            base_file_path=os.path.join(self.base_storage_path, file_hash)
        )

    def store(self, document_node: DocumentNode) -> FilePathSidecars:
        """Store the document node in the local file system.

        Args:
            document_node: The document node to store

        Returns:
            FilePathSidecars: The file paths for the document node
        """

        file_paths = self.paths(document_node.file_hash)

        local_fs_write(file_paths.pdf, document_node.document.get_bytes(), mode="wb")

        if document_node.metadata is not None:
            local_fs_write(
                file_paths.metadata,
                json.dumps(document_node.metadata.model_dump(mode="json")),
                encoding="utf-8",
            )
        else:
            # If there is no metadata, then we should delete the metadata file
            local_fs_delete(file_paths.metadata)

        return file_paths

    def retrieve(self, file_hash: str) -> DocumentNode:
        """Retrieve the document node from the local file system.

        Args:
            file_hash: The hash of the document to retrieve

        Returns:
            DocumentNode: The document node

        Raises:
            FileNotFoundError: If the document node is not found
        """

        file_paths = self.paths(file_hash)

        pdf_bytes = local_fs_read(file_paths.pdf, mode="rb")
        document = Document.from_bytes(pdf_bytes, name=os.path.basename(file_paths.pdf))

        try:
            metadata_bytes = local_fs_read(
                file_paths.metadata, mode="r", encoding="utf-8"
            )
            metadata = self.document_metadata_class.model_validate(
                json.loads(metadata_bytes)
            )
        except FileNotFoundError:
            metadata = None

        # Create our document node
        return self.document_node_class.from_document(  # pylint: disable=no-member
            document=document,
            document_metadata=metadata,
            storage_provider_class=type(self),
        )


def local_fs_write(
    path: str, value: Any, mode: str = "w", encoding: Optional[str] = None
) -> None:
    """Store a document node to the local FS.

    Args:
        path: The path to store the value
        value: The value to store
        mode: The mode to open the file with
        encoding: The encoding to use when writing the file

    Raises:
        ValueError: If the directory does not exist
    """

    if not os.path.exists(os.path.dirname(path)):
        raise ValueError(f"Invalid file path: {path} -- Directory does not exist.")

    with open(path, mode, encoding=encoding) as f:
        f.write(value)


def local_fs_read(path: str, mode: str = "r", encoding: Optional[str] = None) -> bytes:
    """Read a document node from the local FS.

    Args:
        path: The path to read the value from
        mode: The mode to open the file with
        encoding: The encoding to use when reading the file

    Returns:
        bytes: The value read from the file

    Raises:
        FileNotFoundError: If the file does not exist
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, mode, encoding=encoding) as f:
        return f.read()


def local_fs_delete(path: str) -> None:
    """Delete a file from the local FS.

    Args:
        path: The path to delete
    """

    if not os.path.exists(path):
        return

    os.remove(path)
