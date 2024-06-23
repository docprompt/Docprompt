"""The File System class provides a wrapper around FSSpec with the required operations for reading,
writing, and deleting files from an arbitrary file system backend.

When instantiating the model, you must review the `fsspec` documentation to determine what
additional environment and credential kwargs you must provide, based on your specific
selection of file system.
"""

import os
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import fsspec
from pydantic import BaseModel, Field, computed_field, model_validator
from pydantic_core import core_schema


class FileSidecarsPathManager(BaseModel):
    """The FileSidecarsPathManager provides a wrapper around fsspec to provide a clean interface for
    reading and writing sidecar directories for storing document nodes.

    Attributes:
        base_path (str): The base path for the sidecar files.
        file_hash (str): The hash of the file to be stored.
        pdf (str): The path for the PDF file.
        metadata (str): The path for the metadata file.
        page_metadata (str): The path for the page metadata file.
    """

    base_path: str = Field(...)
    file_hash: str = Field(...)

    @computed_field
    @property
    def pdf(self) -> str:
        """The path for the PDF file."""
        return f"{self.base_path}/{self.file_hash}/base.pdf"

    @computed_field
    @property
    def metadata(self) -> str:
        """The path for the metadata file."""
        return f"{self.base_path}/{self.file_hash}/base.json"

    @computed_field
    @property
    def page_metadata(self) -> str:
        """The path for the page metadata file."""
        return f"{self.base_path}/{self.file_hash}/pages.json"


class FileSystemAnnotation:
    """FileSystemType is a custom type for fsspec.AbstractFileSystem instances.

    This class handles defining the necessary information for pydantic to be able to validate
    and fsspec file system as a field in a Pydanitc model.
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        """Validate that it is a valid fsspec file system."""
        if not isinstance(v, fsspec.AbstractFileSystem):
            raise TypeError("Must be an fsspec.AbstractFileSystem instance")
        return v

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: Any
    ) -> core_schema.CoreSchema:
        """Get the Pydantic Core Schema for the FileSystemType."""
        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.any_schema(),
            serialization=core_schema.to_string_ser_schema(),
        )


class FileSystemManager(BaseModel):
    """The FileSystemManager provides a wrapper around fsspec to provide a clean interface for
    reading and writing sidecar directories for storing document nodes.
    """

    path: str = Field(...)
    fs: FileSystemAnnotation = Field(...)
    fs_kwargs: Dict[str, Any] = Field(...)

    def __init__(self, url: str, **kwargs: Any) -> None:
        """Initialize the FileSystemManager with the path and file system backend."""
        super().__init__(path=url, fs_kwargs=kwargs)

    @model_validator(mode="before")
    @classmethod
    def validate_filesystem_protocol_and_kwargs(cls, data: Any) -> Any:
        """Validate that the path is a valid filesystem path."""

        path = data.get("path", None)
        fs_kwargs = data.get("fs_kwargs", {})

        # Validate that the path is a valid filesystem path
        file_system = fsspec.url_to_fs(path, **fs_kwargs)

        # Set the data values on the model
        data["fs"] = file_system[0]
        data["path"] = file_system[1]
        data["fs_kwargs"] = fs_kwargs

        return data

    def get_pdf_name(self, file_hash: str) -> FileSidecarsPathManager:
        """Get the file manager for a specific file hash."""
        path_manager = FileSidecarsPathManager(base_path=self.path, file_hash=file_hash)
        return os.path.basename(path_manager.pdf)

    def _write(self, value: bytes, *args, **kwargs) -> None:
        """Write a value to the file system.

        Args:
            value: The value to write.
            *args: `fsspec.open` positional arguments.
            **kwargs: `fsspec.open` keyword arguments.
        """
        # Make sure the path exists
        parent_dir = os.path.dirname(args[0])

        kwargs = {**self.fs_kwargs, **kwargs}

        if not self.fs.exists(parent_dir, **kwargs):
            self.fs.mkdirs(parent_dir, **kwargs)

        with self.fs.open(*args, **kwargs) as f:
            f.write(value)

    def _read(self, *args, **kwargs) -> bytes:
        """A wrapper for reading with fsspec.

        Args:
            *args: `fsspec.open` positional arguments.
            **kwargs: `fsspec.open` keyword arguments.

        Returns:
            bytes: The read value.
        """

        kwargs = {**self.fs_kwargs, **kwargs}

        with self.fs.open(*args, **kwargs) as f:
            return f.read()

    def _delete(self, path: str, **kwargs):
        """Delete a file from the file system.

        Args:
            path (str): The path to the file to delete.
            **kwargs: Additional keyword arguments to pass to the file system.
        """

        kwargs = {**self.fs_kwargs, **kwargs}

        if self.fs.exists(path, **kwargs):
            self.fs.rm(path, **kwargs)

    def write(
        self,
        pdf_bytes: bytes,
        metadata_bytes: Optional[bytes] = None,
        page_metadata_bytes: Optional[bytes] = None,
        encrypt: bool = False,
        compress: bool = False,
        **kwargs,
    ) -> FileSidecarsPathManager:
        """Write a sidecar to the filesystem."""
        from docprompt.utils.util import hash_from_bytes

        file_hash = hash_from_bytes(pdf_bytes)

        # Craete the sidecar manager
        path_manager = FileSidecarsPathManager(base_path=self.path, file_hash=file_hash)

        if encrypt:
            warnings.warn("Encryption is not yet supported for the FileSystemManager.")

        if compress:
            warnings.warn("Compression is not yet supported for the FileSystemManager.")

        kwargs = {**self.fs_kwargs, **kwargs}

        # Write the file
        self._write(pdf_bytes, path_manager.pdf, "wb", **kwargs)

        # If the metadata is provided, we want to write it
        if metadata_bytes is not None:
            self._write(metadata_bytes, path_manager.metadata, "wb", **kwargs)

        # Otherwise, we need to clear the metadata file, so that the node is read
        # without any metadata (overwriting any existing metadata)
        else:
            # Check if a metadata file exists
            self._delete(path_manager.metadata, **kwargs)

        if page_metadata_bytes is not None:
            self._write(page_metadata_bytes, path_manager.page_metadata, "wb", **kwargs)

        # Otherwise, we need to clear the page metadata file, so that the node is read
        # without any metadata (overwriting any existing metadata)
        else:
            self._delete(path_manager.page_metadata, **kwargs)

        return path_manager

    def read(
        self, file_hash: str, **kwargs
    ) -> Tuple[bytes, Union[bytes, None], Union[bytes, None]]:
        """Read a pair of sidecar files from the filesystem."""

        # Craete the sidecar manager
        path_manager = FileSidecarsPathManager(base_path=self.path, file_hash=file_hash)

        # Read the PDF file
        pdf_bytes = self._read(path_manager.pdf, "rb", **kwargs)

        try:
            metadata_bytes = self._read(path_manager.metadata, "rb", **kwargs)
        except FileNotFoundError:
            metadata_bytes = None

        try:
            page_metadata_bytes = self._read(path_manager.page_metadata, "rb", **kwargs)
        except FileNotFoundError:
            page_metadata_bytes = None

        return pdf_bytes, metadata_bytes, page_metadata_bytes
