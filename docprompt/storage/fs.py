"""The File System class provides a wrapper around FSSpec with the required operations for reading,
writing, and deleting files from an arbitrary file system backend.

When instantiating the model, you must review the `fsspec` documentation to determine what
additional environment and credential kwargs you must provide, based on your specific
selection of file system.
"""

import warnings
from typing import Dict, Any, Optional, Tuple, Union
from typing_extensions import Self

import fsspec
from pydantic import BaseModel, Field, model_validator, computed_field
from pydantic_core import core_schema


class FileSidecarsPathManager(BaseModel):
    """The FileSidecarsPathManager provides a wrapper around fsspec to provide a clean interface for
    reading and writing sidecar directories for storing document nodes.
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
        return f"{self.base_path}/{self.file_hash}/metadata.json"


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

    # Validate that the path is a valid filesystem path here
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
        file_system[0].mkdir(path, create_parents=True)

        # Set the data values on the model
        data["fs"] = file_system[0]
        data["path"] = file_system[1]
        data["fs_kwargs"] = fs_kwargs

        return data

    @model_validator(mode="after")
    def validate_base_path(self) -> Self:
        """Validate that the base_path exists."""

        # Check if the base path exists
        if not self.fs.exists(self.path):  # pylint: disable=not-a-mapping,no-member
            self.fs.mkdir(self.path, create_parents=True, **self.fs_kwargs)  # pylint: disable=not-a-mapping,no-member

        return self

    def write(
        self,
        pdf_bytes: bytes,
        metadata_bytes: Optional[bytes] = None,
        *,
        encrypt: bool = False,  # pylint: disable=unused-argument
        compress: bool = False,  # pylint: disable=unused-argument
    ) -> FileSidecarsPathManager:
        """Write a sidecar to the filesystem."""
        from docprompt.utils.util import hash_from_bytes  # pylint: disable=import-outside-toplevel

        file_hash = hash_from_bytes(pdf_bytes)

        # Craete the sidecar manager
        path_manager = FileSidecarsPathManager(base_path=self.path, file_hash=file_hash)

        if encrypt:
            warnings.warn("Encryption is not yet supported for the FileSystemManager.")

        if compress:
            warnings.warn("Compression is not yet supported for the FileSystemManager.")

        # Write the file
        with self.fs.open(  # pylint: disable=no-member
            path_manager.pdf,
            "wb",
            **self.fs_kwargs,  # pylint: disable=not-a-mapping
        ) as f:
            f.write(pdf_bytes)

        # If the metadata is provided, we want to write it
        if metadata_bytes is not None:
            # Write the metadata file
            with self.fs.open(  # pylint: disable=no-member
                path_manager.metadata,
                "wb",
                **self.fs_kwargs,  # pylint: disable=not-a-mapping
            ) as f:
                f.write(metadata_bytes)

        # Otherwise, we need to clear the metadata file, so that the node is read
        # without any metadata (overwriting any existing metadata)
        else:
            # Check if a metadata file exists
            if self.fs.exists(path_manager.metadata):  # pylint: disable=no-member
                self.fs.rm(path_manager.metadata, **self.fs_kwargs)  # pylint: disable=not-a-mapping,no-member

        return path_manager

    def read(self, file_hash: str) -> Tuple[bytes, Union[bytes, None]]:
        """Read a pair of sidecar files from the filesystem."""

        # Craete the sidecar manager
        path_manager = FileSidecarsPathManager(base_path=self.path, file_hash=file_hash)

        # Read the PDF file
        with self.fs.open(  # pylint: disable=no-member
            path_manager.pdf,
            "rb",
            **self.fs_kwargs,  # pylint: disable=not-a-mapping
        ) as f:
            pdf_bytes = f.read()

        # Check if a metadata file exists
        if self.fs.exists(path_manager.metadata):  # pylint: disable=no-member
            with self.fs.open(  # pylint: disable=no-member
                path_manager.metadata,
                "rb",
                **self.fs_kwargs,  # pylint: disable=not-a-mapping
            ) as f:
                metadata_bytes = f.read()
        else:
            metadata_bytes = None

        return pdf_bytes, metadata_bytes
