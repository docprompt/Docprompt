"""Test the storage wrapper for fsspec."""

from unittest.mock import MagicMock, patch

import fsspec
import pytest
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem

from docprompt.storage import (
    FileSidecarsPathManager,
    FileSystemAnnotation,
    FileSystemManager,
)
from docprompt.utils import hash_from_bytes


def test_file_sidecar_manager_paths():
    """Test that the storage file sidecar manager paths are correct."""

    base_path = "s3://example-bucket/"
    file_hash = "example-hash"

    manager = FileSidecarsPathManager(base_path=base_path, file_hash=file_hash)

    assert manager.base_path == base_path
    assert manager.file_hash == file_hash
    assert manager.pdf == f"{base_path}/{file_hash}/base.pdf"
    assert manager.metadata == f"{base_path}/{file_hash}/base.json"
    assert manager.page_metadata == f"{base_path}/{file_hash}/pages.json"


def test_file_system_annotation_validation():
    """Ensure that the file system custom annotation validates correctly."""

    validator_generator = FileSystemAnnotation.__get_validators__()

    validator = next(validator_generator)

    # Assert that there is only one validator
    with pytest.raises(StopIteration):
        next(validator_generator)

    # Test that the validator raises an error when the input is not an fsspec file system
    fsspec_file_system = fsspec.filesystem("file")
    result = validator(fsspec_file_system)
    assert result == fsspec_file_system

    # Ensure that another object raises a TypeError
    with pytest.raises(TypeError):
        validator("not a file system")


class TestFileSystemManager:
    """Unit tests for the file system manager."""

    @pytest.fixture(scope="class")
    def mock_manager(self):
        """Setup a mock manager to use for testing."""

        mock_fs = MagicMock(spec=AbstractFileSystem)

        with patch.object(
            FileSystemManager, "validate_filesystem_protocol_and_kwargs"
        ) as mock_validator:
            mock_validator.return_value = {
                "path": "/tmp/data",
                "fs": mock_fs,
                "fs_kwargs": {},
            }
            return FileSystemManager(url="/tmp/data")

        return mock_fs

    class TestValidation:
        """Test the validation logic of the file system manager."""

        def test_before_model_validator(self):
            """We want to make sure that the before validator properly updates
            the data that is passed to the model."""

            path = "/tmp/data"
            kwargs = {"example": "kwarg"}

            data = {"path": path, "fs_kwargs": kwargs}
            parsed_data = FileSystemManager.validate_filesystem_protocol_and_kwargs(
                data
            )

            assert parsed_data.get("path") == path
            assert isinstance(parsed_data.get("fs"), LocalFileSystem)
            assert parsed_data.get("fs_kwargs") == kwargs

        def test_before_model_validator_proper_fs_instantiation(self):
            """We want to make sure that the fsspec FileSystem is properly instantiated."""

            path = "/tmp/data"
            kwargs = {"example": "kwarg"}

            data = {"path": path, "fs_kwargs": kwargs}
            with patch.object(fsspec, "url_to_fs") as mock_proto_res:
                FileSystemManager.validate_filesystem_protocol_and_kwargs(data)

                mock_proto_res.assert_called_once_with(path, **kwargs)

        # @pytest.mark.parametrize("exists", [True, False])
        # def test_after_model_validator_craetes_base_path(self, exists, mock_manager):
        #     """We want to make sure that the base path is created in when the FS Manager is instantiated."""

        #     path = "/tmp/data"

        #     with patch.object(mock_manager.fs, "exists") as mock_exists:
        #         with patch.object(mock_manager.fs, "mkdirs") as mock_mkdir:
        #             mock_exists.return_value = exists
        #             mock_manager.validate_base_path_exists()

        #     mock_exists.assert_called_once_with(path, **mock_manager.fs_kwargs)

        #     if not exists:
        #         mock_mkdir.assert_called_once_with(path, **mock_manager.fs_kwargs)
        #     else:
        #         mock_mkdir.assert_not_called()

    class TestImplementationMethods:
        """Test the implementation methods of the File System manager"""

        def test_pdf_name_getter(self, mock_manager):
            """Test that the `get_pdf_name` method returns the correct file name."""

            file_hash = "example-hash"

            result_pdf_name = mock_manager.get_pdf_name(file_hash)

            assert result_pdf_name == "base.pdf"

        def test__write_method_creates_dir(self, mock_manager):
            """Test that the write method creates a direcrory if it does not exist."""

            with patch.object(mock_manager.fs, "exists") as mock_exists:
                mock_exists.return_value = False

                with patch.object(mock_manager.fs, "mkdirs") as mock_mkdir:
                    with patch.object(mock_manager.fs, "open"):
                        mock_manager._write(
                            b"example-value",
                            "/tmp/data/file.txt",
                            "wb",
                            example="kwarg",
                        )

            real_kwargs = {**mock_manager.fs_kwargs, "example": "kwarg"}

            mock_exists.assert_called_once_with("/tmp/data", **real_kwargs)
            mock_mkdir.assert_called_once_with("/tmp/data", **real_kwargs)

        def test__write_method(self, mock_manager):
            """Test that the `_write` implementation method works correctly."""

            with patch.object(mock_manager.fs, "open") as mock_open:
                mock_file = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_file

                mock_manager._write(
                    b"example-value", "example-path", "wb", example="kwarg"
                )

            real_kwargs = {**mock_manager.fs_kwargs, "example": "kwarg"}

            mock_open.assert_called_once_with("example-path", "wb", **real_kwargs)
            mock_file.write.assert_called_once_with(b"example-value")

        def test__read_method(self, mock_manager):
            """Test that the `_read` implementation method works correctly."""

            with patch.object(mock_manager.fs, "open") as mock_open:
                mock_file = MagicMock()
                mock_file.read.return_value = b"example-value"
                mock_open.return_value.__enter__.return_value = mock_file

                result = mock_manager._read("example-path", "rb", example="kwarg")

            real_kwargs = {**mock_manager.fs_kwargs, "example": "kwarg"}

            mock_open.assert_called_once_with("example-path", "rb", **real_kwargs)
            mock_file.read.assert_called_once()
            assert result == b"example-value"

        @pytest.mark.parametrize("exists", [True, False])
        def test__delete_method(self, mock_manager, exists):
            """Test that the `_delete` implementation method works correctly."""

            with patch.object(mock_manager.fs, "exists") as mock_exists:
                with patch.object(mock_manager.fs, "rm") as mock_rm:
                    mock_exists.return_value = exists
                    mock_manager._delete("example-path", example="kwarg")

            real_kwargs = {**mock_manager.fs_kwargs, "example": "kwarg"}

            mock_exists.assert_called_once_with("example-path", **real_kwargs)
            if exists:
                mock_rm.assert_called_once_with("example-path", **real_kwargs)

        def test_write_method_no_metadata(self, mock_manager):
            """Test that the write method works correclty when no metadata is provided."""

            example_bytes = b"example-value"
            example_hash = hash_from_bytes(example_bytes)
            expected_path = f"/tmp/data/{example_hash}/base.pdf"

            with patch.object(mock_manager, "_write") as mock_write:
                result = mock_manager.write(b"example-value", example="kwarg")

            mock_write.assert_called_once_with(
                example_bytes, expected_path, "wb", example="kwarg"
            )
            assert result.pdf == expected_path

        def test_write_method_with_metadata(self, mock_manager):
            """Test that the write method works correctly when metadata is provided."""

            example_bytes = b"example-value"
            example_metadata_bytes = b"example-metadata"
            example_page_metadata_bytes = b"example-page-metadata"
            example_hash = hash_from_bytes(example_bytes)

            expected_path = f"/tmp/data/{example_hash}/base.pdf"
            expected_metadata_path = f"/tmp/data/{example_hash}/base.json"
            expected_page_metadata_path = f"/tmp/data/{example_hash}/pages.json"

            with patch.object(mock_manager, "_write") as mock_write:
                result = mock_manager.write(
                    example_bytes,
                    example_metadata_bytes,
                    example_page_metadata_bytes,
                    example="kwarg",
                )

            mock_write.assert_any_call(
                example_bytes, expected_path, "wb", example="kwarg"
            )
            mock_write.assert_any_call(
                example_metadata_bytes, expected_metadata_path, "wb", example="kwarg"
            )
            mock_write.assert_any_call(
                example_page_metadata_bytes,
                expected_page_metadata_path,
                "wb",
                example="kwarg",
            )
            assert result.pdf == expected_path
            assert result.metadata == expected_metadata_path
            assert result.page_metadata == expected_page_metadata_path

        def test_read_method_no_metadata(self, mock_manager):
            """Test that the read method works correctly when no metadata is provided."""

            example_bytes = b"example-value"
            example_hash = hash_from_bytes(example_bytes)
            expected_path = f"/tmp/data/{example_hash}/base.pdf"
            expected_metadata_path = f"/tmp/data/{example_hash}/base.json"
            expected_page_metadata_path = f"/tmp/data/{example_hash}/pages.json"

            with patch.object(mock_manager, "_read") as mock_read:
                mock_read.side_effect = [
                    example_bytes,
                    FileNotFoundError,
                    FileNotFoundError,
                ]

                mock_read.return_value = example_bytes
                pdf, metadata, page_metadata = mock_manager.read(
                    example_hash, example="kwarg"
                )

            mock_read.assert_any_call(expected_path, "rb", example="kwarg")
            mock_read.assert_any_call(expected_metadata_path, "rb", example="kwarg")
            mock_read.assert_any_call(
                expected_page_metadata_path, "rb", example="kwarg"
            )

            assert pdf == example_bytes
            assert metadata is None
            assert page_metadata is None

        def test_read_method_w_metadata(self, mock_manager):
            """Test that the read method works correctly when metadata is provided."""

            example_bytes = b"example-value"
            example_metadata_bytes = b"example-metadata"
            example_page_metadata_bytes = b"example-page-metadata"
            example_hash = hash_from_bytes(example_bytes)

            expected_path = f"/tmp/data/{example_hash}/base.pdf"
            expected_metadata_path = f"/tmp/data/{example_hash}/base.json"
            expected_page_metadata_path = f"/tmp/data/{example_hash}/pages.json"

            with patch.object(mock_manager, "_read") as mock_read:
                mock_read.side_effect = [
                    example_bytes,
                    example_metadata_bytes,
                    example_page_metadata_bytes,
                ]

                pdf, metadata, page_metadata = mock_manager.read(
                    example_hash, example="kwarg"
                )

            mock_read.assert_any_call(expected_path, "rb", example="kwarg")
            mock_read.assert_any_call(expected_metadata_path, "rb", example="kwarg")
            mock_read.assert_any_call(
                expected_page_metadata_path, "rb", example="kwarg"
            )

            assert pdf == example_bytes
            assert metadata == example_metadata_bytes
            assert page_metadata == example_page_metadata_bytes
