"""Test the storage provider implementations."""

import os
import json
from unittest.mock import MagicMock, patch
from typing import Union

import pytest
from pydantic import BaseModel

from docprompt.schema.document import Document
from docprompt.schema.pipeline import DocumentNode
from docprompt.storage._base import (
    validate_document_metadata_class,
    validate_document_node_class,
)
from docprompt.storage.local import (
    LocalFileSystemStorageProvider,
    FALLBACK_LOCAL_STORAGE_PATH,
    validate_file_path,
    FilePathSidecars,
)


class TestAbstractStorageProviderValidation:
    """Test the model validation functions of the base storage provider."""

    def test_validate_document_node_class_valid(self):
        """Test that the document node class validation function returns the class if it is valid."""

        assert validate_document_node_class(DocumentNode) == DocumentNode

    def test_validate_document_node_class_on_child(self):
        """Test that the model validation allows for a child of the document node class."""

        class ChildDocumentNode(DocumentNode):  # pylint: disable=too-few-public-methods,missing-class-docstring
            pass

        assert validate_document_node_class(ChildDocumentNode) == ChildDocumentNode

    def test_validate_document_node_invalid(self):
        """Test that the model validation raises an error for an invalid document node class."""

        with pytest.raises(ValueError):
            validate_document_node_class(dict)

    def test_validate_document_node_metadata_class_valid(self):
        """Test that the document metadata class validation function returns the class if it is valid."""

        assert validate_document_metadata_class(BaseModel) == BaseModel

    def test_validate_document_node_metadata_class_on_child(self):
        """Test that the model validation allows for a child of the document metadata class."""

        class ChildDocumentMetadata(BaseModel):  # pylint: disable=too-few-public-methods,missing-class-docstring
            pass

        assert (
            validate_document_metadata_class(ChildDocumentMetadata)
            == ChildDocumentMetadata
        )

    def test_validate_document_node_metadata_invalid(self):
        """Test that the model validation raises an error for an invalid document metadata class."""

        with pytest.raises(ValueError):
            validate_document_metadata_class(dict)


class TestLocalStorageProvider:
    """Test that the local storage provider works as expected."""

    class TestFilePathSidecars:
        """Test that the pydantic model for fp sidecars works as expected."""

        def teardown_path(self, path: str):
            """Remove the path if it exists."""

            if os.path.exists(path):
                os.removedirs(path)

        def test_validate_file_path_exists(self):
            """Test that the file path sidecars model works as expected."""

            path = "/tmp/.docprompt/test"

            # Ensure that the path exists BEFORE we validate it
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Validate the path
            assert validate_file_path(path) == path

            # Ensure that the parent directory of the path exists now
            assert os.path.exists(os.path.dirname(path))

            self.teardown_path(path)

        def test_validate_path_does_not_exist(self):
            """We want to ensure that the validator properly creates a path if necessary."""

            # Create a path that we know does not exist
            path = "/tmp/.docprompt/test"

            # Ensure that the path does not exist before we validate it
            self.teardown_path(path)

            # Validate the path
            assert validate_file_path(path) == path

            # Ensure that the parent directory of the path exists now
            assert os.path.exists(os.path.dirname(path))

            self.teardown_path(path)

        def test_sidecars_properties(self):
            """Ensure that the properties of the sidecars model work as expected."""

            path = "/tmp/.docprompt/test"

            sidecars = FilePathSidecars(base_file_path=path)

            assert sidecars.pdf == f"{path}.pdf"
            assert sidecars.metadata == f"{path}.json"

            self.teardown_path(path)

    def test_local_storage_provider_default_path(self):
        """Test that the local storage provider uses the default path if no environment variable is set."""

        value = os.environ.get(
            "DOCPROMPT_LOCAL_STORAGE_PATH", FALLBACK_LOCAL_STORAGE_PATH
        )

        assert LocalFileSystemStorageProvider.base_storage_path == value

    class TestLocalFileSystemStorageProviderImplementation:
        """Test the implementation methods of the local storage provider."""

        def teardown_directories(self, sidecar: FilePathSidecars):
            """Teardown the file path directories if they exist."""

            if os.path.exists(sidecar.base_file_path):
                os.removedirs(sidecar.base_file_path)

        @pytest.fixture
        def provider(self):
            """A fixture for setting up the provider."""

            class ExampleMetadata(BaseModel):  # pylint: disable=missing-class-docstring,too-few-public-methods
                title: str

            return LocalFileSystemStorageProvider(
                document_node_class=DocumentNode,
                document_metadata_class=ExampleMetadata,
            )

        @pytest.fixture(scope="function")
        def create_mock_doc_node(self, provider):
            """A fixture for creating a mock document node."""

            def factory(_hash: str, _bytes: bytes, title: Union[None, str]):
                mock_doc_node = MagicMock(spec=DocumentNode)
                mock_doc_node.file_hash = _hash
                mock_doc_node.document = MagicMock()
                mock_doc_node.document.get_bytes.return_value = _bytes

                if title is not None:
                    metadata = provider.document_metadata_class(title=title)
                    mock_doc_node.metadata = metadata

                return mock_doc_node

            return factory

        def test_file_path_helper(self, provider):
            """Test that the file path helper method works as expected."""

            file_hash = "FILE-HASH"

            sidecar = provider._file_path(file_hash)  # pylint: disable=protected-access

            assert sidecar.base_file_path == os.path.join(
                provider.base_storage_path, file_hash
            )

            self.teardown_directories(sidecar)

        def test_store_method(self, provider, create_mock_doc_node):
            """Test that the store method works as expected."""

            file_hash = "FILE-HASH"
            pdf_bytes = b"PDF BYTES"
            metadata_title = "Test Title"
            mock_doc_node = create_mock_doc_node(file_hash, pdf_bytes, metadata_title)

            side_car = provider._file_path(file_hash)  # pylint: disable=protected-access

            with patch("docprompt.storage.local.local_fs_write") as mock_write:
                provider.store(mock_doc_node)

            mock_write.assert_any_call(side_car.pdf, pdf_bytes, mode="wb")
            mock_write.assert_any_call(
                side_car.metadata,
                json.dumps(mock_doc_node.metadata.model_dump(mode="json")),
                encoding="utf-8",
            )
            assert mock_write.call_count == 2

        def test_store_method_no_metadata(self, provider, create_mock_doc_node):
            """Test that the store method works as expected."""

            file_hash = "FILE-HASH"
            pdf_bytes = b"PDF BYTES"
            metadata_title = "Test Title"
            mock_doc_node = create_mock_doc_node(file_hash, pdf_bytes, metadata_title)
            mock_doc_node.metadata = None

            side_car = provider._file_path(file_hash)  # pylint: disable=protected-access

            with patch("docprompt.storage.local.local_fs_write") as mock_write:
                provider.store(mock_doc_node)

            mock_write.assert_any_call(side_car.pdf, pdf_bytes, mode="wb")
            assert mock_write.call_count == 1

        def test_retrieve_method(self, provider, create_mock_doc_node):
            """Retrieve the document node from the local file system."""

            file_hash = "FILE-HASH"
            pdf_bytes = b"PDF BYTES"
            metadata_title = "Test Title"
            mock_doc_node = create_mock_doc_node(file_hash, pdf_bytes, metadata_title)

            side_car = provider._file_path(file_hash)  # pylint: disable=protected-access

            # Store the document node first
            provider.store(mock_doc_node)

            with patch("docprompt.storage.local.local_fs_read") as mock_read:
                with patch.object(Document, "from_bytes") as mock_from_bytes:
                    with patch.object(
                        DocumentNode, "from_document"
                    ) as mock_from_document:
                        mock_read.side_effect = [
                            pdf_bytes,
                            json.dumps(mock_doc_node.metadata.model_dump(mode="json")),
                        ]
                        mock_from_bytes.return_value = mock_doc_node.document
                        provider.retrieve(file_hash)

            mock_read.assert_any_call(side_car.pdf, mode="rb")
            mock_read.assert_any_call(side_car.metadata, mode="r", encoding="utf-8")
            mock_from_bytes.assert_called_once_with(
                pdf_bytes, name=os.path.basename(side_car.pdf)
            )
            mock_from_document.assert_called_once_with(
                document=mock_doc_node.document,
                document_metadata=mock_doc_node.metadata,
            )

            assert mock_read.call_count == 2

        def test_retrieve_method_no_metadata(self, provider, create_mock_doc_node):
            """Retrieve the document node from the local file system."""

            file_hash = "FILE-HASH"
            pdf_bytes = b"PDF BYTES"
            metadata_title = "Test Title"
            mock_doc_node = create_mock_doc_node(file_hash, pdf_bytes, metadata_title)

            side_car = provider._file_path(file_hash)  # pylint: disable=protected-access

            # Store the document node first
            provider.store(mock_doc_node)

            with patch("docprompt.storage.local.local_fs_read") as mock_read:
                with patch.object(Document, "from_bytes") as mock_from_bytes:
                    with patch.object(
                        DocumentNode, "from_document"
                    ) as mock_from_document:
                        mock_read.side_effect = [pdf_bytes, FileNotFoundError()]
                        mock_from_bytes.return_value = mock_doc_node.document
                        provider.retrieve(file_hash)

            mock_read.assert_any_call(side_car.pdf, mode="rb")
            mock_read.assert_any_call(side_car.metadata, mode="r", encoding="utf-8")
            mock_from_bytes.assert_called_once_with(
                pdf_bytes, name=os.path.basename(side_car.pdf)
            )
            mock_from_document.assert_called_once_with(
                document=mock_doc_node.document, document_metadata=None
            )

            assert mock_read.call_count == 2
