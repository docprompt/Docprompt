"""Test the storage provider implementations."""

import os
import json
from unittest.mock import MagicMock, patch
from typing import Union

import pytest
from botocore.exceptions import ClientError
from pydantic import BaseModel

from docprompt.schema.document import Document
from docprompt.schema.pipeline import DocumentNode
from docprompt.storage._base import (
    AbstractStorageProvider,
    validate_document_metadata_class,
    validate_document_node_class,
)
from docprompt.storage import local, s3


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

    def test_from_document_node_with_metadata(self):
        """Test that the class method for creating a provider from a document node with metadata
        works as expected.
        """

        class ConcreteProvider(AbstractStorageProvider[BaseModel]):  # pylint: disable=missing-class-docstring
            def paths(self, file_hash: str) -> BaseModel:
                """Generate the paths for the document node."""

            def store(self, document_node: DocumentNode) -> BaseModel:
                """Store the document node from the storage provider."""

            def retrieve(self, file_hash: str) -> DocumentNode:
                """Retrieve the document node from the storage provider."""

        class ExampleMetadata(BaseModel):  # pylint: disable=too-few-public-methods
            """Sample Metadata container"""

            title: str

        document_node = MagicMock(spec=DocumentNode)
        document_node.metadata = ExampleMetadata(title="Test Title")

        provider = ConcreteProvider.from_document_node(document_node=document_node)

        assert provider.document_metadata_class == ExampleMetadata
        assert provider.document_node_class == DocumentNode

    def test_from_document_node_wo_metadata(self):
        """Test that the class method for creating a provider from a document node without metadata"""

        class ConcreteProvider(AbstractStorageProvider[BaseModel]):
            """Concrete provider for testing."""

            def paths(self, file_hash: str) -> BaseModel:
                """Generate the paths for the document node."""

            def store(self, document_node: DocumentNode) -> BaseModel:
                """Store the document node from the storage provider."""

            def retrieve(self, file_hash: str) -> DocumentNode:
                """Retrieve the document node from the storage provider."""

        document_node = MagicMock(spec=DocumentNode)

        provider = ConcreteProvider.from_document_node(document_node=document_node)

        assert provider.document_metadata_class is None
        assert provider.document_node_class == DocumentNode


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
            assert local.validate_file_path(path) == path

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
            assert local.validate_file_path(path) == path

            # Ensure that the parent directory of the path exists now
            assert os.path.exists(os.path.dirname(path))

            self.teardown_path(path)

        def test_sidecars_properties(self):
            """Ensure that the properties of the sidecars model work as expected."""

            path = "/tmp/.docprompt/test"

            sidecars = local.FilePathSidecars(base_file_path=path)

            assert sidecars.pdf == f"{path}.pdf"
            assert sidecars.metadata == f"{path}.json"

            self.teardown_path(path)

    def test_local_storage_provider_default_path(self):
        """Test that the local storage provider uses the default path if no environment variable is set."""

        value = os.environ.get(
            "DOCPROMPT_LOCAL_STORAGE_PATH", local.FALLBACK_LOCAL_STORAGE_PATH
        )

        assert local.LocalFileSystemStorageProvider.base_storage_path == value

    class TestFileSystemMethods:
        """Test the functions that operate directly on the FS."""

        def test_local_fs_write(self):
            """Ensure that the local file system write function works as expected."""

            path = "/test/path/test.txt"
            data = b"TEST DATA"

            with patch("builtins.open") as mock_open:
                with patch("os.path.exists") as mock_exists:
                    mock_exists.return_value = True

                    local.local_fs_write(path, data, mode="mode", encoding="encoding")

            mock_open.assert_called_once_with(path, "mode", encoding="encoding")

        def test_local_fs_write_dir_doesnot_exist(self):
            """Ensure that the local file system writer throws a ValueError if the
            directory does not exist."""

            path = "/test/path/test.txt"
            data = b"TEST DATA"

            with patch("builtins.open", create=True):
                with patch("os.path.exists") as mock_exists:
                    mock_exists.return_value = False

                    with pytest.raises(ValueError):
                        local.local_fs_write(
                            path, data, mode="mode", encoding="encoding"
                        )

        def test_local_fs_read(self):
            """Ensure that the local file system read function works as expected."""

            path = "/test/path/test.txt"

            with patch("builtins.open") as mock_open:
                with patch("os.path.exists") as mock_exists:
                    mock_exists.return_value = True

                    local.local_fs_read(path, mode="mode", encoding="encoding")

            mock_open.assert_called_once_with(path, "mode", encoding="encoding")

        def test_local_fs_read_file_not_found(self):
            """Ensure that the local file system reader throws a FileNotFoundError if the
            file does not exist."""

            path = "/test/path/test.txt"

            with patch("builtins.open", create=True):
                with patch("os.path.exists") as mock_exists:
                    mock_exists.return_value = False

                    with pytest.raises(FileNotFoundError):
                        local.local_fs_read(path, mode="mode", encoding="encoding")

        def test_local_fs_delete(self):
            """Ensure that the local file system delete function works as expected."""

            path = "/test/path/test.txt"

            with patch("os.path.exists") as mock_exists:
                with patch("os.remove") as mock_remove:
                    mock_exists.return_value = True

                    local.local_fs_delete(path)

            mock_remove.assert_called_once_with(path)

        def test_local_fs_delete_does_not_exist(self):
            """Ensure that the local file system delete function does not raise an error if the
            file does not exist."""

            path = "/test/path/test.txt"

            with patch("os.path.exists") as mock_exists:
                with patch("os.remove") as mock_remove:
                    mock_exists.return_value = False

                    local.local_fs_delete(path)

            mock_remove.assert_not_called()

    class TestLocalFileSystemStorageProviderImplementation:
        """Test the implementation methods of the local storage provider."""

        def teardown_directories(self, sidecar: local.FilePathSidecars):
            """Teardown the file path directories if they exist."""

            if os.path.exists(sidecar.base_file_path):
                os.removedirs(sidecar.base_file_path)

        @pytest.fixture
        def provider(self):
            """A fixture for setting up the provider."""

            class ExampleMetadata(BaseModel):  # pylint: disable=missing-class-docstring,too-few-public-methods
                title: str

            return local.LocalFileSystemStorageProvider(
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

            sidecar = provider.paths(file_hash)  # pylint: disable=protected-access

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

            side_car = provider.paths(file_hash)  # pylint: disable=protected-access

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

            side_car = provider.paths(file_hash)  # pylint: disable=protected-access

            with patch("docprompt.storage.local.local_fs_write") as mock_write:
                with patch("docprompt.storage.local.local_fs_delete") as mock_delete:
                    provider.store(mock_doc_node)

            mock_write.assert_called_once_with(side_car.pdf, pdf_bytes, mode="wb")
            mock_delete.assert_called_once_with(side_car.metadata)
            assert mock_write.call_count == 1

        def test_retrieve_method(self, provider, create_mock_doc_node):
            """Retrieve the document node from the local file system."""

            file_hash = "FILE-HASH"
            pdf_bytes = b"PDF BYTES"
            metadata_title = "Test Title"
            mock_doc_node = create_mock_doc_node(file_hash, pdf_bytes, metadata_title)

            side_car = provider.paths(file_hash)  # pylint: disable=protected-access

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
                storage_provider_class=type(provider),
            )

            assert mock_read.call_count == 2

        def test_retrieve_method_no_metadata(self, provider, create_mock_doc_node):
            """Retrieve the document node from the local file system."""

            file_hash = "FILE-HASH"
            pdf_bytes = b"PDF BYTES"
            metadata_title = "Test Title"
            mock_doc_node = create_mock_doc_node(file_hash, pdf_bytes, metadata_title)

            side_car = provider.paths(file_hash)  # pylint: disable=protected-access

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
                document=mock_doc_node.document,
                document_metadata=None,
                storage_provider_class=type(provider),
            )

            assert mock_read.call_count == 2


MOCK_AWS_ACCESS_KEY_ID = "FAKE-ACCESS-KEY-ID"
MOCK_AWS_SECRET_ACCESS_KEY = "FAKE-SECRET-ACCESS-KEY"
MOCK_AWS_BUCKET_KEY = "s3://FAKE-BUCKET-KEY"
MOCK_AWS_REGION = "FAKE-REGION"


class TestS3StorageProvider:
    """Test the S3 storage provider implementation."""

    @pytest.fixture(scope="class", autouse=True)
    def mock_aws_credentials(self):
        """Setup the AWS credentials for the test class."""

        os.environ["DOCPROMPT_AWS_ACCESS_KEY_ID"] = MOCK_AWS_ACCESS_KEY_ID
        os.environ["DOCPROMPT_AWS_SECRET_ACCESS_KEY"] = MOCK_AWS_SECRET_ACCESS_KEY
        os.environ["DOCPROMPT_AWS_BUCKET_KEY"] = MOCK_AWS_BUCKET_KEY
        os.environ["DOCPROMPT_AWS_REGION"] = MOCK_AWS_REGION

        yield

        del os.environ["DOCPROMPT_AWS_ACCESS_KEY_ID"]
        del os.environ["DOCPROMPT_AWS_SECRET_ACCESS_KEY"]
        del os.environ["DOCPROMPT_AWS_BUCKET_KEY"]
        del os.environ["DOCPROMPT_AWS_REGION"]

    def test_aws_s3_credentials(self):
        """Ensure that the AWS S3 credentials model works as expected."""

        credentials = s3.S3Credentials()

        assert (
            credentials.AWS_ACCESS_KEY_ID.get_secret_value() == MOCK_AWS_ACCESS_KEY_ID
        )  # pylint: disable=no-member
        assert (
            credentials.AWS_SECRET_ACCESS_KEY.get_secret_value()  # pylint: disable=no-member
            == MOCK_AWS_SECRET_ACCESS_KEY
        )
        assert credentials.AWS_BUCKET_KEY == MOCK_AWS_BUCKET_KEY
        assert credentials.AWS_DEFAULT_REGION == MOCK_AWS_REGION

    class TestBucketURISidecars:
        """Test the bucket URI sidecars model."""

        @pytest.mark.parametrize(
            "uri,raises",
            [
                ("s3://bucket/key", False),
                ("s3://bucket/key/", False),
                ("s3://bucket/", False),
                ("s3://bucket", False),
                ("bucket/key", True),
                ("s3://", True),
            ],
        )
        def test_validate_s3_uri(self, uri, raises):
            """Ensure that the validation of the S3 URI works as expected."""

            if raises:
                with pytest.raises(ValueError):
                    s3.validate_s3_uri(uri)
            else:
                assert s3.validate_s3_uri(uri) == uri.rstrip("/")

        def test_bucket_uri_sidecars(self):
            """Ensure that the bucket URI sidecars model works as expected."""

            base_uri = "s3://bucket/key"

            sidecars = s3.S3BucketURISidecars(base_s3_uri=base_uri)

            assert sidecars.pdf == f"{base_uri}.pdf"
            assert sidecars.metadata == f"{base_uri}.json"

    class TestS3NetworkLayer:
        """Test that the network layer functions work as expected."""

        def test_s3_read(self):
            """Ensure that we can read from and S3 bucket."""
            uri = "s3://bucket/key"

            with patch("boto3.client") as mock_client:
                with patch("docprompt.storage.s3.urlparse") as mock_urlparse:
                    mock_urlparse.return_value.path = "/key"
                    mock_urlparse.return_value.netloc = "bucket"
                    s3.aws_s3_read(uri)

            mock_client.assert_called_once_with(
                "s3",
                region_name=MOCK_AWS_REGION,
                aws_access_key_id=MOCK_AWS_ACCESS_KEY_ID,
                aws_secret_access_key=MOCK_AWS_SECRET_ACCESS_KEY,
            )
            mock_urlparse.assert_called_once_with(
                uri,
                allow_fragments=False,
            )
            mock_client.return_value.get_object.assert_called_once_with(
                Bucket="bucket",
                Key="key",
            )

        def test_s3_read_no_such_key(self):
            """Ensure that a no such key error is properly handled."""

            uri = "s3://bucket/key"

            with patch("boto3.client") as mock_client:
                with patch("docprompt.storage.s3.urlparse") as mock_urlparse:
                    mock_urlparse.return_value.path = "/key"
                    mock_urlparse.return_value.netloc = "bucket"

                    # Configure the mock client to raise a NoSuchKey exception
                    mock_client.return_value.get_object.side_effect = ClientError(
                        error_response={
                            "Error": {
                                "Code": "NoSuchKey",
                                "Message": "The specified key does not exist.",
                            }
                        },
                        operation_name="GetObject",
                    )

                    with pytest.raises(FileNotFoundError):
                        s3.aws_s3_read(uri)

        def test_s3_read_other_error(self):
            """Check the other boto3 errors are propageted correctly."""

            uri = "s3://bucket/key"

            with patch("boto3.client") as mock_client:
                with patch("docprompt.storage.s3.urlparse") as mock_urlparse:
                    mock_urlparse.return_value.path = "/key"
                    mock_urlparse.return_value.netloc = "bucket"

                    # Configure the mock client to raise a NoSuchKey exception
                    mock_client.return_value.get_object.side_effect = ClientError(
                        error_response={
                            "Error": {
                                "Code": "OtherError",
                                "Message": "Some other error.",
                            }
                        },
                        operation_name="GetObject",
                    )

                    with pytest.raises(ClientError):
                        s3.aws_s3_read(uri)

        def test_s3_write(self):
            """Ensure that the write function works as expected."""

            uri = "s3://bucket/key"
            data = b"DATA"

            with patch("boto3.client") as mock_client:
                with patch("docprompt.storage.s3.urlparse") as mock_urlparse:
                    mock_urlparse.return_value.path = "/key"
                    mock_urlparse.return_value.netloc = "bucket"
                    s3.aws_s3_write(uri, data)

            mock_client.assert_called_once_with(
                "s3",
                region_name=MOCK_AWS_REGION,
                aws_access_key_id=MOCK_AWS_ACCESS_KEY_ID,
                aws_secret_access_key=MOCK_AWS_SECRET_ACCESS_KEY,
            )
            mock_urlparse.assert_called_once_with(
                uri,
                allow_fragments=False,
            )
            mock_client.return_value.put_object.assert_called_once_with(
                Bucket="bucket",
                Key="key",
                Body=data,
            )

        def test_s3_delete(self):
            """Ensure that the delete function works as expected."""

            uri = "s3://bucket/key"

            with patch("boto3.client") as mock_client:
                with patch("docprompt.storage.s3.urlparse") as mock_urlparse:
                    mock_urlparse.return_value.path = "/key"
                    mock_urlparse.return_value.netloc = "bucket"
                    s3.aws_s3_delete(uri)

            mock_client.assert_called_once_with(
                "s3",
                region_name=MOCK_AWS_REGION,
                aws_access_key_id=MOCK_AWS_ACCESS_KEY_ID,
                aws_secret_access_key=MOCK_AWS_SECRET_ACCESS_KEY,
            )
            mock_urlparse.assert_called_once_with(
                uri,
                allow_fragments=False,
            )
            mock_client.return_value.delete_object.assert_called_once_with(
                Bucket="bucket",
                Key="key",
            )

    class TestS3StorageProviderImplementation:
        """Test the implementation methods of the S3 storage provider."""

        def teardown_directories(self, sidecar: s3.S3BucketURISidecars):
            """Teardown the file path directories if they exist."""

            if os.path.exists(sidecar.base_s3_uri):
                os.removedirs(sidecar.base_s3_uri)

        @pytest.fixture
        def provider(self):
            """A fixture for setting up the provider."""

            class ExampleMetadata(BaseModel):  # pylint: disable=missing-class-docstring,too-few-public-methods
                title: str

            return s3.S3StorageProvider(
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

        def test_store_method(self, provider, create_mock_doc_node):
            """Test that the store method works as expected."""

            file_hash = "FILE-HASH"
            pdf_bytes = b"PDF BYTES"
            metadata_title = "Test Title"
            mock_doc_node = create_mock_doc_node(file_hash, pdf_bytes, metadata_title)

            side_car = provider.paths(file_hash)  # pylint: disable=protected-access

            with patch("docprompt.storage.s3.aws_s3_write") as mock_write:
                result = provider.store(mock_doc_node)

            mock_write.assert_any_call(side_car.pdf, pdf_bytes)
            mock_write.assert_any_call(
                side_car.metadata,
                json.dumps(mock_doc_node.metadata.model_dump(mode="json")),
            )
            assert mock_write.call_count == 2
            assert result == side_car

        def test_store_method_no_metadata(self, provider, create_mock_doc_node):
            """Test that the store method works as expected without metdata."""

            file_hash = "FILE-HASH"
            pdf_bytes = b"PDF BYTES"
            metadata_title = "Test Title"
            mock_doc_node = create_mock_doc_node(file_hash, pdf_bytes, metadata_title)
            mock_doc_node.metadata = None

            side_car = provider.paths(file_hash)

            with patch("docprompt.storage.s3.aws_s3_write") as mock_write:
                with patch("docprompt.storage.s3.aws_s3_delete") as mock_delete:
                    result = provider.store(mock_doc_node)

            mock_write.assert_called_once_with(side_car.pdf, pdf_bytes)
            mock_delete.assert_called_once_with(side_car.metadata)

            assert result == side_car

        def test_retrieve_method(self, provider, create_mock_doc_node):
            """Test that the retrieve method works as expected."""

            file_hash = "FILE-HASH"
            pdf_bytes = b"PDF BYTES"
            metadata_title = "Test Title"
            mock_doc_node = create_mock_doc_node(file_hash, pdf_bytes, metadata_title)

            side_car = provider.paths(file_hash)  # pylint: disable=protected-access

            with patch("docprompt.storage.s3.aws_s3_read") as mock_read:
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

            mock_read.assert_any_call(side_car.pdf)
            mock_read.assert_any_call(side_car.metadata)
            mock_from_bytes.assert_called_once_with(
                pdf_bytes, name=os.path.basename(side_car.pdf)
            )
            mock_from_document.assert_called_once_with(
                document=mock_doc_node.document,
                document_metadata=mock_doc_node.metadata,
                storage_provider_class=type(provider),
            )

        def test_retrieve_method_no_metadata(self, provider, create_mock_doc_node):
            """Ensure that the retrieve method works as expected without metadata."""

            file_hash = "FILE-HASH"
            pdf_bytes = b"PDF BYTES"
            mock_doc_node = create_mock_doc_node(file_hash, pdf_bytes, None)

            side_car = provider.paths(file_hash)  # pylint: disable=protected-access

            with patch("docprompt.storage.s3.aws_s3_read") as mock_read:
                with patch.object(Document, "from_bytes") as mock_from_bytes:
                    with patch.object(
                        DocumentNode, "from_document"
                    ) as mock_from_document:
                        mock_read.side_effect = [pdf_bytes, FileNotFoundError()]
                        mock_from_bytes.return_value = mock_doc_node.document
                        provider.retrieve(file_hash)

            mock_read.assert_any_call(side_car.pdf)
            mock_read.assert_any_call(side_car.metadata)
            mock_from_bytes.assert_called_once_with(
                pdf_bytes, name=os.path.basename(side_car.pdf)
            )
            mock_from_document.assert_called_once_with(
                document=mock_doc_node.document,
                document_metadata=None,
                storage_provider_class=type(provider),
            )

            assert mock_read.call_count == 2
