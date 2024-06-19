"""Concrete implementation of a storage provider backed by AWS S3.

In order to utilize the AWS S3 storage provider, you must set the following environment
variables:
    - DOCPROMPT_AWS_ACCESS_KEY_ID: The access key ID for the AWS account
    - DOCPROMPT_AWS_SECRET_ACCESS_KEY: The secret access key for the AWS account
    - DOCPROMPT_AWS_DEFAULT_REGION: The default region for the AWS account
    - DOCPROMPT_AWS_BUCKET_KEY: The key to the base S3 folder for storing documents
"""

import os
import json
from urllib.parse import urlparse
from typing_extensions import Annotated

import boto3
from botocore import exceptions as boto_exceptions
from pydantic import BaseModel, Field, AfterValidator, computed_field
from pydantic_settings import BaseSettings

from docprompt.schema.document import Document

from ._base import AbstractStorageProvider, DocumentNode


# Load the S3 Credentials
class S3Credentials(BaseSettings):
    """The settings for the AWS S3 storage provider.

    Attributes:
        AWS_ACCESS_KEY_ID: The access key ID for the AWS account
        AWS_SECRET_ACCESS_KEY: The secret access key for the AWS account
        AWS_DEFAULT_REGION: The default region for the AWS account
        AWS_BUCKET_KEY: The key to the base S3 folder for storing documents
    """

    AWS_ACCESS_KEY_ID: str = Field(..., alias="DOCPROMPT_AWS_ACCESS_KEY_ID", repr=False)
    AWS_SECRET_ACCESS_KEY: str = Field(
        ..., alias="DOCPROMPT_AWS_SECRET_ACCESS_KEY", repr=False
    )
    AWS_DEFAULT_REGION: str = Field(..., alias="DOCPROMPT_AWS_REGION")
    AWS_BUCKET_KEY: str = Field(..., alias="DOCPROMPT_AWS_BUCKET_KEY")


def validate_s3_uri(value: str) -> str:
    """Validate that an S3 URI is valid and exits.

    Args:
        value: The S3 URI to validate

    Returns:
        str: The S3 URI if it is valid

    Raises:
        ValueError: If the S3 URI is invalid
    """
    parsed_uri = urlparse(value)
    if parsed_uri.scheme != "s3":
        raise ValueError("Invalid S3 URI scheme. URI must start with 's3://'")
    if not parsed_uri.netloc:
        raise ValueError("Invalid S3 URI. No bucket name specified.")
    return value.rstrip("/")


class S3BucketURISidecars(BaseModel):
    """The pair of S3 bucket URIs for a document node.

    Attributes:
        base_s3_uri: The base s3 URI for a document node
        pdf: The URI for the PDF file of the document node
        metadata: The URI for the metadata of the document node
    """

    base_s3_uri: Annotated[str, AfterValidator(validate_s3_uri)] = Field(
        ..., repr=False
    )

    @computed_field
    @property
    def pdf(self) -> str:
        """Get the file path for the pdf bytes."""
        return f"{self.base_s3_uri}.pdf"

    @computed_field
    @property
    def metadata(self) -> str:
        """Get the file path for the metadata."""
        return f"{self.base_s3_uri}.json"


class S3StorageProvider(AbstractStorageProvider[S3BucketURISidecars]):
    """The concrete implementation of an AWS S3 storage provider.

    Attributes:
        document_node_class: The document node class to store and retrieve
        document_metadata_class: The document metadata class to store and retrieve
    """

    def _file_paths(self, file_hash: str) -> S3BucketURISidecars:
        """Get the S3 bucket URIs for a document node.

        Args:
            file_hash: The hash of the document node

        Returns:
            S3BucketURISidecars: The S3 bucket URIs for the document node

        Raises:
            pydantic_core.ValidationError: If the S3 URI is invalid
        """
        # Ensure the base S3 URI ends with a slash
        s3_credentials = S3Credentials()
        base_s3_uri = (
            s3_credentials.AWS_BUCKET_KEY
            if s3_credentials.AWS_BUCKET_KEY.endswith("/")  # pylint: disable=no-member
            else s3_credentials.AWS_BUCKET_KEY + "/"
        )
        bucket_path = os.path.join(base_s3_uri, file_hash)
        return S3BucketURISidecars(base_s3_uri=bucket_path)

    def store(self, document_node: DocumentNode) -> S3BucketURISidecars:
        """Store a document in the S3 bucket.

        Args:
            document: The document to store

        Returns:
            S3BucketURISidecars: The S3 bucket URIs for the document
        """

        file_paths = self._file_paths(document_node.file_hash)

        aws_s3_write(file_paths.pdf, document_node.document.get_bytes())

        if document_node.metadata is not None:
            aws_s3_write(
                file_paths.metadata,
                json.dumps(document_node.metadata.model_dump(mode="json")),
            )
        else:
            aws_s3_delete(file_paths.metadata)

        return file_paths

    def retrieve(self, file_hash: str) -> DocumentNode:
        """Retrieve a document from the S3 bucket.

        Args:
            file_hash: The hash of the document to retrieve

        Returns:
            DocumentNode: The document node
        """
        file_paths = self._file_paths(file_hash)

        pdf_bytes = aws_s3_read(file_paths.pdf)
        document = Document.from_bytes(
            pdf_bytes, name=os.path.basename(urlparse(file_paths.pdf).path)
        )

        try:
            metadata_bytes = aws_s3_read(file_paths.metadata)
            metadata = self.document_metadata_class.model_validate(
                json.loads(metadata_bytes)
            )
        except FileNotFoundError:
            metadata = None

        return self.document_node_class.from_document(
            document=document, document_metadata=metadata
        )


def aws_s3_read(uri: str) -> bytes:
    """Read the contents of a file from an S3 bucket.

    Args:
        uri: The S3 URI of the file to read

    Returns:
        bytes: The contents of the file

    Raises:
        FileNotFoundError: If the file is not found in the S3 bucket under the specified URI
        botocore.exceptions.ClientError: If there is an error reading the file from the S3 bucket
    """

    s3_credentials = S3Credentials()
    s3_client = boto3.client(
        "s3",
        region_name=s3_credentials.AWS_DEFAULT_REGION,
        aws_access_key_id=s3_credentials.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=s3_credentials.AWS_SECRET_ACCESS_KEY,
    )

    s3_uri = urlparse(uri, allow_fragments=False)

    try:
        response = s3_client.get_object(
            Bucket=s3_uri.netloc, Key=s3_uri.path.lstrip("/")
        )
    except boto_exceptions.ClientError as exp:
        if exp.response.get("Error", {}).get("Code", None) == "NoSuchKey":
            raise FileNotFoundError(
                f"The file {uri} was not found in the S3 bucket"
            ) from exp
        raise exp

    return response["Body"].read()


def aws_s3_write(uri: str, data: bytes) -> None:
    """Write the contents of a file to an S3 bucket.

    Args:
        uri: The S3 URI of the file to write
        data: The contents of the file
    """

    s3_credentials = S3Credentials()
    s3_client = boto3.client(
        "s3",
        region_name=s3_credentials.AWS_DEFAULT_REGION,
        aws_access_key_id=s3_credentials.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=s3_credentials.AWS_SECRET_ACCESS_KEY,
    )

    s3_uri = urlparse(uri, allow_fragments=False)
    s3_client.put_object(Bucket=s3_uri.netloc, Key=s3_uri.path.lstrip("/"), Body=data)


def aws_s3_delete(uri: str) -> None:
    """Destroy an object in an S3 bucket.

    Args:
        uri: The S3 URI of the object to destroy
    """
    s3_credentials = S3Credentials()
    s3_client = boto3.client(
        "s3",
        region_name=s3_credentials.AWS_DEFAULT_REGION,
        aws_access_key_id=s3_credentials.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=s3_credentials.AWS_SECRET_ACCESS_KEY,
    )

    s3_uri = urlparse(uri, allow_fragments=False)
    s3_client.delete_object(Bucket=s3_uri.netloc, Key=s3_uri.path.lstrip("/"))
