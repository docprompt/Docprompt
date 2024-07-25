"""The credentials module defines a simple model schema for storing credentials."""

import os
from typing import Dict, Mapping, Optional

from pydantic import BaseModel, Field, HttpUrl, SecretStr, model_validator
from typing_extensions import Self


class BaseCredentials(BaseModel):
    """The base credentials model."""

    @property
    def kwargs(self) -> Dict[str, str]:
        """Return the credentials as a dictionary with secrets exposed."""
        data = self.model_dump(exclude_none=True)
        for key, value in data.items():
            if isinstance(value, SecretStr):
                data[key] = value.get_secret_value()
        return data


class APIKeyCredential(BaseCredentials):
    """The API key credential model."""

    api_key: SecretStr

    def __init__(self, environ_path: Optional[str] = None, **data):
        api_key = data.get("api_key", None)
        if api_key is None and environ_path:
            api_key = os.environ.get(environ_path, None)
            data["api_key"] = api_key

        super().__init__(**data)


class GenericOpenAICredentials(APIKeyCredential):
    """Credentials that are common for OpenAI API requests."""

    base_url: Optional[HttpUrl] = Field(None)
    timeout: Optional[int] = Field(None)
    max_retries: Optional[int] = Field(None)

    default_headers: Optional[Mapping[str, str]] = Field(None)
    default_query_params: Optional[Mapping[str, object]] = Field(None)


class AWSCredentials(BaseCredentials):
    """The AWS credentials model."""

    aws_access_key_id: Optional[SecretStr] = Field(None)
    aws_secret_access_key: Optional[SecretStr] = Field(None)
    aws_session_token: Optional[SecretStr] = Field(None)
    aws_region: Optional[str] = Field(None)

    def __init__(self, **data):
        aws_access_key_id = data.get(
            "aws_access_key_id", os.environ.get("AWS_ACCESS_KEY_ID", None)
        )
        aws_secret_access_key = data.get(
            "aws_secret_access_key", os.environ.get("AWS_SECRET_ACCESS_KEY", None)
        )
        aws_session_token = data.get(
            "aws_session_token", os.environ.get("AWS_SESSION_TOKEN", None)
        )
        aws_region = data.get("aws_region", os.environ.get("AWS_DEFAULT_REGION", None))

        super().__init__(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            aws_region=aws_region,
        )

    @model_validator(mode="after")
    def _validate_aws_credentials(self) -> Self:
        """Ensure the provided AWS credentials are valid."""

        key_pair_is_set = self.aws_access_key_id and self.aws_secret_access_key

        if not key_pair_is_set and not self.aws_session_token:
            raise ValueError(
                "You must provide either an AWS session token or an access key and secret key."
            )

        if key_pair_is_set and not self.aws_region:
            raise ValueError(
                "You must provide an AWS region when using an access key and secret key."
            )

        if key_pair_is_set and self.aws_session_token:
            raise ValueError(
                "You cannot provide both an AWS session token and an access key and secret key."
            )

        return self


class GCPServiceFileCredentials(BaseCredentials):
    """The GCP service account credentials model."""

    service_account_info: Optional[Dict[str, str]] = Field(None)
    service_account_file: Optional[str] = Field(None)

    def __init__(self, **data):
        service_account_info = data.get("service_account_info", None)
        service_account_file = data.get(
            "service_account_file", os.environ.get("GCP_SERVICE_ACCOUNT_FILE", None)
        )

        super().__init__(
            service_account_info=service_account_info,
            service_account_file=service_account_file,
        )

    @model_validator(mode="after")
    def _validate_gcp_credentials(self) -> Self:
        """Ensure the provided GCP credentials are valid."""
        if self.service_account_info is None and self.service_account_file is None:
            raise ValueError(
                "You must provide either service_account_info or service_account_file. You may set the `GCP_SERVICE_ACCOUNT_FILE` environment variable to the path of the service account file."
            )
        if (
            self.service_account_info is not None
            and self.service_account_file is not None
        ):
            raise ValueError(
                "You must provide either service_account_info or service_account_file, not both"
            )
        return self
