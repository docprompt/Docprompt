"""
We want to test that our credential managers properly handle fallbacks to environment variables,
and can properly expose a set of kwargs for factories to use.
"""

import contextlib
import os
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from docprompt.tasks.credentials import (
    APIKeyCredential,
    AWSCredentials,
    BaseCredentials,
    GCPServiceFileCredentials,
)


def test_base_credentials_kwargs():
    class SampleCreds(BaseCredentials):
        test: SecretStr

    creds = SampleCreds(test="test")
    assert creds.kwargs == {"test": "test"}


class TestAPIKeyCredentials:
    @pytest.fixture(scope="class")
    def api_key(self):
        return "test-key"

    def test_api_key_credential_direct_init(self, api_key):
        creds = APIKeyCredential(api_key=api_key)
        assert creds.kwargs == {"api_key": api_key}

    def test_api_key_credential_direct_init_ignores_environ(self, api_key):
        with patch.object(os.environ, "get") as mock_get:
            creds = APIKeyCredential(api_key=api_key, environ_path="TEST")

        mock_get.assert_not_called()
        assert creds.kwargs == {"api_key": api_key}

    def test_api_key_fallback_to_environ(self, api_key):
        with patch.object(os.environ, "get", return_value="test-key") as mock_get:
            mock_get.return_value = api_key
            creds = APIKeyCredential(environ_path="TEST_API_KEY")

        mock_get.assert_called_once_with("TEST_API_KEY", None)
        assert creds.kwargs == {"api_key": api_key}

    def test_error_with_invalid_kwargs(self):
        with pytest.raises(ValueError):
            APIKeyCredential()


class TestAWSCredentials:
    @classmethod
    @pytest.fixture(scope="class", autouse=True)
    def setup_class(cls):
        # Unset all AWS-related environment variables
        aws_env_vars = [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_DEFAULT_REGION",
            "AWS_SESSION_TOKEN",
        ]
        for var in aws_env_vars:
            with contextlib.suppress(KeyError):
                del os.environ[var]

    @pytest.fixture
    def aws_creds(self):
        return {
            "aws_access_key_id": "test_access_key",
            "aws_secret_access_key": "test_secret_key",
            "aws_region": "us-west-2",
        }

    def test_aws_credentials_direct_init(self, aws_creds):
        creds = AWSCredentials(**aws_creds)
        assert creds.kwargs == aws_creds

    def test_aws_credentials_fallback_to_environ(self, aws_creds):
        with patch.dict(
            os.environ,
            {
                "AWS_ACCESS_KEY_ID": aws_creds["aws_access_key_id"],
                "AWS_SECRET_ACCESS_KEY": aws_creds["aws_secret_access_key"],
                "AWS_DEFAULT_REGION": aws_creds["aws_region"],
            },
        ):
            creds = AWSCredentials()
            assert creds.kwargs == aws_creds

    def test_aws_credentials_mixed_init(self, aws_creds):
        with patch.dict(
            os.environ, {"AWS_SECRET_ACCESS_KEY": aws_creds["aws_secret_access_key"]}
        ):
            creds = AWSCredentials(
                aws_access_key_id=aws_creds["aws_access_key_id"],
                aws_region=aws_creds["aws_region"],
            )
            assert creds.kwargs == aws_creds

    def test_aws_credentials_with_session_token(self):
        session_token = "test_session_token"
        creds = AWSCredentials(aws_session_token=session_token)
        assert creds.kwargs == {"aws_session_token": session_token}

    def test_aws_credentials_invalid_no_credentials(self):
        with pytest.raises(ValueError):
            AWSCredentials()

    def test_aws_credentials_invalid_missing_region(self, aws_creds):
        with pytest.raises(ValueError):
            AWSCredentials(
                aws_access_key_id=aws_creds["aws_access_key_id"],
                aws_secret_access_key=aws_creds["aws_secret_access_key"],
            )

    def test_aws_credentials_invalid_mixed_auth(self, aws_creds):
        with pytest.raises(ValueError):
            AWSCredentials(
                aws_access_key_id=aws_creds["aws_access_key_id"],
                aws_secret_access_key=aws_creds["aws_secret_access_key"],
                aws_region=aws_creds["aws_region"],
                aws_session_token="test_session_token",
            )

    def test_aws_credentials_kwargs_property(self, aws_creds):
        creds = AWSCredentials(**aws_creds)
        kwargs = creds.kwargs
        assert isinstance(kwargs, dict)
        assert all(isinstance(v, str) for v in kwargs.values())
        assert kwargs == aws_creds


class TestCloudCredentials:
    @classmethod
    @pytest.fixture(scope="class", autouse=True)
    def setup_class(cls):
        # Unset all AWS and GCP-related environment variables
        cloud_env_vars = [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_DEFAULT_REGION",
            "AWS_SESSION_TOKEN",
            "GCP_SERVICE_ACCOUNT_FILE",
        ]
        for var in cloud_env_vars:
            with contextlib.suppress(KeyError):
                del os.environ[var]

    @pytest.fixture
    def aws_creds(self):
        return {
            "aws_access_key_id": "test_access_key",
            "aws_secret_access_key": "test_secret_key",
            "aws_region": "us-west-2",
        }

    # ... existing AWS test methods remain unchanged ...

    @pytest.fixture
    def gcp_service_account_info(self):
        return {
            "type": "service_account",
            "project_id": "test-project",
            "private_key_id": "test-key-id",
            "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQC9hSuP...\n-----END PRIVATE KEY-----\n",
            "client_email": "test@test-project.iam.gserviceaccount.com",
            "client_id": "123456789",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/test%40test-project.iam.gserviceaccount.com",
        }

    def test_gcp_credentials_with_service_account_info(self, gcp_service_account_info):
        creds = GCPServiceFileCredentials(service_account_info=gcp_service_account_info)
        assert creds.service_account_info == gcp_service_account_info
        assert creds.service_account_file is None

    def test_gcp_credentials_with_service_account_file(self):
        file_path = "/path/to/service_account.json"
        creds = GCPServiceFileCredentials(service_account_file=file_path)
        assert creds.service_account_file == file_path
        assert creds.service_account_info is None

    def test_gcp_credentials_fallback_to_environ(self):
        file_path = "/path/to/service_account.json"
        with patch.dict(os.environ, {"GCP_SERVICE_ACCOUNT_FILE": file_path}):
            creds = GCPServiceFileCredentials()
            assert creds.service_account_file == file_path
            assert creds.service_account_info is None

    def test_gcp_credentials_invalid_no_credentials(self):
        with pytest.raises(ValueError):
            GCPServiceFileCredentials()

    def test_gcp_credentials_invalid_both_provided(self, gcp_service_account_info):
        with pytest.raises(ValueError):
            GCPServiceFileCredentials(
                service_account_info=gcp_service_account_info,
                service_account_file="/path/to/service_account.json",
            )

    def test_gcp_credentials_kwargs_property(self, gcp_service_account_info):
        creds = GCPServiceFileCredentials(service_account_info=gcp_service_account_info)
        kwargs = creds.kwargs
        assert isinstance(kwargs, dict)
        assert kwargs == {"service_account_info": gcp_service_account_info}
