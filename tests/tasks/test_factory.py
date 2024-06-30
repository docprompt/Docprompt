"""
Unit tests for the factory module to ensure task providers are instantiated correctly.
"""

import pytest

from docprompt.tasks.factory import (
    AmazonTaskProviderFactory,
    AnthropicTaskProviderFactory,
    GCPTaskProviderFactory,
)


class TestAnthropicTaskProviderFactory:
    @pytest.fixture
    def api_key(self):
        return "test-api-key"

    @pytest.fixture
    def environ_key(self, monkeypatch, api_key):
        monkeypatch.setenv("ANTHROPIC_API_KEY", api_key)

        yield api_key

        monkeypatch.delenv("ANTHROPIC_API_KEY")

    def test_validate_provider_w_explicit_key(self, api_key):
        api_key = api_key
        factory = AnthropicTaskProviderFactory(api_key=api_key)
        assert factory._credentials.kwargs == {"api_key": api_key}

    def test_validate_provider_w_environ_key(self, environ_key):
        factory = AnthropicTaskProviderFactory()
        assert factory._credentials.kwargs == {"api_key": environ_key}

    def test_get_page_classification_provider(self, environ_key):
        factory = AnthropicTaskProviderFactory()
        provider = factory.get_page_classification_provider()

        assert provider._default_invoke_kwargs == {"api_key": environ_key}
        assert provider.name == "anthropic"

    def test_get_page_table_extraction_provider(self, environ_key):
        factory = AnthropicTaskProviderFactory()
        provider = factory.get_page_table_extraction_provider()

        assert provider._default_invoke_kwargs == {"api_key": environ_key}
        assert provider.name == "anthropic"

    def test_get_page_markerization_provider(self, environ_key):
        factory = AnthropicTaskProviderFactory()
        provider = factory.get_page_markerization_provider()

        assert provider._default_invoke_kwargs == {"api_key": environ_key}
        assert provider.name == "anthropic"


class TestGoogleTaskProviderFactory:
    @pytest.fixture
    def sa_file(self):
        return "/path/to/file"

    @pytest.fixture
    def environ_file(self, monkeypatch, sa_file):
        monkeypatch.setenv("GCP_SERVICE_ACCOUNT_FILE", sa_file)

        yield sa_file

        monkeypatch.delenv("GCP_SERVICE_ACCOUNT_FILE")

    def test_validate_provider_w_explicit_sa_file(self, sa_file):
        factory = GCPTaskProviderFactory(service_account_file=sa_file)
        assert factory._credentials.kwargs == {"service_account_file": sa_file}

    def test_validate_provider_w_environ_sa_file(self, environ_file):
        factory = GCPTaskProviderFactory()
        assert factory._credentials.kwargs == {"service_account_file": environ_file}

    def test_get_page_ocr_provider(self, environ_file):
        factory = GCPTaskProviderFactory()

        project_id = "project-id"
        processor_id = "processor-id"

        provider = factory.get_page_ocr_provider(project_id, processor_id)

        assert provider._default_invoke_kwargs == {"service_account_file": environ_file}
        assert provider.name == "gcp_documentai"


class TestAmazonTaskProviderFactory:
    @pytest.fixture
    def aws_creds(self):
        return {
            "aws_access_key_id": "test_access_key",
            "aws_secret_access_key": "test_secret_key",
            "aws_region": "us-west-2",
        }

    @pytest.fixture
    def environ_creds(self, monkeypatch, aws_creds):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", aws_creds["aws_access_key_id"])
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", aws_creds["aws_secret_access_key"])
        monkeypatch.setenv("AWS_DEFAULT_REGION", aws_creds["aws_region"])

        yield aws_creds

        monkeypatch.delenv("AWS_ACCESS_KEY_ID")
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY")
        monkeypatch.delenv("AWS_DEFAULT_REGION")

    def test_validate_provider_w_explicict_creds(self, aws_creds):
        factory = AmazonTaskProviderFactory(**aws_creds)
        assert factory._credentials.kwargs == aws_creds

    def test_validate_provider_w_environ_creds(self, environ_creds):
        factory = AmazonTaskProviderFactory()
        assert factory._credentials.kwargs == environ_creds

    def test_get_page_ocr_provider(self, environ_creds):
        factory = AmazonTaskProviderFactory()
        provider = factory.get_page_ocr_provider()

        assert provider._default_invoke_kwargs == environ_creds
        assert provider.name == "aws_textract"
