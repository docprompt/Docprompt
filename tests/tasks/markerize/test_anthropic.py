"""
Test the Anthropic implementation of the markerize task.
"""

from unittest.mock import patch

import pytest

from docprompt.tasks.markerize.anthropic import (
    AnthropicMarkerizeProvider,
    _parse_result,
    _prepare_messages,
)
from docprompt.tasks.markerize.base import MarkerizeResult
from docprompt.tasks.message import OpenAIComplexContent, OpenAIImageURL, OpenAIMessage


@pytest.fixture
def mock_image_bytes():
    return b"mock_image_bytes"


class TestAnthropicMarkerizeProvider:
    @pytest.fixture
    def provider(self):
        return AnthropicMarkerizeProvider()

    def test_provider_name(self, provider):
        assert provider.name == "anthropic"

    @pytest.mark.asyncio
    async def test_ainvoke(self, provider, mock_image_bytes):
        mock_completions = ["<md># Test Markdown</md>", "<md>## Another Test</md>"]

        with patch(
            "docprompt.tasks.markerize.anthropic._prepare_messages"
        ) as mock_prepare:
            with patch(
                "docprompt.utils.inference.run_batch_inference_anthropic"
            ) as mock_inference:
                mock_prepare.return_value = "mock_messages"
                mock_inference.return_value = mock_completions

                test_kwargs = {
                    "test": "test"
                }  # Test that kwargs are passed through to inference
                result = await provider._ainvoke(
                    [mock_image_bytes, mock_image_bytes], **test_kwargs
                )

                assert len(result) == 2
                assert all(isinstance(r, MarkerizeResult) for r in result)
                assert result[0].raw_markdown == "# Test Markdown"
                assert result[1].raw_markdown == "## Another Test"
                assert all(r.provider_name == "anthropic" for r in result)

                mock_prepare.assert_called_once_with(
                    [mock_image_bytes, mock_image_bytes]
                )
                mock_inference.assert_called_once_with(
                    "claude-3-haiku-20240307", "mock_messages", **test_kwargs
                )


def test_prepare_messages(mock_image_bytes):
    messages = _prepare_messages([mock_image_bytes])

    assert len(messages) == 1
    assert len(messages[0]) == 1
    assert isinstance(messages[0][0], OpenAIMessage)
    assert messages[0][0].role == "user"
    assert len(messages[0][0].content) == 2
    assert isinstance(messages[0][0].content[0], OpenAIComplexContent)
    assert messages[0][0].content[0].type == "image_url"
    assert isinstance(messages[0][0].content[0].image_url, OpenAIImageURL)
    assert messages[0][0].content[0].image_url.url == mock_image_bytes.decode("utf-8")
    assert isinstance(messages[0][0].content[1], OpenAIComplexContent)
    assert messages[0][0].content[1].type == "text"
    assert "Convert the image into markdown" in messages[0][0].content[1].text


@pytest.mark.parametrize(
    "raw_markdown,expected",
    [
        ("<md># Test Markdown</md>", "# Test Markdown"),
        ("<md>## Another Test</md>", "## Another Test"),
        ("Invalid markdown", ""),
        ("<md>  Trimmed  </md>", "Trimmed"),
    ],
)
def test_parse_result(raw_markdown, expected):
    assert _parse_result(raw_markdown) == expected
