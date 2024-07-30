from unittest.mock import AsyncMock, patch

import pytest

from docprompt.tasks.classification.anthropic import (
    AnthropicClassificationProvider,
    AnthropicPageClassificationOutputParser,
    _prepare_messages,
)
from docprompt.tasks.classification.base import (
    ClassificationConfig,
    ClassificationOutput,
    ClassificationTypes,
    ConfidenceLevel,
)


class TestAnthropicPageClassificationOutputParser:
    @pytest.fixture
    def parser(self):
        return AnthropicPageClassificationOutputParser(
            name="anthropic",
            type=ClassificationTypes.SINGLE_LABEL,
            labels=["A", "B", "C"],
            confidence=True,
        )

    def test_parse_single_label(self, parser):
        text = "Reasoning: This is a test.\nAnswer: B\nConfidence: high"
        result = parser.parse(text)

        assert isinstance(result, ClassificationOutput)
        assert result.type == ClassificationTypes.SINGLE_LABEL
        assert result.labels == "B"
        assert result.score == ConfidenceLevel.HIGH
        assert result.provider_name == "anthropic"

    def test_parse_multi_label(self):
        parser = AnthropicPageClassificationOutputParser(
            name="anthropic",
            type=ClassificationTypes.MULTI_LABEL,
            labels=["X", "Y", "Z"],
            confidence=True,
        )
        text = (
            "Reasoning: This is a multi-label test.\nAnswer: X, Z\nConfidence: medium"
        )
        result = parser.parse(text)

        assert isinstance(result, ClassificationOutput)
        assert result.type == ClassificationTypes.MULTI_LABEL
        assert result.labels == ["X", "Z"]
        assert result.score == ConfidenceLevel.MEDIUM
        assert result.provider_name == "anthropic"

    def test_parse_binary(self):
        parser = AnthropicPageClassificationOutputParser(
            name="anthropic",
            type=ClassificationTypes.BINARY,
            labels=["YES", "NO"],
            confidence=False,
        )
        text = "Reasoning: This is a binary test.\nAnswer: YES"
        result = parser.parse(text)

        assert isinstance(result, ClassificationOutput)
        assert result.type == ClassificationTypes.BINARY
        assert result.labels == "YES"
        assert result.score is None
        assert result.provider_name == "anthropic"

    def test_parse_invalid_answer(self, parser):
        text = "Reasoning: This is an invalid test.\nAnswer: D\nConfidence: low"
        with pytest.raises(ValueError, match="Invalid label: D"):
            parser.parse(text)


class TestAnthropicClassificationProvider:
    @pytest.fixture
    def provider(self):
        return AnthropicClassificationProvider()

    @pytest.fixture
    def mock_config(self):
        return ClassificationConfig(
            type=ClassificationTypes.SINGLE_LABEL,
            labels=["A", "B", "C"],
            confidence=True,
        )

    @pytest.mark.asyncio()
    async def test_ainvoke(self, provider, mock_config):
        mock_input = [b"image1", b"image2"]
        mock_completions = [
            "Reasoning: Test 1\nAnswer: A\nConfidence: high",
            "Reasoning: Test 2\nAnswer: B\nConfidence: medium",
        ]

        with patch(
            "docprompt.tasks.classification.anthropic._prepare_messages"
        ) as mock_prepare:
            mock_prepare.return_value = "mock_messages"

            with patch(
                "docprompt.utils.inference.run_batch_inference_anthropic",
                new_callable=AsyncMock,
            ) as mock_inference:
                mock_inference.return_value = mock_completions

                test_kwargs = {
                    "test": "test"
                }  # Test that kwargs are passed through to inference
                results = await provider._ainvoke(
                    mock_input, mock_config, **test_kwargs
                )

        assert len(results) == 2
        assert all(isinstance(result, ClassificationOutput) for result in results)
        assert results[0].labels == "A"
        assert results[0].score == ConfidenceLevel.HIGH
        assert results[1].labels == "B"
        assert results[1].score == ConfidenceLevel.MEDIUM

        mock_prepare.assert_called_once_with(mock_input, mock_config)
        mock_inference.assert_called_once_with(
            "claude-3-haiku-20240307", "mock_messages", **test_kwargs
        )

    @pytest.mark.asyncio()
    async def test_ainvoke_with_error(self, provider, mock_config):
        mock_input = [b"image1"]
        mock_completions = ["Reasoning: Error test\nAnswer: Invalid\nConfidence: low"]

        with patch(
            "docprompt.tasks.classification.anthropic._prepare_messages"
        ) as mock_prepare:
            mock_prepare.return_value = "mock_messages"

            with patch(
                "docprompt.utils.inference.run_batch_inference_anthropic",
                new_callable=AsyncMock,
            ) as mock_inference:
                mock_inference.return_value = mock_completions

                with pytest.raises(ValueError, match="Invalid label: Invalid"):
                    await provider._ainvoke(mock_input, mock_config)

    def test_prepare_messages(self, mock_config):
        imgs = [b"image1", b"image2"]
        config = mock_config

        result = _prepare_messages(imgs, config)

        assert len(result) == 2
        for msg_group in result:
            assert len(msg_group) == 1
            msg = msg_group[0]
            assert msg.role == "user"
            assert len(msg.content) == 2

            types = set([content.type for content in msg.content])
            assert types == set(["image_url", "text"])
