"""
Test the basic funcitonality of the atomic components for the classification task.
"""

import re
from enum import Enum
from unittest.mock import MagicMock, patch

import pytest

from docprompt.schema.pipeline.node.document import DocumentNode
from docprompt.schema.pipeline.node.page import PageNode
from docprompt.tasks.classification.base import (
    BaseClassificationProvider,
    BasePageClassificationOutputParser,
    ClassificationConfig,
    ClassificationOutput,
    ClassificationTypes,
    ConfidenceLevel,
)


class TestClassificationConfig:
    """
    Test the classification config class to make sure that the labels are properly handled.
    """

    def test_single_label_classification(self):
        config = ClassificationConfig(
            type=ClassificationTypes.SINGLE_LABEL, labels=["A", "B", "C"]
        )
        assert config.type == ClassificationTypes.SINGLE_LABEL
        assert config.labels == ["A", "B", "C"]

    def test_multi_label_classification(self):
        config = ClassificationConfig(
            type=ClassificationTypes.MULTI_LABEL, labels=["X", "Y", "Z"]
        )
        assert config.type == ClassificationTypes.MULTI_LABEL
        assert config.labels == ["X", "Y", "Z"]

    def test_binary_classification(self):
        config = ClassificationConfig(
            type=ClassificationTypes.BINARY, instructions="Is this a cat?"
        )
        assert config.type == ClassificationTypes.BINARY
        assert config.labels == ["YES", "NO"]
        assert config.instructions == "Is this a cat?"

    def test_binary_classification_without_instructions(self):
        with pytest.raises(ValueError):
            ClassificationConfig(type=ClassificationTypes.BINARY)

    def test_single_label_classification_without_labels(self):
        with pytest.raises(ValueError):
            ClassificationConfig(type=ClassificationTypes.SINGLE_LABEL)

    def test_multi_label_classification_without_labels(self):
        with pytest.raises(ValueError):
            ClassificationConfig(type=ClassificationTypes.MULTI_LABEL)

    def test_validate_descriptions_length_valid(self):
        config = ClassificationConfig(
            type=ClassificationTypes.SINGLE_LABEL,
            labels=["A", "B"],
            descriptions=["Description A", "Description B"],
        )
        assert config.descriptions == ["Description A", "Description B"]

    def test_validate_descriptions_length_invalid(self):
        with pytest.raises(ValueError):
            ClassificationConfig(
                type=ClassificationTypes.SINGLE_LABEL,
                labels=["A", "B"],
                descriptions=["Description A"],
            )

    def test_formatted_labels_without_descriptions(self):
        config = ClassificationConfig(
            type=ClassificationTypes.SINGLE_LABEL, labels=["A", "B", "C"]
        )
        assert list(config.formatted_labels) == ['"A"', '"B"', '"C"']

    def test_formatted_labels_with_descriptions(self):
        config = ClassificationConfig(
            type=ClassificationTypes.SINGLE_LABEL,
            labels=["A", "B"],
            descriptions=["Description A", "Description B"],
        )
        assert list(config.formatted_labels) == [
            '"A": Description A',
            '"B": Description B',
        ]

    def test_formatted_enum_labels_with_descriptions(self):
        class TestEnum(Enum):
            A = "A"
            B = "B"

        config = ClassificationConfig(
            type=ClassificationTypes.SINGLE_LABEL,
            labels=TestEnum,
            descriptions=["Description A", "Description B"],
        )
        assert list(config.formatted_labels) == [
            '"A": Description A',
            '"B": Description B',
        ]

    def test_confidence_default_value(self):
        config = ClassificationConfig(
            type=ClassificationTypes.SINGLE_LABEL, labels=["A", "B"]
        )
        assert config.confidence is False

    def test_confidence_custom_value(self):
        config = ClassificationConfig(
            type=ClassificationTypes.SINGLE_LABEL, labels=["A", "B"], confidence=True
        )
        assert config.confidence is True


class ConcreteClassificationOutputParser(BasePageClassificationOutputParser):
    def parse(self, text: str) -> ClassificationOutput:
        # Simple implementation for testing
        match = re.search(r"Label: (.*)", text)
        label = self.resolve_match(match)

        confidence_match = re.search(r"Confidence: (.*)", text)
        confidence = (
            self.resolve_confidence(confidence_match) if self.confidence else None
        )

        return ClassificationOutput(
            type=self.type,
            labels=label,
            score=confidence,
            task_name="classification",
            provider_name=self.name,
        )


class TestBasePageClassificationOutputParser:
    @pytest.fixture
    def parser(self):
        return ConcreteClassificationOutputParser(
            name="test_parser",
            type=ClassificationTypes.SINGLE_LABEL,
            labels=["A", "B", "C"],
            confidence=True,
        )

    # NOTE: For some reason, getting this VERY strange warning, but only when running the
    # tests with other tests in the `test_anthropic` file. This is the only test that is causing
    # this warning to be raised. We just ignore the warning for now.
    @pytest.mark.filterwarnings(
        "ignore:coroutine 'AsyncMockMixin._execute_mock_call' was never awaited"
    )
    def test_from_task_input(self):
        config = ClassificationConfig(
            type=ClassificationTypes.SINGLE_LABEL,
            labels=["X", "Y", "Z"],
            confidence=True,
        )
        parser = ConcreteClassificationOutputParser.from_task_input(
            config, "test_provider"
        )

        assert parser.name == "test_provider"
        assert parser.type == ClassificationTypes.SINGLE_LABEL
        assert parser.labels == ["X", "Y", "Z"]
        assert parser.confidence is True

    @pytest.mark.filterwarnings(
        "ignore:coroutine 'AsyncMockMixin._execute_mock_call' was never awaited"
    )
    def test_resolve_match_binary(self, parser):
        parser.type = ClassificationTypes.BINARY
        parser.labels = ["YES", "NO"]

        match = re.match(r"Label: (YES)", "Label: YES")
        assert parser.resolve_match(match) == "YES"

        with pytest.raises(ValueError):
            parser.resolve_match(re.match(r"Label: (MAYBE)", "Label: MAYBE"))

    @pytest.mark.filterwarnings(
        "ignore:coroutine 'AsyncMockMixin._execute_mock_call' was never awaited"
    )
    def test_resolve_match_single_label(self, parser):
        pattern = re.compile(r"Label: (B)")
        match = pattern.match("Label: B")
        assert parser.resolve_match(match) == "B"

        with pytest.raises(ValueError):
            invalid_pattern = re.compile(r"Label: (D)")
            parser.resolve_match(invalid_pattern.match("Label: D"))

    @pytest.mark.filterwarnings(
        "ignore:coroutine 'AsyncMockMixin._execute_mock_call' was never awaited"
    )
    def test_resolve_match_multi_label(self, parser):
        parser.type = ClassificationTypes.MULTI_LABEL
        pattern = re.compile(r"Label: (A, B)")
        match = pattern.match("Label: A, B")
        assert parser.resolve_match(match) == ["A", "B"]

        with pytest.raises(ValueError):
            invalid_pattern = re.compile(r"Label: (A, D)")
            parser.resolve_match(invalid_pattern.match("Label: A, D"))

    @pytest.mark.filterwarnings(
        "ignore:coroutine 'AsyncMockMixin._execute_mock_call' was never awaited"
    )
    def test_resolve_match_invalid_type(self, parser):
        parser.type = "INVALID_TYPE"
        match = re.match(r"Label: (A)", "Label: A")
        with pytest.raises(ValueError):
            parser.resolve_match(match)

    @pytest.mark.filterwarnings(
        "ignore:coroutine 'AsyncMockMixin._execute_mock_call' was never awaited"
    )
    def test_resolve_match_no_match(self, parser):
        with pytest.raises(ValueError):
            parser.resolve_match(None)

    @pytest.mark.filterwarnings(
        "ignore:coroutine 'AsyncMockMixin._execute_mock_call' was never awaited"
    )
    def test_resolve_confidence(self, parser):
        match = re.match(r"Confidence: (high)", "Confidence: high")
        assert parser.resolve_confidence(match) == ConfidenceLevel.HIGH

        match = re.match(r"Confidence: (medium)", "Confidence: medium")
        assert parser.resolve_confidence(match) == ConfidenceLevel.MEDIUM

        match = re.match(r"Confidence: (low)", "Confidence: low")
        assert parser.resolve_confidence(match) == ConfidenceLevel.LOW

    def test_resolve_confidence_no_match(self, parser):
        assert parser.resolve_confidence(None) is None

    def test_parse(self, parser):
        text = "Label: B\nConfidence: high"
        output = parser.parse(text)

        assert isinstance(output, ClassificationOutput)
        assert output.type == ClassificationTypes.SINGLE_LABEL
        assert output.labels == "B"
        assert output.score == ConfidenceLevel.HIGH
        assert output.task_name == "classification"

    def test_parse_without_confidence(self, parser):
        parser.confidence = False
        text = "Label: C\nConfidence: low"
        output = parser.parse(text)

        assert isinstance(output, ClassificationOutput)
        assert output.type == ClassificationTypes.SINGLE_LABEL
        assert output.labels == "C"
        assert output.score is None
        assert output.task_name == "classification"


class TestBasePageClassificationProvider:
    """
    Test the base classification provider to ensure that the task is properly handled.
    """

    def test_async_method_is_present(self):
        assert hasattr(BaseClassificationProvider, "aprocess_document_node")
        assert hasattr(BaseClassificationProvider, "process_document_node")

    @pytest.fixture
    def mock_document_node(self):
        mock_node = MagicMock(spec=DocumentNode)
        mock_node.page_nodes = [MagicMock(spec=PageNode) for _ in range(5)]
        for pnode in mock_node.page_nodes:
            pnode.rasterizer.rasterize.return_value = b"image"
        mock_node.__len__.return_value = len(mock_node.page_nodes)
        return mock_node

    @pytest.mark.parametrize(
        "start,stop,expected_keys,expected_results",
        [
            (2, 4, [2, 3, 4], {2: "RESULT-0", 3: "RESULT-1", 4: "RESULT-2"}),
            (3, None, [3, 4, 5], {3: "RESULT-0", 4: "RESULT-1", 5: "RESULT-2"}),
            (None, 2, [1, 2], {1: "RESULT-0", 2: "RESULT-1"}),
            (
                None,
                None,
                [1, 2, 3, 4, 5],
                {
                    1: "RESULT-0",
                    2: "RESULT-1",
                    3: "RESULT-2",
                    4: "RESULT-3",
                    5: "RESULT-4",
                },
            ),
        ],
    )
    def test_process_document_node_with_start_stop(
        self, mock_document_node, start, stop, expected_keys, expected_results
    ):
        class TestProvider(BaseClassificationProvider):
            name = "test"

            def _invoke(self, input, config, **kwargs):
                return [f"RESULT-{i}" for i in range(len(input))]

        config = object
        provider = TestProvider()
        result = provider.process_document_node(
            mock_document_node, start=start, stop=stop, task_config=config
        )

        assert list(result.keys()) == expected_keys
        assert result == expected_results

        with patch.object(provider, "_invoke"):
            provider.process_document_node(
                mock_document_node, start=start, stop=stop, task_config=config
            )
            # mock_invoke.assert_called_once()
            # expected_invoke_length = len(expected_keys)
            # assert len(mock_invoke.call_args[0][0]) == expected_invoke_length

    def test_no_config_raises_error(self):
        class TestProvider(BaseClassificationProvider):
            name = "test"

        provider = TestProvider()
        with pytest.raises(
            AssertionError,
            match="task_config must be provided for classification tasks",
        ):
            provider.process_document_node(MagicMock(spec=DocumentNode))
