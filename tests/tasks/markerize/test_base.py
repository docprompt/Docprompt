"""
Test the basic functionality of the atomic components for the markerize task.
"""

from unittest.mock import MagicMock, patch

import pytest

from docprompt.schema.pipeline.node.document import DocumentNode
from docprompt.schema.pipeline.node.page import PageNode
from docprompt.tasks.markerize.base import BaseMarkerizeProvider, MarkerizeResult


class TestBaseMarkerizeProvider:
    """
    Test the base markerize provider to ensure that the task is properly handled.
    """

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
        class TestProvider(BaseMarkerizeProvider):
            name = "test"

            def _invoke(self, input, config, **kwargs):
                return [
                    MarkerizeResult(raw_markdown=f"RESULT-{i}", provider_name="test")
                    for i in range(len(input))
                ]

        provider = TestProvider()
        result = provider.process_document_node(
            mock_document_node, start=start, stop=stop
        )

        assert list(result.keys()) == expected_keys
        assert all(isinstance(v, MarkerizeResult) for v in result.values())
        assert {k: v.raw_markdown for k, v in result.items()} == expected_results

        with patch.object(provider, "_invoke") as mock_invoke:
            provider.process_document_node(mock_document_node, start=start, stop=stop)
            mock_invoke.assert_called_once()
            expected_invoke_length = len(expected_keys)
            assert len(mock_invoke.call_args[0][0]) == expected_invoke_length

    def test_process_document_node_rasterization(self, mock_document_node):
        class TestProvider(BaseMarkerizeProvider):
            name = "test"

            def _invoke(self, input, config, **kwargs):
                return [
                    MarkerizeResult(raw_markdown=f"RESULT-{i}", provider_name="test")
                    for i in range(len(input))
                ]

        provider = TestProvider()
        provider.process_document_node(mock_document_node)

        for page_node in mock_document_node.page_nodes:
            page_node.rasterizer.rasterize.assert_called_once_with("default")
