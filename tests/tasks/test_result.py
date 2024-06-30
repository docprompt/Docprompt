"""
Test the Page Level and Document Level task results operate as expected.
"""

from unittest.mock import MagicMock

from docprompt import DocumentNode
from docprompt.schema.pipeline import BaseMetadata
from docprompt.tasks.result import BaseDocumentResult, BasePageResult, BaseResult


def test_task_key():
    class TestResult(BaseResult):
        task_name = "test"

        def contribute_to_document_node(self, document_node, page_number=None):
            pass

    result = TestResult(provider_name="test")
    assert result.task_key == "test_test"


def test_base_document_result_contribution():
    class TestDocumentResult(BaseDocumentResult):
        task_name = "test"

    result = TestDocumentResult(
        provider_name="test", document_name="test", file_hash="test"
    )

    mock_meta = MagicMock(spec=BaseMetadata)
    mock_meta.task_results = {}
    mock_node = MagicMock(spec=DocumentNode)
    mock_node.metadata = mock_meta

    result.contribute_to_document_node(mock_node)

    assert mock_meta.task_results["test_test"] == result


def test_base_page_result_contribution():
    class TestPageResult(BasePageResult):
        task_name = "test"

    result = TestPageResult(provider_name="test")

    num_pages = 3
    mock_meta = MagicMock(spec=BaseMetadata)
    mock_meta.task_results = {}
    mock_node = MagicMock(spec=DocumentNode)
    mock_node.page_nodes = [MagicMock() for _ in range(num_pages)]

    # Test contributing to a specific page
    mock_node.page_nodes[0].metadata = mock_meta

    mock_node.__len__.return_value = num_pages

    result.contribute_to_document_node(mock_node, page_number=1)

    assert mock_node.page_nodes[0].metadata.task_results["test_test"] == result
