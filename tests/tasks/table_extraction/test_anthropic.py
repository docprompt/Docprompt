"""
Test the Anthropic implementation of the table extraction task.
"""

from unittest.mock import patch

import pytest
from bs4 import BeautifulSoup

from docprompt.tasks.message import OpenAIComplexContent, OpenAIImageURL, OpenAIMessage
from docprompt.tasks.table_extraction.anthropic import (
    AnthropicTableExtractionProvider,
    _headers_from_tree,
    _prepare_messages,
    _rows_from_tree,
    _title_from_tree,
    parse_response,
)
from docprompt.tasks.table_extraction.schema import (
    TableCell,
    TableExtractionPageResult,
    TableHeader,
    TableRow,
)


@pytest.fixture
def mock_image_bytes():
    return b"mock_image_bytes"


class TestAnthropicTableExtractionProvider:
    @pytest.fixture
    def provider(self):
        return AnthropicTableExtractionProvider()

    def test_provider_name(self, provider):
        assert provider.name == "anthropic"

    @pytest.mark.asyncio
    async def test_ainvoke(self, provider, mock_image_bytes):
        mock_completions = [
            "<table><title>Test Table</title><headers><header>Col1</header></headers><rows><row><column>Data1</column></row></table>",
            "<table><headers><header>Col2</header></headers><rows><row><column>Data2</column></row></table>",
        ]

        with patch(
            "docprompt.tasks.table_extraction.anthropic._prepare_messages"
        ) as mock_prepare:
            with patch(
                "docprompt.utils.inference.run_batch_inference_anthropic"
            ) as mock_inference:
                mock_prepare.return_value = "mock_messages"
                mock_inference.return_value = mock_completions

                result = await provider._ainvoke([mock_image_bytes, mock_image_bytes])

                assert len(result) == 2
                assert all(isinstance(r, TableExtractionPageResult) for r in result)
                assert result[0].tables[0].title == "Test Table"
                assert result[1].tables[0].title is None
                assert all(r.provider_name == "anthropic" for r in result)

                mock_prepare.assert_called_once_with(
                    [mock_image_bytes, mock_image_bytes]
                )
                mock_inference.assert_called_once_with(
                    "claude-3-haiku-20240307", "mock_messages"
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
    assert messages[0][0].content[0].image_url.url == mock_image_bytes.decode()
    assert isinstance(messages[0][0].content[1], OpenAIComplexContent)
    assert messages[0][0].content[1].type == "text"
    assert (
        "Identify and extract all tables from the document"
        in messages[0][0].content[1].text
    )


def test_parse_response():
    response = """
    <table>
    <title>Test Table</title>
    <headers>
    <header>Col1</header>
    <header>Col2</header>
    </headers>
    <rows>
    <row>
    <column>Data1</column>
    <column>Data2</column>
    </row>
    </rows>
    </table>
    """
    result = parse_response(response)

    assert isinstance(result, TableExtractionPageResult)
    assert len(result.tables) == 1
    assert result.tables[0].title == "Test Table"
    assert len(result.tables[0].headers) == 2
    assert result.tables[0].headers[0].text == "Col1"
    assert len(result.tables[0].rows) == 1
    assert result.tables[0].rows[0].cells[0].text == "Data1"


def test_title_from_tree():
    soup = BeautifulSoup("<table><title>Test Title</title></table>")
    assert _title_from_tree(soup.table) == "Test Title"

    soup = BeautifulSoup("<table></table>")
    assert _title_from_tree(soup.table) is None


def test_headers_from_tree():
    soup = BeautifulSoup(
        "<table><headers><header>Col1</header><header>Col2</header></headers></table>",
    )
    headers = _headers_from_tree(soup.table)
    assert len(headers) == 2
    assert all(isinstance(h, TableHeader) for h in headers)
    assert headers[0].text == "Col1"

    soup = BeautifulSoup("<table></table>")
    assert _headers_from_tree(soup.table) == []


def test_rows_from_tree():
    soup = BeautifulSoup(
        "<table><rows><row><column>Data1</column><column>Data2</column></row></rows></table>",
    )
    rows = _rows_from_tree(soup.table)
    assert len(rows) == 1
    assert isinstance(rows[0], TableRow)
    assert len(rows[0].cells) == 2
    assert all(isinstance(c, TableCell) for c in rows[0].cells)
    assert rows[0].cells[0].text == "Data1"

    soup = BeautifulSoup("<table></table>")
    assert _rows_from_tree(soup.table) == []


@pytest.mark.parametrize(
    "input_str,sub_str,expected",
    [
        ("abc<table>def</table>ghi<table>jkl</table>", "<table>", [3, 24]),
        ("notables", "<table>", []),
        ("<table><table><table>", "<table>", [0, 7, 14]),
    ],
)
def test_find_start_indices(input_str, sub_str, expected):
    from docprompt.tasks.table_extraction.anthropic import _find_start_indices

    assert _find_start_indices(input_str, sub_str) == expected


@pytest.mark.parametrize(
    "input_str,sub_str,expected",
    [
        ("abc</table>def</table>ghi", "</table>", [11, 22]),
        ("notables", "</table>", []),
        ("</table></table></table>", "</table>", [8, 16, 24]),
    ],
)
def test_find_end_indices(input_str, sub_str, expected):
    from docprompt.tasks.table_extraction.anthropic import _find_end_indices

    assert _find_end_indices(input_str, sub_str) == expected
