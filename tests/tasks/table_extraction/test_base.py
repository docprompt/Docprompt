from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from docprompt import DocumentNode
from docprompt.schema.layout import NormBBox
from docprompt.schema.pipeline.node.page import PageNode
from docprompt.tasks.table_extraction.base import BaseTableExtractionProvider
from docprompt.tasks.table_extraction.schema import (
    ExtractedTable,
    TableCell,
    TableExtractionPageResult,
    TableHeader,
    TableRow,
)


class TestExtractedTable:
    def test_to_markdown_string_with_title(self):
        table = ExtractedTable(
            title="Sample Table",
            headers=[TableHeader(text="Header 1"), TableHeader(text="Header 2")],
            rows=[
                TableRow(
                    cells=[
                        TableCell(text="Row 1, Col 1"),
                        TableCell(text="Row 1, Col 2"),
                    ]
                ),
                TableRow(
                    cells=[
                        TableCell(text="Row 2, Col 1"),
                        TableCell(text="Row 2, Col 2"),
                    ]
                ),
            ],
        )
        expected_markdown = (
            "# Sample Table\n\n"
            "|Header 1|Header 2|\n"
            "|---|---|\n"
            "|Row 1, Col 1|Row 1, Col 2|\n"
            "|Row 2, Col 1|Row 2, Col 2|"
        )

        assert table.to_markdown_string() == expected_markdown

    def test_to_markdown_string_without_title(self):
        table = ExtractedTable(
            headers=[TableHeader(text="Header 1"), TableHeader(text="Header 2")],
            rows=[
                TableRow(
                    cells=[
                        TableCell(text="Row 1, Col 1"),
                        TableCell(text="Row 1, Col 2"),
                    ]
                ),
            ],
        )
        expected_markdown = (
            "|Header 1|Header 2|\n" "|---|---|\n" "|Row 1, Col 1|Row 1, Col 2|"
        )
        assert table.to_markdown_string() == expected_markdown

    def test_extracted_table_with_bbox(self):
        bbox = NormBBox(x0=0.1, top=0.1, x1=0.9, bottom=0.9)
        table = ExtractedTable(bbox=bbox)
        assert table.bbox == bbox


class TestTableExtractionPageResult:
    def test_initialization(self):
        table1 = ExtractedTable(title="Table 1")
        table2 = ExtractedTable(title="Table 2")
        result = TableExtractionPageResult(
            tables=[table1, table2], provider_name="test"
        )
        assert len(result.tables) == 2
        assert result.tables[0].title == "Table 1"
        assert result.tables[1].title == "Table 2"

    def test_task_name(self):
        result = TableExtractionPageResult(provider_name="test")
        assert result.task_name == "table_extraction"

    def test_empty_tables(self):
        result = TableExtractionPageResult(provider_name="test")
        assert result.tables == []

    def test_invalid_table_type(self):
        with pytest.raises(ValidationError):
            TableExtractionPageResult(tables=["Not a table"], provider_name="test")


class TestTableComponents:
    def test_table_header(self):
        header = TableHeader(text="Header 1")
        assert header.text == "Header 1"
        assert header.bbox is None

        bbox = NormBBox(x0=0.1, top=0.1, x1=0.2, bottom=0.2)
        header_with_bbox = TableHeader(text="Header 2", bbox=bbox)
        assert header_with_bbox.text == "Header 2"
        assert header_with_bbox.bbox == bbox

    def test_table_cell(self):
        cell = TableCell(text="Cell 1")
        assert cell.text == "Cell 1"
        assert cell.bbox is None

        bbox = NormBBox(x0=0.1, top=0.1, x1=0.2, bottom=0.2)
        cell_with_bbox = TableCell(text="Cell 2", bbox=bbox)
        assert cell_with_bbox.text == "Cell 2"
        assert cell_with_bbox.bbox == bbox

    def test_table_row(self):
        cell1 = TableCell(text="Cell 1")
        cell2 = TableCell(text="Cell 2")
        row = TableRow(cells=[cell1, cell2])
        assert len(row.cells) == 2
        assert row.cells[0].text == "Cell 1"
        assert row.cells[1].text == "Cell 2"
        assert row.bbox is None

        bbox = NormBBox(x0=0.1, top=0.1, x1=0.9, bottom=0.2)
        row_with_bbox = TableRow(cells=[cell1, cell2], bbox=bbox)
        assert row_with_bbox.bbox == bbox

    def test_normbbox_validation(self):
        with pytest.raises(ValidationError):
            NormBBox(x0=1.1, top=0.1, x1=0.9, bottom=0.9)  # x0 > 1

        with pytest.raises(ValidationError):
            NormBBox(x0=0.1, top=-0.1, x1=0.9, bottom=0.9)  # top < 0

        with pytest.raises(ValidationError):
            NormBBox(x0=0.1, top=0.1, x1=1.1, bottom=0.9)  # x1 > 1

        with pytest.raises(ValidationError):
            NormBBox(x0=0.1, top=0.1, x1=0.9, bottom=1.1)  # bottom > 1

        # Valid NormBBox should not raise an exception
        NormBBox(x0=0.1, top=0.1, x1=0.9, bottom=0.9)


class TestBaseTableExtractionProvider:
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
            (2, 4, [2, 3, 4], {2: "TABLE-0", 3: "TABLE-1", 4: "TABLE-2"}),
            (3, None, [3, 4, 5], {3: "TABLE-0", 4: "TABLE-1", 5: "TABLE-2"}),
            (None, 2, [1, 2], {1: "TABLE-0", 2: "TABLE-1"}),
            (
                None,
                None,
                [1, 2, 3, 4, 5],
                {
                    1: "TABLE-0",
                    2: "TABLE-1",
                    3: "TABLE-2",
                    4: "TABLE-3",
                    5: "TABLE-4",
                },
            ),
        ],
    )
    def test_process_document_node_with_start_stop(
        self, mock_document_node, start, stop, expected_keys, expected_results
    ):
        class TestProvider(BaseTableExtractionProvider):
            name = "test"

            def _invoke(self, input, config, **kwargs):
                return [
                    TableExtractionPageResult(
                        tables=[ExtractedTable(title=f"TABLE-{i}")],
                        provider_name="test",
                    )
                    for i in range(len(input))
                ]

        provider = TestProvider()
        result = provider.process_document_node(
            mock_document_node, start=start, stop=stop
        )

        assert list(result.keys()) == expected_keys
        assert all(isinstance(v, TableExtractionPageResult) for v in result.values())
        assert {k: v.tables[0].title for k, v in result.items()} == expected_results

        with patch.object(provider, "_invoke") as mock_invoke:
            provider.process_document_node(mock_document_node, start=start, stop=stop)
            mock_invoke.assert_called_once()
            expected_invoke_length = len(expected_keys)
            assert len(mock_invoke.call_args[0][0]) == expected_invoke_length

    def test_process_document_node_rasterization(self, mock_document_node):
        class TestProvider(BaseTableExtractionProvider):
            name = "test"

            def _invoke(self, input, config, **kwargs):
                return [
                    TableExtractionPageResult(
                        tables=[ExtractedTable(title=f"TABLE-{i}")],
                        provider_name="test",
                    )
                    for i in range(len(input))
                ]

        provider = TestProvider()
        provider.process_document_node(mock_document_node)

        for page_node in mock_document_node.page_nodes:
            page_node.rasterizer.rasterize.assert_called_once_with("default")
