from typing import Union

from docprompt.schema.document import PdfDocument
from docprompt.tasks.message import OpenAIMessage, OpenAIComplexContent, OpenAIImageURL
from ..base import BaseTableExtractionProvider
from ..result import (
    ExtractedTable,
    TableRow,
    TableHeader,
    TableCell,
    TableExtractionPageResult,
)
import xml.etree.ElementTree as ET
import re


SYSTEM_PROMPT = """
You are given an image. Identify and extract all tables from the document.

For each table, respond in the following format:

<table>
<title>(value)</title>
<headers>
<header>(value)</header>
...
</headers>
<rows>
<row>
<column>(value)</column>
...
</row>
...
</table>
""".strip()


def _title_from_tree(tree: ET.Element) -> Union[str, None]:
    title = tree.find("title")
    if title is not None:
        return title.text
    return None


def _headers_from_tree(tree: ET.Element) -> list[TableHeader]:
    headers = tree.find("headers")
    if headers is not None:
        return [
            TableHeader(text=header.text or "") for header in headers.findall("header")
        ]
    return []


def _rows_from_tree(tree: ET.Element) -> list[TableRow]:
    rows = tree.find("rows")
    if rows is not None:
        return [
            TableRow(
                cells=[
                    TableCell(text=cell.text or "") for cell in row.findall("column")
                ]
            )
            for row in rows.findall("row")
        ]
    return []


def _find_start_indices(s: str, sub: str) -> list[int]:
    return [m.start() for m in re.finditer(sub, s)]


def _find_end_indices(s: str, sub: str) -> list[int]:
    return [m.end() for m in re.finditer(sub, s)]


def parse_response(response: str) -> list[ExtractedTable]:
    table_start_indices = _find_start_indices(response, "<table>")
    table_end_indices = _find_end_indices(response, "</table>")

    tables: list[ExtractedTable] = []

    for table_start, table_end in zip(table_start_indices, table_end_indices):
        table_str = response[table_start:table_end]

        table_element = ET.fromstring(table_str)

        title = _title_from_tree(table_element)
        headers = _headers_from_tree(table_element)
        rows = _rows_from_tree(table_element)

        tables.append(ExtractedTable(title=title, headers=headers, rows=rows))

    return tables


class ClaudeTableExtractionProvider(BaseTableExtractionProvider):
    def process_document_pages(
        self,
        document: PdfDocument,
        start: int | None = None,
        stop: int | None = None,
        **kwargs,
    ) -> dict[int, TableExtractionPageResult]:
        messages = []
        for page_number in range(start or 1, (stop or len(document)) + 1):
            rastered_page = document.rasterize_page_to_data_uri(page_number)

            messages.append(
                [
                    OpenAIMessage(
                        role="system",
                        content=SYSTEM_PROMPT,
                    ),
                    OpenAIMessage(
                        role="user",
                        content=[
                            OpenAIComplexContent(
                                type="image_url",
                                image_url=OpenAIImageURL(url=rastered_page),
                            )
                        ],
                    ),
                ]
            )

        return {}
