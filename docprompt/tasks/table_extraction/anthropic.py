import re
from typing import Iterable, List, Optional, Union

from bs4 import BeautifulSoup, Tag
from pydantic import Field

from docprompt.tasks.message import OpenAIComplexContent, OpenAIImageURL, OpenAIMessage
from docprompt.utils import inference

from .base import BaseTableExtractionProvider
from .schema import (
    ExtractedTable,
    TableCell,
    TableExtractionPageResult,
    TableHeader,
    TableRow,
)

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


def _title_from_tree(tree: Tag) -> Union[str, None]:
    title = tree.find("title")
    if title is not None:
        return title.text
    return None


def _headers_from_tree(tree: Tag) -> List[TableHeader]:
    headers = tree.find("headers")
    if headers is not None:
        return [
            TableHeader(text=header.text or "") for header in headers.find_all("header")
        ]
    return []


def _rows_from_tree(tree: Tag) -> List[TableRow]:
    rows = tree.find("rows")
    if rows is not None:
        return [
            TableRow(
                cells=[
                    TableCell(text=cell.text or "") for cell in row.find_all("column")
                ]
            )
            for row in rows.find_all("row")
        ]
    return []


def _find_start_indices(s: str, sub: str) -> List[int]:
    return [m.start() for m in re.finditer(sub, s)]


def _find_end_indices(s: str, sub: str) -> List[int]:
    return [m.end() for m in re.finditer(sub, s)]


def parse_response(response: str, **kwargs) -> TableExtractionPageResult:
    table_start_indices = _find_start_indices(response, "<table>")
    table_end_indices = _find_end_indices(response, "</table>")

    tables: List[ExtractedTable] = []
    provider_name = kwargs.pop("provider_name", "anthropic")

    for table_start, table_end in zip(table_start_indices, table_end_indices):
        table_str = response[table_start:table_end]

        soup = BeautifulSoup(table_str, "html.parser")

        table_element = soup.find("table")

        title = _title_from_tree(table_element)
        headers = _headers_from_tree(table_element)
        rows = _rows_from_tree(table_element)

        tables.append(ExtractedTable(title=title, headers=headers, rows=rows))

    result = TableExtractionPageResult(tables=tables, provider_name=provider_name)
    return result


def _prepare_messages(
    document_images: Iterable[bytes],
    start: Optional[int] = None,
    stop: Optional[int] = None,
):
    messages = []

    for image_bytes in document_images:
        messages.append(
            [
                OpenAIMessage(
                    role="user",
                    content=[
                        OpenAIComplexContent(
                            type="image_url",
                            image_url=OpenAIImageURL(url=image_bytes),
                        ),
                        OpenAIComplexContent(type="text", text=SYSTEM_PROMPT),
                    ],
                ),
            ]
        )

    return messages


class AnthropicTableExtractionProvider(BaseTableExtractionProvider):
    name = "anthropic"

    anthropic_model_name: str = Field("claude-3-haiku-20240307")

    async def _ainvoke(
        self, input: Iterable[bytes], config: Optional[None] = None, **kwargs
    ) -> List[TableExtractionPageResult]:
        messages = _prepare_messages(input)

        model_name = kwargs.pop("model_name", self.anthropic_model_name)
        completions = await inference.run_batch_inference_anthropic(
            model_name, messages, **kwargs
        )

        return [parse_response(x, provider_name=self.name) for x in completions]
