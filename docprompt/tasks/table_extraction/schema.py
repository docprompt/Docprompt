from typing import List, Optional

from pydantic import BaseModel, Field

from docprompt.schema.layout import NormBBox
from docprompt.tasks.base import BasePageResult


class TableHeader(BaseModel):
    text: str
    bbox: Optional[NormBBox] = None


class TableCell(BaseModel):
    text: str
    bbox: Optional[NormBBox] = None


class TableRow(BaseModel):
    cells: List[TableCell] = Field(default_factory=list)
    bbox: Optional[NormBBox] = None


class ExtractedTable(BaseModel):
    title: Optional[str] = None
    bbox: Optional[NormBBox] = None

    headers: List[TableHeader] = Field(default_factory=list)
    rows: List[TableRow] = Field(default_factory=list)

    def to_markdown_string(self) -> str:
        markdown = ""

        # Add title if present
        if self.title:
            markdown += f"# {self.title}\n\n"

        # Create header row
        header_row = "|" + "|".join(header.text for header in self.headers) + "|\n"
        markdown += header_row

        # Create separator row
        separator_row = "|" + "|".join("---" for _ in self.headers) + "|\n"
        markdown += separator_row

        # Create data rows
        for row in self.rows:
            data_row = "|" + "|".join(cell.text for cell in row.cells) + "|\n"
            markdown += data_row

        return markdown.strip()


class TableExtractionPageResult(BasePageResult):
    tables: List[ExtractedTable] = Field(default_factory=list)

    task_name = "table_extraction"
