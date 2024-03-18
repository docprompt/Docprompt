from docprompt.tasks.base import BasePageResult
from pydantic import BaseModel, Field
from typing import Optional
from docprompt.schema.layout import NormBBox


class TableHeader(BaseModel):
    text: str
    bbox: Optional[NormBBox] = None


class TableCell(BaseModel):
    text: str
    bbox: Optional[NormBBox] = None


class TableRow(BaseModel):
    cells: list[TableCell] = Field(default_factory=list)
    bbox: Optional[NormBBox] = None


class ExtractedTable(BaseModel):
    title: Optional[str] = None
    bbox: Optional[NormBBox] = None

    headers: list[TableHeader] = Field(default_factory=list)
    rows: list[TableRow] = Field(default_factory=list)


class TableExtractionPageResult(BasePageResult):
    tables: list[ExtractedTable] = Field(default_factory=list)
