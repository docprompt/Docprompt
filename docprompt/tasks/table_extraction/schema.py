from docprompt.tasks.base import BasePageResult
from pydantic import BaseModel, Field
from typing import List, Optional
from docprompt.schema.layout import NormBBox


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


class TableExtractionPageResult(BasePageResult):
    tables: List[ExtractedTable] = Field(default_factory=list)
