from pydantic import BaseModel, Field, PositiveInt, TypeAdapter
from pathlib import Path
from typing import Dict, Optional

from docprompt.tasks.ocr.result import OcrPageResult


FIXTURE_PATH = Path(__file__).parent / "fixtures"


OCR_ADAPTER = TypeAdapter(Dict[int, OcrPageResult])


class PdfFixture(BaseModel):
    name: str = Field(description="The name of the fixture")
    page_count: PositiveInt = Field(description="The number of pages in the fixture")
    file_hash: str = Field(description="The expected hash of the fixture")
    ocr_name: Optional[str] = Field(
        description="The path to the OCR results for the fixture", default=None
    )

    def get_full_path(self):
        return FIXTURE_PATH / self.name

    def get_bytes(self):
        return self.get_full_path().read_bytes()

    def get_ocr_results(self):
        if not self.ocr_name:
            return None

        ocr_path = FIXTURE_PATH / self.ocr_name

        return OCR_ADAPTER.validate_json(ocr_path.read_text())
