from pydantic import BaseModel, Field, PositiveInt
from pathlib import Path


FIXTURE_PATH = Path(__file__).parent / "fixtures"


class PdfFixture(BaseModel):
    name: str = Field(description="The name of the fixture")
    page_count: PositiveInt = Field(description="The number of pages in the fixture")
    file_hash: str = Field(description="The expected hash of the fixture")

    def get_full_path(self):
        return FIXTURE_PATH / self.name

    def get_bytes(self):
        return self.get_full_path().read_bytes()
