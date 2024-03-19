from typing import Optional
from pydantic import Field
from docprompt.schema.layout import TextBlock
from docprompt.tasks.base import BasePageResult


class OcrPageResult(BasePageResult):
    page_text: str = Field(description="The text for the entire page in reading order")

    word_level_blocks: list[TextBlock] = Field(
        default_factory=list,
        description="The provider-sourced words for the page",
        repr=False,
    )
    line_level_blocks: list[TextBlock] = Field(
        default_factory=list,
        description="The provider-sourced lines for the page",
        repr=False,
    )
    block_level_blocks: list[TextBlock] = Field(
        default_factory=list,
        description="The provider-sourced blocks for the page",
        repr=False,
    )

    raster_image: Optional[bytes] = Field(
        default=None,
        description="The rasterized image of the page used in OCR",
        repr=False,
    )

    @property
    def words(self):
        return self.word_level_blocks

    @property
    def lines(self):
        return self.line_level_blocks

    @property
    def blocks(self):
        return self.block_level_blocks
