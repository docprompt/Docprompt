from typing import List, Literal, Optional

from pydantic import BaseModel, Field, PositiveInt, computed_field

from docprompt.schema.layout import TextBlock


class PageTextLocation(BaseModel):
    """
    Specifies the location of a piece of text in a page
    """

    source_blocks: List[TextBlock] = Field(
        description="The source text blocks", repr=False
    )
    text: str  # Sometimes the source text is less than the textblock's text.
    score: float
    granularity: Literal["word", "line", "block"] = "block"

    merged_source_block: Optional[TextBlock] = Field(default=None)


class ProvenanceSource(BaseModel):
    """
    Bundled with some data, specifies exactly where a piece of verbatim text came from
    in a document.
    """

    document_name: str
    page_number: PositiveInt
    text_location: Optional[PageTextLocation] = None

    @computed_field  # type: ignore
    @property
    def source_block(self) -> Optional[TextBlock]:
        if self.text_location:
            if self.text_location.merged_source_block:
                return self.text_location.merged_source_block
            if self.text_location.source_blocks:
                return self.text_location.source_blocks[0]

            return None

    @property
    def text(self) -> str:
        if self.text_location:
            return "\n".join([block.text for block in self.text_location.source_blocks])

        return ""
