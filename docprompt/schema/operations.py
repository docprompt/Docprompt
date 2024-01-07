from typing import Optional

from pydantic import BaseModel, Field

from .layout import TextBlock


class PageTextExtractionOutput(BaseModel):
    """
    Represents the output of text extraction for a single page
    """

    text: str = Field(description="The provider-sourced text for the page")

    words: list[TextBlock] = Field(default_factory=list, description="The provider-sourced words for the page")
    lines: list[TextBlock] = Field(default_factory=list, description="The provider-sourced lines for the page")
    blocks: list[TextBlock] = Field(default_factory=list, description="The provider-sourced blocks for the page")


class PageResult(BaseModel):
    """
    Represents the result of processing a single page
    (page-wise)
    """

    provider_name: str
    page_number: int

    ocr_result: Optional[PageTextExtractionOutput] = None


class DocumentResult(BaseModel):
    """
    Represents the result of operations that consider the
    document as a whole (document-wise)

    Examples:

    * Multi-page DocVQA
    """

    provider_name: str


class ProviderResult(BaseModel):
    """
    Represents the result of processing a document
    """

    provider_name: str

    page_results: Optional[list[PageResult]] = None
    document_result: Optional[DocumentResult] = None
