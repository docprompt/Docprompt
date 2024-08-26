from typing import TYPE_CHECKING, Any, Dict, Generic, Optional, Union

from pydantic import Field, PositiveInt, PrivateAttr

from docprompt.schema.pipeline.metadata import BaseMetadata
from docprompt.schema.pipeline.rasterizer import PageRasterizer

from .base import BaseNode
from .typing import PageNodeMetadata

if TYPE_CHECKING:
    from docprompt.tasks.ocr.result import OcrPageResult

    from .document import DocumentNode


class SimplePageNodeMetadata(BaseMetadata):
    """
    A simple metadata class for a page node
    """

    ocr_results: Optional["OcrPageResult"] = Field(
        None, description="The OCR results for the page"
    )


class PageNode(BaseNode, Generic[PageNodeMetadata]):
    """
    Represents a single page in a document, with some metadata
    """

    document: "DocumentNode" = Field(exclude=True, repr=False)
    page_number: PositiveInt = Field(description="The page number")
    metadata: Union[PageNodeMetadata, SimplePageNodeMetadata] = Field(
        description="Application-specific metadata for the page",
        default_factory=SimplePageNodeMetadata,
    )
    extra: Dict[str, Any] = Field(
        description="Extra data that can be stored on the page node",
        default_factory=dict,
    )

    _raster_cache: Dict[str, bytes] = PrivateAttr(default_factory=dict)

    def __getstate__(self):
        state = super().__getstate__()

        state["__pydantic_private__"]["_raster_cache"] = {}

        return state

    @property
    def rasterizer(self):
        return PageRasterizer(self._raster_cache, self)

    @property
    def ocr_results(self) -> Optional["OcrPageResult"]:
        from docprompt.tasks.ocr.result import OcrPageResult

        return self.metadata.find_by_type(OcrPageResult)

    @ocr_results.setter
    def ocr_results(self, value):
        if not hasattr(self.metadata, "ocr_results"):
            raise AttributeError(
                "Page metadata does not have an `ocr_results` attribute"
            )

        self.metadata.ocr_results = value

    def search(
        self, query: str, refine_to_words: bool = True, require_exact_match: bool = True
    ):
        return self.document.locator.search(
            query,
            page_number=self.page_number,
            refine_to_word=refine_to_words,
            require_exact_match=require_exact_match,
        )

    def get_layout_aware_text(self, **kwargs) -> str:
        if not self.ocr_results:
            raise ValueError("Calculate OCR results before calling layout_aware_text")

        from docprompt.utils.layout import build_layout_aware_page_representation

        word_blocks = self.ocr_results.word_level_blocks

        line_blocks = self.ocr_results.line_level_blocks

        if not len(line_blocks):
            line_blocks = None

        return build_layout_aware_page_representation(
            word_blocks, line_blocks=line_blocks, **kwargs
        )

    @property
    def layout_aware_text(self):
        return self.get_layout_aware_text()
