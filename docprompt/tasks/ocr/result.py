from io import BytesIO
from typing import Any, Dict, List, Optional

from pydantic import Field

from docprompt.schema.layout import TextBlock
from docprompt.schema.pipeline.node.document import DocumentNode
from docprompt.tasks.base import BasePageResult


class OcrPageResult(BasePageResult):
    page_text: str = Field(description="The text for the entire page in reading order")

    word_level_blocks: List[TextBlock] = Field(
        default_factory=list,
        description="The provider-sourced words for the page",
        repr=False,
    )
    line_level_blocks: List[TextBlock] = Field(
        default_factory=list,
        description="The provider-sourced lines for the page",
        repr=False,
    )
    block_level_blocks: List[TextBlock] = Field(
        default_factory=list,
        description="The provider-sourced blocks for the page",
        repr=False,
    )

    raster_image: Optional[bytes] = Field(
        default=None,
        description="The rasterized image of the page used in OCR",
        repr=False,
    )

    extra: Optional[Dict[str, Any]] = Field(default_factory=dict)

    task_name = "ocr"

    @property
    def pil_image(self):
        from PIL import Image

        return Image.open(BytesIO(self.raster_image))

    @property
    def words(self):
        return self.word_level_blocks

    @property
    def lines(self):
        return self.line_level_blocks

    @property
    def blocks(self):
        return self.block_level_blocks

    def contribute_to_document_node(
        self, document_node: DocumentNode, page_number: Optional[int] = None, **kwargs
    ) -> None:
        if not page_number:
            raise ValueError("Page number must be provided for page level results")

        page_node = document_node.page_nodes[page_number - 1]
        if hasattr(page_node.metadata, "ocr_results"):
            page_node.metadata.ocr_results = self
        else:
            super().contribute_to_document_node(document_node, page_number=page_number)
