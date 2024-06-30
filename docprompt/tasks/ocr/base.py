from typing import TYPE_CHECKING, Dict, Union

from docprompt.schema.document import PdfDocument
from docprompt.schema.pipeline import DocumentNode
from docprompt.tasks.base import AbstractPageTaskProvider
from docprompt.tasks.ocr.result import OcrPageResult

if TYPE_CHECKING:
    from docprompt.schema.pipeline import DocumentNode


ImageBytes = bytes


class BaseOCRProvider(
    AbstractPageTaskProvider[Union[PdfDocument, ImageBytes], None, OcrPageResult]
):
    def _populate_ocr_results(
        self, document_node: "DocumentNode", results: Dict[int, OcrPageResult]
    ) -> None:
        for page_number, result in results.items():
            document_node.page_nodes[page_number - 1].ocr_results.results[self.name] = (
                result
            )
