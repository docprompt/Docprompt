from typing import TYPE_CHECKING, Dict

from docprompt.schema.document import PdfDocument
from docprompt.schema.pipeline import DocumentNode
from docprompt.tasks.base import AbstractPageTaskProvider
from docprompt.tasks.ocr.result import OcrPageResult

if TYPE_CHECKING:
    from docprompt.schema.pipeline import DocumentNode


class OcrMixin(AbstractPageTaskProvider[PdfDocument, None, OcrPageResult]):
    def _populate_ocr_results(
        self, document_node: "DocumentNode", results: Dict[int, OcrPageResult]
    ) -> None:
        for page_number, result in results.items():
            document_node.page_nodes[page_number - 1].ocr_results.results[self.name] = (
                result
            )


class DocumentOcrProvider(
    OcrMixin, AbstractPageTaskProvider[PdfDocument, None, OcrPageResult]
):
    def process_document_node(
        self,
        document_node: DocumentNode,
        task_config: None = None,
        start: int | None = None,
        stop: int | None = None,
        contribute_to_document: bool = True,
        **kwargs,
    ) -> Dict[int, OcrPageResult]:
        base_result = self.invoke(
            [document_node.document.file_bytes], start=start, stop=stop, **kwargs
        )

        # For OCR, we also need to populate the ocr_results for powered search
        self._populate_ocr_results(document_node, base_result)

        return base_result


class ImageOcrProvider(OcrMixin, AbstractPageTaskProvider[bytes, None, OcrPageResult]):
    def process_document_node(
        self,
        document_node: DocumentNode,
        task_config: None = None,
        start: int | None = None,
        stop: int | None = None,
        contribute_to_document: bool = True,
        **kwargs,
    ) -> Dict[int, OcrPageResult]:
        base_result = self.invoke(
            [document_node.document.file_bytes], start=start, stop=stop, **kwargs
        )

        # For OCR, we also need to populate the ocr_results for powered search
        self._populate_ocr_results(document_node, base_result)

        return base_result
