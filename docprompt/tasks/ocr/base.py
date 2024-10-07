from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, Optional, Union

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
        self,
        document_node: "DocumentNode",
        results: Dict[int, OcrPageResult],
        add_images_to_raster_cache: bool = False,
        raster_cache_key: str = "default",
    ) -> None:
        for page_number, result in results.items():
            result.contribute_to_document_node(
                document_node,
                page_number=page_number,
                add_images_to_raster_cache=add_images_to_raster_cache,
                raster_cache_key=raster_cache_key,
            )

    @abstractmethod
    def process_document_node(
        self,
        document_node: "DocumentNode",
        task_config: Optional[None] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        contribute_to_document: bool = True,
        **kwargs,
    ) -> Dict[int, OcrPageResult]: ...
