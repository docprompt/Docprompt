import logging
from typing import Dict, Optional

from pydantic import BaseModel

from docprompt import PdfDocument
from docprompt.tasks.capabilities import PageLevelCapabilities
from docprompt.tasks.ocr.base import BaseOCRProvider
from docprompt.tasks.ocr.result import OcrPageResult

logger = logging.getLogger(__name__)


class TesseractPageMetadata(BaseModel):
    pass


class TesseractOcrProvider(BaseOCRProvider):
    name = "tesseract"
    capabilities = [
        PageLevelCapabilities.PAGE_TEXT_OCR,
    ]

    def __init__(
        self,
        *,
        tesseract_cmd: Optional[str] = None,
        exclude_bounding_poly: bool = False,
        return_images: bool = False,
    ):
        self.exclude_bounding_poly = exclude_bounding_poly
        self.return_images = return_images

    def process_document_pages(
        self,
        document: PdfDocument,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        **kwargs,
    ) -> Dict[int, OcrPageResult]:
        images = document.rasterize_pdf()

        results: Dict[int, OcrPageResult] = {}

        for idx, image in enumerate(images[start:stop], start=(start or 0) + 1):
            pass

        return results
