import logging
from typing import Dict, List, Optional
from pathlib import Path

import pytesseract
from PIL import Image
import pdf2image
from pydantic import BaseModel, Field

from docprompt.schema.document import Document
from docprompt.schema.layout import (
    BoundingPoly,
    NormBBox,
    Point,
    SegmentLevels,
    TextBlock,
    TextBlockMetadata,
    TextSpan,
)
from docprompt.tasks.ocr.base import BaseOCRProvider
from docprompt.tasks.ocr.result import OcrPageResult

from ..base import CAPABILITIES

logger = logging.getLogger(__name__)

class TesseractPageMetadata(BaseModel):
    pass

def process_page(
    image: Image.Image,
    doc_page_num: int,
    provider_name: str,
    document_name: str,
    file_hash: str,
    exclude_bounding_poly: bool = False,
    return_image: bool = False,
) -> OcrPageResult:
    page_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    page_text = " ".join(page_data['text'])
    
    word_boxes = []
    line_boxes = []
    block_boxes = []
    
    width, height = image.size
    
    for i in range(len(page_data['text'])):
        if page_data['text'][i].strip():
            x, y, w, h = page_data['left'][i], page_data['top'][i], page_data['width'][i], page_data['height'][i]
            
            normalized_bbox = NormBBox(
                x0=round(x / width, 5),
                top=round(y / height, 5),
                x1=round((x + w) / width, 5),
                bottom=round((y + h) / height, 5),
            )
            
            bounding_poly = None if exclude_bounding_poly else BoundingPoly(
                normalized_vertices=[
                    Point(x=round(x / width, 5), y=round(y / height, 5)),
                    Point(x=round((x + w) / width, 5), y=round(y / height, 5)),
                    Point(x=round((x + w) / width, 5), y=round((y + h) / height, 5)),
                    Point(x=round(x / width, 5), y=round((y + h) / height, 5)),
                ]
            )
            
            text_block = TextBlock(
                text=page_data['text'][i],
                type="word",
                bounding_box=normalized_bbox,
                bounding_poly=bounding_poly,
                metadata=TextBlockMetadata(
                    confidence=round(page_data['conf'][i] / 100, 5),
                ),
                text_spans=[TextSpan(
                    start_index=len(" ".join(page_data['text'][:i])),
                    end_index=len(" ".join(page_data['text'][:i+1])),
                    level="page",
                )],
            )
            
            word_boxes.append(text_block)
    
    if return_image:
        raster_image = image.tobytes()
    else:
        raster_image = None

    return OcrPageResult[TesseractPageMetadata](
        provider_name=provider_name,
        document_name=document_name,
        file_hash=file_hash,
        page_number=doc_page_num,
        page_text=page_text,
        word_level_blocks=word_boxes,
        line_level_blocks=line_boxes,
        block_level_blocks=block_boxes,
        raster_image=raster_image,
        extra=TesseractPageMetadata(),
    )

class TesseractOcrProvider(BaseOCRProvider):
    name = "Tesseract OCR"
    capabilities = [
        CAPABILITIES.PAGE_TEXT_OCR.value,
        CAPABILITIES.PAGE_LAYOUT_OCR.value,
        CAPABILITIES.PAGE_RASTERIZATION.value,
    ]

    def __init__(
        self,
        *,
        tesseract_cmd: Optional[str] = None,
        exclude_bounding_poly: bool = False,
        return_images: bool = False,
    ):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        self.exclude_bounding_poly = exclude_bounding_poly
        self.return_images = return_images

    def process_document_pages(
        self,
        document: Document,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        **kwargs,
    ) -> Dict[int, OcrPageResult]:
        file_bytes = document.get_bytes()
        images = pdf2image.convert_from_bytes(file_bytes)
        
        results: Dict[int, OcrPageResult] = {}
        
        for idx, image in enumerate(images[start:stop], start=(start or 0) + 1):
            page_result = process_page(
                image,
                idx,
                self.name,
                document.name,
                document.document_hash,
                exclude_bounding_poly=self.exclude_bounding_poly,
                return_image=self.return_images,
            )
            
            results[idx] = page_result
        
        return results
