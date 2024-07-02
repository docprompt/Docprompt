import logging
import multiprocessing as mp
import tempfile
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Dict, List, Optional

from pydantic import BaseModel

from docprompt._exec.tesseract import OCRResult as TesseractResult
from docprompt._exec.tesseract import check_tesseract_installed, process_image_to_dict
from docprompt.schema.layout import NormBBox, TextBlock, TextBlockMetadata, TextSpan
from docprompt.tasks.capabilities import PageLevelCapabilities
from docprompt.tasks.ocr.base import BaseOCRProvider, ImageBytes
from docprompt.tasks.ocr.result import OcrPageResult

if TYPE_CHECKING:
    from docprompt.schema.pipeline import DocumentNode

logger = logging.getLogger(__name__)


class TesseractPageMetadata(BaseModel):
    pass


def _process_words(words: List[Dict]) -> List[TextBlock]:
    return [
        TextBlock(
            text=word["text"],
            type="word",
            bounding_box=NormBBox(
                x0=word["bbox"]["x"],
                top=word["bbox"]["y"],
                x1=word["bbox"]["x"] + word["bbox"]["width"],
                bottom=word["bbox"]["y"] + word["bbox"]["height"],
            ),
            metadata=TextBlockMetadata(
                direction="UP",  # Tesseract doesn't provide orientation info
                confidence=1.0,  # Tesseract doesn't provide confidence scores
            ),
            text_spans=[
                TextSpan(start_index=0, end_index=len(word["text"]), level="page")
            ],
        )
        for word in words
    ]


def _process_lines(lines: List[Dict]) -> List[TextBlock]:
    return [
        TextBlock(
            text=line["text"],
            type="line",
            bounding_box=NormBBox(
                x0=line["bbox"]["x"],
                top=line["bbox"]["y"],
                x1=line["bbox"]["x"] + line["bbox"]["width"],
                bottom=line["bbox"]["y"] + line["bbox"]["height"],
            ),
            metadata=TextBlockMetadata(
                direction="UP",  # Tesseract doesn't provide orientation info
                confidence=1.0,  # Tesseract doesn't provide confidence scores
            ),
            text_spans=[
                TextSpan(start_index=0, end_index=len(line["text"]), level="page")
            ],
        )
        for line in lines
    ]


def _process_blocks(blocks: List[Dict]) -> List[TextBlock]:
    return [
        TextBlock(
            text=block["text"],
            type="block",
            bounding_box=NormBBox(
                x0=block["bbox"]["x"],
                top=block["bbox"]["y"],
                x1=block["bbox"]["x"] + block["bbox"]["width"],
                bottom=block["bbox"]["y"] + block["bbox"]["height"],
            ),
            metadata=TextBlockMetadata(
                direction="UP",  # Tesseract doesn't provide orientation info
                confidence=1.0,  # Tesseract doesn't provide confidence scores
            ),
            text_spans=[
                TextSpan(start_index=0, end_index=len(block["text"]), level="page")
            ],
        )
        for block in blocks
    ]


def _tesseract_result_to_page_result(result: TesseractResult) -> OcrPageResult:
    words = _process_words(result["words"])
    lines = _process_lines(result["lines"])
    blocks = _process_blocks(result["blocks"])

    page_text = " ".join(block["text"] for block in result["blocks"])

    return OcrPageResult(
        provider_name="tesseract",
        page_number=-1,  # Set to 0 because we don't know the page number
        page_text=page_text,
        word_level_blocks=words,
        line_level_blocks=lines,
        block_level_blocks=blocks,
        raster_image=None,  # Tesseract doesn't provide raster image
    )


def _process_image_to_page_result(image: bytes):
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        f.write(image)
        f.flush()
        result = process_image_to_dict(f.name)

    return _tesseract_result_to_page_result(result)


class TesseractOcrProvider(BaseOCRProvider):
    name = "tesseract"
    capabilities = [
        PageLevelCapabilities.PAGE_TEXT_OCR,
    ]

    def __init__(self, **data):
        super().__init__(**data)
        if not check_tesseract_installed():
            raise RuntimeError(
                "Tesseract is not installed. Please install Tesseract to use this provider."
            )

            # Need to set this since we aren't instantiaed from a factory
        self._default_invoke_kwargs = {}

    def _process_images(
        self,
        images: List[bytes],
        start: Optional[int] = None,
        stop: Optional[int] = None,
        **kwargs,
    ) -> List[OcrPageResult]:
        results = []

        ctx = mp.get_context("spawn")

        with ProcessPoolExecutor(mp.cpu_count(), mp_context=ctx) as executor:
            for result in executor.map(_process_image_to_page_result, images):
                results.append(result)

        # Page range is used here for the indexing of results
        page_range = range(start or 1, (stop or len(images)) + 1)
        return {page_i: results[real_i] for real_i, page_i in enumerate(page_range)}

    def _invoke(
        self,
        input: List[ImageBytes],
        config: None = None,
        **kwargs,
    ):
        return self._process_images(input, **kwargs)

    def process_document_node(
        self,
        document_node: "DocumentNode",
        task_config: None = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        contribute_to_document: bool = True,
        **kwargs,
    ) -> Dict[int, OcrPageResult]:
        rasterized_images = document_node.rasterizer.rasterize("default")

        # Trim the rasterized images by the start and stop
        page_range = range(start or 1, (stop or len(document_node)) + 1)
        rasterized_images = [rasterized_images[i - 1] for i in page_range]

        result = self.invoke(rasterized_images, start=start, stop=stop, **kwargs)

        # For OCR, we also need to populate the ocr_results for powered search
        self._populate_ocr_results(document_node, result)

        return result
