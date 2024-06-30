import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, List, Optional

import boto3
import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

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
from docprompt.schema.pipeline import DocumentNode
from docprompt.tasks.ocr.base import BaseOCRProvider
from docprompt.utils.splitter import pdf_split_iter_with_max_bytes

from ..base import CAPABILITIES
from .result import OcrPageResult

logger = logging.getLogger(__name__)

service_account_file_read_lock = Lock()

# This will wait up to ~8 minutes before giving up, which covers almost all high-contention cases
default_retry_decorator = retry(
    wait=wait_exponential(multiplier=1, max=60), stop=stop_after_attempt(10)
)


def bounding_box_from_geometry(geometry: Dict):
    return NormBBox(
        x0=round(geometry["BoundingBox"]["Left"], 5),
        top=round(geometry["BoundingBox"]["Top"], 5),
        x1=round(geometry["BoundingBox"]["Left"] + geometry["BoundingBox"]["Width"], 5),
        bottom=round(
            geometry["BoundingBox"]["Top"] + geometry["BoundingBox"]["Height"], 5
        ),
    )


def bounding_poly_from_geometry(geometry: Dict):
    return BoundingPoly(
        normalized_vertices=[
            Point(x=round(point["X"], 5), y=round(point["Y"], 5))
            for point in geometry["Polygon"]
        ]
    )


def text_block_from_item(item: Dict, block_type: SegmentLevels) -> TextBlock:
    return TextBlock(
        text=item["Text"],
        type=block_type,
        bounding_box=bounding_box_from_geometry(item["Geometry"]),
        bounding_poly=bounding_poly_from_geometry(item["Geometry"]),
        metadata=TextBlockMetadata(
            direction="UP",  # Textract doesn't provide orientation information
            confidence=round(item["Confidence"] / 100, 5),
        ),
        text_spans=[
            TextSpan(
                start_index=0,
                end_index=len(item["Text"]),
                level="page",
            )
        ],
    )


def process_page(
    page: Dict,
    doc_page_num: int,
    provider_name: str,
    document_name: str,
    file_hash: str,
    exclude_bounding_poly: bool = False,
    return_image: bool = False,
) -> OcrPageResult:
    word_boxes = [text_block_from_item(word, "word") for word in page.get("Words", [])]
    line_boxes = [text_block_from_item(line, "line") for line in page.get("Lines", [])]
    block_boxes = [
        text_block_from_item(block, "block")
        for block in page.get("Blocks", [])
        if block["BlockType"] == "LINE"
    ]

    page_text = " ".join([word["Text"] for word in page.get("Words", [])])

    return OcrPageResult(
        provider_name=provider_name,
        document_name=document_name,
        file_hash=file_hash,
        page_number=doc_page_num,
        page_text=page_text,
        word_level_blocks=word_boxes,
        line_level_blocks=line_boxes,
        block_level_blocks=block_boxes,
        raster_image=None,  # Textract doesn't return rasterized images
    )


def textract_documents_to_result(
    documents: List[Dict],
    provider_name: str,
    document_name: str,
    file_hash: str,
    *,
    exclude_bounding_poly: bool = False,
    return_images: bool = False,
) -> Dict[int, OcrPageResult]:
    results: Dict[int, OcrPageResult] = {}
    page_offset = 1  # We want pages to be 1-indexed

    for document in documents:
        for doc_page_num, page in enumerate(document["Blocks"]):
            if page["BlockType"] == "PAGE":
                page_result = process_page(
                    page,
                    page_offset + doc_page_num,
                    provider_name,
                    document_name,
                    file_hash,
                    exclude_bounding_poly=exclude_bounding_poly,
                    return_image=return_images,
                )
                results[page_offset + doc_page_num] = page_result

        page_offset += doc_page_num + 1

    return results


class AmazonTextractProvider(BaseOCRProvider):
    name = "aws_textract"

    capabilities = [
        CAPABILITIES.PAGE_TEXT_OCR.value,
        CAPABILITIES.PAGE_LAYOUT_OCR.value,
    ]

    max_bytes_per_request = (
        1024 * 1024 * 5
    )  # 5MB is the max size for a single sync request
    max_page_count = 15

    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: Optional[str] = None,
        *,
        max_workers: int = multiprocessing.cpu_count() * 2,
        exclude_bounding_poly: bool = False,
        return_images: bool = False,
    ):
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region_name = region_name
        self.max_workers = max_workers
        self.exclude_bounding_poly = exclude_bounding_poly
        self.return_images = return_images

    def get_textract_client(self):
        kwargs = {}
        if self.aws_access_key_id and self.aws_secret_access_key:
            kwargs["aws_access_key_id"] = self.aws_access_key_id
            kwargs["aws_secret_access_key"] = self.aws_secret_access_key
        if self.region_name:
            kwargs["region_name"] = self.region_name

        return boto3.client("textract", **kwargs)

    @default_retry_decorator
    def process_byte_chunk(self, split_bytes: bytes):
        client = self.get_textract_client()
        response = client.analyze_document(
            Document={"Bytes": split_bytes}, FeatureTypes=["FORMS", "TABLES"]
        )
        return response

    def _process_document_concurrent(
        self,
        document: Document,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ):
        file_bytes = document.get_bytes()

        logger.info("Splitting document into chunks...")
        document_byte_splits = list(
            pdf_split_iter_with_max_bytes(
                file_bytes,
                max_page_count=self.max_page_count,
                max_bytes=self.max_bytes_per_request,
            )
        )

        max_workers = min(len(document_byte_splits), self.max_workers)

        logger.info(f"Processing {len(document_byte_splits)} chunks...")
        with tqdm.tqdm(
            total=len(document_byte_splits), desc="Processing document"
        ) as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(self.process_byte_chunk, split): index
                    for index, split in enumerate(document_byte_splits)
                }

                documents: List[Dict] = [None] * len(document_byte_splits)

                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    documents[index] = future.result()
                    pbar.update(1)

        logger.info("Recombining OCR results...")
        return textract_documents_to_result(
            documents,
            self.name,
            document_name=document.name,
            file_hash=document.document_hash,
            exclude_bounding_poly=self.exclude_bounding_poly,
            return_images=self.return_images,
        )

    def process_document_pages(
        self,
        document: Document,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        **kwargs,
    ) -> Dict[int, OcrPageResult]:
        return self._process_document_concurrent(document, start=start, stop=stop)

    def contribute_to_document_node(
        self, document_node: "DocumentNode", results: Dict[int, OcrPageResult]
    ) -> None:
        for page_number, result in results.items():
            document_node.page_nodes[page_number - 1].ocr_results.results[self.name] = (
                result
            )
