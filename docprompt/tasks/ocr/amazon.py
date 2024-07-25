import asyncio
import logging
import multiprocessing
from threading import Lock
from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional

import tqdm
from pydantic import Field, PrivateAttr, model_validator
from tenacity import retry, stop_after_attempt, wait_exponential
from typing_extensions import Self

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
from docprompt.tasks.capabilities import PageLevelCapabilities
from docprompt.tasks.ocr.base import BaseOCRProvider, ImageBytes

from .result import OcrPageResult

if TYPE_CHECKING:
    pass

# TODO: Fix this
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
    document: Dict,
    doc_page_num: int,
    provider_name: str,
    document_name: str,
    file_hash: str,
    exclude_bounding_poly: bool = False,
    return_image: bool = False,
) -> OcrPageResult:
    blocks = document.get("Blocks", [])
    word_boxes = [
        text_block_from_item(block, "word")
        for block in blocks
        if block["BlockType"] == "WORD"
    ]
    line_boxes = [
        text_block_from_item(block, "line")
        for block in blocks
        if block["BlockType"] == "LINE"
    ]
    block_boxes = [
        text_block_from_item(block, "block")
        for block in blocks
        if block["BlockType"] == "LINE"
    ]

    page_text = " ".join(
        [block["Text"] for block in blocks if block["BlockType"] == "WORD"]
    )

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
    start: Optional[int] = None,
    stop: Optional[int] = None,
) -> Dict[int, OcrPageResult]:
    results: Dict[int, OcrPageResult] = {}

    page_range = range(start or 1, (stop or len(documents)) + 1)
    for page_num, document in zip(page_range, documents):
        page_result = process_page(
            document,
            page_num,
            provider_name,
            document_name,
            file_hash,
            exclude_bounding_poly=exclude_bounding_poly,
            return_image=return_images,
        )
        results[page_num] = page_result

    return results


class AmazonTextractOCRProvider(BaseOCRProvider):
    name = "aws_textract"

    capabilities = [
        PageLevelCapabilities.PAGE_TEXT_OCR,
        PageLevelCapabilities.PAGE_LAYOUT_OCR,
    ]

    max_bytes_per_request: ClassVar[int] = (
        1024 * 1024 * 5
    )  # 5MB is the max size for a single sync request
    max_page_count: ClassVar[int] = 15

    _aws_access_key_id: Optional[str] = PrivateAttr(None)
    _aws_secret_access_key: Optional[str] = PrivateAttr(None)
    _region_name: Optional[str] = PrivateAttr(None)
    _aws_session_token: Optional[str] = PrivateAttr(None)

    max_workers: int = Field(multiprocessing.cpu_count() * 2)
    exclude_bounding_poly: bool = Field(False)
    return_images: bool = Field(False)

    @model_validator(mode="after")
    def validate_aws_credentials(self) -> Self:
        # Set the AWS credentials from the environment if not provided
        if self._aws_session_token is not None:
            return self

        if self._aws_access_key_id is None:
            self._aws_access_key_id = self._default_invoke_kwargs.get(
                "aws_access_key_id", None
            )
        if self._aws_secret_access_key is None:
            self._aws_secret_access_key = self._default_invoke_kwargs.get(
                "aws_secret_access_key", None
            )
        if self._region_name is None:
            self._region_name = self._default_invoke_kwargs.get("aws_region", None)

        _explict_keys_provided = self._aws_access_key_id and self._aws_secret_access_key

        if not _explict_keys_provided:
            # Check for the session key
            self._aws_session_token = self._default_invoke_kwargs.get(
                "aws_session_token", None
            )

        # Ensure that we have valid AWS credentials
        if not (_explict_keys_provided and not self._aws_session_token):
            raise ValueError(
                "You must provide either an AWS session token or an access key and secret key."
            )

        return self

    @model_validator(mode="after")
    def validate_aioboto3(self) -> Self:
        try:
            import aioboto3  # noqa
        except ImportError as e:
            raise ValueError(
                "The aioboto3 library is required to use the AWS Textract provider."
            ) from e

    def _get_session(self):
        import aioboto3

        return aioboto3.Session(
            aws_access_key_id=self._aws_access_key_id,
            aws_secret_access_key=(
                self._aws_secret_access_key if self._aws_secret_access_key else None
            ),
            region_name=self._region_name,
            aws_session_token=self._aws_session_token
            if self._aws_session_token
            else None,
        )

    @default_retry_decorator
    async def process_byte_chunk(self, image_bytes: bytes, index: int):
        session = self._get_session()

        size = len(image_bytes)
        sess_id = hex(id(session))
        logger.debug(
            "Session - %s: Processing chunk of size %d bytes with", sess_id, size
        )

        import time

        async with session.client("textract") as textract_client:
            try:
                start = time.perf_counter()
                response = await textract_client.detect_document_text(
                    Document={"Bytes": image_bytes}
                )
                end = time.perf_counter()
                logger.debug(
                    "Session - %s: Successfully processed chunk of size %d bytes in %.2f seconds",
                    sess_id,
                    size,
                    end - start,
                )
            except Exception as e:
                logger.error(
                    "Session -%s: Error processing chunk of size %d bytes: %s",
                    sess_id,
                    size,
                    e,
                )
                raise e

        return response, index

    async def _process_document_concurrent(
        self,
        document_images: List[ImageBytes],
        document_name: Optional[str] = None,
        document_hash: Optional[str] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ):
        tasks = [
            self.process_byte_chunk(image_bytes, i)
            for i, image_bytes in enumerate(document_images)
        ]

        documents = {}
        errors = []
        with tqdm.tqdm(total=len(tasks), desc="Processing document") as pbar:
            for task in asyncio.as_completed(tasks):
                try:
                    result, index = await task
                    documents[index] = result
                    pbar.update(1)
                except Exception as e:
                    errors.append(e)
                    logger.error(f"Error processing chunk: {e}")

        if errors:
            raise Exception(
                f"Encountered {len(errors)} errors during processing: {errors}"
            )

        logger.info("Recombining OCR results...")
        ordered_documents = [documents[i] for i in range(len(document_images))]
        return textract_documents_to_result(
            ordered_documents,
            self.name,
            document_name=document_name,
            file_hash=document_hash,
            exclude_bounding_poly=self.exclude_bounding_poly,
            return_images=self.return_images,
            start=start,
            stop=stop,
        )

    async def _ainvoke(
        self,
        input: List[ImageBytes],
        config: None = None,
        **kwargs,
    ):
        return await self._process_document_concurrent(
            input, start=kwargs.get("start", None), stop=kwargs.get("stop", None)
        )

    async def aprocess_document_node(
        self,
        document_node: "DocumentNode",
        task_config: None = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        contribute_to_document: bool = True,
        **kwargs,
    ) -> Dict[int, OcrPageResult]:
        rasterized_images = document_node.rasterizer.rasterize("default")

        # Get the start and stop range (in pages not indexes)
        start = start or 1
        stop = stop or len(document_node)

        page_range = range(start, stop + 1)
        rasterized_images = [rasterized_images[i - 1] for i in page_range]

        result = await self.ainvoke(rasterized_images, start=start, stop=stop, **kwargs)

        # For OCR, we also need to populate the ocr_results for powered search
        if contribute_to_document:
            self._populate_ocr_results(document_node, result)

        return result
