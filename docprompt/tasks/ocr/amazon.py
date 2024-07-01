import asyncio
import logging
import multiprocessing
from threading import Lock
from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional

import tqdm
from pydantic import Field, SecretStr, model_validator
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
) -> Dict[int, OcrPageResult]:
    results: Dict[int, OcrPageResult] = {}

    for page_num, document in enumerate(documents, start=1):
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

    aws_access_key_id: Optional[str] = Field(None)
    aws_secret_access_key: Optional[SecretStr] = Field(None)
    region_name: Optional[str] = Field(None)
    aws_session_token: Optional[SecretStr] = Field(None)

    max_workers: int = Field(multiprocessing.cpu_count() * 2)
    exclude_bounding_poly: bool = Field(False)
    return_images: bool = Field(False)

    @model_validator(mode="after")
    def validate_aws_credentials(self) -> Self:
        # Set the AWS credentials from the environment if not provided
        if self.aws_session_token is not None:
            return self

        if self.aws_access_key_id is None:
            self.aws_access_key_id = self._default_invoke_kwargs.get(
                "aws_access_key_id", None
            )
        if self.aws_secret_access_key is None:
            self.aws_secret_access_key = self._default_invoke_kwargs.get(
                "aws_secret_access_key", None
            )
        if self.region_name is None:
            self.region_name = self._default_invoke_kwargs.get("aws_region", None)

        _explict_keys_provided = self.aws_access_key_id and self.aws_secret_access_key

        if not _explict_keys_provided:
            # Check for the session key
            self.aws_session_token = self._default_invoke_kwargs.get(
                "aws_session_token", None
            )

        # Ensure that we have valid AWS credentials
        if not (_explict_keys_provided and not self.aws_session_token):
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
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key.get_secret_value()
            if self.aws_secret_access_key
            else None,
            region_name=self.region_name,
            aws_session_token=self.aws_session_token.get_secret_value()
            if self.aws_session_token
            else None,
        )

    @default_retry_decorator
    async def process_byte_chunk(self, image_bytes: bytes):
        session = self._get_session()

        async with session.client("textract") as textract_client:
            response = await textract_client.detect_document_text(
                Document={"Bytes": image_bytes}
            )

        return response

    async def _process_document_concurrent(
        self,
        document_images: List[ImageBytes],
        document_name: Optional[str] = None,
        document_hash: Optional[str] = None,
    ):
        tasks = [
            self.process_byte_chunk(image_bytes) for image_bytes in document_images
        ]

        documents = []
        for task in tqdm.tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Processing document"
        ):
            result = await task
            documents.append(result)

        logger.info("Recombining OCR results...")
        return textract_documents_to_result(
            documents,
            self.name,
            document_name=document_name,
            file_hash=document_hash,
            exclude_bounding_poly=self.exclude_bounding_poly,
            return_images=self.return_images,
        )

    async def _ainvoke(
        self,
        input: List[ImageBytes],
        config: None = None,
        **kwargs,
    ):
        return await self._process_document_concurrent(input)

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

        base_result = self.invoke(rasterized_images, start=start, stop=stop, **kwargs)

        # For OCR, we also need to populate the ocr_results for powered search
        self._populate_ocr_results(document_node, base_result)

        return base_result
