import asyncio
import logging
import multiprocessing
import multiprocessing.spawn
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
from threading import Lock
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Literal, Optional, Union

import tqdm
from pydantic import BaseModel, Field, PrivateAttr
from tenacity import retry, stop_after_attempt, wait_exponential

from docprompt.schema.document import Document, PdfDocument
from docprompt.schema.layout import (
    BoundingPoly,
    DirectionChoices,
    NormBBox,
    Point,
    SegmentLevels,
    TextBlock,
    TextBlockMetadata,
    TextSpan,
)
from docprompt.tasks.capabilities import PageLevelCapabilities
from docprompt.tasks.ocr.base import BaseOCRProvider
from docprompt.utils.splitter import pdf_split_iter_with_max_bytes_pypdf

from .result import OcrPageResult

logger = logging.getLogger("TEST-LOGGER")

if TYPE_CHECKING:
    from google.cloud import documentai

    from docprompt.schema.document import Document
    from docprompt.schema.pipeline.node.document import DocumentNode


service_account_file_read_lock = Lock()


orientation_rotation_mapping = {
    0: 0,
    1: 0,
    2: 90,
    3: 180,
    4: -90,
}

type_mapping: Dict[str, SegmentLevels] = {
    "line": "line",
    "paragraph": "block",
    "block": "block",
    "token": "word",
}

orientation_mapping: Dict[int, DirectionChoices] = {
    1: "UP",
    2: "RIGHT",
    3: "DOWN",
    4: "LEFT",
}

service_account_file_read_lock = Lock()

# This will wait up to ~8 minutes before giving up, which covers almost all high-contention cases
# TODO: Scope this to only retry on 429 and 5xx
default_retry_decorator = retry(
    wait=wait_exponential(multiplier=1, max=60), stop=stop_after_attempt(10)
)


class GCPDefectTypes(str, Enum):
    BLURRY = "BLURRY"
    NOISY = "NOISY"
    DARK = "DARK"
    FAINT = "FAINT"
    TOO_SMALL = "TOO_SMALL"
    DOCUMENT_CUTOFF = "CUTOFF"
    TEXT_CUTOFF = "TEXT_CUTOFF"
    GLARE = "GLARE"


GCP_DEFECT_TYPE_MAPPING = {
    "quality/defect_blurry": GCPDefectTypes.BLURRY,
    "quality/defect_noisy": GCPDefectTypes.NOISY,
    "quality/defect_dark": GCPDefectTypes.DARK,
    "quality/defect_faint": GCPDefectTypes.FAINT,
    "quality/defect_text_too_small": GCPDefectTypes.TOO_SMALL,
    "quality/defect_document_cutoff": GCPDefectTypes.DOCUMENT_CUTOFF,
    "quality/defect_text_cutoff": GCPDefectTypes.TEXT_CUTOFF,
    "quality/defect_glare": GCPDefectTypes.GLARE,
}


class GCPPageMetadata(BaseModel):
    quality_score: Optional[float] = None
    defect_scores: Dict[GCPDefectTypes, float] = Field(default_factory=dict)


def bounding_poly_from_layout(
    layout: Union["documentai.Document.Page.Layout", "documentai.Document.Page.Token"],
):
    return BoundingPoly(
        normalized_vertices=[
            Point(x=round(vertex.x, 5), y=round(vertex.y, 5))
            for vertex in layout.bounding_poly.normalized_vertices
        ]
    )


def bounding_box_from_layout(
    layout: Union["documentai.Document.Page.Layout", "documentai.Document.Page.Token"],
):
    top = min(vertex.y for vertex in layout.bounding_poly.normalized_vertices)
    bottom = max(vertex.y for vertex in layout.bounding_poly.normalized_vertices)

    left = min(vertex.x for vertex in layout.bounding_poly.normalized_vertices)
    right = max(vertex.x for vertex in layout.bounding_poly.normalized_vertices)

    return NormBBox(
        x0=left,
        top=top,
        x1=right,
        bottom=bottom,
    )


def geometry_from_layout(
    layout: Union["documentai.Document.Page.Layout", "documentai.Document.Page.Token"],
    exclude_bounding_poly: bool = False,
):
    bounding_poly = None if exclude_bounding_poly else bounding_poly_from_layout(layout)

    bounding_box = bounding_box_from_layout(layout)

    return {
        "bounding_poly": bounding_poly,
        "bounding_box": bounding_box,
    }


def text_from_layout(
    layout: Union["documentai.Document.Page.Layout", "documentai.Document.Page.Token"],
    document_text: str,
    offset: int = 0,
) -> str:
    """
    Offset is used to account for the fact that text references
    are relative to the entire document.
    """
    working_text = ""

    for segment in sorted(layout.text_anchor.text_segments, key=lambda x: x.end_index):
        start = getattr(segment, "start_index", 0)
        end = segment.end_index

        working_text += document_text[start - offset : end - offset]

    return working_text


def text_spans_from_layout(
    layout: Union["documentai.Document.Page.Layout", "documentai.Document.Page.Token"],
    level: Literal["page", "document"],
    offset: int = 0,
) -> List[TextSpan]:
    text_spans = []

    for segment in sorted(layout.text_anchor.text_segments, key=lambda x: x.end_index):
        start = getattr(segment, "start_index", 0)
        end = segment.end_index

        text_spans.append(
            TextSpan(
                start_index=start - offset,
                end_index=end - offset,
                level=level,
            )
        )

    return text_spans


def text_blocks_from_page(
    page: "documentai.Document.Page",
    document_text: str,
    type: Literal["line", "block", "token", "paragraph"],
    *,
    exclude_bounding_poly: bool = False,
) -> List[TextBlock]:
    text_blocks = []

    # Offset is used to account for the fact that text references are relative to the entire document.
    # while we need to compute spans relative to the page.
    offset_low = page.layout.text_anchor.text_segments[0].start_index or 0

    for item in getattr(page, f"{type}s"):
        layout = item.layout
        block_text = text_from_layout(layout, document_text)

        bounding_box = bounding_box_from_layout(layout)
        bounding_poly = (
            bounding_poly_from_layout(layout) if not exclude_bounding_poly else None
        )

        confidence = layout.confidence
        orientation = orientation_mapping.get(layout.orientation, "UP")

        text_spans = text_spans_from_layout(layout, level="page", offset=offset_low)

        block_type = type_mapping[type]
        text_blocks.append(
            TextBlock(
                text=block_text,
                type=block_type,
                bounding_box=bounding_box,
                bounding_poly=bounding_poly,
                metadata=TextBlockMetadata(
                    direction=orientation,
                    confidence=round(confidence, 5),
                ),
                text_spans=text_spans,
            )
        )

    return text_blocks


def metadata_from_page(page: "documentai.Document.Page") -> GCPPageMetadata:
    if not hasattr(page, "image_quality_scores"):
        return GCPPageMetadata()

    scores = page.image_quality_scores

    quality_score = scores.quality_score

    defect_scores = {}

    for defect in scores.detected_defects:
        defect_type = GCP_DEFECT_TYPE_MAPPING.get(defect.type_)

        if defect_type is not None:
            defect_scores[defect_type] = round(defect.confidence, 5)

    return GCPPageMetadata(
        quality_score=round(quality_score, 5),
        defect_scores=defect_scores,
    )


def process_page(
    document_text: str,
    page,
    doc_page_num: int,
    provider_name: str,
    document_name: str,
    file_hash: str,
    exclude_bounding_poly: bool = False,
    return_image: bool = False,
) -> OcrPageResult:
    layout = page.layout

    page_text = text_from_layout(layout, document_text)

    word_boxes = text_blocks_from_page(
        page, document_text, "token", exclude_bounding_poly=exclude_bounding_poly
    )
    line_boxes = text_blocks_from_page(
        page, document_text, "line", exclude_bounding_poly=exclude_bounding_poly
    )
    block_boxes = text_blocks_from_page(
        page, document_text, "block", exclude_bounding_poly=exclude_bounding_poly
    )

    metadata = metadata_from_page(page)

    if return_image:
        image = page.image.content
    else:
        image = None

    return OcrPageResult(
        provider_name=provider_name,
        document_name=document_name,
        file_hash=file_hash,
        page_number=doc_page_num,
        page_text=page_text,
        word_level_blocks=word_boxes,
        line_level_blocks=line_boxes,
        block_level_blocks=block_boxes,
        raster_image=image,
        extra=metadata.model_dump(),
    )


def postprocess_gcp_document(
    document: "documentai.Document",
    provider_name: str,
    document_name: str,
    file_hash: str,
    exclude_bounding_poly: bool = False,
    return_images: bool = False,
) -> Dict[int, OcrPageResult]:
    results: Dict[int, OcrPageResult] = {}

    for doc_page_num, page in enumerate(
        document.pages, start=1
    ):  # Start from 1 for 1-indexing
        page_result = process_page(
            document.text,
            page,
            doc_page_num + 1,
            provider_name,
            document_name,
            file_hash,
            exclude_bounding_poly=exclude_bounding_poly,
            return_image=return_images,
        )

        results[doc_page_num] = page_result

    return results


def postprocess_gcp_document_in_executor(
    pages_in_split: int,
    idx: int,
    document: "documentai.Document",
    provider_name: str,
    document_name: str,
    file_hash: str,
    exclude_bounding_poly: bool = False,
    return_images: bool = False,
):
    result = postprocess_gcp_document(
        document,
        provider_name,
        document_name,
        file_hash,
        exclude_bounding_poly=exclude_bounding_poly,
        return_images=return_images,
    )

    return result, pages_in_split, idx


class GoogleOcrProvider(BaseOCRProvider):
    name = "gcp_documentai"

    capabilities = [
        PageLevelCapabilities.PAGE_TEXT_OCR,
        PageLevelCapabilities.PAGE_LAYOUT_OCR,
        PageLevelCapabilities.PAGE_RASTERIZATION,
    ]

    max_bytes_per_request: ClassVar[int] = (
        1024 * 1024 * 20
    )  # 20MB is the max size for a single sync request
    max_page_count: ClassVar[int] = 15

    project_id: str = Field(...)
    processor_id: str = Field(...)

    service_account_info: Optional[Dict[str, str]] = Field(None)
    service_account_file: Optional[str] = Field(None)
    location: str = Field("us")
    max_workers: int = Field(multiprocessing.cpu_count() * 2)
    exclude_bounding_poly: bool = Field(False)
    return_images: bool = Field(False)
    add_images_to_raster_cache: bool = Field(False)
    image_raster_cache_key: str = "default"
    return_image_quality_scores: bool = Field(False)

    _documentai: "documentai" = PrivateAttr()

    def __init__(
        self,
        project_id: str,
        processor_id: str,
        service_account_info: Optional[Dict[str, str]] = None,
        service_account_file: Optional[str] = None,
        return_images: bool = False,
        add_images_to_raster_cache: bool = False,
        image_raster_cache_key: str = "default",
        **kwargs,
    ):
        super().__init__(project_id=project_id, processor_id=processor_id, **kwargs)

        self.service_account_info = self._default_invoke_kwargs.get(
            "service_account_info", service_account_info
        )
        self.service_account_file = self._default_invoke_kwargs.get(
            "service_account_file", service_account_file
        )

        self.return_images = return_images
        self.add_images_to_raster_cache = add_images_to_raster_cache
        self.image_raster_cache_key = image_raster_cache_key

        try:
            from google.cloud import documentai

            self._documentai = documentai
        except ImportError:
            raise ImportError(
                "Please install 'google-cloud-documentai' to use the GoogleCloudVisionTextExtractionProvider"
            )

    async def get_documentai_client(self, client_option_kwargs: dict = {}, **kwargs):
        from google.api_core.client_options import ClientOptions

        opts = ClientOptions(
            **{
                "api_endpoint": "us-documentai.googleapis.com",
                **client_option_kwargs,
            }
        )

        base_service_client_kwargs = {
            **kwargs,
            "client_options": opts,
        }

        if self.service_account_info is not None:
            return self._documentai.DocumentProcessorServiceAsyncClient.from_service_account_info(
                info=self.service_account_info,
                **base_service_client_kwargs,
            )
        elif self.service_account_file is not None:
            with service_account_file_read_lock:
                return self._documentai.DocumentProcessorServiceAsyncClient.from_service_account_file(
                    filename=self.service_account_file,
                    **base_service_client_kwargs,
                )
        else:
            raise ValueError("Missing account info and service file path.")

    def _get_process_options(self):
        if not self.return_image_quality_scores:
            return None

        return self._documentai.ProcessOptions(
            ocr_config=self._documentai.OcrConfig(
                enable_image_quality_scores=True,
            )
        )

    @default_retry_decorator
    async def process_byte_chunk(
        self, split_bytes: bytes, client: Any
    ):  # TODO: Fix `client` typing
        try:
            raw_document = self._documentai.RawDocument(
                content=split_bytes,
                mime_type="application/pdf",
            )

            processor_name = client.processor_path(
                project=self.project_id,
                location=self.location,
                processor=self.processor_id,
            )

            field_mask = (
                "text,pages.layout,pages.words,pages.lines,pages.tokens,pages.blocks"
            )

            if self.return_images:
                field_mask += ",pages.image"

            if self.return_image_quality_scores:
                field_mask += ",image_quality_scores"

            request = self._documentai.ProcessRequest(
                name=processor_name,
                raw_document=raw_document,
                process_options=self._get_process_options(),
            )

            result = await client.process_document(request=request)

            document = result.document

            return document
        except Exception as exp:
            logger.error("Error processing byte chunk %s", exp, exc_info=True)
            raise exp

    async def _process_document(
        self,
        document: Document,
        include_raster: bool = False,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        **kwargs,
    ):
        # Process page chunks concurrently
        client = await self.get_documentai_client()

        file_bytes = document.file_bytes

        if document.bytes_per_page > 1024 * 1024 * 2:
            logger.info("Document has few pages but is large, compressing first")
            file_bytes = document.to_compressed_bytes()

        logger.info("Processing document chunks as they are generated...")
        to_merge = []
        tasks = []
        futures = []

        async def process_single_document_chunk(
            split_bytes, client, pages_in_split, idx
        ):
            result = await self.process_byte_chunk(split_bytes, client)
            return result, pages_in_split, idx

        with tqdm.tqdm(
            desc="API Processing", total=len(document)
        ) as api_pbar, tqdm.tqdm(
            desc="Postprocessing", total=len(document)
        ) as post_pbar:
            with ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=multiprocessing.get_context("spawn"),
            ) as executor:
                # Start tasks as chunks are generated
                for idx, (split_bytes, pages_in_split) in enumerate(
                    pdf_split_iter_with_max_bytes_pypdf(
                        file_bytes,
                        max_page_count=self.max_page_count,
                        max_bytes=self.max_bytes_per_request,
                    )
                ):
                    task = asyncio.create_task(
                        process_single_document_chunk(
                            split_bytes, client, pages_in_split, idx
                        )
                    )
                    tasks.append(task)

                # Process completed tasks in order
                for task in asyncio.as_completed(tasks):
                    result, pages_in_split, order_idx = await task
                    api_pbar.update(pages_in_split)

                    def postprocess_done_cb(future, pages_in_split=pages_in_split):
                        post_pbar.update(pages_in_split)

                    future = executor.submit(
                        postprocess_gcp_document_in_executor,
                        pages_in_split,
                        order_idx,
                        result,
                        self.name,
                        document.name,
                        document.document_hash,
                        self.exclude_bounding_poly,
                        self.return_images,
                    )

                    future.add_done_callback(postprocess_done_cb)
                    futures.append(future)

                for future in as_completed(futures):
                    result, pages_in_split, order_idx = future.result()
                    to_merge.append((result, pages_in_split, order_idx))

        logger.info("Recombining OCR results...")
        results: Dict[int, OcrPageResult] = {}

        # Sort by original chunk order
        to_merge.sort(key=lambda x: x[2])  # Sort by order_idx

        current_page = 1  # Start with page 1 (1-indexed)

        # Merge results maintaining correct page numbers
        for chunk_results, pages_in_split, _ in to_merge:
            # Add results for this chunk with correct page numbers
            for page_num, result in chunk_results.items():
                results[current_page] = result  # +1 because chunk_results is 1-indexed
                current_page += 1

        return results

    async def _ainvoke(
        self,
        input: List[PdfDocument],
        config: None = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        **kwargs,
    ):
        if len(input) != 1:
            raise ValueError(
                "GoogleOcrProvider only supports processing a single document at a time."
            )

        return await self._process_document(input[0], start=start, stop=stop, **kwargs)

    def _invoke(
        self,
        input: List[PdfDocument],
        config: None = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        **kwargs,
    ):
        try:
            return asyncio.run(self._ainvoke(input, config, start, stop, **kwargs))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(
                self._ainvoke(input, config, start, stop, **kwargs)
            )

    def process_document_node(
        self,
        document_node: "DocumentNode",
        task_config: None = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        contribute_to_document: bool = True,
        **kwargs,
    ) -> Dict[int, OcrPageResult]:
        if start or stop:
            logger.warning(
                "GoogleOcrProvider does not currently support `start` and `stop`."
            )

        result = self.invoke([document_node.document], start=start, stop=stop, **kwargs)

        # For OCR, we also need to populate the ocr_results for powered search
        self._populate_ocr_results(
            document_node,
            result,
            add_images_to_raster_cache=self.add_images_to_raster_cache,
            raster_cache_key=self.image_raster_cache_key,
        )

        return result
