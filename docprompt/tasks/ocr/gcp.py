from typing import Optional, TYPE_CHECKING, Union, Literal, Dict, List
from docprompt.schema.document import Document

from docprompt.tasks.base import AbstractTaskProvider
from docprompt.utils.splitter import pdf_split_iter_with_max_bytes
from ..base import CAPABILITIES
from .result import OcrPageResult
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
from threading import Lock

from docprompt.schema.layout import (
    DirectionChoices,
    TextBlock,
    NormBBox,
    TextSpan,
    BoundingPoly,
    Point,
    SegmentLevels,
    TextBlockMetadata,
)
from tenacity import retry, stop_after_attempt, wait_exponential
import tqdm
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from google.cloud import documentai

    from docprompt.schema.document import Document
    from docprompt.schema.pipeline import DocumentNode


service_account_file_read_lock = Lock()


orientation_rotation_mapping = {
    0: 0,
    1: 0,
    2: 90,
    3: 180,
    4: -90,
}

service_account_file_read_lock = Lock()

# This will wait up to ~8 minutes before giving up, which covers almost all high-contention cases
# TODO: Scope this to only retry on 429 and 5xx
default_retry_decorator = retry(
    wait=wait_exponential(multiplier=1, max=60), stop=stop_after_attempt(10)
)


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
    sorted_vertices = sorted(
        layout.bounding_poly.normalized_vertices, key=lambda x: (x.x, x.y)
    )
    upper_left = sorted_vertices[0]
    lower_right = sorted_vertices[-1]

    return NormBBox(
        x0=round(upper_left.x, 5),
        top=round(upper_left.y, 5),
        x1=round(lower_right.x, 5),
        bottom=round(lower_right.y, 5),
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

    # Offset is used to account for the fact that text references are relative to the entire document.
    # while we need to compute spans relative to the page.
    offset_low = page.layout.text_anchor.text_segments[0].start_index or 0

    for item in getattr(page, f"{type}s"):
        layout = item.layout
        block_text = text_from_layout(layout, document_text)
        geometry_kwargs = geometry_from_layout(
            layout, exclude_bounding_poly=exclude_bounding_poly
        )
        confidence = layout.confidence
        orientation = orientation_mapping.get(layout.orientation, "UP")

        text_spans = text_spans_from_layout(layout, level="page", offset=offset_low)

        block_type = type_mapping[type]
        text_blocks.append(
            TextBlock(
                text=block_text,
                type=block_type,
                bounding_box=geometry_kwargs["bounding_box"],
                bounding_poly=geometry_kwargs["bounding_poly"],
                metadata=TextBlockMetadata(
                    direction=orientation,
                    confidence=round(confidence, 5),
                ),
                text_spans=text_spans,
            )
        )

    return text_blocks


def process_page(
    document_text: str,
    page,
    doc_page_num: int,
    provider_name: str,
    document_name: str,
    file_hash: str,
    exclude_bounding_poly: bool = False,
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

    return OcrPageResult(
        provider_name=provider_name,
        document_name=document_name,
        file_hash=file_hash,
        page_number=doc_page_num,
        page_text=page_text,
        word_level_blocks=word_boxes,
        line_level_blocks=line_boxes,
        block_level_blocks=block_boxes,
    )


def gcp_documents_to_result_single(
    documents: List["documentai.Document"],
    provider_name: str,
    document_name: str,
    file_hash: str,
    *,
    exclude_bounding_poly: bool = False,
):
    page_offset = 1  # We want pages to be 1-indexed

    results: Dict[int, OcrPageResult] = {}

    for document in tqdm.tqdm(documents):
        for doc_page_num, page in enumerate(document.pages):
            page_result = process_page(
                document.text,
                page,
                page_offset + doc_page_num,
                provider_name,
                document_name,
                file_hash,
                exclude_bounding_poly=exclude_bounding_poly,
            )

            results[page_offset + doc_page_num] = page_result

        page_offset += doc_page_num + 1

    return results


def multi_process_page(args):
    idx, document, provider_name, page_offset, exclude_bounding_poly = args

    results: Dict[int, OcrPageResult] = {}

    for doc_page_num, page in enumerate(document.pages):
        page_result = process_page(
            document.text,
            page,
            page_offset + doc_page_num,
            provider_name,
            exclude_bounding_poly=exclude_bounding_poly,
        )

        results[page_offset + doc_page_num] = page_result

    return idx, results


def gcp_documents_to_result_multi(
    documents: List["documentai.Document"],
    provider_name: str,
    *,
    exclude_bounding_poly: bool = False,
):
    tasks = []
    page_offset = 1  # Pages are 1-indexed

    # Prepare tasks for processing, each with a document index
    for idx, document in enumerate(documents):
        args = (
            idx,
            document,
            provider_name,
            page_offset,
            exclude_bounding_poly,
        )
        tasks.append(args)
        page_offset += len(document.pages)

    document_results = [None] * len(documents)

    worker_count = min(len(documents), min(4, max(multiprocessing.cpu_count(), 1)))

    futures = []

    with tqdm.tqdm(total=len(documents), desc="Processing documents") as pbar:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for task in tasks:
                future = executor.submit(multi_process_page, task)
                future.add_done_callback(lambda x: pbar.update(1))
                futures.append(future)

            for future in as_completed(futures):
                idx, page_results = future.result()
                document_results[idx] = page_results

    results: Dict[int, OcrPageResult] = {}

    for document_result in document_results:
        results.update(document_result)  # type: ignore

    return results


def gcp_documents_to_result(
    documents: List["documentai.Document"],
    provider_name: str,
    document_name: str,
    file_hash: str,
    *,
    mode: Literal["single", "multi"] = "single",
    exclude_bounding_poly: bool = False,
) -> Dict[int, OcrPageResult]:
    if mode == "single" or len(documents) == 1 or multiprocessing.cpu_count() == 1:
        logger.info("Using single process")
        return gcp_documents_to_result_single(
            documents,
            provider_name,
            document_name,
            file_hash,
            exclude_bounding_poly=exclude_bounding_poly,
        )
    elif mode == "multi":
        logger.info("Using multiprocessing")
        return gcp_documents_to_result_multi(
            documents,
            provider_name,
            exclude_bounding_poly=exclude_bounding_poly,
        )
    else:
        raise ValueError("Invalid mode for GCP document processing")


class GoogleOcrProvider(AbstractTaskProvider):
    name = "Google Document AI"
    capabilities = [
        CAPABILITIES.PAGE_TEXT_OCR.value,
        CAPABILITIES.PAGE_LAYOUT_OCR.value,
        CAPABILITIES.PAGE_RASTERIZATION.value,
    ]

    max_bytes_per_request = (
        1024 * 1024 * 20
    )  # 20MB is the max size for a single sync request
    max_page_count = 15

    def __init__(
        self,
        project_id: str,
        processor_id: str,
        *,
        service_account_info: Optional[dict] = None,
        service_account_file: Optional[str] = None,
        location: str = "us",
        max_workers: int = multiprocessing.cpu_count() * 2,
        exclude_bounding_poly: bool = False,
    ):
        if service_account_info is None and service_account_file is None:
            raise ValueError(
                "You must provide either service_account_info or service_account_file"
            )
        if service_account_info is not None and service_account_file is not None:
            raise ValueError(
                "You must provide either service_account_info or service_account_file, not both"
            )

        self.project_id = project_id
        self.processor_id = processor_id
        self.location = location

        self.max_workers = max_workers

        self.service_account_info = service_account_info
        self.service_account_file = service_account_file

        self.exclude_bounding_poly = exclude_bounding_poly

        try:
            from google.cloud import documentai

            self.documentai = documentai
        except ImportError:
            raise ImportError(
                "Please install 'google-cloud-documentai' to use the GoogleCloudVisionTextExtractionProvider"
            )

    @classmethod
    def from_service_account_file(
        cls,
        project_id: str,
        processor_id: str,
        service_account_file: str,
    ):
        return cls(
            project_id,
            processor_id,
            service_account_file=service_account_file,
        )

    def get_documentai_client(self, client_option_kwargs: dict = {}, **kwargs):
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
            return self.documentai.DocumentProcessorServiceClient.from_service_account_info(
                info=self.service_account_info,
                **base_service_client_kwargs,
            )
        elif self.service_account_file is not None:
            with service_account_file_read_lock:
                return self.documentai.DocumentProcessorServiceClient.from_service_account_file(
                    filename=self.service_account_file,
                    **base_service_client_kwargs,
                )
        else:
            raise ValueError("Missing account info and service file path.")

    def _process_document_sync(self, document: Document):
        """
        Split the document into chunks of 15 pages or less, and process each chunk
        synchronously.
        """
        client = self.get_documentai_client()
        processor_name = client.processor_path(
            project=self.project_id,
            location=self.location,
            processor=self.processor_id,
        )

        documents: List["documentai.Document"] = []

        file_bytes = document.get_bytes()

        @default_retry_decorator
        def process_byte_chunk(split_bytes: bytes) -> "documentai.Document":
            raw_document = self.documentai.RawDocument(
                content=split_bytes,
                mime_type="application/pdf",
            )

            field_mask = (
                "text,pages.layout,pages.words,pages.lines,pages.tokens,pages.blocks"
            )

            request = self.documentai.ProcessRequest(
                name=processor_name, raw_document=raw_document, field_mask=field_mask
            )

            result = client.process_document(request=request)

            return result.document

        with tqdm.tqdm(
            total=len(file_bytes), unit="B", unit_scale=True, desc="Processing document"
        ) as pbar:
            for split_bytes in pdf_split_iter_with_max_bytes(
                file_bytes,
                max_page_count=self.max_page_count,
                max_bytes=self.max_bytes_per_request,
            ):
                document = process_byte_chunk(split_bytes)

                documents.append(document)

                pbar.update(len(split_bytes))

        return gcp_documents_to_result(
            documents,
            self.name,
            document_name=document.name,
            file_hash=document.document_hash,
            exclude_bounding_poly=self.exclude_bounding_poly,
        )

    def _process_document_concurrent(
        self,
        document: Document,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        include_raster: bool = False,
    ):
        # Process page chunks concurrently
        client = self.get_documentai_client()
        processor_name = client.processor_path(
            project=self.project_id,
            location=self.location,
            processor=self.processor_id,
        )

        logger.info("Splitting document into chunks...")
        document_byte_splits = list(
            pdf_split_iter_with_max_bytes(
                document.get_bytes(),
                max_page_count=self.max_page_count,
                max_bytes=self.max_bytes_per_request,
            )
        )

        max_workers = min(len(document_byte_splits), self.max_workers)

        @default_retry_decorator
        def process_byte_chunk(split_bytes: bytes):
            raw_document = self.documentai.RawDocument(
                content=split_bytes,
                mime_type="application/pdf",
            )

            field_mask = (
                "text,pages.layout,pages.words,pages.lines,pages.tokens,pages.blocks"
            )

            request = self.documentai.ProcessRequest(
                name=processor_name, raw_document=raw_document, field_mask=field_mask
            )

            result = client.process_document(request=request)

            document = result.document

            return document

        logger.info(f"Processing {len(document_byte_splits)} chunks...")
        with tqdm.tqdm(
            total=len(document_byte_splits), desc="Processing document"
        ) as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(process_byte_chunk, split): index
                    for index, split in enumerate(document_byte_splits)
                }

                documents: List["documentai.Document"] = [None] * len(  # type: ignore
                    document_byte_splits
                )

                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    documents[index] = future.result()
                    pbar.update(1)

        logger.info("Recombining")
        return gcp_documents_to_result(
            documents,
            self.name,
            document_name=document.name,
            file_hash=document.document_hash,
            exclude_bounding_poly=self.exclude_bounding_poly,
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
