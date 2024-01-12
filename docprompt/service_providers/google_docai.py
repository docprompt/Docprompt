import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import TYPE_CHECKING, Dict, Literal, Optional, Union

import tqdm

from docprompt.schema.document import Document
from docprompt.schema.layout import BoundingPoly, Geometry, NormBBox, Point, SegmentLevels, TextBlock, TextSpan
from docprompt.schema.operations import PageResult, PageTextExtractionOutput
from docprompt.service_providers.base import ProviderResult
from docprompt.service_providers.types import OPERATIONS
from docprompt.utils.splitter import pdf_split_iter

from .base import BaseProvider, ProviderResult

if TYPE_CHECKING:
    from google.cloud import documentai

    from docprompt.schema.document import Document


orientation_rotation_mapping = {
    0: 0,
    1: 0,
    2: 90,
    3: 180,
    4: -90,
}

service_account_file_read_lock = Lock()


def bounding_poly_from_layout(layout: Union["documentai.Document.Page.Layout", "documentai.Document.Page.Token"]):
    return BoundingPoly(
        normalized_vertices=[Point(x=vertex.x, y=vertex.y) for vertex in layout.bounding_poly.normalized_vertices]
    )


def geometry_from_layout(layout: Union["documentai.Document.Page.Layout", "documentai.Document.Page.Token"]):
    bounding_poly = bounding_poly_from_layout(layout)

    bounding_box = NormBBox.from_bounding_poly(bounding_poly)

    return Geometry(
        bounding_box=bounding_box,
        bounding_poly=bounding_poly,
    )


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
) -> list[TextSpan]:
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
) -> list[TextBlock]:
    text_blocks = []

    type_mapping: Dict[str, SegmentLevels] = {
        "line": "line",
        "paragraph": "paragraph",
        "block": "block",
        "token": "word",
    }

    orientation_mapping = {
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
        geometry = geometry_from_layout(layout)
        confidence = layout.confidence
        orientation = orientation_mapping.get(layout.orientation, "UP")

        text_spans = text_spans_from_layout(layout, level="page", offset=offset_low)

        block_type = type_mapping[type]
        text_blocks.append(
            TextBlock(
                text=block_text,
                type=block_type,
                geometry=geometry,
                confidence=confidence,
                direction=orientation,
                text_spans=text_spans,
            )
        )

    return text_blocks


class GoogleDocumentAIProvider(BaseProvider):
    name = "GoogleDocumentAIProvider"

    max_bytes_per_request = 1024 * 1024 * 20  # 20MB is the max size for a single sync request
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
        **kwargs,
    ):
        if service_account_info is None and service_account_file is None:
            raise ValueError("You must provide either service_account_info or service_account_file")
        if service_account_info is not None and service_account_file is not None:
            raise ValueError("You must provide either service_account_info or service_account_file, not both")

        self.project_id = project_id
        self.processor_id = processor_id
        self.location = location

        self.max_workers = max_workers

        self.service_account_info = service_account_info
        self.service_account_file = service_account_file

        try:
            from google.cloud import documentai

            self.documentai = documentai
        except ImportError:
            raise ImportError(
                "Please install 'google-cloud-documentai' to use the GoogleCloudVisionTextExtractionProvider"
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

    @property
    def capabilities(self) -> list[OPERATIONS]:
        return [
            OPERATIONS.TEXT_EXTRACTION,
            OPERATIONS.LAYOUT_ANALYSIS,
            OPERATIONS.IMAGE_PROCESSING,
        ]

    def _gcp_documents_to_result(
        self, documents: list["documentai.Document"], get_images: bool = False
    ) -> ProviderResult:
        page_offset = 1  # We want pages to be 1-indexed

        page_results = []

        for document in documents:
            for doc_page_num, page in enumerate(document.pages):
                layout = page.layout

                page_text = text_from_layout(layout, document.text)

                word_boxes = text_blocks_from_page(page, document.text, "token")
                line_boxes = text_blocks_from_page(page, document.text, "line")
                block_boxes = text_blocks_from_page(page, document.text, "block")

                ocr_result = PageTextExtractionOutput(
                    text=page_text,
                    words=word_boxes,
                    lines=line_boxes,
                    blocks=block_boxes,
                )

                page_result = PageResult(
                    provider_name=self.name,
                    page_number=page_offset + doc_page_num,
                    ocr_result=ocr_result,
                )

                page_results.append(page_result)

            page_offset += doc_page_num + 1

        return ProviderResult(
            provider_name="GoogleDocumentAIProvider",
            page_results=page_results,
        )

    def _process_document_sync(self, file_bytes: bytes):
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

        documents = []

        with tqdm.tqdm(total=len(file_bytes), unit="B", unit_scale=True, desc="Processing document") as pbar:
            for split_bytes in pdf_split_iter(
                file_bytes, max_page_count=self.max_page_count, max_bytes=self.max_bytes_per_request
            ):
                raw_document = self.documentai.RawDocument(
                    content=split_bytes,
                    mime_type="application/pdf",
                )

                field_mask = "text,pages.layout,pages.words,pages.lines,pages.tokens"

                request = self.documentai.ProcessRequest(
                    name=processor_name, raw_document=raw_document, field_mask=field_mask
                )

                result = client.process_document(request=request)

                documents.append(result.document)

                pbar.update(len(split_bytes))

        return self._gcp_documents_to_result(documents)

    def _process_document_concurrent(self, file_bytes: bytes):
        # Process page chunks concurrently
        client = self.get_documentai_client()
        processor_name = client.processor_path(
            project=self.project_id,
            location=self.location,
            processor=self.processor_id,
        )

        print("Splitting document into chunks...")
        document_byte_splits = list(
            pdf_split_iter(file_bytes, max_page_count=self.max_page_count, max_bytes=self.max_bytes_per_request)
        )

        max_workers = min(len(document_byte_splits), self.max_workers)

        def process_byte_chunk(split_bytes: bytes):
            raw_document = self.documentai.RawDocument(
                content=split_bytes,
                mime_type="application/pdf",
            )

            field_mask = "text,pages.layout,pages.words,pages.lines,pages.tokens"

            request = self.documentai.ProcessRequest(
                name=processor_name, raw_document=raw_document, field_mask=field_mask
            )

            result = client.process_document(request=request)

            return result.document

        print(f"Processing {len(document_byte_splits)} chunks...")
        with tqdm.tqdm(total=len(document_byte_splits), desc="Processing document") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(process_byte_chunk, split): index
                    for index, split in enumerate(document_byte_splits)
                }

                documents = [None] * len(document_byte_splits)

                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    documents[index] = future.result()
                    pbar.update(1)

        print("Recombining")
        return self._gcp_documents_to_result(documents)

    def _call(self, document: Document, pages=...) -> ProviderResult:
        return self._process_document_concurrent(document.get_bytes())
