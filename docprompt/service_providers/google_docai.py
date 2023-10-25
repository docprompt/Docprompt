from typing import TYPE_CHECKING, Dict, Literal, Optional, Union

from docprompt.schema.document import Document
from docprompt.schema.layout import BoundingPoly, Geometry, NormBBox, Point, SegmentLevels, TextBlock
from docprompt.service_providers.base import ProviderResult
from docprompt.service_providers.types import OPERATIONS, ImageProcessResult, PageTextExtractionOutput
from docprompt.utils.splitter import pdf_split_iter

from .base import BaseProvider, PageResult, ProviderResult

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
    start = getattr(layout.text_anchor.text_segments[0], "start_index", 0)
    end = layout.text_anchor.text_segments[0].end_index

    return document_text[start - offset : end - offset]


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

    for item in getattr(page, f"{type}s"):
        layout = item.layout
        block_text = text_from_layout(layout, document_text)
        geometry = geometry_from_layout(layout)
        confidence = layout.confidence
        orientation = orientation_mapping.get(layout.orientation, "UP")

        block_type = type_mapping[type]
        text_blocks.append(
            TextBlock(
                text=block_text,
                type=block_type,
                geometry=geometry,
                confidence=confidence,
                direction=orientation,
            )
        )

    return text_blocks


class GoogleDocumentAIProvider(BaseProvider):
    name = "GoogleDocumentAIProvider"

    def __init__(
        self,
        project_id: str,
        processor_id: str,
        *,
        service_account_info: Optional[dict] = None,
        service_account_file: Optional[str] = None,
        location: str = "us",
    ):
        if service_account_info is None and service_account_file is None:
            raise ValueError("You must provide either service_account_info or service_account_file")
        if service_account_info is not None and service_account_file is not None:
            raise ValueError("You must provide either service_account_info or service_account_file, not both")

        self.project_id = project_id
        self.processor_id = processor_id
        self.location = location

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
                    blocks={
                        "word": word_boxes,
                        "line": line_boxes,
                        "block": block_boxes,
                    },
                )

                if get_images:
                    image_process_result = ImageProcessResult(
                        type="regularize",
                        image_data=page.image.content,
                        width=page.image.width,
                        height=page.image.height,
                    )
                else:
                    image_process_result = None

                page_result = PageResult(
                    provider_name=self.name,
                    page_number=page_offset + doc_page_num,
                    ocr_result=ocr_result,
                    image_process_result=image_process_result,
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
        max_page_count = 15

        client = self.get_documentai_client()
        processor_name = client.processor_path(
            project=self.project_id,
            location=self.location,
            processor=self.processor_id,
        )

        documents = []

        max_bytes = 1024 * 1024 * 20  # 20MB is the max size for a single sync request

        for split_bytes in pdf_split_iter(file_bytes, max_page_count=max_page_count, max_bytes=max_bytes):
            raw_document = self.documentai.RawDocument(
                content=split_bytes,
                mime_type="application/pdf",
            )

            request = self.documentai.ProcessRequest(name=processor_name, raw_document=raw_document)

            result = client.process_document(request=request)
            documents.append(result.document)

        return self._gcp_documents_to_result(documents)

    def _call(self, document: Document, pages=...) -> ProviderResult:
        return self._process_document_sync(document.file_bytes)
