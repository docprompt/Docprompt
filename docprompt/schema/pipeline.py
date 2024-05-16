import base64
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    TYPE_CHECKING,
    Union,
)

from pydantic import BaseModel, Field, PositiveInt, PrivateAttr

from docprompt.rasterize import AspectRatioRule, ResizeModes, process_raster_image
from docprompt.tasks.base import ResultContainer
from docprompt.tasks.ocr.result import OcrPageResult
from PIL import Image

if TYPE_CHECKING:
    from docprompt.provenance.search import DocumentProvenanceLocator

from .document import Document

DocumentCollectionMetadata = TypeVar("DocumentCollectionMetadata", bound=BaseModel)
DocumentNodeMetadata = TypeVar("DocumentNodeMetadata", bound=BaseModel)
PageNodeMetadata = TypeVar("PageNodeMetadata", bound=BaseModel)


class PageRasterizer:
    def __init__(self, raster_cache: Dict[str, bytes], owner: "PageNode"):
        self.raster_cache = raster_cache
        self.owner = owner

    def _construct_cache_key(self, **kwargs):
        key = ""

        for k, v in kwargs.items():
            if isinstance(v, Iterable) and not isinstance(v, str):
                v = ",".join(str(i) for i in v)
            elif isinstance(v, bool):
                v = str(v).lower()
            else:
                v = str(v)

            key += f"{k}={v},"

        return key

    def rasterize(
        self,
        name: Optional[str] = None,
        *,
        return_mode: Literal["bytes", "pil"] = "bytes",
        dpi: int = 100,
        downscale_size: Optional[Tuple[int, int]] = None,
        resize_mode: ResizeModes = "thumbnail",
        resize_aspect_ratios: Optional[Iterable[AspectRatioRule]] = None,
        do_convert: bool = False,
        image_convert_mode: str = "L",
        do_quantize: bool = False,
        quantize_color_count: int = 8,
        max_file_size_bytes: Optional[int] = None,
    ) -> Union[bytes, Image.Image]:
        cache_key = (
            name
            if name
            else self._construct_cache_key(
                dpi=dpi,
                downscale_size=downscale_size,
                resize_mode=resize_mode,
                resize_aspect_ratios=resize_aspect_ratios,
                do_convert=do_convert,
                image_convert_mode=image_convert_mode,
                do_quantize=do_quantize,
                quantize_color_count=quantize_color_count,
                max_file_size_bytes=max_file_size_bytes,
            )
        )

        if cache_key in self.raster_cache:
            rastered = self.raster_cache[cache_key]
        else:
            rastered = self.owner.document.document.rasterize_page(
                self.owner.page_number,
                dpi=dpi,
                downscale_size=downscale_size,
                resize_mode=resize_mode,
                resize_aspect_ratios=resize_aspect_ratios,
                do_convert=do_convert,
                image_convert_mode=image_convert_mode,
                do_quantize=do_quantize,
                quantize_color_count=quantize_color_count,
                max_file_size_bytes=max_file_size_bytes,
            )

            self.raster_cache[cache_key] = rastered

        if return_mode == "pil" and isinstance(rastered, bytes):
            from PIL import Image
            import io

            return Image.open(io.BytesIO(rastered))
        elif return_mode == "bytes" and isinstance(rastered, bytes):
            return rastered

        return rastered

    def rasterize_to_data_uri(
        self,
        name: str,
        *,
        dpi: int = 100,
        downscale_size: Optional[Tuple[int, int]] = None,
        resize_mode: ResizeModes = "thumbnail",
        resize_aspect_ratios: Optional[Iterable[AspectRatioRule]] = None,
        do_convert: bool = False,
        image_convert_mode: str = "L",
        do_quantize: bool = False,
        quantize_color_count: int = 8,
        max_file_size_bytes: Optional[int] = None,
    ) -> str:
        rastered = self.rasterize(
            name,
            return_mode="bytes",
            dpi=dpi,
            downscale_size=downscale_size,
            resize_mode=resize_mode,
            resize_aspect_ratios=resize_aspect_ratios,
            do_convert=do_convert,
            image_convert_mode=image_convert_mode,
            do_quantize=do_quantize,
            quantize_color_count=quantize_color_count,
            max_file_size_bytes=max_file_size_bytes,
        )

        return f"data:image/png;base64,{base64.b64encode(rastered).decode('utf-8')}"

    def clear_cache(self):
        self.raster_cache.clear()

    def pop(self, name: str, default=None):
        return self.raster_cache.pop(name, default=default)


def process_bytes(
    rastered: bytes,
    *,
    resize_width: Optional[int] = None,
    resize_height: Optional[int] = None,
    resize_mode: ResizeModes = "thumbnail",
    aspect_ratios: Optional[Iterable[AspectRatioRule]] = None,
    do_convert: bool = False,
    image_convert_mode: str = "L",
    do_quantize: bool = False,
    quantize_color_count: int = 8,
    max_file_size_bytes: Optional[int] = None,
):
    rastered = process_raster_image(
        rastered,
        resize_width=resize_width,
        resize_height=resize_height,
        resize_mode=resize_mode,
        resize_aspect_ratios=aspect_ratios,
        do_convert=do_convert,
        do_quantize=do_quantize,
        image_convert_mode=image_convert_mode,
        quantize_color_count=quantize_color_count,
        max_file_size_bytes=max_file_size_bytes,
    )

    return rastered


class DocumentRasterizer:
    def __init__(self, owner: "DocumentNode"):
        self.owner = owner

    def rasterize(
        self,
        name: str,
        *,
        return_mode: Literal["bytes", "pil"] = "bytes",
        dpi: int = 100,
        downscale_size: Optional[Tuple[int, int]] = None,
        resize_mode: ResizeModes = "thumbnail",
        resize_aspect_ratios: Optional[Iterable[AspectRatioRule]] = None,
        do_convert: bool = False,
        image_convert_mode: str = "L",
        do_quantize: bool = False,
        quantize_color_count: int = 8,
        max_file_size_bytes: Optional[int] = None,
        render_grayscale: bool = False,
    ) -> List[Union[bytes, Image.Image]]:
        images = self.owner.document.rasterize_pdf(
            dpi=dpi,
            downscale_size=downscale_size,
            resize_mode=resize_mode,
            resize_aspect_ratios=resize_aspect_ratios,
            do_convert=do_convert,
            image_convert_mode=image_convert_mode,
            do_quantize=do_quantize,
            quantize_color_count=quantize_color_count,
            max_file_size_bytes=max_file_size_bytes,
            render_grayscale=render_grayscale,
            return_mode=return_mode,
        )

        for page_number, image in images.items():
            page_node = self.owner.page_nodes[page_number - 1]

            page_node._raster_cache[name] = image

        return list(images.values())


class PageNode(BaseModel, Generic[PageNodeMetadata]):
    """
    Represents a single page in a document, with some metadata
    """

    document: "DocumentNode" = Field(exclude=True, repr=False)
    page_number: PositiveInt = Field(description="The page number")
    metadata: Optional[PageNodeMetadata] = Field(
        description="Application-specific metadata for the page", default=None
    )
    extra: Dict[str, Any] = Field(
        description="Extra data that can be stored on the page node",
        default_factory=dict,
    )

    ocr_results: ResultContainer[OcrPageResult] = Field(
        default_factory=lambda: ResultContainer(),
        description="The OCR results for the page",
        repr=False,
    )

    _raster_cache: Dict[str, bytes] = PrivateAttr(default_factory=dict)

    def __getstate__(self):
        state = super().__getstate__()

        state["__pydantic_private__"]["_raster_cache"] = {}

        return state

    @property
    def rasterizer(self):
        return PageRasterizer(self._raster_cache, self)

    def search(
        self, query: str, refine_to_words: bool = True, require_exact_match: bool = True
    ):
        return self.document.locator.search(
            query,
            page_number=self.page_number,
            refine_to_word=refine_to_words,
            require_exact_match=require_exact_match,
        )


class DocumentNode(BaseModel, Generic[DocumentNodeMetadata, PageNodeMetadata]):
    """
    Represents a single document, with some metadata
    """

    document: Document
    page_nodes: List[PageNode[PageNodeMetadata]] = Field(
        description="The pages in the document", default_factory=list, repr=False
    )
    metadata: Optional[DocumentNodeMetadata] = Field(
        description="Application-specific metadata for the document", default=None
    )

    _locator: Optional["DocumentProvenanceLocator"] = PrivateAttr(default=None)

    def __getstate__(self):
        state = super().__getstate__()

        state["__pydantic_private__"]["_locator"] = None

        return state

    def __len__(self):
        return len(self.page_nodes)

    def __getitem__(self, index):
        return self.page_nodes[index]

    def __iter__(self):
        return iter(self.page_nodes)

    @property
    def rasterizer(self):
        return DocumentRasterizer(self)

    @property
    def locator(self):
        if self._locator is None:
            self.refresh_locator()

        return self._locator

    def refresh_locator(self):
        """
        Refreshes the locator for this document node
        """
        from docprompt.provenance.search import DocumentProvenanceLocator

        if any(not page.ocr_results.result for page in self.page_nodes):
            raise ValueError(
                "Cannot create a locator for a document node with missing OCR results"
            )

        self._locator = DocumentProvenanceLocator.from_document_node(self)

        return self.locator

    @classmethod
    def from_document(
        cls,
        document: Document,
        document_metadata: Optional[DocumentNodeMetadata] = None,
    ):
        document_node: "DocumentNode[DocumentNodeMetadata, PageNodeMetadata]" = (
            DocumentNode(document=document, metadata=document_metadata)
        )

        for page_number in range(1, len(document) + 1):
            document_node.page_nodes.append(
                PageNode(document=document_node, page_number=page_number)
            )

        return document_node

    @property
    def file_hash(self):
        return self.document.document_hash

    @property
    def document_name(self):
        return self.document.name


class DocumentCollection(
    BaseModel,
    Generic[DocumentCollectionMetadata, DocumentNodeMetadata, PageNodeMetadata],
):
    """
    Represents a collection of documents with some common metadata
    """

    document_nodes: List[DocumentNode[DocumentNodeMetadata, PageNodeMetadata]]
    metadata: DocumentCollectionMetadata
