import base64
import json
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    ForwardRef,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from PIL import Image
from pydantic import BaseModel, Field, PositiveInt, PrivateAttr
from typing_extensions import Self

from docprompt.rasterize import AspectRatioRule, ResizeModes, process_raster_image
from docprompt.storage import FileSidecarsPathManager, FileSystemManager
from docprompt.tasks.base import ResultContainer
from docprompt.tasks.ocr.result import OcrPageResult

if TYPE_CHECKING:
    from docprompt.provenance.search import DocumentProvenanceLocator


from .document import Document
from .metadata import BaseMetadata

DocumentCollectionMetadata = TypeVar("DocumentCollectionMetadata", bound=BaseMetadata)
DocumentNodeMetadata = TypeVar("DocumentNodeMetadata", bound=BaseMetadata)
PageNodeMetadata = TypeVar("PageNodeMetadata", bound=BaseMetadata)


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
            import io

            from PIL import Image

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

    def propagate_cache(self, name: str, rasters: Dict[int, Union[bytes, Image.Image]]):
        """
        Should be one-indexed
        """
        for page_number, raster in rasters.items():
            page_node = self.owner.page_nodes[page_number - 1]

            page_node._raster_cache[name] = raster


class PageNode(BaseModel, Generic[PageNodeMetadata]):
    """
    Represents a single page in a document, with some metadata
    """

    document: "DocumentNode" = Field(exclude=True, repr=False)
    page_number: PositiveInt = Field(description="The page number")
    metadata: PageNodeMetadata = Field(
        description="Application-specific metadata for the page",
        default_factory=BaseMetadata,
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
    metadata: DocumentNodeMetadata = Field(
        description="Application-specific metadata for the document",
        default_factory=BaseMetadata,
    )

    _locator: Optional["DocumentProvenanceLocator"] = PrivateAttr(default=None)

    _persistance_path: Optional[str] = PrivateAttr(default=None)

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
        page_metadata: Optional[List[PageNodeMetadata]] = None,
    ):
        document_node: "DocumentNode[DocumentNodeMetadata, PageNodeMetadata]" = cls(
            document=document,
        )
        document_node.metadata = document_metadata or cls.metadata_class().from_owner(
            document_node, **{}
        )

        if page_metadata is not None and len(page_metadata) != len(document):
            raise ValueError(
                "The number of page metadata items must match the number of pages in the document."
            )

        for page_number in range(1, len(document) + 1):
            if page_metadata is not None:
                page_node = PageNode(
                    document=document_node,
                    page_number=page_number,
                    metadata=page_metadata[page_number - 1],
                )
            else:
                page_node = PageNode(document=document_node, page_number=page_number)

            document_node.page_nodes.append(page_node)

        return document_node

    @property
    def file_hash(self):
        return self.document.document_hash

    @property
    def document_name(self):
        return self.document.name

    @classmethod
    def metadata_class(cls) -> Type[BaseMetadata]:
        """Get the metadata class for instantiating metadata from the model."""

        fields = cls.model_fields

        # NOTE: The indexing is important here, and relies on the generic type being
        # the SECOND of the two arguments in the `Union` annotation
        metadata_field_annotation = fields["metadata"].annotation

        # If no override has been provided to the metadata model, we want to retrieve
        # it as a TypedDict
        if metadata_field_annotation == DocumentNodeMetadata:
            return BaseMetadata

        if isinstance(metadata_field_annotation, ForwardRef):
            raise ValueError(
                "You cannot define DocumentNode with a ForwardRef for Generic metadata model types."
            )

        # Get the overriden Generic type of th DocumentNodeMetadata
        return metadata_field_annotation

    @classmethod
    def page_metadata_class(cls) -> Type[Union[dict, BaseModel]]:
        """Get the metadata class for the page nodes in the document."""
        fields = cls.model_fields

        # NOTE: The indexing is important here, and it allows us to get the type of each
        # page node in the `List` annotation
        page_nodes_field_class = fields["page_nodes"].annotation.__args__[0]

        # NOTE: The indexing is important here, and relies on the generic type being
        # the SECOND of the two arguments in the `Union` annotation
        page_node_metadata_field_annotation = page_nodes_field_class.model_fields[
            "metadata"
        ].annotation

        if page_node_metadata_field_annotation == PageNodeMetadata:
            return BaseMetadata

        if isinstance(page_node_metadata_field_annotation, ForwardRef):
            raise ValueError(
                "You cannot define PageNode with a ForwardRef for Generic metadata model types."
            )

        return page_node_metadata_field_annotation

    @property
    def persistance_path(self):
        """The base path to storage location."""
        return self._persistance_path

    @persistance_path.setter
    def persistance_path(self, path: str):
        """Set the base path to storage location."""
        self._persistance_path = path

    @classmethod
    def from_storage(cls, path: str, file_hash: str, **kwargs) -> Self:
        """Load the document node from storage.

        Args:
            path (str): The base path to storage location.
                - Example (S3): "s3://bucket-name/key/to/folder"
                - Example (Local FS): "/tmp/docprompt/storage"
            file_hash (str): The hash of the document.
            **kwargs: Additional keyword arguments for fsspec FileSystem

        Returns:
            DocumentNode: The loaded document node.
        """

        fs_manager = FileSystemManager(path, **kwargs)

        pdf_bytes, metadata_bytes, page_metadata_bytes = fs_manager.read(
            file_hash, **kwargs
        )

        doc = Document.from_bytes(pdf_bytes, name=fs_manager.get_pdf_name(file_hash))
        node = cls.from_document(doc)

        if metadata_bytes:
            metadata_json = json.loads(metadata_bytes.decode("utf-8"))
            metadata = cls.metadata_class().from_owner(node, **metadata_json)
        else:
            metadata = cls.metadata_class().from_owner(node, **{})

        if page_metadata_bytes:
            page_metadata_json = [
                json.loads(page_str)
                for page_str in json.loads(page_metadata_bytes.decode("utf-8"))
            ]
            page_metadata = [
                cls.page_metadata_class().from_owner(node, **page)
                for page in page_metadata_json
            ]
        else:
            page_metadata = [
                cls.page_metadata_class().from_owner(node, **{})
                for _ in range(len(doc))
            ]

        # Store the metadata on the node and page nodes
        node.metadata = metadata
        for page, meta in zip(node.page_nodes, page_metadata):
            page.metadata = meta

        # Make sure to set the persistance path on the node
        node.persistance_path = path

        return node

    def persist(self, path: Optional[str] = None, **kwargs) -> FileSidecarsPathManager:
        """Persist a document node to storage.

        Args:
            path (Optional[str]): Overwrites the current `persistance_path` property
                - If `persistance_path` is not currently set, path must be provided.
            **kwargs: Additional keyword arguments for fsspec FileSystem

        Returns:
            FileSidecarsPathManager: The file path manager for the persisted document node.
        """

        path = path or self.persistance_path

        if path is None:
            raise ValueError("The path must be provided to persist the document node.")

        # Make sure to update the persistance path
        self.persistance_path = path

        fs_manager = FileSystemManager(path, **kwargs)

        pdf_bytes = self.document.get_bytes()
        metadata_bytes = bytes(self.metadata.model_dump_json(), encoding="utf-8")
        page_metadata_bytes = bytes(
            json.dumps([page.metadata.model_dump_json() for page in self.page_nodes]),
            encoding="utf-8",
        )

        return fs_manager.write(
            pdf_bytes, metadata_bytes, page_metadata_bytes, **kwargs
        )


class DocumentCollection(
    BaseModel,
    Generic[DocumentCollectionMetadata, DocumentNodeMetadata, PageNodeMetadata],
):
    """
    Represents a collection of documents with some common metadata
    """

    document_nodes: List[DocumentNode[DocumentNodeMetadata, PageNodeMetadata]]
    metadata: DocumentCollectionMetadata = Field(..., default_factory=dict)
