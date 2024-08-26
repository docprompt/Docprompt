import json
from typing import (
    TYPE_CHECKING,
    ForwardRef,
    Generic,
    List,
    Optional,
    Type,
    Union,
)

from pydantic import BaseModel, Field, PrivateAttr
from typing_extensions import Self

from docprompt.schema.document import PdfDocument
from docprompt.schema.pipeline.metadata import BaseMetadata
from docprompt.schema.pipeline.node.page import PageNode
from docprompt.schema.pipeline.rasterizer import DocumentRasterizer
from docprompt.storage import FileSidecarsPathManager, FileSystemManager

from .base import BaseNode
from .typing import DocumentNodeMetadata, PageNodeMetadata

if TYPE_CHECKING:
    from docprompt.provenance.search import DocumentProvenanceLocator


class DocumentNode(BaseNode, Generic[DocumentNodeMetadata, PageNodeMetadata]):
    """
    Represents a single document, with some metadata
    """

    document: PdfDocument
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

        has_all_ocr_results = all(x.ocr_results for x in self.page_nodes)

        if not has_all_ocr_results:
            raise ValueError(
                "Cannot create a locator for a document without any OCR results."
            )

        self._locator = DocumentProvenanceLocator.from_document_node(self)

        return self.locator

    @classmethod
    def from_document(
        cls,
        document: PdfDocument,
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

        doc = PdfDocument.from_bytes(pdf_bytes, name=fs_manager.get_pdf_name(file_hash))
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
                cls.page_metadata_class()(**page) for page in page_metadata_json
            ]
        else:
            page_metadata = [cls.page_metadata_class()(**{}) for _ in range(len(doc))]

        # Store the metadata on the node and page nodes
        node.metadata = metadata
        for page, meta in zip(node.page_nodes, page_metadata):
            meta.owner = page
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
