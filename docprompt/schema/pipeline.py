from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, Field, PositiveInt

from docprompt.tasks.base import ResultContainer
from docprompt.tasks.ocr.result import OcrPageResult

from .document import Document

DocumentCollectionMetadata = TypeVar("DocumentCollectionMetadata", bound=BaseModel)
DocumentNodeMetadata = TypeVar("DocumentNodeMetadata", bound=BaseModel)
PageNodeMetadata = TypeVar("PageNodeMetadata", bound=BaseModel)


class PageNode(BaseModel, Generic[PageNodeMetadata]):
    """
    Represents a single page in a document, with some metadata
    """

    document: "DocumentNode" = Field(exclude=True, repr=False)
    page_number: PositiveInt = Field(description="The page number")
    metadata: Optional[PageNodeMetadata] = Field(
        description="Application-specific metadata for the page", default=None
    )

    ocr_results: ResultContainer[OcrPageResult] = Field(
        default_factory=lambda: ResultContainer(),
        description="The OCR results for the page",
        repr=False,
    )
    table_extraction_results: ResultContainer = Field(
        default_factory=lambda: ResultContainer(),
        description="The table extraction results for the page",
        repr=False,
    )


class DocumentNode(BaseModel, Generic[DocumentNodeMetadata, PageNodeMetadata]):
    """
    Represents a single document, with some metadata
    """

    document: Document
    page_nodes: list[PageNode[PageNodeMetadata]] = Field(
        description="The pages in the document", default_factory=list, repr=False
    )
    metadata: Optional[DocumentNodeMetadata] = Field(
        description="Application-specific metadata for the document", default=None
    )

    def __len__(self):
        return len(self.page_nodes)

    def __getitem__(self, index):
        return self.page_nodes[index]

    def __iter__(self):
        return iter(self.page_nodes)

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

    document_nodes: list[DocumentNode[DocumentNodeMetadata, PageNodeMetadata]]
    metadata: DocumentCollectionMetadata
