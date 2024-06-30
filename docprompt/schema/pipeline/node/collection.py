from typing import TYPE_CHECKING, Generic, List

from pydantic import BaseModel, Field

from .typing import DocumentCollectionMetadata, DocumentNodeMetadata, PageNodeMetadata

if TYPE_CHECKING:
    from .document import DocumentNode


class DocumentCollection(
    BaseModel,
    Generic[DocumentCollectionMetadata, DocumentNodeMetadata, PageNodeMetadata],
):
    """
    Represents a collection of documents with some common metadata
    """

    document_nodes: List["DocumentNode[DocumentNodeMetadata, PageNodeMetadata]"]
    metadata: DocumentCollectionMetadata = Field(..., default_factory=dict)
