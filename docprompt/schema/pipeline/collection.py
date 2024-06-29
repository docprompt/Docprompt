from typing import Generic, List, TypeVar

from pydantic import BaseModel, Field

from .metadata import BaseMetadata
from .node import DocumentNode, DocumentNodeMetadata, PageNodeMetadata

DocumentCollectionMetadata = TypeVar("DocumentCollectionMetadata", bound=BaseMetadata)


class DocumentCollection(
    BaseModel,
    Generic[DocumentCollectionMetadata, DocumentNodeMetadata, PageNodeMetadata],
):
    """
    Represents a collection of documents with some common metadata
    """

    document_nodes: List[DocumentNode[DocumentNodeMetadata, PageNodeMetadata]]
    metadata: DocumentCollectionMetadata = Field(..., default_factory=dict)
