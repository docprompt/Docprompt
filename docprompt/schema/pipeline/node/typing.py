from typing import TypeVar

from ..metadata import BaseMetadata

ImageNodeMetadata = TypeVar("ImageNodeMetadata", bound=BaseMetadata)
PageNodeMetadata = TypeVar("PageNodeMetadata", bound=BaseMetadata)
DocumentNodeMetadata = TypeVar("DocumentNodeMetadata", bound=BaseMetadata)
DocumentCollectionMetadata = TypeVar("DocumentCollectionMetadata", bound=BaseMetadata)
