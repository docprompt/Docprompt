from typing import TypeVar

from ..metadata import BaseMetadata

PageNodeMetadata = TypeVar("PageNodeMetadata", bound=BaseMetadata)
DocumentNodeMetadata = TypeVar("DocumentNodeMetadata", bound=BaseMetadata)
