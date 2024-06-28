from typing import Dict

from pydantic import BaseModel

from docprompt.schema.pipeline import DocumentNode
from docprompt.tasks.base import AbstractPageTaskProvider


class MarkerizeResult(BaseModel):
    raw_markdown: str


class BaseMarkerizeProvider(AbstractPageTaskProvider[None, MarkerizeResult]):
    class Meta:
        abstract = True

    def contribute_to_document_node(
        self, document_node: DocumentNode, results: Dict[int, MarkerizeResult]
    ) -> None:
        for page_number, page_result in results.items():
            document_node.page_nodes[page_number - 1].extra["raw_markdown"] = (
                page_result.raw_markdown
            )
