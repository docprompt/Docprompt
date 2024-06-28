from typing import Dict

from docprompt.schema.pipeline import DocumentNode
from docprompt.tasks.base import AbstractPageTaskProvider
from docprompt.tasks.capabilities import PageLevelCapabilities

from .schema import TableExtractionPageResult


class BaseTableExtractionProvider(
    AbstractPageTaskProvider[None, TableExtractionPageResult]
):
    capabilities = [
        PageLevelCapabilities.PAGE_TABLE_EXTRACTION,
        PageLevelCapabilities.PAGE_TABLE_IDENTIFICATION,
    ]

    class Meta:
        abstract = True

    def contribute_to_document_node(
        self, document_node: DocumentNode, results: Dict[int, TableExtractionPageResult]
    ) -> None:
        for page_number, page_result in results.items():
            document_node.page_nodes[page_number - 1].extra["extracted_tables"] = (
                page_result.tables
            )
