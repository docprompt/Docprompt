from typing import Dict
from docprompt.schema.pipeline import DocumentNode
from docprompt.tasks.base import AbstractLanguageModelTaskProvider, CAPABILITIES
from .result import TableExtractionPageResult


class BaseTableExtractionProvider(AbstractLanguageModelTaskProvider):
    capabilities = [
        CAPABILITIES.PAGE_TABLE_IDENTIFICATION.value,
        CAPABILITIES.PAGE_TABLE_EXTRACTION.value,
    ]

    def contribute_to_document_node(
        self, document_node: DocumentNode, results: Dict[int, TableExtractionPageResult]
    ) -> None:
        for page_number, page_result in results.items():
            document_node.page_nodes[page_number - 1].table_extraction_results.results[
                self.name
            ] = page_result
