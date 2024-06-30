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
