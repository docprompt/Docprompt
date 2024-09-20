"""Top-level package for Docprompt."""

__author__ = """Frankie Colson"""
__email__ = "frank@pageleaf.io"
__version__ = "0.8.2"

from docprompt.rasterize import ProviderResizeRatios
from docprompt.schema.document import Document, PdfDocument  # noqa
from docprompt.schema.layout import NormBBox, TextBlock  # noqa
from docprompt.schema.pipeline import DocumentCollection, DocumentNode, PageNode  # noqa
from docprompt.tasks.ocr.result import OcrPageResult  # noqa
from docprompt.utils import (  # noqa
    hash_from_bytes,
    load_document,
    load_document_node,
    load_documents,
    load_pdf_document,
    load_pdf_documents,
)

# PdfDocument.model_rebuild()
DocumentNode.model_rebuild()


__all__ = [
    "Document",
    "PdfDocument",
    "DocumentCollection",
    "DocumentNode",
    "NormBBox",
    "PageNode",
    "TextBlock",
    "load_document",
    "load_documents",
    "hash_from_bytes",
    "ProviderResizeRatios",
    "load_pdf_document",
]
