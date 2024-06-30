"""Top-level package for Docprompt."""

__author__ = """Frankie Colson"""
__email__ = "frank@pageleaf.io"
__version__ = "0.7.0"

from docprompt.rasterize import ProviderResizeRatios
from docprompt.schema.document import Document, PdfDocument  # noqa
from docprompt.schema.layout import NormBBox, TextBlock  # noqa
from docprompt.schema.pipeline import DocumentCollection, DocumentNode, PageNode  # noqa
from docprompt.utils import hash_from_bytes, load_document, load_documents  # noqa

PdfDocument.model_rebuild()
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
]
