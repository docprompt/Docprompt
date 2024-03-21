"""Top-level package for Docprompt."""

__author__ = """Frankie Colson"""
__email__ = "frank@pageleaf.io"
__version__ = "0.2.1"

from docprompt.schema.document import Document, PdfDocument  # noqa
from docprompt.schema.layout import NormBBox, TextBlock  # noqa
from docprompt.schema.pipeline import DocumentCollection, DocumentNode, PageNode  # noqa
from docprompt.utils import load_document  # noqa

Document.model_rebuild()


__all__ = [
    "Document",
    "PdfDocument",
    "DocumentCollection",
    "DocumentNode",
    "NormBBox",
    "PageNode",
    "TextBlock",
    "load_document",
]
