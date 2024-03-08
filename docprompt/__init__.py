"""Top-level package for Docprompt."""

__author__ = """Frankie Colson"""
__email__ = 'frank@pageleaf.io'
__version__ = '0.1.0'


import docprompt.silenceable_tqdm  # noqa
from docprompt.schema.document import Document
from docprompt.schema.layout import NormBBox, TextBlock
from docprompt.schema.pipeline import DocumentCollection, DocumentNode, PageNode
from docprompt.utils import load_document

Document.model_rebuild()
