import base64
import gzip
import pickle
import tempfile
from contextlib import contextmanager
from datetime import datetime
from functools import cached_property
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import magic
from PIL import ImageDraw
from pydantic import BaseModel, Field, PositiveInt, computed_field, field_serializer, field_validator

from docprompt._exec.ghostscript import compress_pdf_to_bytes, rasterize_page_to_bytes, rasterize_pdf_to_bytes
from docprompt.schema.operations import PageTextExtractionOutput

from .layout import TextBlock

if TYPE_CHECKING:
    from docprompt.service_providers.base import BaseProvider

import pdfplumber

DEFAULT_DPI = 100


def get_page_render_size_from_bytes(file_bytes: bytes, page_number: int, dpi: int = DEFAULT_DPI):
    """
    Returns the render size of a page in pixels
    """
    with pdfplumber.PDF.open(BytesIO(file_bytes)) as pdf:
        page = pdf.pages[page_number]

        width_pt, height_pt = page.mediabox.upper_right

    width_in = width_pt / 72
    height_in = height_pt / 72

    width_px = int(width_in * dpi)
    height_px = int(height_in * dpi)

    return (width_px, height_px)


class Document(BaseModel):
    """
    Represents a PDF document
    """

    name: str = Field(description="The name of the document")
    file_bytes: bytes = Field(description="The bytes of the document", repr=False)
    file_path: Optional[str]
    text_sidecars: Dict[str, Dict[int, PageTextExtractionOutput]] = Field(default_factory=dict, repr=False)

    def __len__(self):
        return len(self.pages)

    def __hash__(self):
        return hash(self.document_hash)

    @computed_field
    @cached_property
    def page_count(self) -> PositiveInt:
        with pdfplumber.PDF.open(BytesIO(self.file_bytes)) as pdf:
            return len(pdf.pages)

    @property
    def num_pages(self):
        return self.page_count

    @computed_field
    @cached_property
    def document_hash(self) -> str:
        from docprompt.utils.util import hash_from_bytes

        return hash_from_bytes(self.file_bytes)

    @field_serializer("file_bytes")
    def serialize_file_bytes(self, v: bytes, _info):
        compressed = gzip.compress(v)

        return base64.b64encode(compressed).decode("utf-8")

    @field_validator("file_bytes")
    def validate_file_bytes(cls, v: bytes):
        if not isinstance(v, bytes):
            raise ValueError("File bytes must be bytes")

        if len(v) == 0:
            raise ValueError("File bytes must not be empty")

        if magic.from_buffer(v, mime=True) == "text/plain":
            v = base64.b64decode(v, validate=True)

        if magic.from_buffer(v, mime=True) == "application/gzip":
            v = gzip.decompress(v)

        if magic.from_buffer(v, mime=True) != "application/pdf":
            raise ValueError("File bytes must be a PDF")

        return v

    @classmethod
    def from_path(cls, file_path: Union[PathLike, str]):
        file_path = Path(file_path)

        if not file_path.is_file():
            raise ValueError(f"File path {file_path} is not a file")

        file_bytes = file_path.read_bytes()

        return cls(name=file_path.name, file_path=str(file_path), file_bytes=file_bytes)

    @classmethod
    def from_bytes(cls, file_bytes: bytes, name: Optional[str] = None):
        if name is None:
            name = f"PDF-{datetime.now().isoformat()}.pdf"

        return cls(name=name, file_bytes=file_bytes)

    def get_bytes(self) -> bytes:
        return self.file_bytes  # Deprecated

    @property
    def path(self):
        return self.file_path

    def get_page_render_size(self, page_number: int, dpi: int = DEFAULT_DPI) -> Tuple[int, int]:
        """
        Returns the render size of a page in pixels
        """
        return get_page_render_size_from_bytes(self.get_bytes(), page_number, dpi=dpi)

    def to_compressed_bytes(self, compression_kwargs: dict = {}) -> bytes:
        """
        Compresses the document using Ghostscript
        """
        with self.as_tempfile() as temp_path:
            return compress_pdf_to_bytes(temp_path, **compression_kwargs)

    def rasterize_page(self, page_number: int, dpi: int = DEFAULT_DPI, device="png16m") -> bytes:
        """
        Rasterizes a page of the document using Ghostscript
        """
        if page_number < 0 or page_number > self.num_pages:
            raise ValueError(f"Page number must be between 0 and {self.num_pages}")

        with self.as_tempfile() as temp_path:
            return rasterize_page_to_bytes(temp_path, page_number, dpi=dpi, device=device)

    def rasterize_pdf(
        self, dpi: int = DEFAULT_DPI, device="pnggray", downscale_factor: Optional[int] = None
    ) -> Dict[int, bytes]:
        """
        Rasterizes the entire document using Ghostscript
        """
        with self.as_tempfile() as temp_path:
            return rasterize_pdf_to_bytes(temp_path, dpi=dpi, device=device, downscale_factor=downscale_factor)

    @contextmanager
    def as_tempfile(self, **kwargs) -> str:
        """
        Returns a tempfile of the document
        """
        tempfile_kwargs = {"mode": "wb", "delete": True, "suffix": ".pdf", **kwargs}

        with tempfile.NamedTemporaryFile(**tempfile_kwargs) as f:
            f.write(self.file_bytes)
            f.flush()
            yield f.name

    def write_to_path(self, path: Union[PathLike, str], **kwargs):
        """
        Writes the document to a path
        """
        path = Path(path)

        if path.is_dir():
            path = path / self.name

        with path.open("wb") as f:
            f.write(self.file_bytes)

    @property
    def text_data(self):
        try:
            return next(iter(self.text_sidecars.values()))
        except StopIteration:
            return None

    def perform_text_extraction(
        self, provider: "BaseProvider", cache: bool = True
    ) -> Dict[int, PageTextExtractionOutput]:
        """
        Performs text extraction for a given provider
        """

        result = provider.process_document(self)

        sidecars = {}

        if not result or not result.page_results:
            return sidecars

        for page_result in result.page_results:
            if not page_result.ocr_result:
                continue

            sidecars[page_result.page_number] = page_result.ocr_result

        if cache:
            self.text_sidecars[provider.name] = sidecars

        return sidecars
