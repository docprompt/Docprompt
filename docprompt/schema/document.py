import pickle
import tempfile
from contextlib import contextmanager
from datetime import datetime
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union
from urllib.parse import unquote

from attrs import define, field, frozen
from PIL import ImageDraw

from docprompt._exec.ghostscript import compress_pdf_to_bytes, rasterize_page_to_bytes, rasterize_pdf_to_bytes

from .layout import TextBlock

if TYPE_CHECKING:
    from docprompt.service_providers.types import PageTextExtractionOutput
    from docprompt.service_providers.base import BaseProvider

import pdfplumber

DEFAULT_DPI = 100


@define
class Document:
    """
    Represents a PDF document
    """

    name: str
    file_path: str

    file_bytes: Optional[bytes] = field(default=None, repr=False)
    num_pages: int = field(init=False)

    def __attrs_post_init__(self):
        file_bytes = self.get_bytes()

        if file_bytes:
            from docprompt.utils import get_page_count

            self.num_pages = get_page_count(file_bytes)

    def get_bytes(self) -> bytes:
        if self.file_bytes:
            return self.file_bytes

        self.open()

        return self.file_bytes

    def document_hash(self):
        from docprompt.utils.util import hash_from_bytes

        return hash_from_bytes(self.get_bytes())

    def get_page_render_size(self, page_number: int, dpi: int = DEFAULT_DPI) -> Tuple[int, int]:
        """
        Returns the render size of a page in pixels
        """
        with pdfplumber.PDF.open(BytesIO(self.get_bytes())) as pdf:
            page = pdf.pages[page_number]

            width_pt, height_pt = page.mediabox.upper_right

        width_in = width_pt / 72
        height_in = height_pt / 72

        width_px = int(width_in * dpi)
        height_px = int(height_in * dpi)

        return (width_px, height_px)

    @property
    def path(self):
        return self.file_path

    def _clone(self):
        """
        Lightweight alternative to deepcopy
        """
        raise NotImplementedError("Not implemented yet")

    def to_compressed_bytes(self, compression_kwargs: dict = {}) -> bytes:
        """
        Compresses the document using Ghostscript
        """
        with self.as_tempfile() as temp_path:
            return compress_pdf_to_bytes(temp_path, **compression_kwargs)

    def rasterize_page(self, page_number: int, dpi: int = DEFAULT_DPI) -> bytes:
        """
        Rasterizes a page of the document using Ghostscript
        """
        if page_number < 0 or page_number > self.num_pages:
            raise ValueError(f"Page number must be between 0 and {self.num_pages}")

        with self.as_tempfile() as temp_path:
            return rasterize_page_to_bytes(temp_path, page_number, dpi=dpi)

    def rasterize_pdf(self, dpi: int = DEFAULT_DPI) -> Dict[int, bytes]:
        """
        Rasterizes the entire document using Ghostscript
        """
        with self.as_tempfile() as temp_path:
            return rasterize_pdf_to_bytes(temp_path, dpi=dpi)

    @contextmanager
    def as_tempfile(self, **kwargs) -> str:
        """
        Returns a tempfile of the document
        """
        tempfile_kwargs = {"mode": "wb", "delete": True, "suffix": ".pdf", **kwargs}

        with tempfile.NamedTemporaryFile(**tempfile_kwargs) as f:
            f.write(self.get_bytes())
            f.flush()
            yield f.name

    def close(self):
        """
        Close the file and clear the bytes from memory
        """
        self.file_bytes = None

    def open(self):
        """
        Reopens the file and reads the bytes into memory
        """
        if self.file_bytes is not None:
            return

        file_path = Path(self.file_path)

        if file_path.is_file():
            with open(file_path, "rb") as f:
                self.file_bytes = f.read()

    def __len__(self):
        return len(self.pages)

    def __hash__(self):
        return hash(self.document_hash())


@frozen
class TextExtractionSidecar:
    """
    Represents a sidecar file that contains the output of
    text extraction / OCR for a document. Provides
    utilities

    A sidecar is unique to a document and a provider.
    """

    provider_name: str

    words: List[TextBlock] = field(repr=False)
    lines: List[TextBlock] = field(repr=False)
    blocks: List[TextBlock] = field(repr=False)

    when: datetime = field(factory=datetime.now)

    @classmethod
    def from_textextractionoutput(cls, provider_name: str, output: "PageTextExtractionOutput"):
        blocks = output.blocks
        return cls(
            provider_name=provider_name,
            words=blocks.get("word", []),
            lines=blocks.get("line", []),
            blocks=blocks.get("block", []),
        )

    def draw_bounding_boxes(self, image, level: Literal["word", "line", "block"] = "word", outline="red"):
        """
        Draws bounding boxes on an image
        """
        draw = ImageDraw.Draw(image)

        for block in getattr(self, level + "s"):
            draw.rectangle(block.bounding_box, outline=outline)

        return image


@define
class DocumentContainer:
    """
    A document container encapsulates a document and all the operations
    that can be performed on it.
    """

    document: Document
    text_sidecars: Dict[str, Dict[int, TextExtractionSidecar]] = field(factory=dict, repr=False)

    def dump(self, fp: Union[Path, PathLike], compress: bool = True):
        if not isinstance(fp, Path):
            fp = Path(fp)

        if fp.is_dir():
            fp = fp / f"{self.document.name}.pickle"

        with fp.open("wb") as f:
            pickle.dump(self, f)

    def dumps(self, compress: bool = True):
        return pickle.dumps(self)

    @classmethod
    def load(cls, fp: Union[Path, PathLike, str]):
        if not isinstance(fp, Path):
            fp = Path(fp)

        with fp.open("rb") as f:
            return pickle.load(f)

    @classmethod
    def loads(cls, bytes_: bytes):
        return pickle.loads(bytes_)

    @property
    def text_data(self):
        try:
            return next(iter(self.text_sidecars.values()))
        except StopIteration:
            return None

    def perform_text_extraction(self, provider: "BaseProvider", cache: bool = True) -> Dict[int, TextExtractionSidecar]:
        """
        Performs text extraction for a given provider
        """

        result = provider.process_document(self.document)

        sidecars = {}

        if not result or not result.page_results:
            return sidecars

        for page_result in result.page_results:
            if not page_result.ocr_result:
                continue

            sidecars[page_result.page_number] = TextExtractionSidecar.from_textextractionoutput(
                provider.name, page_result.ocr_result
            )

        if cache:
            self.text_sidecars[provider.name] = sidecars

        return sidecars
