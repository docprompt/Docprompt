import base64
import gzip
import tempfile
from contextlib import contextmanager
from datetime import datetime
from functools import cached_property
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import magic
import pypdfium2 as pdfium
from PIL import Image
from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    PrivateAttr,
    computed_field,
    field_serializer,
    field_validator,
)

from docprompt._exec.ghostscript import (
    compress_pdf_to_bytes,
)

DEFAULT_DPI = 100


def get_page_render_size_from_bytes(
    file_bytes: bytes, page_number: int, dpi: int = DEFAULT_DPI
):
    """
    Returns the render size of a page in pixels
    """

    pdf = pdfium.PdfDocument(BytesIO(file_bytes))
    page = pdf.get_page(page_number)

    mediabox = page.get_mediabox()

    base_width = int(mediabox[2] - mediabox[0])
    base_height = int(mediabox[3] - mediabox[1])

    width = int(base_width * dpi / 72)
    height = int(base_height * dpi / 72)

    return width, height


class PdfDocument(BaseModel):
    """
    Represents a PDF document
    """

    name: str = Field(description="The name of the document")
    file_bytes: bytes = Field(description="The bytes of the document", repr=False)
    file_path: Optional[str] = None

    _raster_cache: Dict[int, Dict[int, bytes]] = PrivateAttr(default_factory=dict)

    def __len__(self):
        return self.num_pages

    def __hash__(self):
        return hash(self.document_hash)

    @computed_field
    @cached_property
    def page_count(self) -> PositiveInt:
        from docprompt.utils.util import get_page_count

        return get_page_count(self.file_bytes)

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

    def get_page_render_size(
        self, page_number: int, dpi: int = DEFAULT_DPI
    ) -> Tuple[int, int]:
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

    def rasterize_page(
        self,
        page_number: int,
        *,
        dpi: int = DEFAULT_DPI,
        downscale_size: Optional[Tuple[int, int]] = None,
        use_cache: bool = True,
    ) -> bytes:
        """
        Rasterizes a page of the document using Ghostscript
        """
        generated_image = False

        if page_number < 0 or page_number > self.num_pages:
            raise ValueError(f"Page number must be between 0 and {self.num_pages}")

        if use_cache and self._raster_cache.get(dpi, {}).get(page_number):
            rastered = self._raster_cache[dpi][page_number]
        else:
            pdf = pdfium.PdfDocument(BytesIO(self.file_bytes))
            page = pdf[page_number - 1]

            bitmap = page.render(scale=(1 / 72) * dpi)

            pil = bitmap.to_pil().convert("RGB")

            img_bytes = BytesIO()
            pil.save(img_bytes, format="PNG")
            rastered = img_bytes.getvalue()

            generated_image = True

        if use_cache and generated_image:
            self._raster_cache.setdefault(dpi, {})
            self._raster_cache[dpi][page_number] = rastered

        if downscale_size:
            with Image.open(BytesIO(rastered)) as img:
                width, height = img.size
                if not (width < downscale_size[0] or height < downscale_size[1]):
                    img = img.resize(downscale_size)
                    img_bytes = BytesIO()
                    img.save(img_bytes, format="PNG")
                    rastered = img_bytes.getvalue()

        return rastered

    def rasterize_page_to_data_uri(
        self,
        page_number: int,
        *,
        dpi: int = DEFAULT_DPI,
        downscale_size: Optional[Tuple[int, int]] = None,
        use_cache: bool = True,
    ) -> str:
        """
        Rasterizes a page of the document using Ghostscript and returns a data URI, which can
        be embedded into HTML or passed to large language models
        """
        image_bytes = self.rasterize_page(
            page_number,
            dpi=dpi,
            downscale_size=downscale_size,
            use_cache=use_cache,
        )
        return f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"

    def rasterize_pdf(
        self,
        dpi: int = DEFAULT_DPI,
        use_cache: bool = True,
    ) -> Dict[int, bytes]:
        """
        Rasterizes the entire document using Ghostscript
        """
        if (
            use_cache
            and self._raster_cache.get(dpi)
            and len(self._raster_cache[dpi]) == self.num_pages
        ):
            return self._raster_cache[dpi]

        result = {}

        for page_number in range(1, self.num_pages + 1):
            result[page_number] = self.rasterize_page(
                page_number, dpi=dpi, use_cache=False
            )

        if use_cache:
            self._raster_cache[dpi] = result.copy()  # Shallow copy should be OK

        return result

    def split(self, start: Optional[int] = None, stop: Optional[int] = None):
        """
        Splits a document into multiple documents
        """
        if start is None and stop is None:
            raise ValueError("Must specify either start or stop")

        start = start or 0

        from docprompt.utils.splitter import split_pdf_to_bytes

        split_bytes = split_pdf_to_bytes(
            self.file_bytes, start_page=start, stop_page=stop
        )

        return Document.from_bytes(split_bytes, name=self.name)

    def as_tempfile(self, **kwargs):
        """
        Returns a tempfile of the document
        """

        @contextmanager
        def tempfile_context() -> str:
            tempfile_kwargs = {"mode": "wb", "delete": True, "suffix": ".pdf", **kwargs}

            with tempfile.NamedTemporaryFile(**tempfile_kwargs) as f:
                f.write(self.file_bytes)
                f.flush()
                yield f.name

        return tempfile_context()

    def write_to_path(self, path: Union[PathLike, str], **kwargs):
        """
        Writes the document to a path
        """
        path = Path(path)

        if path.is_dir():
            path = path / self.name

        with path.open("wb") as f:
            f.write(self.file_bytes)


Document = PdfDocument  # Alias
