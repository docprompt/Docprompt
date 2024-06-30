import base64
import gzip
import logging
import tempfile
from contextlib import contextmanager
from datetime import datetime
from functools import cached_property, partial
from os import PathLike
from pathlib import Path
from typing import Dict, Generator, Iterable, Literal, Optional, Tuple, Union

import filetype
from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    SecretStr,
    computed_field,
    field_serializer,
    field_validator,
)

from docprompt._exec.ghostscript import compress_pdf_to_bytes
from docprompt._pdfium import (
    get_pdfium_document,
    rasterize_page_with_pdfium,
    rasterize_pdf_with_pdfium,
)
from docprompt.rasterize import AspectRatioRule, ResizeModes, process_raster_image

DEFAULT_DPI = 100

logger = logging.getLogger(__name__)


def get_page_render_size_from_bytes(
    file_bytes: bytes, page_number: int, dpi: int = DEFAULT_DPI
):
    """
    Returns the render size of a page in pixels
    """

    with get_pdfium_document(file_bytes) as pdf:
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

    password: Optional[SecretStr] = None

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

    @property
    def bytes_per_page(self):
        return len(self.file_bytes) / self.num_pages

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

        if filetype.guess_mime(v) == "text/plain":
            v = base64.b64decode(v, validate=True)

        if filetype.guess_mime(v) == "application/gzip":
            v = gzip.decompress(v)

        if filetype.guess_mime(v) != "application/pdf":
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
        resize_mode: ResizeModes = "thumbnail",
        max_file_size_bytes: Optional[int] = None,
        resize_aspect_ratios: Optional[Iterable[AspectRatioRule]] = None,
        do_convert: bool = False,
        image_convert_mode: str = "L",
        do_quantize: bool = False,
        quantize_color_count: int = 8,
        return_mode: Literal["pil", "bytes"] = "bytes",
    ):
        """
        Rasterizes a page of the document using Pdfium
        """
        if page_number <= 0 or page_number > self.num_pages:
            raise ValueError(f"Page number must be between 0 and {self.num_pages}")

        post_process_fn = None

        if any(
            (
                downscale_size,
                max_file_size_bytes,
                resize_aspect_ratios,
                do_convert,
                do_quantize,
            )
        ):
            post_process_fn = partial(
                process_raster_image,
                resize_width=downscale_size[0] if downscale_size else None,
                resize_height=downscale_size[1] if downscale_size else None,
                resize_mode=resize_mode,
                resize_aspect_ratios=resize_aspect_ratios,
                do_convert=do_convert,
                image_convert_mode=image_convert_mode,
                do_quantize=do_quantize,
                quantize_color_count=quantize_color_count,
                max_file_size_bytes=max_file_size_bytes,
            )

        rastered = rasterize_page_with_pdfium(
            self.file_bytes,
            page_number,
            return_mode=return_mode,
            post_process_fn=post_process_fn,
            scale=(1 / 72) * dpi,
        )

        return rastered

    def rasterize_page_to_data_uri(
        self,
        page_number: int,
        *,
        dpi: int = DEFAULT_DPI,
        downscale_size: Optional[Tuple[int, int]] = None,
        resize_mode: ResizeModes = "thumbnail",
        max_file_size_bytes: Optional[int] = None,
        resize_aspect_ratios: Optional[Iterable[AspectRatioRule]] = None,
        do_convert: bool = False,
        image_convert_mode: str = "L",
        do_quantize: bool = False,
        quantize_color_count: int = 8,
        render_grayscale: bool = False,
    ) -> str:
        """
        Rasterizes a page of the document using Pdfium and returns a data URI, which can
        be embedded into HTML or passed to large language models
        """
        image_bytes = self.rasterize_page(
            page_number,
            dpi=dpi,
            downscale_size=downscale_size,
            do_convert=do_convert,
            image_convert_mode=image_convert_mode,
            do_quantize=do_quantize,
            quantize_color_count=quantize_color_count,
            resize_mode=resize_mode,
            max_file_size_bytes=max_file_size_bytes,
            resize_aspect_ratios=resize_aspect_ratios,
            return_mode="bytes",
        )
        return f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"

    def rasterize_pdf(
        self,
        dpi: int = DEFAULT_DPI,
        downscale_size: Optional[Tuple[int, int]] = None,
        resize_mode: ResizeModes = "thumbnail",
        max_file_size_bytes: Optional[int] = None,
        resize_aspect_ratios: Optional[Iterable[AspectRatioRule]] = None,
        do_convert: bool = False,
        image_convert_mode: str = "L",
        do_quantize: bool = False,
        quantize_color_count: int = 8,
        return_mode: Literal["pil", "bytes"] = "bytes",
        render_grayscale: bool = False,
    ) -> Dict[int, bytes]:
        """
        Rasterizes the entire document using Pdfium
        """
        result = {}

        post_process_fn = None

        if any(
            (
                downscale_size,
                max_file_size_bytes,
                resize_aspect_ratios,
                do_convert,
                do_quantize,
            )
        ):
            post_process_fn = partial(
                process_raster_image,
                resize_width=downscale_size[0] if downscale_size else None,
                resize_height=downscale_size[1] if downscale_size else None,
                resize_mode=resize_mode,
                resize_aspect_ratios=resize_aspect_ratios,
                do_convert=do_convert,
                image_convert_mode=image_convert_mode,
                do_quantize=do_quantize,
                quantize_color_count=quantize_color_count,
                max_file_size_bytes=max_file_size_bytes,
            )

        for idx, rastered in enumerate(
            rasterize_pdf_with_pdfium(
                self.file_bytes,
                scale=(1 / 72) * dpi,
                grayscale=render_grayscale,
                return_mode=return_mode,
                post_process_fn=post_process_fn,
            )
        ):
            result[idx + 1] = rastered

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
        def tempfile_context() -> Generator[str, None, None]:
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
