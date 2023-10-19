import logging
from io import BytesIO
from typing import Optional, Tuple

import magic
from pypdf import PdfReader, PdfWriter

from docprompt.util import get_page_count

logger = logging.getLogger(__name__)


class UnsupportedDocumentType(ValueError):
    pass


class DocumentSplitter:
    @classmethod
    def get_mime_type(cls, bytes: bytes):
        return magic.from_buffer(buffer=bytes, mime=True)

    @classmethod
    def split(
        cls,
        bytes: bytes,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: int = 1,
    ) -> Tuple[bytes, str]:
        TYPE_MAPPING = {"application/pdf": cls.split_pdf}

        mime_type = cls.get_mime_type(bytes)
        split_func = TYPE_MAPPING.get(mime_type, None)

        if not split_func:
            raise UnsupportedDocumentType()

        start = start or 0

        return split_func(bytes, start, stop, step), mime_type

    @classmethod
    def split_pdf(cls, bytes: bytes, start: int = 0, stop: Optional[int] = None, step: int = 1) -> bytes:
        reader = PdfReader(BytesIO(bytes))
        writer = PdfWriter()
        output_stream = BytesIO()

        if start == 0 and stop is None:  # Don't bother if we dont get paramaters specified
            return bytes

        for i in range(start, stop or len(reader.pages), step):
            writer.add_page(reader.pages[i])

        writer.write(output_stream)

        return output_stream.getvalue()


def pdf_split_iter(file_bytes: bytes, max_page_count: int, max_bytes: Optional[int] = None):
    """
    Split a PDF into batches of pages up to `max_page_count` pages and possibly
    `max_bytes` bytes.
    """

    page_count = get_page_count(file_bytes)

    if page_count <= max_page_count and (max_bytes is None or len(file_bytes) <= max_bytes):
        yield file_bytes
        return

    start_page = 0

    while start_page < page_count:
        end_page = min(start_page + max_page_count, page_count)

        split_bytes = DocumentSplitter.split_pdf(file_bytes, start_page, end_page)

        while max_bytes is not None and len(split_bytes) > max_bytes:
            if end_page <= start_page:
                raise ValueError("Cannot shrink chunk size any further. This PDF is HUGE!")
            end_page -= 1
            print(f"Shrinking chunk size by 1 due to byte constraints")
            split_bytes = DocumentSplitter.split_pdf(file_bytes, start_page, end_page)

        yield split_bytes

        start_page = end_page
