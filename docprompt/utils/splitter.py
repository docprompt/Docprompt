import io
import logging
import tempfile
from typing import Iterator, Optional

import pypdfium2 as pdfium

from docprompt._exec.ghostscript import compress_pdf_to_bytes
from docprompt.utils import get_page_count

from docprompt._pdfium import get_pdfium_document, writable_temp_pdf


logger = logging.getLogger(__name__)


class UnsupportedDocumentType(ValueError):
    pass


def split_pdf_to_bytes(
    file_bytes: bytes,
    *,
    start_page: Optional[int] = None,
    stop_page: Optional[int] = None,
):
    """
    Splits a PDF into a list of bytes.
    """
    if start_page is None:
        start_page = 0
    if stop_page is None:
        stop_page = get_page_count(file_bytes)

    if stop_page <= start_page:
        raise ValueError("stop_page must be greater than start_page")

    # Load the PDF from bytes
    with get_pdfium_document(file_bytes) as src_pdf:
        # Create a new PDF for the current batch
        dst_pdf = pdfium.PdfDocument.new()

        # Append pages to the batch
        dst_pdf.import_pages(src_pdf, list(range(start_page, stop_page)))

        # Save the batch PDF to a bytes buffer
        pdf_bytes_buffer = io.BytesIO()
        dst_pdf.save(pdf_bytes_buffer)
        pdf_bytes_buffer.seek(0)  # Reset buffer pointer to the beginning

        # Yield the bytes of the batch PDF
        return pdf_bytes_buffer.getvalue()


def pdf_split_iter_fast(file_bytes: bytes, max_page_count: int) -> Iterator[bytes]:
    """
    Splits a PDF into batches of pages up to `max_page_count` pages quickly.
    """
    with get_pdfium_document(file_bytes) as src_pdf:
        current_page = 0
        total_pages = len(src_pdf)

        while current_page < total_pages:
            # Determine the last page for the current batch
            last_page = min(current_page + max_page_count, total_pages)

            with writable_temp_pdf() as dst_pdf:
                # Append pages to the batch
                dst_pdf.import_pages(src_pdf, list(range(current_page, last_page)))

                # Save the batch PDF to a bytes buffer
                pdf_bytes_buffer = io.BytesIO()
                dst_pdf.save(pdf_bytes_buffer)
                pdf_bytes_buffer.seek(0)  # Reset buffer pointer to the beginning

            # Yield the bytes of the batch PDF
            yield pdf_bytes_buffer.getvalue()

            # Update the current page for the next batch
            current_page += max_page_count


def pdf_split_iter_with_max_bytes(
    file_bytes: bytes, max_page_count: int, max_bytes: int
) -> Iterator[bytes]:
    """
    Splits a PDF into batches of pages up to `max_page_count` pages and `max_bytes` bytes.
    """
    for batch_bytes in pdf_split_iter_fast(file_bytes, max_page_count):
        if len(batch_bytes) <= max_bytes:
            yield batch_bytes
        else:
            # If batch size is greater than max_bytes, reduce the number of pages
            pages_in_batch = max_page_count
            while len(batch_bytes) > max_bytes and pages_in_batch > 1:
                pages_in_batch -= 1
                batch_bytes = next(pdf_split_iter_fast(file_bytes, pages_in_batch))

            if len(batch_bytes) > max_bytes and pages_in_batch == 1:
                # If a single page is still too large, compress it
                with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
                    f.write(batch_bytes)
                    f.flush()
                    compressed_bytes = compress_pdf_to_bytes(f.name)
                yield compressed_bytes
            else:
                yield batch_bytes
