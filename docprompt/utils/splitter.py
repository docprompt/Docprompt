import io
import logging
from typing import Iterator, Optional

import pypdfium2 as pdfium

from docprompt._pdfium import get_pdfium_document, writable_temp_pdf
from docprompt.utils import get_page_count
from docprompt.utils.compressor import compress_pdf_bytes

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
    Compresses individual pages if they exceed max_bytes.
    Raises an error if compression fails to bring a page under the byte limit.
    """
    current_pages = 0
    current_byte_size = 0
    current_batch = io.BytesIO()

    single_page_splits = pdf_split_iter_fast(file_bytes, 1)

    for page in single_page_splits:
        page_size = len(page)

        # Check if a single page exceeds the byte limit
        if page_size > max_bytes:
            try:
                compressed_page = compress_pdf_bytes(page)
                if len(compressed_page) > max_bytes:
                    raise ValueError(
                        f"Page size ({len(compressed_page)} bytes) exceeds max_bytes ({max_bytes}) even after compression."
                    )
                page = compressed_page
                page_size = len(page)
            except Exception as e:
                raise RuntimeError(f"Failed to compress page: {str(e)}")

        if current_pages == 0 or (
            current_pages < max_page_count
            and current_byte_size + page_size <= max_bytes
        ):
            # Add page to the current batch
            if current_pages == 0:
                current_batch = io.BytesIO(page)
            else:
                with writable_temp_pdf() as merged_pdf:
                    merged_pdf.import_pages(
                        pdfium.PdfDocument(io.BytesIO(current_batch.getvalue()))
                    )
                    merged_pdf.import_pages(pdfium.PdfDocument(io.BytesIO(page)))
                    current_batch = io.BytesIO()
                    merged_pdf.save(current_batch)

            current_pages += 1
            current_byte_size = len(current_batch.getvalue())
        else:
            # Yield the current batch and start a new one
            yield current_batch.getvalue()
            current_batch = io.BytesIO(page)
            current_pages = 1
            current_byte_size = page_size

    # Don't forget to yield the last batch
    if current_pages > 0:
        yield current_batch.getvalue()
