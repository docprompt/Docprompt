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
    Uses page deletion to efficiently reduce batch size if needed.
    Compresses batches if they exceed max_bytes.
    """
    with get_pdfium_document(file_bytes) as src_pdf:
        total_pages = len(src_pdf)
        current_page = 0

        while current_page < total_pages:
            # Start with the maximum allowed pages or remaining pages
            pages_in_batch = min(max_page_count, total_pages - current_page)

            with writable_temp_pdf() as batch_pdf:
                # Create a batch with the current number of pages
                batch_pdf.import_pages(
                    src_pdf, list(range(current_page, current_page + pages_in_batch))
                )

                while pages_in_batch > 0:
                    # Save the batch to bytes
                    pdf_bytes_buffer = io.BytesIO()
                    batch_pdf.save(pdf_bytes_buffer)
                    batch_bytes = pdf_bytes_buffer.getvalue()

                    if len(batch_bytes) <= max_bytes:
                        # If the batch is within the byte limit, yield it
                        yield batch_bytes
                        current_page += pages_in_batch
                        break
                    else:
                        # If the batch exceeds the byte limit, try compressing
                        try:
                            compressed_batch = compress_pdf_bytes(batch_bytes)
                            if len(compressed_batch) <= max_bytes:
                                yield compressed_batch
                                current_page += pages_in_batch
                                break
                        except Exception as e:
                            logger.warning(f"Compression failed: {str(e)}")

                        # If compression fails or is still too large, remove the last page
                        batch_pdf.del_page(pages_in_batch - 1)
                        pages_in_batch -= 1

            # If we can't fit even one page, raise an error
            if pages_in_batch == 0:
                raise ValueError(
                    f"Unable to fit even a single page within max_bytes ({max_bytes})"
                )
