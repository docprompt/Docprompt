import pypdfium2 as pdfium
from contextlib import contextmanager
from threading import Lock
from typing import Union, Optional
from os import PathLike
from pathlib import Path


PDFIUM_LOAD_LOCK = Lock()  # PDF fails to load without this lock
PDFIUM_WRITE_LOCK = Lock()  # Deadlocks occur in threaded environments without this lock


@contextmanager
def get_pdfium_document(
    fp: Union[PathLike, Path, bytes], password: Optional[str] = None
):
    """
    Loads a PDF document with a lock to prevent race conditions in threaded environments
    """
    with PDFIUM_LOAD_LOCK:
        pdf = pdfium.PdfDocument(fp, password=password, autoclose=False)

    try:
        yield pdf
    finally:
        pdf.close()


@contextmanager
def writable_temp_pdf():
    with PDFIUM_WRITE_LOCK:
        pdf = pdfium.PdfDocument.new()

        try:
            yield pdf
        finally:
            pdf.close()
