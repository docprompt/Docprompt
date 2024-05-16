from io import BytesIO
import os
import pypdfium2 as pdfium
from contextlib import contextmanager
from threading import Lock
from typing import Any, Callable, Dict, List, Literal, Union, Optional
from os import PathLike
from pathlib import Path
import logging
import multiprocessing as mp
import concurrent.futures as ft
from PIL import Image

logger = logging.getLogger(__name__)


PDFIUM_LOAD_LOCK = Lock()  # PDF fails to load without this lock
PDFIUM_WRITE_LOCK = Lock()  # Deadlocks occur in threaded environments without this lock
PDFIUM_RASTERIZE_LOCK = (
    Lock()
)  # Rasterization fails without this lock in threaded environments


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


def _render_parallel_init(
    extra_init,
    input,
    password,
    may_init_forms,
    kwargs,
    return_mode="pil",
    post_process_fn=None,
):
    if extra_init:
        extra_init()

    logger.info(f"Initializing data for process {os.getpid()}")

    pdf = pdfium.PdfDocument(input, password=password, autoclose=True)
    if may_init_forms:
        pdf.init_forms()

    global ProcObjs
    ProcObjs = (pdf, kwargs, return_mode, post_process_fn)


def _render_job(
    i: int,
    pdf: pdfium.PdfDocument,
    raster_kwargs: Dict[str, Any],
    return_mode: Literal["pil", "bytes"],
    post_process_fn: Optional[Callable[[Image.Image], Image.Image]] = None,
):
    # logger.info(f"Started page {i+1} ...")
    page = pdf[i]
    image = page.render(**raster_kwargs).to_pil().convert("RGB")

    if post_process_fn:
        image = post_process_fn(image)

    if return_mode == "pil":
        if isinstance(image, Image.Image):
            return image
        else:
            image = Image.open(BytesIO(image))
    else:
        if isinstance(image, bytes):
            return image
        else:
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()


def _render_parallel_job(i):
    global ProcObjs
    return _render_job(i, *ProcObjs)


def rasterize_page_with_pdfium(
    fp: Union[PathLike, Path, bytes],
    page_number: int,
    *,
    return_mode: Literal["pil", "bytes"] = "pil",
    post_process_fn: Optional[Callable[[Image.Image], Image.Image]] = None,
    **kwargs,
) -> Union[Image.Image, bytes]:
    """
    Rasterizes a page of a PDF document
    """
    with get_pdfium_document(fp) as pdf:
        return _render_job(
            page_number - 1,
            pdf,
            kwargs,
            return_mode=return_mode,
            post_process_fn=post_process_fn,
        )


def rasterize_pdf_with_pdfium(
    fp: Union[PathLike, Path, bytes],
    password: Optional[str] = None,
    *,
    return_mode: Literal["pil", "bytes"] = "pil",
    post_process_fn: Optional[Callable[[Image.Image], Image.Image]] = None,
    **kwargs,
) -> List[Union[Image.Image, bytes]]:
    """
    Rasterizes a page of a PDF document
    """
    with get_pdfium_document(fp, password=password) as pdf:
        total_pages = len(pdf)

    max_workers = min(mp.cpu_count(), total_pages)

    ctx = mp.get_context("spawn")

    initargs = (None, fp, password, False, kwargs, return_mode, post_process_fn)

    with ft.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_render_parallel_init,
        initargs=initargs,
        mp_context=ctx,
    ) as executor:
        results = executor.map(_render_parallel_job, range(total_pages), chunksize=1)

    return list(results)
