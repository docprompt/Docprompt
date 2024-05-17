from io import BytesIO
import os
import random
import tempfile
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
import tqdm

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


def _render_parallel_multi_doc_init(
    extra_init,
    inputs,
    passwords,
    may_init_forms,
    kwargs,
    return_mode="pil",
    post_process_fn=None,
):
    if extra_init:
        extra_init()

    logger.info(f"Initializing data for process {os.getpid()}")

    pdfs_map = {}

    for i, (input, password) in enumerate(zip(inputs, passwords)):
        pdf = pdfium.PdfDocument(input, password=password, autoclose=True)
        if may_init_forms:
            pdf.init_forms()

        pdfs_map[i] = pdf

    global ProcObjsMultiDoc
    ProcObjsMultiDoc = (pdfs_map, kwargs, return_mode, post_process_fn)


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


def _render_parallel_multi_doc_job(pdf_indice, page_indice):
    global ProcObjsMultiDoc

    pdf = ProcObjsMultiDoc[0][pdf_indice]

    return pdf_indice, page_indice, _render_job(page_indice, pdf, *ProcObjsMultiDoc[1:])


def _render_parallel_job(page_indice):
    global ProcObjs
    return _render_job(page_indice, *ProcObjs)


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


@contextmanager
def potential_temporary_file(fp: Union[PathLike, Path, bytes]):
    if isinstance(fp, bytes):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_fp:
            temp_fp.write(fp)
            temp_fp.flush()
            yield temp_fp.name
    else:
        yield fp


def rasterize_pdf_with_pdfium(
    fp: Union[PathLike, Path, bytes],
    password: Optional[str] = None,
    *,
    return_mode: Literal["pil", "bytes"] = "pil",
    post_process_fn: Optional[Callable[[Image.Image], Image.Image]] = None,
    **kwargs,
) -> List[Union[Image.Image, bytes]]:
    """
    Rasterizes an entire PDF using PDFium and a pool of workers
    """
    with get_pdfium_document(fp, password=password) as pdf:
        total_pages = len(pdf)

    max_workers = min(mp.cpu_count(), total_pages)

    ctx = mp.get_context("spawn")

    with potential_temporary_file(fp) as temp_fp:
        initargs = (
            None,
            temp_fp,
            password,
            False,
            kwargs,
            return_mode,
            post_process_fn,
        )

        with ft.ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_render_parallel_init,
            initargs=initargs,
            mp_context=ctx,
        ) as executor:
            results = executor.map(
                _render_parallel_job, range(total_pages), chunksize=1
            )

        return list(results)


def rasterize_pdfs_with_pdfium(
    fps: List[Union[PathLike, Path, bytes]],
    passwords: Optional[List[str]] = None,
    *,
    return_mode: Literal["pil", "bytes"] = "pil",
    post_process_fn: Optional[Callable[[Image.Image], Image.Image]] = None,
    **kwargs,
) -> Dict[int, Dict[int, Union[Image.Image, bytes]]]:
    """
    Like 'rasterize_pdf_with_pdfium', but optimized for multiple PDFs by loading all PDF's into the workers memory space
    """
    if passwords and len(passwords) != len(fps):
        raise ValueError(
            "If specifying passwords, must provide one for each PDF. Use None for no password."
        )

    passwords = passwords or [None] * len(fps)

    ctx = mp.get_context("spawn")

    page_counts = []
    total_to_process = 0

    for fp, password in zip(fps, passwords):
        with get_pdfium_document(fp, password) as pdf:
            page_counts.append(len(pdf))
            total_to_process += len(pdf)

    writable_fps = []

    with tempfile.TemporaryDirectory(prefix="docprompt_raster_tmp") as tempdir:
        for i, fp in enumerate(fps):
            if isinstance(fp, bytes):
                temp_fp = os.path.join(
                    tempdir, f"{i}_{random.randint(10000, 50000)}.pdf"
                )
                with open(temp_fp, "wb") as f:
                    f.write(fp)
                writable_fps.append(temp_fp)
            else:
                writable_fps.append(str(fp))

        initargs = (
            None,
            writable_fps,
            passwords,
            False,
            kwargs,
            return_mode,
            post_process_fn,
        )

        results = {}

        futures = []

        max_workers = min(mp.cpu_count(), total_to_process)

        with tqdm.tqdm(total=total_to_process) as pbar:
            with ft.ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_render_parallel_multi_doc_init,
                initargs=initargs,
                mp_context=ctx,
            ) as executor:
                for i, page_count in enumerate(page_counts):
                    for j in range(page_count):
                        futures.append(
                            executor.submit(_render_parallel_multi_doc_job, i, j)
                        )

            for future in ft.as_completed(futures):
                pdf_indice, page_indice, result = future.result()

                results.setdefault(pdf_indice, {})[page_indice + 1] = result
                pbar.update(1)

    return results
