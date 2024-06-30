import concurrent.futures as ft
import logging
import multiprocessing as mp
import os
import queue
import random
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from io import BytesIO
from math import ceil
from os import PathLike
from pathlib import Path
from threading import Lock
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import pypdfium2 as pdfium
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


PDFIUM_LOAD_LOCK = Lock()  # PDF fails to load without this lock
PDFIUM_WRITE_LOCK = Lock()  # Deadlocks occur in threaded environments without this lock
PDFIUM_RASTERIZE_LOCK = (
    Lock()
)  # Rasterization fails without this lock in threaded environments


@contextmanager
def get_pdfium_document(
    fp: Union[PathLike, Path, bytes, str], password: Optional[str] = None
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


T = TypeVar("T")


def chunk_iterable(iterable: Iterable[T], chunk_size: int) -> List[List[T]]:
    """
    Splits an iterable into chunks of specified size, distributing the remainder evenly.

    Args:
        iterable (Iterable[T]): The iterable to be chunked.
        chunk_size (int): The desired size of each chunk.

    Returns:
        List[List[T]]: A list of lists, where each sublist is a chunk.
    """
    # Convert the iterable to a list
    items = list(iterable)
    total_items = len(items)

    # Calculate the number of chunks needed
    num_chunks = (total_items + chunk_size - 1) // chunk_size

    # Calculate the ideal size of each chunk
    ideal_chunk_size = total_items // num_chunks
    remainder = total_items % num_chunks

    # Create the chunks
    chunks = []
    start = 0
    for i in range(num_chunks):
        end = start + ideal_chunk_size + (1 if i < remainder else 0)
        chunks.append(items[start:end])
        start = end

    return chunks


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


def _get_writable_temp_fp_paths(fps: List[Union[PathLike, Path, bytes]], tempdir: str):
    writable_fps = []

    for i, fp in enumerate(fps):
        if isinstance(fp, bytes):
            temp_fp = os.path.join(tempdir, f"{i}_{random.randint(10000, 50000)}.pdf")
            with open(temp_fp, "wb") as f:
                f.write(fp)
            writable_fps.append(temp_fp)
        else:
            writable_fps.append(str(fp))

    return writable_fps


def _get_page_counts_from_pdfs(fps: List[Union[PathLike, Path, bytes]]):
    page_counts = []

    for fp in fps:
        with get_pdfium_document(fp) as pdf:
            page_counts.append(len(pdf))

    return page_counts


def distribute_pdfs(pdf_page_counts, num_cores):
    total_pages = sum(pdf_page_counts.values())
    average_pages_per_core = ceil(total_pages / num_cores)

    # Sort PDFs by page count in descending order
    sorted_pdfs = sorted(
        pdf_page_counts.items(), key=lambda item: item[1], reverse=True
    )

    core_pdf_assignment = {i: defaultdict(list) for i in range(num_cores)}
    core_page_counts = [0] * num_cores
    for pdf, page_count in sorted_pdfs:
        divisor, remainder = divmod(page_count, average_pages_per_core)

        if divisor == 0:
            # Send this PDF to the core with the least amount of pages
            min_core = core_page_counts.index(min(core_page_counts))

            core_pdf_assignment[min_core][pdf].extend(range(page_count))
            core_page_counts[min_core] += page_count
        else:
            page_chunks = chunk_iterable(range(page_count), min(divisor, num_cores))

            # Round robin the chunks to the cores with the least amount of pages

            for i, chunk in enumerate(page_chunks):
                min_core = core_page_counts.index(min(core_page_counts))

                core_pdf_assignment[min_core][pdf].extend(chunk)
                core_page_counts[min_core] += len(chunk)

    return core_pdf_assignment


def process_work(
    mapping: Dict[Tuple[str, str], List[int]],
    post_process_fn,
    return_mode,
    queue: mp.Queue,
):
    for (pdf, password), pages in mapping.items():
        pdf_doc = pdfium.PdfDocument(pdf, password=password, autoclose=True)
        for page in pages:
            pdf_page = pdf_doc[page]
            image = pdf_page.render().to_pil().convert("RGB")

            if post_process_fn:
                image = post_process_fn(image)

            if return_mode == "pil":
                if isinstance(image, Image.Image):
                    result = image
                else:
                    result = Image.open(BytesIO(image))
            else:
                if isinstance(image, bytes):
                    result = image
                else:
                    buffer = BytesIO()
                    image.save(buffer, format="PNG")
                    result = buffer.getvalue()

            queue.put((pdf, page, result), block=True)


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

    with tempfile.TemporaryDirectory(prefix="docprompt_raster_tmp") as tempdir:
        writable_fps = _get_writable_temp_fp_paths(fps, tempdir)
        page_counts = _get_page_counts_from_pdfs(writable_fps)
        total_to_process = sum(page_counts)

        max_workers = min(mp.cpu_count(), total_to_process)

        pdf_page_map = dict(enumerate(page_counts))
        name_to_idx = {fp: i for i, fp in enumerate(writable_fps)}

        core_pdf_assignments = distribute_pdfs(pdf_page_map, max_workers)

        with mp.Manager() as manager:
            mp_queue = manager.Queue()

            processes = []

            with tqdm(total=total_to_process, desc="Rasterizing PDF's") as pbar:
                for core_id, pdf_page_map in core_pdf_assignments.items():
                    data = {
                        (writable_fps[i], passwords[i]): pages
                        for i, pages in pdf_page_map.items()
                    }

                    p = ctx.Process(
                        target=process_work,
                        args=(data, post_process_fn, return_mode, mp_queue),
                    )
                    p.start()
                    processes.append(p)

                results: Dict[int, Dict[int, Union[Image.Image, bytes]]] = {}

                while any(p.is_alive() for p in processes) or not mp_queue.empty():
                    try:
                        pdf, page, result = mp_queue.get(timeout=0.5)
                        i = name_to_idx[pdf]
                        results.setdefault(i, {})[page] = result
                        pbar.update(1)
                    except queue.Empty:
                        pass

    return results
