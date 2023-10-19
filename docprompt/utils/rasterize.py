import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from typing import Dict, List, Optional

from PIL import Image
from tqdm import tqdm

from docprompt._exec.ghostscript import rasterize_page_to_bytes
from docprompt.schema.document import DocumentContainer


def process_func(args):
    file_path, page_number = args

    raster_bytes = rasterize_page_to_bytes(file_path, page_number)

    return raster_bytes, page_number


def rasterize_single_page(
    container: DocumentContainer,
    page_number: int,
):
    """
    Rasterize a single page from a container
    """
    fp_object = container.document.file_path

    return Image.open(BytesIO(rasterize_page_to_bytes(fp_object, page_number)))


def rasterize_container(
    container: DocumentContainer,
    pages: Optional[List[int]] = None,
    num_workers=6,
    **kwargs,
) -> Dict[int, Image.Image]:
    """
    Rasterize images in a container using concurrent.futures

    Warning: Rasterizing all pages at once can be very memory intensive.
    """

    results = {}
    pages = pages or list(range(1, container.document.num_pages))

    worker_count = max(min(multiprocessing.cpu_count() - 1, min(len(pages), num_workers)), 1)

    fp_object = container.document.file_path

    with tqdm(total=len(pages)) as pbar:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = []
            for page_number in pages:
                future = executor.submit(process_func, (fp_object, page_number))
                future.add_done_callback(lambda x: pbar.update(1))
                futures.append(future)

            for future in as_completed(futures):
                raster_bytes, page_number = future.result()

                results[page_number] = Image.open(BytesIO(raster_bytes))

    return results
