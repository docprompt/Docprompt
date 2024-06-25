from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from docprompt import load_document
from docprompt.utils.splitter import pdf_split_iter_with_max_bytes


def do_split(document):
    return list(
        pdf_split_iter_with_max_bytes(document.file_bytes, 15, 1024 * 1024 * 15)
    )


@pytest.mark.skip(reason="Fixures are missing for this test")
def test_document_split_in_threadpool__does_not_hang():
    source_dir = Path(__file__).parent.parent / "data" / "threadpool_test"

    documents = [load_document(file) for file in source_dir.iterdir()]

    futures = []

    with ThreadPoolExecutor() as executor:
        for document in documents:
            future = executor.submit(do_split, document)

            futures.append(future)

    for future in as_completed(futures):
        assert future.result() is not None
