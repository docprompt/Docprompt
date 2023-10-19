import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Union

import fsspec
import magic
from pypdf import PdfReader

from docprompt._exec.ghostscript import compress_pdf_to_path
from docprompt.schema import Document, DocumentContainer


def ensure_path(fp: Union[Path, PathLike]) -> Path:
    """
    Ensures that a file path is a Path object
    """
    if not isinstance(fp, Path):
        fp = Path(fp)

    return fp


def is_pdf(fd: Union[Path, PathLike, bytes]) -> bool:
    """
    Determines if a file is a PDF
    """
    if not isinstance(fd, bytes):
        with fsspec.open(fd, "rb") as f:
            fd: bytes = f.read(1024)
            # We only need the first 1024 bytes to determine if it's a PDF

    mime = magic.from_buffer(fd, mime=True)

    return mime == "application/pdf"


def get_page_count(fd: Union[Path, PathLike, bytes]) -> int:
    """
    Determines the number of pages in a PDF
    """
    if not isinstance(fd, bytes):
        with fsspec.open(fd, "rb") as f:
            fd: bytes = f.read()

    reader = PdfReader(BytesIO(fd))

    return len(reader.pages)


def load_document(fp: Union[Path, PathLike], do_compress: bool = False, do_clean: bool = False) -> DocumentContainer:
    """
    Loads a document from a file path
    """
    if not isinstance(fp, Path):
        fp = Path(fp)

    if fp.is_symlink():
        fp = fp.resolve()

    with open(fp, "rb") as f:
        file_bytes: bytes = f.read()

    if not is_pdf(file_bytes):
        raise ValueError("File is not a PDF")

    if do_compress or do_clean:
        with tempfile.TemporaryDirectory(f"_process_{fp.name}") as temp_dir:
            temp_path = Path(temp_dir)
            temp_file = temp_path / fp.name

            with temp_file.open("wb") as f:
                f.write(file_bytes)

            if do_compress:
                compress_pdf_to_path(temp_file, temp_path / "compressed.pdf")
                file_bytes = (temp_path / "compressed.pdf").read_bytes()

            if do_clean:
                raise NotImplementedError("Cleaning with unpaper is not yet implemented")
                # compress_pdf_to_path(temp_file, temp_path / "cleaned.pdf", clean=True)
                # file_bytes = (temp_path / "cleaned.pdf").read_bytes()

    document = Document(name=fp.name, file_path=str(fp), file_bytes=file_bytes)

    return DocumentContainer(document)


def load_document_from_url(url: str, **kwargs):
    with fsspec.open(url, "rb") as f:
        file_bytes: bytes = f.read()

    file_name = url.split("/")[-1]

    if not is_pdf(file_bytes):
        raise ValueError("File is not a PDF")

    document = Document(name=file_name, file_path=url, file_bytes=file_bytes)

    return DocumentContainer(document)


def load_documents_from_urls(urls: list[str], max_workers: int = 5, **kwargs) -> list[DocumentContainer]:
    documents = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(load_document_from_url, url): url for url in urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                documents.append(future.result())
            except Exception as exc:
                print(f"{url} generated an exception: {exc}")

    return documents


def load_containers(fp: Union[Path, PathLike]):
    fp = ensure_path(fp)

    if fp.is_dir():
        return [DocumentContainer.load(p) for p in fp.iterdir()]

    return [DocumentContainer.load(fp)]
