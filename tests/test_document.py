from docprompt import load_document
from PIL import Image
import io

from .fixtures import PDF_FIXTURES

from docprompt.utils.splitter import pdf_split_iter_fast


def test_load_document():
    for fixture in PDF_FIXTURES:
        doc = load_document(fixture.get_full_path())
        assert doc.page_count == fixture.page_count
        assert doc.document_hash == fixture.file_hash


def test_rasterize():
    # Fow now just test PIL can open the image
    for fixture in PDF_FIXTURES:
        doc = load_document(fixture.get_full_path())
        img_bytes = doc.rasterize_page(1)
        Image.open(io.BytesIO(img_bytes))


def test_split():
    doc = load_document(PDF_FIXTURES[0].get_full_path())

    new_docs = doc.split(start=2)

    assert len(new_docs) == len(doc) - 2


def test_pdf_split_iter_fast_1_sized():
    doc = load_document(PDF_FIXTURES[0].get_full_path())

    splits = list(pdf_split_iter_fast(doc.file_bytes, 1))

    assert len(splits) == len(doc)
