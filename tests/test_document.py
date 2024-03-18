from docprompt import load_document
from PIL import Image
import io

from .fixtures import PDF_FIXTURES


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
