from docprompt import load_document

from .fixtures import PDF_FIXTURES


def test_load_document():
    for fixture in PDF_FIXTURES:
        doc = load_document(fixture.get_full_path())
        assert doc.page_count == fixture.page_count
        assert doc.document_hash == fixture.file_hash
