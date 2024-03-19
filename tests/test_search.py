from .fixtures import PDF_FIXTURES

from docprompt import DocumentNode, load_document
from pytest import raises


def test_search():
    document = load_document(PDF_FIXTURES[0].get_full_path())
    document_node = DocumentNode.from_document(document)

    ocr_results = PDF_FIXTURES[0].get_ocr_results()

    with raises(ValueError):
        document_node.refresh_locator()

    for page_num, ocr_results in ocr_results.items():
        document_node.page_nodes[page_num - 1].ocr_results.results[
            ocr_results.provider_name
        ] = ocr_results

    assert document_node._locator is None  # Ensure the locator is not set

    locator = document_node.locator

    result = locator.search("word that doesn't exist")

    assert len(result) == 0

    result_all_pages = locator.search("and")

    assert len(result_all_pages) == 50

    result_page_1 = locator.search("rooted", page_number=1)

    assert len(result_page_1) == 1
