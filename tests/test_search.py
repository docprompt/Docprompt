import pickle

from pytest import raises

from docprompt import DocumentNode, load_document

from .fixtures import PDF_FIXTURES


def test_search():
    document = load_document(PDF_FIXTURES[0].get_full_path())
    document_node = DocumentNode.from_document(document)

    ocr_results = PDF_FIXTURES[0].get_ocr_results()

    with raises(ValueError):
        document_node.refresh_locator()

    for page_num, ocr_result in ocr_results.items():
        ocr_result.contribute_to_document_node(document_node, page_number=page_num)

    print(document_node[0].metadata.task_results)

    assert document_node._locator is None  # Ensure the locator is not set

    # Need to make sure an ocr_key is set to avoid ValueError
    locator = document_node.locator

    result = locator.search("word that doesn't exist")

    assert len(result) == 0

    result_all_pages = locator.search("and")

    assert len(result_all_pages) == 50

    result_page_1 = locator.search("rooted", page_number=1)

    assert len(result_page_1) == 1

    result_multiple_words = locator.search("MMAX2 system", page_number=1)

    assert len(result_multiple_words) == 1

    sources = result_multiple_words[0].text_location.source_blocks

    assert len(sources) == 2

    result_multiple_words = locator.search(
        "MMAX2 system", page_number=1, refine_to_word=False
    )

    assert len(result_multiple_words) == 1

    sources = result_multiple_words[0].text_location.source_blocks

    assert len(sources) == 1

    n_best = locator.search_n_best("and", n=3)

    assert len(n_best) == 3

    raw_search = locator.search_raw('content:"rooted"')

    assert len(raw_search) == 1


def test_pickling__removes_locator_document_basis():
    document = load_document(PDF_FIXTURES[0].get_full_path())
    document_node = DocumentNode.from_document(document)

    ocr_results = PDF_FIXTURES[0].get_ocr_results()

    for page_num, ocr_result in ocr_results.items():
        ocr_result.contribute_to_document_node(document_node, page_number=page_num)

    result_page_1 = document_node.locator.search("rooted", page_number=1)

    assert len(result_page_1) == 1

    dumped = pickle.dumps(document_node)

    loaded = pickle.loads(dumped)

    assert loaded._locator is None


def test_pickling__removes_locator_page_basis():
    document = load_document(PDF_FIXTURES[0].get_full_path())
    document_node = DocumentNode.from_document(document)

    ocr_results = PDF_FIXTURES[0].get_ocr_results()

    for page_num, ocr_result in ocr_results.items():
        ocr_result.contribute_to_document_node(document_node, page_number=page_num)

    page = document_node.page_nodes[0]

    result_page_1 = page.search("rooted")

    assert len(result_page_1) == 1

    dumped = pickle.dumps(page)

    loaded = pickle.loads(dumped)

    assert loaded.document._locator is None
