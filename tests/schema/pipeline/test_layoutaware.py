from tests.fixtures import PDF_FIXTURES


def test_direct__page_node_layout_aware_text():
    # Create a sample PageNode with some TextBlocks
    fixture = PDF_FIXTURES[0]

    document = fixture.get_document_node()

    page = document.page_nodes[0]

    assert page.ocr_results, "The OCR results should be populated"

    layout_text__property = page.layout_aware_text

    layout_len = 4786

    assert (
        len(layout_text__property) == layout_len
    ), f"The layout-aware text should be {layout_len} characters long"

    layout_text__direct = page.get_layout_aware_text()

    assert (
        layout_text__property == layout_text__direct
    ), "The layout-aware text should be the same"
