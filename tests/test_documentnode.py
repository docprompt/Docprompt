from docprompt import load_document, DocumentNode
from .fixtures import PDF_FIXTURES

from PIL import Image


def test_rasterize_via_page_node():
    document = load_document(PDF_FIXTURES[0].get_full_path())

    document_node = DocumentNode.from_document(document)

    page_node = document_node.page_nodes[0]

    image = page_node.rasterizer.rasterize()

    assert isinstance(image, bytes)

    image = page_node.rasterizer.rasterize(return_mode="pil")

    assert isinstance(image, Image.Image)

    page_node.rasterizer.clear_cache()

    assert not page_node.rasterizer.raster_cache
