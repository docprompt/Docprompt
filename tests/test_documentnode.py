import pickle
from docprompt import load_document, DocumentNode
from .fixtures import PDF_FIXTURES

from PIL import Image
import base64


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

    image_uri = page_node.rasterizer.rasterize_to_data_uri("test")

    assert image_uri.startswith("data:image/png;base64,")

    image_bytes = base64.b64decode(
        image_uri.split("data:image/png;base64,")[1].encode("utf-8")
    )

    assert image_bytes == page_node.rasterizer.rasterize("test", return_mode="bytes")


def test_rasterize_via_document_node():
    document = load_document(PDF_FIXTURES[0].get_full_path())

    document_node = DocumentNode.from_document(document)

    images = document_node.rasterizer.rasterize("default")

    assert len(images) == len(document_node)
    assert all(isinstance(image, bytes) for image in images)
    assert all(
        (
            "default" in page_node.rasterizer.raster_cache
            for page_node in document_node.page_nodes
        )
    )


def test__pickling_drops_cache():
    document = load_document(PDF_FIXTURES[0].get_full_path())

    document_node = DocumentNode.from_document(document)

    page_node = document_node.page_nodes[0]

    page_node.rasterizer.rasterize()

    assert page_node._raster_cache

    dumped = pickle.dumps(page_node)

    loaded = pickle.loads(dumped)

    assert not loaded._raster_cache
