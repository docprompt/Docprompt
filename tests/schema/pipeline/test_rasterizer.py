import io
from unittest.mock import patch

import pytest
from PIL import Image

from docprompt import DocumentNode, load_document
from docprompt.schema.document import PdfDocument
from docprompt.schema.pipeline.rasterizer import (
    DocumentRasterCache,
    DocumentRasterizer,
    FilesystemCache,
    InMemoryCache,
    PageRasterizer,
)
from tests.fixtures import PDF_FIXTURES


@pytest.fixture
def real_document_node():
    document = load_document(PDF_FIXTURES[0].get_full_path())
    return DocumentNode.from_document(document)


@pytest.fixture
def real_page_node(real_document_node):
    return real_document_node.page_nodes[0]


@pytest.fixture
def pdf_document():
    return PdfDocument.from_path(PDF_FIXTURES[0].get_full_path())


@pytest.fixture
def in_memory_cache():
    return InMemoryCache()


@pytest.fixture
def filesystem_cache():
    return FilesystemCache()


@pytest.fixture
def document_raster_cache(pdf_document):
    return DocumentRasterCache(document=pdf_document, cache_url="memory")


@pytest.fixture
def page_rasterizer(real_page_node):
    return PageRasterizer(owner=real_page_node)


@pytest.fixture
def document_rasterizer(mock_document_node):
    return DocumentRasterizer(owner=mock_document_node, cache_url="memory")


class TestInMemoryCache:
    def test_set_and_get(self, in_memory_cache):
        in_memory_cache.set("key1", b"value1")
        assert in_memory_cache.get("key1") == b"value1"

    def test_has_key(self, in_memory_cache):
        in_memory_cache.set("key1", b"value1")
        assert in_memory_cache.has_key("key1")
        assert not in_memory_cache.has_key("key2")

    def test_list_prefix(self, in_memory_cache):
        in_memory_cache.set("prefix1/key1", b"value1")
        in_memory_cache.set("prefix1/key2", b"value2")
        in_memory_cache.set("prefix2/key3", b"value3")
        assert set(in_memory_cache.list_prefix("prefix1")) == {
            "prefix1/key1",
            "prefix1/key2",
        }

    def test_clear(self, in_memory_cache):
        in_memory_cache.set("key1", b"value1")
        in_memory_cache.clear()
        assert not in_memory_cache.has_key("key1")

    def test_pop(self, in_memory_cache):
        in_memory_cache.set("key1", b"value1")
        assert in_memory_cache.pop("key1") == b"value1"
        assert not in_memory_cache.has_key("key1")


class TestFilesystemCache:
    def test_set_and_get(self, filesystem_cache):
        filesystem_cache.set("key1", b"value1")
        assert filesystem_cache.get("key1") == b"value1"

    def test_has_key(self, filesystem_cache):
        filesystem_cache.set("key1", b"value1")
        assert filesystem_cache.has_key("key1")
        assert not filesystem_cache.has_key("key2")

    def test_list_prefix(self, filesystem_cache):
        filesystem_cache.set("prefix1/key1", b"value1")
        filesystem_cache.set("prefix1/key2", b"value2")
        filesystem_cache.set("prefix2/key3", b"value3")
        assert set(filesystem_cache.list_prefix("prefix1")) == {
            f"{filesystem_cache.cache_dir}prefix1/key1",
            f"{filesystem_cache.cache_dir}prefix1/key2",
        }

    def test_clear(self, filesystem_cache):
        filesystem_cache.set("key1", b"value1")
        filesystem_cache.clear()
        assert not filesystem_cache.has_key("key1")

    def test_pop(self, filesystem_cache):
        filesystem_cache.set("key1", b"value1")
        assert filesystem_cache.pop("key1") == b"value1"
        assert not filesystem_cache.has_key("key1")


class TestDocumentRasterCache:
    def test_cached_pages(self, document_raster_cache):
        document_raster_cache.set_image_for_page("test", 1, b"image1")
        document_raster_cache.set_image_for_page("test", 2, b"image2")
        assert set(document_raster_cache.cached_pages("test")) == {1, 2}

    def test_cache_proportion(self, document_raster_cache):
        total_pages = len(document_raster_cache.document)
        document_raster_cache.set_image_for_page("test", 1, b"image1")
        document_raster_cache.set_image_for_page("test", 2, b"image2")
        assert document_raster_cache.cache_proportion("test") == 2 / total_pages

    def test_fully_cached(self, document_raster_cache):
        total_pages = len(document_raster_cache.document)
        for i in range(1, total_pages + 1):
            document_raster_cache.set_image_for_page("test", i, b"image")

        assert document_raster_cache.fully_cached("test")

    def test_get_and_set_image_for_page(self, document_raster_cache):
        document_raster_cache.set_image_for_page("test", 1, b"image1")
        assert document_raster_cache.get_image_for_page("test", 1) == b"image1"


class TestPageRasterizer:
    def test_rasterize(self, real_page_node):
        page_rasterizer = PageRasterizer(owner=real_page_node)
        result = page_rasterizer.rasterize(name="test", dpi=100)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_rasterize_to_data_uri(self, real_page_node):
        page_rasterizer = PageRasterizer(owner=real_page_node)
        result = page_rasterizer.rasterize_to_data_uri(name="test", dpi=100)
        assert result.startswith("data:image/png;base64,")

    def test_clear_cache(self, real_page_node):
        page_rasterizer = PageRasterizer(owner=real_page_node)
        page_rasterizer.rasterize(name="test", dpi=100)
        page_rasterizer.clear_cache("test")
        assert page_rasterizer.document_cache.get_image_for_page("test", 1) is None


class TestDocumentRasterizer:
    def test_rasterize(self, real_document_node):
        document_rasterizer = DocumentRasterizer(owner=real_document_node)
        result = document_rasterizer.rasterize(
            name="test", return_mode="bytes", dpi=100
        )
        assert isinstance(result, list)
        assert all(isinstance(img, bytes) for img in result)
        assert len(result) == len(real_document_node.page_nodes)

    def test_rasterize_cached(self, real_document_node: DocumentNode):
        # Clear the cache before the test
        real_document_node.rasterizer.cache.cache.clear()

        # First call should rasterize and cache the results
        initial_result = real_document_node.rasterizer.rasterize()

        assert isinstance(initial_result, list), "Initial result should be a list"
        assert (
            len(initial_result) == len(real_document_node.page_nodes)
        ), f"Expected {len(real_document_node.page_nodes)} pages, got {len(initial_result)}"
        assert all(
            isinstance(img, bytes) for img in initial_result
        ), "All items in initial_result should be bytes"
        assert (
            len(real_document_node.rasterizer.cache.cache.list_prefix("default")) == 6
        ), "All pages should be cached after initial rasterization"

        # Verify that all pages are cached
        assert real_document_node.rasterizer.cache.fully_cached(
            "default"
        ), "All pages should be cached after initial rasterization"

        # Second call should use cached results
        with patch(
            "docprompt.schema.document.PdfDocument.rasterize_pdf"
        ) as mock_rasterize_pdf:
            cached_result = real_document_node.rasterizer.rasterize()
            mock_rasterize_pdf.assert_not_called()

        assert (
            len(real_document_node.rasterizer.cache.cache.list_prefix("default")) == 6
        ), "All pages should be cached"
        assert len(cached_result) == len(
            initial_result
        ), "Cached result should have the same length as the initial result"

    def test_memory_cache(self, real_document_node):
        document_rasterizer = DocumentRasterizer(
            owner=real_document_node, cache_url="memory"
        )
        assert isinstance(document_rasterizer.cache.cache, InMemoryCache)

        real_document_node.rasterizer = document_rasterizer

        initial_result = real_document_node.rasterizer.rasterize()

        assert isinstance(initial_result, list), "Initial result should be a list"
        assert (
            len(initial_result) == len(real_document_node.page_nodes)
        ), f"Expected {len(real_document_node.page_nodes)} pages, got {len(initial_result)}"
        assert all(
            isinstance(img, bytes) for img in initial_result
        ), "All items in initial_result should be bytes"
        assert (
            len(real_document_node.rasterizer.cache.cache.list_prefix("default")) == 6
        ), "All pages should be cached after initial rasterization"


@pytest.mark.parametrize("cache_type", ["InMemoryCache", "FilesystemCache"])
def test_cache_integration(cache_type, tmp_path):
    if cache_type == "InMemoryCache":
        cache = InMemoryCache()
    else:
        cache = FilesystemCache()

    # Test set and get
    cache.set("key1", b"value1")
    assert cache.get("key1") == b"value1"

    # Test has_key
    assert cache.has_key("key1")
    assert not cache.has_key("key2")

    # Test list_prefix
    cache.set("prefix1/key1", b"value1")
    cache.set("prefix1/key2", b"value2")
    cache.set("prefix2/key3", b"value3")
    prefix1_keys = cache.list_prefix("prefix1")
    assert len(prefix1_keys) == 2
    assert all(key.endswith(("key1", "key2")) for key in prefix1_keys)

    # Test clear
    cache.clear()
    assert not cache.has_key("key1")
    assert not cache.has_key("prefix1/key1")

    # Test pop
    cache.set("key2", b"value2")
    assert cache.pop("key2") == b"value2"
    assert not cache.has_key("key2")


def test_document_rasterizer_integration(real_document_node, tmp_path):
    rasterizer = DocumentRasterizer(owner=real_document_node)

    result = rasterizer.rasterize(name="test", return_mode="bytes", dpi=100)

    assert isinstance(result, list)
    assert all(isinstance(img, bytes) for img in result)
    assert len(result) == len(real_document_node.page_nodes)


def test_page_rasterizer_integration(real_page_node, tmp_path):
    rasterizer = PageRasterizer(owner=real_page_node)

    # Test rasterization and caching
    result1 = rasterizer.rasterize(name="test", return_mode="bytes", dpi=100)
    assert isinstance(result1, bytes)

    # Test that the results are cached
    result2 = rasterizer.rasterize(name="test", return_mode="bytes", dpi=100)
    assert result2 == result1

    # Test rasterization with different parameters
    result3 = rasterizer.rasterize(name="test_hires", return_mode="bytes", dpi=300)
    assert isinstance(result3, bytes)
    assert result3 != result1

    # Test clear cache
    rasterizer.clear_cache("test")
    result4 = rasterizer.rasterize(name="test", return_mode="bytes", dpi=100)
    assert result4 == result1

    # Test rasterize_to_data_uri
    data_uri = rasterizer.rasterize_to_data_uri(name="test", dpi=100)
    assert data_uri.startswith("data:image/png;base64,")


def test_insert_image(real_page_node, tmp_path):
    rasterizer = PageRasterizer(owner=real_page_node)

    # Create a small valid PNG image
    img = Image.new("RGB", (10, 10), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    # Test inserting a valid image
    rasterizer.insert_image("test", 1, img_bytes)
    assert rasterizer.document_cache.get_image_for_page("test", 1) == img_bytes

    # Test inserting an invalid image
    with pytest.raises(ValueError):
        rasterizer.insert_image("test", 2, b"invalid_image_data")

    # Test inserting an invalid image with skip_validation
    rasterizer.insert_image("test", 3, b"invalid_image_data", skip_validation=True)
    assert (
        rasterizer.document_cache.get_image_for_page("test", 3) == b"invalid_image_data"
    )


def test_rasterize_via_page_node():
    document = load_document(PDF_FIXTURES[0].get_full_path())
    document_node = DocumentNode.from_document(document)

    page_node = document_node.page_nodes[0]
    page_rasterizer = PageRasterizer(owner=page_node)

    result = page_rasterizer.rasterize(name="test", dpi=100)
    assert isinstance(result, bytes)
    assert len(result) > 0

    data_uri = page_rasterizer.rasterize_to_data_uri(name="test", dpi=100)
    assert data_uri.startswith("data:image/png;base64,")
