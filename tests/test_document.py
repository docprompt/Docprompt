from docprompt import load_document
from PIL import Image
import io

from .fixtures import PDF_FIXTURES

from docprompt.utils.splitter import pdf_split_iter_fast
from docprompt.rasterize import ProviderResizeRatios


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


def test_rasterize_convert_and_quantize():
    # Fow now just test PIL can open the image
    convert_mode = "L"
    quantize_color_count = 8

    for fixture in PDF_FIXTURES:
        doc = load_document(fixture.get_full_path())
        img_bytes = doc.rasterize_page(
            1,
            do_convert=True,
            image_covert_mode=convert_mode,
            do_quantize=True,
            quantize_color_count=quantize_color_count,
        )
        Image.open(io.BytesIO(img_bytes))


def test_rasterize_resize__regular():
    # Fow now just test PIL can open the image
    resize_width = 100
    resize_height = 100

    for fixture in PDF_FIXTURES:
        doc = load_document(fixture.get_full_path())
        img_bytes = doc.rasterize_page(
            1, downscale_size=(resize_width, resize_height), resize_mode="resize"
        )
        result = Image.open(io.BytesIO(img_bytes))

        assert result.size == (resize_width, resize_height)


def test_rasterize_resize__thumbnail():
    # Fow now just test PIL can open the image
    resize_width = 100
    resize_height = 100

    for fixture in PDF_FIXTURES:
        doc = load_document(fixture.get_full_path())
        img_bytes = doc.rasterize_page(
            1, downscale_size=(resize_width, resize_height), resize_mode="thumbnail"
        )
        result = Image.open(io.BytesIO(img_bytes))

        result_width, result_height = result.size

        assert result_width == resize_width or result_height == resize_height


def test_max_image_file_size():
    for fixture in PDF_FIXTURES:
        doc = load_document(fixture.get_full_path())

        initial_img_bytes = doc.rasterize_page(1)

        resize_bytes = doc.rasterize_page(
            1, max_file_size_bytes=len(initial_img_bytes) - 10000
        )

        assert len(resize_bytes) < len(initial_img_bytes)


def test_rasterize_aspect_ratio_rules():
    document = load_document(PDF_FIXTURES[0].get_full_path())
    ratios = ProviderResizeRatios.ANTHROPIC.value

    img_bytes = document.rasterize_page(
        1, resize_aspect_ratios=ratios, resize_mode="resize"
    )

    img = Image.open(io.BytesIO(img_bytes))

    for ratio in ratios:
        if all(
            [
                img.width <= ratio.max_width,
                img.height <= ratio.max_height,
                img.width / img.height == ratio.ratio,
            ]
        ):
            break


def test_split():
    doc = load_document(PDF_FIXTURES[0].get_full_path())

    new_docs = doc.split(start=2)

    assert len(new_docs) == len(doc) - 2


def test_pdf_split_iter_fast_1_sized():
    doc = load_document(PDF_FIXTURES[0].get_full_path())

    splits = list(pdf_split_iter_fast(doc.file_bytes, 1))

    assert len(splits) == len(doc)


def test_compression():
    for fixture in PDF_FIXTURES:
        doc = load_document(fixture.get_full_path())
        compressed_bytes = doc.to_compressed_bytes()

        assert len(compressed_bytes) < len(doc.file_bytes)
