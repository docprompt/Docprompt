import warnings
from typing import Dict, List, Literal, NamedTuple, Optional, TypedDict

from PIL import Image

from docprompt.schema import DocumentContainer
from docprompt.schema.layout import TextBlock
from docprompt.utils.rasterize import rasterize_single_page

try:
    from transformers import pipeline
except ImportError:
    print("transformers not installed, table extraction will not work")
    pipeline = None


class ExtractedTable(NamedTuple):
    column_names: list[str]
    rows: list[list[str]]


class TableLayoutAnalysisItem(TypedDict):
    score: float
    label: Literal["table", "row", "column", "column_header"]
    text: str

    x0: int
    y0: int
    x1: int
    y1: int


class DetectedTable(NamedTuple):
    image_width: int
    image_height: int

    score: float
    x0: int
    y0: int
    x1: int
    y1: int


def get_table_detector(device: str = "cpu"):
    pipe = pipeline("object-detection", model="microsoft/table-transformer-detection", device=device)
    return pipe


def get_table_layout_detector(device: str = "cpu"):
    pipe = pipeline(
        "table-question-answering", model="microsoft/table-transformer-structure-recognition", device=device
    )
    return pipe


def prepare_image_for_table_detection(image: Image.Image) -> Image.Image:
    return image.copy()


def detect_tables_in_image(image: Image.Image, detector=None) -> list[DetectedTable]:
    detector = detector or get_table_detector()

    working_image = prepare_image_for_table_detection(image)

    width, height = working_image.size

    detection_results = detector(working_image)

    detected_tables = []

    for detection in detection_results:
        detected_tables.append(
            DetectedTable(
                image_width=width,
                image_height=height,
                score=detection["score"],
                x0=detection["box"]["xmin"],
                y0=detection["box"]["ymin"],
                x1=detection["box"]["xmax"],
                y1=detection["box"]["ymax"],
            )
        )

    return detected_tables


def detect_layout_from_table_image(table_image: Image.Image, detector=None) -> list[TableLayoutAnalysisItem]:
    detector = detector or get_table_layout_detector

    detector_results = detector(table_image)

    result_map = {
        "table row": "row",
        "table column": "column",
        "table column header": "column_header",
        "table": "table",
    }

    results = []

    for detection in detector_results:
        results.append(
            TableLayoutAnalysisItem(
                score=detection["score"],
                text="",
                label=result_map[detection["label"]],
                x0=detection["box"]["xmin"],
                y0=detection["box"]["ymin"],
                x1=detection["box"]["xmax"],
                y1=detection["box"]["ymax"],
            )
        )

    return results


def get_cropped_table_image(image: Image.Image, table: DetectedTable) -> Image.Image:
    return image.crop((table.x0, table.y0, table.x1, table.y1))


def show_found_tables_in_image(image: Image.Image, detected_tables: list[DetectedTable]):
    from PIL import ImageDraw

    image_copy = image.convert("RGB")

    draw = ImageDraw.Draw(image_copy)

    for table in detected_tables:
        draw.rectangle(
            (table.x0, table.y0, table.x1, table.y1),
            outline="red",
            width=3,
        )

    return image_copy


def get_text_for_table_layout_item(
    item: TableLayoutAnalysisItem,
    word_blocks: List[TextBlock],
    original_width: int,
    original_height: int,
    table_image_width: int,
    table_image_height: int,
) -> str:
    width_ratio = original_width / table_image_width
    height_ratio = original_height / table_image_height

    # Convert to normalized coordinates 0-1 for the original image
    # in order to match with the word blocks
    x0 = (item["x0"] * width_ratio) / original_width
    y0 = (item["y0"] * height_ratio) / original_height
    x1 = (item["x1"] * width_ratio) / original_width
    y1 = (item["y1"] * height_ratio) / original_height

    valid_word_blocks = []

    for block in word_blocks:
        if (
            block.bounding_box.x0 >= x0
            and block.bounding_box.y0 >= y0
            and block.bounding_box.x1 <= x1
            and block.bounding_box.y1 <= y1
        ):
            valid_word_blocks.append(block)

    # Sort in reading order
    valid_word_blocks.sort(key=lambda block: (block.bounding_box.y0, block.bounding_box.x0))

    return " ".join([block.text for block in valid_word_blocks])


def extract_tables_from_page(
    image: Image.Image,
    word_blocks: List[TextBlock],
    table_detector=None,
    table_layout_detector=None,
) -> list[ExtractedTable]:
    extracted_tables = []

    detected_tables = detect_tables_in_image(image, table_detector)

    original_width, original_height = image.size

    for table in detected_tables:
        cropped_image = get_cropped_table_image(image, table)
        cropped_width, cropped_height = cropped_image.size

        layout_results = detect_layout_from_table_image(cropped_image, table_layout_detector)

        for result in layout_results:
            result["text"] = get_text_for_table_layout_item(
                result,
                word_blocks,
                original_width=original_width,
                original_height=original_height,
                table_image_width=cropped_width,
                table_image_height=cropped_height,
            )

        extracted_tables.append(layout_results)

    return extracted_tables


def extract_tables_from_container(
    container: DocumentContainer,
    pages: Optional[List[int]] = None,
    table_detector=None,
    table_layout_detector=None,
):
    pages = pages or list(range(1, container.document.num_pages))

    detected_tables = {}

    for page_number in pages:
        if container.text_data[page_number] is None:
            warnings.warn(f"No text data found for page {page_number}, skipping...")
            detected_tables[page_number] = []
            continue

        word_blocks = container.text_data[page_number].words
        rasterized_page = rasterize_single_page(container, page_number)

        extracted_table = extract_tables_from_page(rasterized_page, word_blocks, table_detector, table_layout_detector)

        detected_tables[page_number] = extracted_table

    return detected_tables
