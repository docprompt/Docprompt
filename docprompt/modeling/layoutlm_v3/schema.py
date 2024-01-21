from dataclasses import dataclass
from dataclasses import field as dataclass_field
from io import BytesIO
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from docprompt._exec.ghostscript import rasterize_page_to_bytes
from docprompt.schema.document import Document
from docprompt.schema.layout import NormBBox, TextBlock, deskew_bounding_poly


@dataclass
class LayoutLMV3BaseInput:
    image: Image.Image
    tokens: List[str] = dataclass_field(default_factory=list)
    bboxes: List[Tuple[int, int, int, int]] = dataclass_field(default_factory=list)

    def show_image_with_bboxes(self):
        image_copy = self.image.copy()
        draw = ImageDraw.Draw(image_copy)
        width, height = image_copy.size
        for bbox in self.bboxes:
            bbox_in_pixels = [
                bbox[0] * width / 1000,
                bbox[1] * height / 1000,
                bbox[2] * width / 1000,
                bbox[3] * height / 1000,
            ]
            draw.rectangle(bbox_in_pixels, outline="red", width=2)
        image_copy.show()

    def as_dict(self):
        return {
            "image": self.image,
            "tokens": self.tokens,
            "bboxes": self.bboxes,
        }


def convert_bbox(bbox: NormBBox):
    # Convert to LayoutLMv3 format (x0, y0, x1, y1) in 0-1000 scale
    x0_scaled = max(0, round(bbox.x0 * 1000))
    y0_scaled = max(0, round(bbox.top * 1000))
    x1_scaled = min(1000, round(bbox.x1 * 1000))
    y1_scaled = min(1000, round(bbox.bottom * 1000))

    if y0_scaled > y1_scaled:
        print(bbox)
        raise ValueError("y1 must be less then y0")

    return (x0_scaled, y0_scaled, x1_scaled, y1_scaled)


def text_block_to_norm_input(
    text_block: TextBlock,
) -> Tuple[str, Tuple[int, int, int, int]]:
    """
    Converts a text block to a LayoutLMv3 input object
    """

    text = text_block["text"].replace("\n", " ")
    scaled_bbox = convert_bbox(text_block.bounding_box)

    return text, scaled_bbox


def merge_adjacent_tokens_and_bboxes(tokens: list[str], bboxes: list[Tuple[int, int, int, int]]):
    return tokens, bboxes


def layoutlmv3_inputs_from_document_page(
    document: Document,
    page_number: int,
    image: Optional[Image.Image] = None,
    merge_adjacent_tokens: bool = True,
):
    """
    Returns a list of LayoutLMv3 input objects for a given document
    """

    if page_number > document.num_pages:
        raise ValueError(f"Page number {page_number} is out of range for document {document}")

    if document.text_data is None:
        raise ValueError(f"Document {document} does not have text data. Try running `perform_text_extraction` first")

    text_data = document.text_data

    if image is None:
        image_bytes = rasterize_page_to_bytes(document.file_path, page_number)

        image = Image.open(BytesIO(image_bytes))

    image = image.convert("RGB")  # LayoutLMV3 needs 3 channels

    assert image.size[0] > 0 and image.size[1] > 0, "Image must have non-zero dimensions"

    word_blocks = text_data[page_number].words

    tokens = []
    bboxes = []

    for block in word_blocks:
        if block.confidence is not None and block.confidence < 0.5:
            continue

        if block.direction is not None and block.direction != "UP":
            continue

        if block.bounding_box.top >= block.bounding_box.bottom:
            print("Skipping block due to top >= bottom", block)
            continue

        text, bbox = text_block_to_norm_input(block)
        tokens.append(text)
        bboxes.append(bbox)

    if merge_adjacent_tokens:
        tokens, bboxes = merge_adjacent_tokens_and_bboxes(tokens, bboxes)

    assert len(tokens) == len(bboxes), "Tokens and bboxes must be the same length"

    return LayoutLMV3BaseInput(image=image, tokens=tokens, bboxes=bboxes)
