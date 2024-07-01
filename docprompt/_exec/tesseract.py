import io
import re
import xml.etree.ElementTree as ET
from os import PathLike
from subprocess import PIPE, CompletedProcess, run
from typing import List, TypedDict, Union

from PIL import Image

TESSERACT = "tesseract"


def check_tesseract_installed() -> bool:
    result = run(
        [TESSERACT, "--version"], stdout=PIPE, stderr=PIPE, check=False, text=True
    )
    return result.returncode == 0


class TesseractError(Exception):
    def __init__(self, message: str, process: CompletedProcess) -> None:
        self.process = process
        super().__init__(message)


class BoundingBox(TypedDict):
    x: int
    y: int
    width: int
    height: int


class Word(TypedDict):
    id: str
    bbox: BoundingBox
    text: str
    line_id: str
    block_id: str


class Line(TypedDict):
    id: str
    bbox: BoundingBox
    text: str
    words: List[str]
    block_id: str


class Block(TypedDict):
    id: str
    bbox: BoundingBox
    text: str


class OCRResult(TypedDict):
    blocks: List[Block]
    lines: List[Line]
    words: List[Word]
    language: str


def process_image(
    fp: Union[PathLike, str],
    *,
    lang: str = "eng",
    config: List[str] = None,
) -> str:
    args_tesseract = [
        TESSERACT,
        str(fp),
        "stdout",
        "-l",
        lang,
    ]

    if config is None:
        config = []

    # Add HOCR output format to get word bounding boxes
    config.extend(["-c", "tessedit_create_hocr=1"])

    args_tesseract.extend(config)

    result = run(args_tesseract, stdout=PIPE, stderr=PIPE, check=False, text=True)

    if result.returncode != 0:
        raise TesseractError(
            f"Tesseract failed to process the image, {result.stderr}", result
        )

    return result.stdout


def get_bbox(element) -> BoundingBox:
    title = element.get("title")
    bbox = [int(x) for x in title.split(";")[0].split(" ")[1:]]
    return BoundingBox(
        x=bbox[0], y=bbox[1], width=bbox[2] - bbox[0], height=bbox[3] - bbox[1]
    )


def clean_text(text: str) -> str:
    # Remove extra whitespace
    text = re.sub(r" \n ", " ", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r" +", " ", text).strip()
    return text.strip()


def process_image_to_dict(
    fp: Union[PathLike, str],
    *,
    lang: str = "eng",
    config: List[str] = None,
) -> OCRResult:
    hocr_content = process_image(fp, lang=lang, config=config)

    # Use StringIO to create a file-like object from the string
    hocr_file = io.StringIO(hocr_content)
    root = ET.parse(hocr_file).getroot()

    # Get image dimensions
    image = Image.open(fp)
    img_width, img_height = image.size
    image.close()

    blocks: List[Block] = []
    lines: List[Line] = []
    words: List[Word] = []

    block_id = 0
    line_id = 0
    word_id = 0

    def normalize_bbox(bbox: BoundingBox) -> BoundingBox:
        return BoundingBox(
            x=bbox["x"] / img_width,
            y=bbox["y"] / img_height,
            width=bbox["width"] / img_width,
            height=bbox["height"] / img_height,
        )

    for block in root.findall(".//*[@class='ocr_carea']"):
        block_bbox = normalize_bbox(get_bbox(block))
        block_text = clean_text(" ".join(block.itertext()))
        blocks.append(Block(id=f"block_{block_id}", bbox=block_bbox, text=block_text))

        for line in block.findall(".//*[@class='ocr_line']"):
            line_bbox = normalize_bbox(get_bbox(line))
            line_words: List[str] = []
            line_text: List[str] = []

            for word in line.findall(".//*[@class='ocrx_word']"):
                word_bbox = normalize_bbox(get_bbox(word))
                word_text = clean_text(word.text) if word.text else ""
                words.append(
                    Word(
                        id=f"word_{word_id}",
                        bbox=word_bbox,
                        text=word_text,
                        line_id=f"line_{line_id}",
                        block_id=f"block_{block_id}",
                    )
                )
                line_words.append(f"word_{word_id}")
                line_text.append(word_text)
                word_id += 1

            lines.append(
                Line(
                    id=f"line_{line_id}",
                    bbox=line_bbox,
                    text=" ".join(line_text),
                    words=line_words,
                    block_id=f"block_{block_id}",
                )
            )
            line_id += 1

        block_id += 1

    return OCRResult(
        blocks=blocks,
        lines=lines,
        words=words,
        language=lang,
    )
