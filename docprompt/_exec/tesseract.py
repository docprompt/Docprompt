from os import PathLike
from pathlib import Path
from subprocess import PIPE, CompletedProcess, run
from typing import Dict, List, Union

TESSERACT = "tesseract"


class TesseractError(Exception):
    def __init__(self, message: str, process: CompletedProcess) -> None:
        self.process = process
        super().__init__(message)


def process_image(
    fp: Union[PathLike, str],
    output_path: str,
    *,
    lang: str = "eng",
    config: List[str] = None,
):
    args_tesseract = [
        TESSERACT,
        str(fp),
        output_path,
        "-l",
        lang,
    ]

    if config:
        args_tesseract.extend(config)

    result = run(args_tesseract, stdout=PIPE, stderr=PIPE, check=False)

    if result.returncode != 0:
        raise TesseractError("Tesseract failed to process the image", result)

    return result


def process_image_to_string(
    fp: Union[PathLike, str],
    *,
    lang: str = "eng",
    config: List[str] = None,
) -> str:
    result = process_image(fp, output_path="stdout", lang=lang, config=config)
    return result.stdout.decode().strip()


def process_image_to_file(
    fp: Union[PathLike, str],
    output_path: PathLike,
    *,
    lang: str = "eng",
    config: List[str] = None,
) -> Path:
    process_image(fp, output_path=str(output_path), lang=lang, config=config)
    return Path(output_path)


def process_image_to_dict(
    fp: Union[PathLike, str],
    *,
    lang: str = "eng",
    config: List[str] = None,
) -> Dict[str, str]:
    result = process_image_to_string(fp, lang=lang, config=config)
    lines = result.split("\n")
    return {
        "text": result,
        "lines": lines,
        "word_count": sum(len(line.split()) for line in lines),
        "language": lang,
    }
