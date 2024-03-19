from os import PathLike
from pathlib import Path
from subprocess import PIPE, CompletedProcess, run
from typing import Literal, Union

GS = "gs"


class GhostscriptError(Exception):
    def __init__(self, message: str, process: CompletedProcess) -> None:
        self.process = process
        super().__init__(message)


def compress_pdf(
    fp: Union[PathLike, str],  # Ghostscript insists on a file instead of bytes
    output_path: str,
    *,
    compression: Literal["jpeg", "lossless"] = "jpeg",
):
    compression_args = []
    if compression == "jpeg":
        compression_args = [
            "-dAutoFilterColorImages=false",
            "-dColorImageFilter=/DCTEncode",
            "-dAutoFilterGrayImages=false",
            "-dGrayImageFilter=/DCTEncode",
        ]
    elif compression == "lossless":
        compression_args = [
            "-dAutoFilterColorImages=false",
            "-dColorImageFilter=/FlateEncode",
            "-dAutoFilterGrayImages=false",
            "-dGrayImageFilter=/FlateEncode",
        ]
    else:
        compression_args = [
            "-dAutoFilterColorImages=true",
            "-dAutoFilterGrayImages=true",
        ]

    args_gs = (
        [
            GS,
            "-q",
            "-dBATCH",
            "-dNOPAUSE",
            "-dSAFER",
            "-dCompatibilityLevel=1.5",
            "-sDEVICE=pdfwrite",
            "-dAutoRotatePages=/None",
            "-sColorConversionStrategy=LeaveColorUnchanged",
        ]
        + compression_args
        + [
            "-dJPEGQ=95",
            "-dPDFA=2",
            "-dPDFACompatibilityPolicy=1",
            "-sOutputFile=" + output_path,
            str(fp),
        ]
    )

    result = run(args_gs, stdout=PIPE, stderr=PIPE, check=False)

    if result.returncode != 0:
        raise GhostscriptError("Ghostscript failed to compress the document", result)

    return result


def compress_pdf_to_bytes(
    fp: Union[PathLike, str], *, compression: Literal["jpeg", "lossless"] = "jpeg"
) -> bytes:
    result = compress_pdf(fp, output_path="%stdout", compression=compression)

    return result.stdout


def compress_pdf_to_path(
    fp: Union[PathLike, str],
    output_path: PathLike,
    *,
    compression: Literal["jpeg", "lossless"] = "jpeg",
) -> Path:
    compress_pdf(fp, output_path=str(output_path), compression=compression)

    return Path(output_path)
