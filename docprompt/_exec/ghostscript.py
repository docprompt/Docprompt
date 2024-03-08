import multiprocessing
import tempfile
from os import PathLike
from pathlib import Path
from subprocess import PIPE, CompletedProcess, run
from typing import Dict, List, Literal, Optional, Union

GS = "gs"

_RENDER_THREAD_COUNT = max(min(multiprocessing.cpu_count() - 2, 8), 1)


class GhostscriptError(Exception):
    def __init__(self, message: str, process: CompletedProcess) -> None:
        self.process = process
        super().__init__(message)


def _validate_device(device: str) -> str:
    if device not in ["pnggray", "png16m", "png256"]:
        raise ValueError("Invalid device")

    return device


def rasterize_page(
    fp: Union[PathLike, str],
    output_path: str,
    idx: int,
    *,
    dpi: int = 200,
    device="pnggray",
):
    device = _validate_device(device)
    args = [
        GS,
        "-q",
        "-dSAFER",
        "-dBATCH",
        "-dNOPAUSE",
        "-sDEVICE=png16m",
        "-dTextAlphaBits=4",
        "-dGraphicsAlphaBits=4",
        f"-sDEVICE={device}",
        f"-dFirstPage={idx}",
        f"-dLastPage={idx}",
        f"-r{dpi}",
        f"-sOutputFile={output_path}",
        "-f",
        str(fp),
    ]

    result = run(args, stdout=PIPE, stderr=PIPE, check=False)

    if result.returncode != 0:
        raise GhostscriptError("Ghostscript failed to rasterize the document: ", result)

    return result


def rasterize_pdf(
    fp: Union[PathLike, str],
    output_path: str,
    *,
    dpi: int = 100,
    device="pnggray",
    downscale_factor: Optional[int] = None,
):
    device = _validate_device(device)
    base_args = [
        GS,
        "-q",
        "-dSAFER",
        "-dBATCH",
        "-dNOPAUSE",
        "-dTextAlphaBits=4",
        "-dGraphicsAlphaBits=4",
        "-dBufferSpace=250000000",  # 250 Mb of buffer space.
        f"-dNumRenderingThreads={_RENDER_THREAD_COUNT}",
    ]

    if downscale_factor is not None:
        base_args += [f"-dDownScaleFactor={downscale_factor}"]

    args = base_args + [
        f"-sDEVICE={device}",
        f"-r{dpi}",
        f"-sOutputFile={output_path}",
        "-f",
        str(fp),
    ]

    result = run(args, stdout=PIPE, stderr=PIPE, check=False)

    if result.returncode != 0:
        raise GhostscriptError("Ghostscript failed to rasterize the document: ", result)

    return result


def split_png_images(data: bytes) -> List[bytes]:
    # PNG signature
    png_signature = b'\x89PNG\r\n\x1a\n'
    images = []

    # Find the first PNG signature
    start = data.find(png_signature)
    if start == -1:
        # No PNG images found in the data
        return []

    while True:
        # Find the next PNG signature in the data
        next_start = data.find(png_signature, start + 1)
        if next_start == -1:
            # No more images
            images.append(data[start:])
            break
        images.append(data[start:next_start])
        start = next_start

    return images


def rasterize_pdf_to_bytes(
    fp: Union[PathLike, str], *, dpi: int = 100, device="pnggray", downscale_factor: Optional[int] = None
) -> Dict[int, bytes]:
    if "png" not in device:
        raise ValueError("Device must be a PNG device for rasterize_pdf_to_bytes")

    result = rasterize_pdf(fp, "%stdout", dpi=dpi, device=device, downscale_factor=downscale_factor)

    images = split_png_images(result.stdout)

    return {idx: image for idx, image in enumerate(images, start=1)}


def rasterize_page_to_bytes(fp: Union[PathLike, str], idx: int, *, dpi: int = 200, device="pnggray") -> bytes:
    if isinstance(fp, bytes):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            f.write(fp)
            f.flush()
            result = rasterize_page(f.file.name, "%stdout", idx, dpi=dpi, device=device)
    else:
        result = rasterize_page(fp, "%stdout", idx, dpi=dpi, device=device)

    return result.stdout


def rasterize_page_to_path(
    fp: Union[PathLike, str],
    idx: int,
    output_path: PathLike,
    *,
    dpi: int = 200,
    device="pnggray",
) -> Path:
    if isinstance(fp, bytes):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            f.write(fp)
            f.flush()
            rasterize_page(f.file.name, str(output_path), idx, dpi=dpi, device=device)
    else:
        rasterize_page(fp, str(output_path), idx, dpi=dpi, device=device)

    return Path(output_path)


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


def compress_pdf_to_bytes(fp: Union[PathLike, str], *, compression: Literal["jpeg", "lossless"] = "jpeg") -> bytes:
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
