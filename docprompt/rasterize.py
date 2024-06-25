import logging
from enum import Enum
from io import BytesIO
from typing import Iterable, Literal, Optional, Union

from PIL import Image, ImageDraw
from pydantic import BaseModel

from docprompt.schema.layout import NormBBox

logger = logging.getLogger(__name__)


ResizeModes = Literal["thumbnail", "resize"]

PILOrBytes = Union[Image.Image, bytes]


class AspectRatioRule(BaseModel):
    ratio: float
    max_width: int
    max_height: int


def save_image_to_bytes(image: Image.Image, format: str = "PNG", **kwargs) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format=format, **kwargs)
    return buffer.getvalue()


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(image_bytes))


def estimate_png_byte_size(
    image: Image.Image,
    assummed_compression_ratio: float = 4.0,
    overhead_bytes: int = 1024,
) -> int:
    """
    Provides an estimate of the size of a PNG image given the uncompressed size and an assumed compression ratio.

    The default compression ratio of 4.0 is based on the assumption that the image is a document, and represents a
    pessimistic estimate.
    """
    width, height = image.size
    mode = image.mode

    # Determine bytes per pixel based on image mode
    if mode == "1":
        bytes_per_pixel = 1 / 8  # 1 bit per pixel
    elif mode == "L":
        bytes_per_pixel = 1  # 1 byte per pixel
    elif mode == "LA":
        bytes_per_pixel = 2  # 2 bytes per pixel
    elif mode == "RGB":
        bytes_per_pixel = 3  # 3 bytes per pixel
    elif mode == "RGBA":
        bytes_per_pixel = 4  # 4 bytes per pixel
    else:
        raise ValueError(f"Unsupported image mode: {mode}")

    uncompressed_size = width * height * bytes_per_pixel
    compressed_size = uncompressed_size / assummed_compression_ratio

    return int(compressed_size + overhead_bytes)


def resize_image_to_closest_aspect_ratio(
    image: PILOrBytes,
    ratios: Iterable[AspectRatioRule],
    *,
    resize_mode: ResizeModes = "thumbnail",
) -> Image.Image:
    if isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    original_width, original_height = image.size

    original_ratio = original_width / original_height

    closest_aspect_ratio = min(
        ratios,
        key=lambda x: abs(x.ratio - original_ratio),
    )

    if (
        closest_aspect_ratio.ratio == original_ratio
        and original_width <= closest_aspect_ratio.max_width
        and original_height <= closest_aspect_ratio.max_height
    ):
        return image

    if resize_mode == "thumbnail":
        image = image.copy()
        image.thumbnail(
            (closest_aspect_ratio.max_width, closest_aspect_ratio.max_height)
        )
    elif resize_mode == "resize":
        image = image.resize(
            (closest_aspect_ratio.max_width, closest_aspect_ratio.max_height)
        )

    return image


def resize_image_to_fize_size_limit(
    image: PILOrBytes,
    max_file_size_bytes: int,
    *,
    resize_mode: ResizeModes = "thumbnail",
    resize_step_size: float = 0.1,
    allow_channel_reduction: bool = True,
    image_convert_mode: str = "L",
) -> Image.Image:
    """
    Incrementally resizes an image until it is under a certain file size
    """
    if resize_step_size <= 0 or resize_step_size >= 0.5:
        raise ValueError("resize_step_size must be between 0 and 0.5")

    if isinstance(image, bytes):
        image = load_image_from_bytes(image)

    estimated_bytes = estimate_png_byte_size(image)

    if estimated_bytes < max_file_size_bytes:
        return image

    # Convert image to the desired mode if it has multiple channels
    if allow_channel_reduction and image.mode in ["LA", "RGBA"]:
        image = image.convert(image_convert_mode)

        if estimate_png_byte_size(image) < max_file_size_bytes:
            return image

    step_count = 0
    working_image = image.copy()

    while estimated_bytes > max_file_size_bytes:
        new_width = int(image.width * (1 - resize_step_size * step_count))
        new_height = int(image.height * (1 - resize_step_size * step_count))

        if new_width <= 200 or new_height <= 200:
            logger.warning(
                f"Image could not be resized to under {max_file_size_bytes} bytes. Reached {estimated_bytes} bytes."
            )
            break

        if resize_mode == "thumbnail":
            working_image.thumbnail((new_width, new_height))
        elif resize_mode == "resize":
            working_image = working_image.resize((new_width, new_height))

        estimated_bytes = estimate_png_byte_size(working_image)

        if estimated_bytes < max_file_size_bytes:
            return working_image

        step_count += 1

    return working_image


def resize_image(
    image: PILOrBytes,
    *,
    width: int,
    height: int,
    resize_mode: ResizeModes = "thumbnail",
):
    from_bytes = False
    if isinstance(image, bytes):
        image = load_image_from_bytes(image)
        from_bytes = True

    if resize_mode == "thumbnail":
        if not from_bytes:
            image = image.copy()

        image.thumbnail((width, height))
    elif resize_mode == "resize":
        image = image.resize((width, height))

    return image


def process_raster_image(
    image: Image.Image,
    *,
    resize_width: Optional[int] = None,
    resize_height: Optional[int] = None,
    resize_mode: ResizeModes = "thumbnail",
    resize_aspect_ratios: Optional[Iterable[AspectRatioRule]] = None,
    do_convert: bool = False,
    image_convert_mode: str = "L",
    do_quantize: bool = False,
    quantize_color_count: int = 8,
    max_file_size_bytes: Optional[int] = None,
) -> Image.Image:
    if resize_aspect_ratios:
        image = resize_image_to_closest_aspect_ratio(
            image,
            resize_aspect_ratios,
            resize_mode=resize_mode,
        )
    elif resize_width and resize_height:
        image = resize_image(
            image,
            width=resize_width,
            height=resize_height,
            resize_mode=resize_mode,
        )

    if do_convert:
        image = image.convert(image_convert_mode)

    if do_quantize:
        image = image.quantize(colors=quantize_color_count)

    if max_file_size_bytes and estimate_png_byte_size(image) > max_file_size_bytes:
        image = resize_image_to_fize_size_limit(
            image,
            max_file_size_bytes,
            resize_mode=resize_mode,
            resize_step_size=0.1,
        )

    return image


def mask_image_from_bboxes(
    image: PILOrBytes,
    bboxes: Iterable[NormBBox],
    *,
    mask_color: Union[str, int] = "black",
):
    """
    Given a set of normalized bounding boxes, masks the image.
    :param image: PIL Image object or bytes object representing an image.
    :param bboxes: Iterable of NormBBox objects.
    :param mask_color: Color used for the mask, can be a string (e.g., "black") or a tuple (e.g., (0, 0, 0)).
    """
    # Convert bytes image to PIL Image if necessary
    if isinstance(image, bytes):
        image = load_image_from_bytes(image)

    # Get image dimensions
    width, height = image.size

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Draw rectangles over the specified bounding boxes
    for bbox in bboxes:
        # Convert normalized coordinates to absolute coordinates
        absolute_bbox = (
            bbox.x0 * width,
            bbox.top * height,
            bbox.x1 * width,
            bbox.bottom * height,
        )
        # Draw rectangle
        draw.rectangle(absolute_bbox, fill=mask_color)

    return image


class ProviderResizeRatios(Enum):
    ANTHROPIC = [
        AspectRatioRule(ratio=1 / 1, max_width=1092, max_height=1092),
        AspectRatioRule(ratio=3 / 4, max_width=951, max_height=1268),
        AspectRatioRule(ratio=2 / 3, max_width=896, max_height=1344),
        AspectRatioRule(ratio=9 / 16, max_width=819, max_height=1456),
        AspectRatioRule(ratio=1 / 2, max_width=784, max_height=1568),
    ]
