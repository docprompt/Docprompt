from typing import Literal, Iterable, Optional, Union
from pydantic import BaseModel
from PIL import Image, ImageDraw
from io import BytesIO
import logging
from enum import Enum

from docprompt.schema.layout import NormBBox

logger = logging.getLogger(__name__)


ResizeModes = Literal["thumbnail", "resize"]

PILOrBytes = Union[Image.Image, bytes]


class AspectRatioRule(BaseModel):
    ratio: float
    max_width: int
    max_height: int


def resize_image_to_closest_aspect_ratio(
    image_bytes: bytes,
    ratios: Iterable[AspectRatioRule],
    *,
    resize_mode: ResizeModes = "thumbnail",
):
    image = Image.open(BytesIO(image_bytes))

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
        return image_bytes

    if resize_mode == "thumbnail":
        image.thumbnail(
            (closest_aspect_ratio.max_width, closest_aspect_ratio.max_height)
        )
    elif resize_mode == "resize":
        image = image.resize(
            (closest_aspect_ratio.max_width, closest_aspect_ratio.max_height)
        )

    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)

    return buffer.getvalue()


def resize_image_to_fize_size_limit(
    image_bytes: bytes,
    max_file_size_bytes: int,
    *,
    resize_mode: ResizeModes = "thumbnail",
    resize_step_size: float = 0.1,
):
    """
    Incrementally resizes an image until it is under a certain file size
    """

    if resize_step_size <= 0 or resize_step_size >= 0.5:
        raise ValueError("resize_step_size must be between 0 and 0.5")

    if len(image_bytes) < max_file_size_bytes:
        return image_bytes

    output_bytes = image_bytes
    step_count = 0

    while len(output_bytes) > max_file_size_bytes:
        image = Image.open(BytesIO(output_bytes))

        new_width = int(image.width * (1 - resize_step_size * step_count))
        new_height = int(image.height * (1 - resize_step_size * step_count))

        if new_width <= 200 or new_height <= 200:
            logger.warning(
                f"Image could not be resized to under {max_file_size_bytes} bytes. Reached {len(output_bytes)} bytes."
            )
            break

        if resize_mode == "thumbnail":
            image.thumbnail((new_width, new_height))
        elif resize_mode == "resize":
            image = image.resize((new_width, new_height))

        buffer = BytesIO()

        image.save(buffer, format="PNG", optimize=True)

        output_bytes = buffer.getvalue()

        step_count += 1

    return output_bytes


def resize_pil_image(
    image: Image.Image,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    resize_mode: ResizeModes = "thumbnail",
    aspect_ratios: Optional[Iterable[AspectRatioRule]] = None,
):
    if width is None and height is None and aspect_ratios is None:
        return image

    if aspect_ratios is not None:
        buffer = BytesIO()
        image.save(buffer, format="PNG")

        result = resize_image_to_closest_aspect_ratio(
            buffer.getvalue(),
            aspect_ratios,
            resize_mode=resize_mode,
        )

        image = Image.open(BytesIO(result))

    elif width is not None and height is not None:
        if resize_mode == "thumbnail":
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
):
    image = resize_pil_image(
        image,
        width=resize_width,
        height=resize_height,
        resize_mode=resize_mode,
        aspect_ratios=resize_aspect_ratios,
    )

    if do_convert:
        image = image.convert(image_convert_mode)

    if do_quantize:
        image = image.quantize(colors=quantize_color_count)

    result = image

    if max_file_size_bytes:
        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=True)

        result = buffer.getvalue()

        # We want to do this last to avoid resizing if optimization would have made it smaller anyway
        if max_file_size_bytes and len(result) > max_file_size_bytes:
            result = resize_image_to_fize_size_limit(
                result,
                max_file_size_bytes,
                resize_mode=resize_mode,
                resize_step_size=0.1,
            )

    return result


def mask_image_from_bboxes(
    image: Union[Image.Image, bytes],
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
        image = Image.open(BytesIO(image))

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
