from typing import Literal

from PIL import Image

from docprompt.schema.layout import NormBBox

ImageMaskModes = Literal["color", "average", "alpha"]


def mask_image_from_bounding_boxes(
    image: Image.Image,
    *bounding_boxes: NormBBox,
    mask_color: str = "#000000",
):
    """
    Create a copy of the image with the positions of the bounding boxes masked.
    """

    width, height = image.size

    mask = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    for bbox in bounding_boxes:
        mask.paste(
            Image.new("RGBA", (bbox.width, bbox.height), mask_color),
            (int(bbox.x0 * width), int(bbox.top * height)),
        )

    return Image.alpha_composite(image, mask)
