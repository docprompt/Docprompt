import base64
from typing import Generic

from pydantic import Field, field_serializer, field_validator

from docprompt.schema.pipeline.metadata import BaseMetadata

from .base import BaseNode
from .typing import ImageNodeMetadata


class ImageNode(BaseNode, Generic[ImageNodeMetadata]):
    """
    Represents a single image of any kind
    """

    image: bytes

    metadata: ImageNodeMetadata = Field(
        description="Application-specific metadata for the image",
        default_factory=BaseMetadata,
    )

    @field_serializer("image")
    def serialize_image(self, value):
        return base64.b64encode(value).decode("utf-8")

    @field_validator("image")
    @classmethod
    def validate_image(cls, value):
        if isinstance(value, bytes):
            return value

        return base64.b64decode(value)

    @property
    def pil_image(self):
        from io import BytesIO

        from PIL import Image

        return Image.open(BytesIO(self.image))

    @property
    def cv2_image(self):
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV is required to use this property")

        try:
            import numpy as np
        except ImportError:
            raise ImportError("Numpy is required to use this property")

        return cv2.imdecode(np.frombuffer(self.image, np.uint8), cv2.IMREAD_COLOR)
