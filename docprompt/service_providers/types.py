from enum import Enum
from typing import Dict, List, Optional, Tuple, TypedDict, TypeVar

from attrs import frozen

from docprompt.schema import SegmentLevels, TextBlock


class OPERATIONS(Enum):
    IMAGE_PROCESSING = "image_processing"
    LAYOUT_ANALYSIS = "layout_analysis"
    TEXT_EXTRACTION = "text_extraction"


class ImageProcessResult(TypedDict):
    """
    Represents the result of processing an image
    """

    type: str
    image_data: bytes
    width: int
    height: int


T = TypeVar("T", bound=str)


class LayoutBlock(TypedDict):
    """
    Represents a single box in the layout, with its bounding box.
    The bounding box is a tuple of (x0, top, x1, bottom) and
    is normalized to the page size.
    """

    type: T
    text: Optional[str]
    bounding_box: Tuple[float, float, float, float]


LayoutAnalysisOutput = Dict[str, List[LayoutBlock]]


@frozen
class PageTextExtractionOutput:
    """
    Represents the output of text extraction for a single page
    """

    text: str
    blocks: Dict[SegmentLevels, list[TextBlock]]
