from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer
from typing_extensions import Annotated

SegmentLevels = Literal["word", "line", "block"]
TextblockSource = Literal["ocr", "derived"]
DirectionChoices = Literal["UP", "DOWN", "LEFT", "RIGHT"]

RoundedFloat = Annotated[float, PlainSerializer(lambda x: round(x, 5))]
BoundedFloat = Annotated[
    RoundedFloat, Field(..., ge=0, le=1, description="A float between 0 and 1")
]


class TextSpan(BaseModel):
    start_index: int
    end_index: int
    level: Literal["page", "document"] = Field(
        default="page", description="The level of the span"
    )


class NormBBox(BaseModel):
    """
    Represents a normalized bounding box with each value in the range [0, 1]

    Where x1 > x0 and bottom > top
    """

    x0: BoundedFloat
    top: BoundedFloat
    x1: BoundedFloat
    bottom: BoundedFloat

    model_config: ConfigDict = {"json_encoders": {float: lambda v: round(v, 5)}}

    def as_tuple(self):
        return (self.x0, self.top, self.x1, self.bottom)

    def __getitem__(self, index):
        # Lots of if statements to prevent new allocations
        if index > 3:
            raise IndexError("Index out of range")

        if index == 0:
            return self.x0
        elif index == 1:
            return self.top
        elif index == 2:
            return self.x1
        elif index == 3:
            return self.bottom

    def __eq__(self, other):
        if not isinstance(other, NormBBox):
            return False

        return self.as_tuple() == other.as_tuple()

    def __hash__(self):
        return hash(self.as_tuple())

    def __and__(self, other):
        if not isinstance(other, NormBBox):
            raise TypeError("Can only compute intersection with NormBBox")
        # Compute the intersection of two bounding boxes
        new_x0 = max(self.x0, other.x0)
        new_top = max(self.top, other.top)
        new_x1 = min(self.x1, other.x1)
        new_bottom = min(self.bottom, other.bottom)

        # Check if there is an actual intersection and if the resulting bounding box is valid
        if new_x0 <= new_x1 and new_top <= new_bottom:
            return NormBBox(x0=new_x0, top=new_top, x1=new_x1, bottom=new_bottom)
        else:
            # Return an empty or non-existent bounding box representation
            return None

    def __add__(self, other):
        if not isinstance(other, NormBBox):
            raise TypeError("Can only add NormBBox to NormBBox")

        return NormBBox(
            x0=min(self.x0, other.x0),
            top=min(self.top, other.top),
            x1=max(self.x1, other.x1),
            bottom=max(self.bottom, other.bottom),
        )

    def __contains__(self, other):
        return (
            self.x0 <= other.x0
            and self.top <= other.top
            and self.x1 >= other.x1
            and self.bottom >= other.bottom
        )

    def intersection_over_union(self, other):
        if not isinstance(other, NormBBox):
            raise TypeError("Can only compute IOU with NormBBox")

        # Compute the intersection
        intersection_bbox = self & other

        if intersection_bbox:
            intersection_area = intersection_bbox.area
            union_area = self.area + other.area - intersection_area
            return intersection_area / union_area

        return 0  # No intersection

    def x_overlap(self, other):
        """
        Get the overlap, between 0 and 1, of the x-axis of two bounding boxes
        """
        return max(0, min(self.x1, other.x1) - max(self.x0, other.x0))

    def y_overlap(self, other):
        """
        Get the overlap, between 0 and 1, of the y-axis of two bounding boxes
        """
        return max(0, min(self.bottom, other.bottom) - max(self.top, other.top))

    @classmethod
    def combine(cls, *bboxes: "NormBBox"):
        """
        Combines multiple bounding boxes into a single bounding box
        """
        if len(bboxes) == 0:
            raise ValueError("Must provide at least one bounding box")

        if len(bboxes) == 1:
            return bboxes[0]

        working_bbox = bboxes[0]
        for bbox in bboxes[1:]:
            working_bbox = working_bbox + bbox

        return working_bbox

    @classmethod
    def from_bounding_poly(cls, bounding_poly: "BoundingPoly"):
        """
        Returns a NormBBox from a BoundingPoly
        """
        if len(bounding_poly.normalized_vertices) != 4:
            raise ValueError(
                "BoundingPoly must have 4 vertices for NormBBox conversion"
            )

        (
            top_left,
            top_right,
            bottom_right,
            bottom_left,
        ) = bounding_poly.normalized_vertices

        return cls(
            x0=top_left.x,
            top=top_left.y,
            x1=bottom_right.x,
            bottom=bottom_right.y,
        )

    @property
    def width(self):
        return self.x1 - self.x0

    @property
    def height(self):
        return self.bottom - self.top

    @property
    def area(self):
        return self.width * self.height

    @property
    def centroid(self):
        return (self.x0 + self.x1) / 2, (self.top + self.bottom) / 2

    @property
    def y_center(self):
        return (self.top + self.bottom) / 2

    @property
    def x_center(self):
        return (self.x0 + self.x1) / 2


class Point(BaseModel):
    """
    Represents a normalized bounding box with each value in the range [0, 1]
    """

    model_config: ConfigDict = {"json_encoders": {float: lambda v: round(v, 5)}}

    x: BoundedFloat
    y: BoundedFloat


class BoundingPoly(BaseModel):
    """
    Represents a normalized bounding poly with each value in the range [0, 1]

    Used for higher order shapes like polygons on a page
    """

    normalized_vertices: List[Point]

    def __getitem__(self, index):
        return self.normalized_vertices[index]


class TextBlockMetadata(BaseModel):
    direction: Optional[DirectionChoices] = None
    confidence: Optional[RoundedFloat] = None
    layout_category: Optional[str] = Field(
        default=None, description="The category of the text block"
    )


class TextBlock(BaseModel):
    """
    Represents a single block of text, with its bounding box.
    The bounding box is a tuple of (x0, top, x1, bottom) and
    is normalized to the page size.
    """

    model_config: ConfigDict = {"json_encoders": {float: lambda v: round(v, 5)}}

    text: str
    type: SegmentLevels
    source: TextblockSource = Field(
        default="derived", description="The source of the text block"
    )

    # Layout information
    bounding_box: NormBBox = Field(default=None, repr=False)
    bounding_poly: Optional[BoundingPoly] = Field(default=None, repr=False)
    text_spans: Optional[List[TextSpan]] = Field(default=None, repr=False)

    metadata: Optional[TextBlockMetadata] = Field(default_factory=TextBlockMetadata)

    def __getitem__(self, index):
        return getattr(self, index)

    def __hash__(self):
        return hash((self.text, self.bounding_box.as_tuple()))

    @property
    def confidence(self):
        return self.metadata.confidence

    @property
    def direction(self):
        return self.metadata.direction
