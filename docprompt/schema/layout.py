from math import atan, cos, degrees, radians, sin
from typing import Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, Field

SegmentLevels = Literal["word", "line", "block", "paragraph"]


class TextBlock(BaseModel):
    """
    Represents a single block of text, with its bounding box.
    The bounding box is a tuple of (x0, top, x1, bottom) and
    is normalized to the page size.
    """

    text: str
    type: SegmentLevels
    geometry: "Geometry"
    direction: Optional[str] = None
    confidence: Optional[float] = None

    def __getitem__(self, index):
        return getattr(self, index)

    @property
    def bounding_box(self):
        return self.geometry.bounding_box

    @property
    def has_vertices(self):
        return self.geometry.bounding_poly is not None


class NormBBox(BaseModel):
    """
    Represents a normalized bounding box with each value in the range [0, 1]
    """

    x0: float
    top: float
    x1: float
    bottom: float

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

    def __div__(self, other):
        """
        Computes the overlap (intersection) between two bounding boxes
        """
        if not isinstance(other, NormBBox):
            raise TypeError("Can only compute overlap with NormBBox")

        # Check if there is overlap in the horizontal direction
        overlap_x0 = max(self.x0, other.x0)
        overlap_x1 = min(self.x1, other.x1)
        if overlap_x0 >= overlap_x1:
            return 0.0  # No horizontal overlap

        # Check if there is overlap in the vertical direction
        overlap_top = max(self.top, other.top)
        overlap_bottom = min(self.bottom, other.bottom)
        if overlap_top >= overlap_bottom:
            return 0.0  # No vertical overlap

        # Calculate the area of overlap
        overlap_width = overlap_x1 - overlap_x0
        overlap_height = overlap_bottom - overlap_top
        return overlap_width * overlap_height

    def __add__(self, other):
        if not isinstance(other, NormBBox):
            raise TypeError("Can only add NormBBox to NormBBox")

        return NormBBox(
            x0=min(self.x0, other.x0),
            top=min(self.top, other.top),
            x1=max(self.x1, other.x1),
            bottom=max(self.bottom, other.bottom),
        )

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
            raise ValueError("BoundingPoly must have 4 vertices for NormBBox conversion")

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

    x: float
    y: float


class BoundingPoly(BaseModel):
    """
    Represents a normalized bounding poly with each value in the range [0, 1]
    """

    normalized_vertices: list[Point]

    def __getitem__(self, index):
        return self.normalized_vertices[index]

    def get_skew_angle(self):
        """
        Determines the skew angle (in degrees) of the bounding poly from
        perfectly horizontal
        """
        if len(self.normalized_vertices) != 4:
            raise ValueError("BoundingPoly must have 4 vertices for skew angle calculation")

        top_left, top_right, bottom_right, bottom_left = self.normalized_vertices

        top_line_vertical_difference = top_left.y - top_right.y
        top_line_horizontal_difference = top_right.x - top_left.x

        if top_line_horizontal_difference == 0:
            return 90 if top_line_vertical_difference > 0 else -90

        skew_angle_radians = atan(top_line_vertical_difference / top_line_horizontal_difference)
        skew_angle_degrees = degrees(skew_angle_radians)

        return skew_angle_degrees

    def get_centroid(self):
        """
        Returns the centroid of the bounding poly
        """
        if len(self.normalized_vertices) != 4:
            raise ValueError("BoundingPoly must have 4 vertices for centroid calculation")

        total_x = sum(vertex.x for vertex in self.normalized_vertices)
        total_y = sum(vertex.y for vertex in self.normalized_vertices)

        centroid_x = total_x / len(self.normalized_vertices)
        centroid_y = total_y / len(self.normalized_vertices)

        return Point(x=centroid_x, y=centroid_y)

    def get_rotation_point(self) -> Point:
        """
        Determines the ideal point to rotate around based on the skew
        """
        if len(self.normalized_vertices) != 4:
            raise ValueError("BoundingPoly must have 4 vertices to determine rotation point")
        top_left, top_right, _, bottom_left = self.normalized_vertices

        # If top right is higher than top left
        if top_right.y > top_left.y:
            return bottom_left
        # If top left is higher or they are the same
        return top_right

    def get_centroid_rotated(self, angle_degrees: float) -> "BoundingPoly":
        """
        Return a new bounding poly that has been rotated by the given angle
        """

        if len(self.normalized_vertices) != 4:
            raise ValueError("BoundingPoly must have 4 vertices for skew angle rotation")

        centroid = self.get_centroid()
        angle_rad = radians(angle_degrees)
        rotated_vertices: list[Point] = []

        for vertex in self.normalized_vertices:
            # Translate to the origin
            translated_x = vertex.x - centroid.x
            translated_y = vertex.y - centroid.y

            # Rotate around the origin
            rotated_x = translated_x * cos(angle_rad) - translated_y * sin(angle_rad)
            rotated_y = translated_x * sin(angle_rad) + translated_y * cos(angle_rad)

            # Translate back
            final_x = rotated_x + centroid.x
            final_y = rotated_y + centroid.y

            rotated_vertices.append(Point(x=final_x, y=final_y))

        return BoundingPoly(normalized_vertices=rotated_vertices)

    def get_rotated_around_point(self, angle_degrees: float, rotation_point: Point) -> "BoundingPoly":
        """
        Return a new bounding poly that has been rotated by the given angle around the given point
        """
        if len(self.normalized_vertices) != 4:
            raise ValueError("BoundingPoly must have 4 vertices for rotation")

        angle_rad = radians(angle_degrees)
        rotated_vertices: list[Point] = []

        for vertex in self.normalized_vertices:
            # Translate to the rotation point
            translated_x = vertex.x - rotation_point.x
            translated_y = vertex.y - rotation_point.y

            # Rotate around the rotation point
            rotated_x = translated_x * cos(angle_rad) - translated_y * sin(angle_rad)
            rotated_y = translated_x * sin(angle_rad) + translated_y * cos(angle_rad)

            # Translate back
            final_x = rotated_x + rotation_point.x
            final_y = rotated_y + rotation_point.y

            rotated_vertices.append(Point(x=final_x, y=final_y))

        return BoundingPoly(normalized_vertices=rotated_vertices)


class Geometry(BaseModel):
    """
    Represnts a "geometry" of an object
    """

    bounding_box: NormBBox
    bounding_poly: Optional[BoundingPoly] = Field(default=None, repr=False)

    def to_deskewed_geometry(self):
        """
        Returns a new geometry object with both bounding box and bounding poly
        rotated to have zero skew angle
        """
        if not self.bounding_poly:
            raise ValueError("Bounding poly must be present to deskew geometry")

        skew_angle = self.bounding_poly.get_skew_angle()

        rotate_point = self.bounding_poly.get_rotation_point()

        rotated_bounding_poly = self.bounding_poly.get_rotated_around_point(skew_angle, rotate_point)
        bounding_box = NormBBox.from_bounding_poly(rotated_bounding_poly)

        return Geometry(
            bounding_box=bounding_box,
            bounding_poly=rotated_bounding_poly,
        )
