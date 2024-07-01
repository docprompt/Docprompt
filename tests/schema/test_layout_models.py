import pytest

from docprompt import NormBBox
from docprompt.schema.layout import BoundingPoly, Point


def test_normbbox_utilities():
    bbox = NormBBox(x0=0, top=0, x1=1, bottom=1)

    # Test simple properties
    assert bbox[0] == 0
    assert bbox[1] == 0
    assert bbox[2] == 1
    assert bbox[3] == 1

    assert bbox.width == 1
    assert bbox.height == 1
    assert bbox.area == 1
    assert bbox.centroid == (0.5, 0.5)
    assert bbox.y_center == 0.5
    assert bbox.x_center == 0.5

    # Test equality
    assert bbox == NormBBox(x0=0, top=0, x1=1, bottom=1)

    # Test out of bounds

    with pytest.raises(ValueError):
        NormBBox(x0=0, top=0, x1=1, bottom=2)

    # Add two bboxes

    bbox_2 = NormBBox(x0=0.5, top=0.5, x1=1, bottom=1.0)
    combined_bbox = bbox + bbox_2
    assert combined_bbox == NormBBox(x0=0, top=0, x1=1.0, bottom=1.0)

    # Add two bboxes via combine

    combined_bbox = NormBBox.combine(bbox, bbox_2)
    assert combined_bbox == NormBBox(x0=0, top=0, x1=1.0, bottom=1.0)

    # Test from bounding poly
    bounding_poly = BoundingPoly(
        normalized_vertices=[
            Point(x=0, y=0),
            Point(x=1, y=0),
            Point(x=1, y=1),
            Point(x=0, y=1),
        ]
    )

    bbox = NormBBox.from_bounding_poly(bounding_poly)

    assert bbox == NormBBox(x0=0, top=0, x1=1, bottom=1)

    # Test contains

    small_bbox = NormBBox(x0=0.25, top=0.25, x1=0.75, bottom=0.75)
    big_bbox = NormBBox(x0=0, top=0, x1=1, bottom=1)

    assert small_bbox in big_bbox
    assert big_bbox not in small_bbox

    # Test Overlap

    assert small_bbox.x_overlap(big_bbox) == 0.5
    assert small_bbox.y_overlap(big_bbox) == 0.5

    # Should be commutative
    assert big_bbox.x_overlap(small_bbox) == 0.5
    assert big_bbox.y_overlap(small_bbox) == 0.5
