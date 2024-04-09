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

    # Add two bboxes

    bbox_2 = NormBBox(x0=0.5, top=0.5, x1=1.5, bottom=1.5)
    combined_bbox = bbox + bbox_2
    assert combined_bbox == NormBBox(x0=0, top=0, x1=1.5, bottom=1.5)

    # Add two bboxes via combine

    combined_bbox = NormBBox.combine(bbox, bbox_2)
    assert combined_bbox == NormBBox(x0=0, top=0, x1=1.5, bottom=1.5)

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
