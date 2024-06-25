# Docprompt Primitives

Docprompt uses several primitive objects that are fundamental to its operation. These primitives are used throughout the library and are essential for understanding how Docprompt processes and represents documents.

## PdfDocument

The `PdfDocument` class is a core primitive in Docprompt, representing a PDF document with various utilities for manipulation and analysis.

```python
class PdfDocument(BaseModel):
    name: str
    file_bytes: bytes
    file_path: Optional[str] = None
```

### Key Features

1. **Document Properties**
   - `name`: The name of the document
   - `file_bytes`: The raw bytes of the PDF file
   - `file_path`: Optional path to the PDF file on disk
   - `page_count`: The number of pages in the document (computed field)
   - `document_hash`: A unique hash of the document (computed field)

2. **Utility Methods**
   - `from_path(file_path)`: Create a PdfDocument from a file path
   - `from_bytes(file_bytes, name)`: Create a PdfDocument from bytes
   - `get_page_render_size(page_number, dpi)`: Get the render size of a specific page
   - `to_compressed_bytes()`: Compress the PDF using Ghostscript
   - `rasterize_page(page_number, ...)`: Rasterize a specific page with various options
   - `rasterize_pdf(...)`: Rasterize the entire PDF
   - `split(start, stop)`: Split the PDF into a new document
   - `as_tempfile()`: Create a temporary file from the PDF
   - `write_to_path(path)`: Write the PDF to a specific path

### Usage Example

```python
from docprompt import PdfDocument

# Load a PDF
pdf = PdfDocument.from_path("path/to/document.pdf")

# Get document properties
print(f"Document name: {pdf.name}")
print(f"Page count: {pdf.page_count}")

# Rasterize a page
page_image = pdf.rasterize_page(1, dpi=300)

# Split the document
new_pdf = pdf.split(start=5, stop=10)
```

## Layout Primitives

Docprompt uses several layout primitives to represent the structure and content of documents.

### NormBBox

`NormBBox` represents a normalized bounding box with values between 0 and 1.

```python
class NormBBox(BaseModel):
    x0: BoundedFloat
    top: BoundedFloat
    x1: BoundedFloat
    bottom: BoundedFloat
```

Key features:
- Intersection operations (`__and__`)
- Union operations (`__add__`)
- Intersection over Union (IoU) calculation
- Area and centroid properties

### TextBlock

`TextBlock` represents a block of text within a document, including its bounding box and metadata.

```python
class TextBlock(BaseModel):
    text: str
    type: SegmentLevels
    source: TextblockSource
    bounding_box: NormBBox
    bounding_poly: Optional[BoundingPoly]
    text_spans: Optional[List[TextSpan]]
    metadata: Optional[TextBlockMetadata]
```

### Point and BoundingPoly

`Point` and `BoundingPoly` are used to represent more complex shapes within a document.

```python
class Point(BaseModel):
    x: BoundedFloat
    y: BoundedFloat

class BoundingPoly(BaseModel):
    normalized_vertices: List[Point]
```

### TextSpan

`TextSpan` represents a span of text within a document or page.

```python
class TextSpan(BaseModel):
    start_index: int
    end_index: int
    level: Literal["page", "document"]
```

### Usage Example

```python
from docprompt.schema.layout import NormBBox, TextBlock, TextBlockMetadata

# Create a bounding box
bbox = NormBBox(x0=0.1, top=0.1, x1=0.9, bottom=0.2)

# Create a text block
text_block = TextBlock(
    text="Example text",
    type="block",
    source="ocr",
    bounding_box=bbox,
    metadata=TextBlockMetadata(confidence=0.95)
)

# Use the text block
print(f"Text: {text_block.text}")
print(f"Bounding box: {text_block.bounding_box}")
print(f"Confidence: {text_block.confidence}")
```

These primitives form the foundation of Docprompt's document processing capabilities, allowing for precise representation and manipulation of document content and structure.
