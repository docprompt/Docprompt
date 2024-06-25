# Provenance in Docprompt

## Overview

Provenance in Docprompt refers to the ability to trace and locate specific pieces of text within a document. The `DocumentProvenanceLocator` class is a powerful tool that enables efficient text search, spatial queries, and fine-grained text location within documents that have been processed with OCR.

## Key Concepts

### DocumentProvenanceLocator

The `DocumentProvenanceLocator` is a class that provides advanced search capabilities for documents in Docprompt. It combines full-text search with spatial indexing to offer fast and accurate text location services.

```python
@dataclass
class DocumentProvenanceLocator:
    document_name: str
    search_index: "tantivy.Index"
    block_mapping: Dict[int, OcrPageResult]
    geo_index: DocumentProvenanceGeoMap
```

Key features:
- Full-text search using the Tantivy search engine
- Spatial indexing using R-tree for efficient bounding box queries
- Support for different granularity levels (word, line, block)
- Ability to refine search results to word-level precision

## Main Functionalities

### 1. Text Search

The `search` method allows you to find specific text within a document:

```python
def search(
    self,
    query: str,
    page_number: Optional[int] = None,
    *,
    refine_to_word: bool = True,
    require_exact_match: bool = True
) -> List[ProvenanceSource]:
    # ... implementation ...
```

This method returns a list of `ProvenanceSource` objects, which contain detailed information about where the text was found, including page number, bounding box, and the surrounding context.

### 2. Spatial Queries

The `DocumentProvenanceLocator` supports spatial queries to find text blocks based on their location on the page:

```python
def get_k_nearest_blocks(
    self,
    bbox: NormBBox,
    page_number: int,
    k: int,
    granularity: BlockGranularity = "block"
) -> List[TextBlock]:
    # ... implementation ...

def get_overlapping_blocks(
    self,
    bbox: NormBBox,
    page_number: int,
    granularity: BlockGranularity = "block"
) -> List[TextBlock]:
    # ... implementation ...
```

These methods allow you to find text blocks that are near or overlapping with a given bounding box on a specific page.

## Usage

### Recommended Usage: Through DocumentNode

The recommended way to use the `DocumentProvenanceLocator` is through the `DocumentNode` class. The `DocumentNode` provides two methods for working with the locator:

1. `locator` property: Lazily creates and returns the `DocumentProvenanceLocator`.
2. `refresh_locator()` method: Explicitly refreshes the locator for the document node.

Here's how to use these methods:

```python
from docprompt import load_document, DocumentNode
from docprompt.tasks.ocr.gcp import GoogleOcrProvider

# Load and process the document
document = load_document("path/to/my.pdf")
document_node = DocumentNode.from_document(document)

# Process the document with OCR
provider = GoogleOcrProvider.from_service_account_file(...)
provider.process_document_node(document_node)

# Access the locator (creates it if it doesn't exist)
locator = document_node.locator

# Perform a search
results = locator.search("Docprompt")

# If you need to refresh the locator (e.g., after updating OCR results)
document_node.refresh_locator()
```

Note: Attempting to access the locator before OCR results are available will raise a `ValueError`.

### Alternative: Standalone Usage

While the recommended approach is to use the locator through `DocumentNode`, you can also create and use a `DocumentProvenanceLocator` independently:

```python
from docprompt.provenance.search import DocumentProvenanceLocator

# Assuming you have a processed DocumentNode
locator = DocumentProvenanceLocator.from_document_node(document_node)

# Now you can use the locator directly
results = locator.search("Docprompt")
```

### Searching for Text

To search for text within the document:

```python
results = locator.search("Docprompt")
for result in results:
    print(f"Found on page {result.page_number}, bbox: {result.text_location.merged_source_block.bounding_box}")
```

### Performing Spatial Queries

To find text blocks near a specific location:

```python
bbox = NormBBox(x0=0.1, y0=0.1, x1=0.2, y1=0.2)
nearby_blocks = locator.get_k_nearest_blocks(bbox, page_number=1, k=5)
```

## Benefits of Using Provenance

1. **Accurate Text Location**: Quickly find the exact location of text within a document, including page number and bounding box.
2. **Efficient Searching**: Combine full-text search with spatial indexing for fast and accurate results.
3. **Flexible Granularity**: Search and retrieve results at different levels of granularity (word, line, block).
4. **Integration with OCR**: Seamlessly works with OCR results to provide comprehensive document analysis capabilities.
5. **Support for Complex Queries**: Perform spatial queries to find text based on location within pages.
6. **Easy Access**: Conveniently access the locator through the `DocumentNode` class, ensuring it's always available when needed.

By leveraging the provenance functionality in Docprompt, you can build sophisticated document analysis workflows that require precise text location and contextual information retrieval.
