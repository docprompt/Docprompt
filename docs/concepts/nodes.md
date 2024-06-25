# Nodes in Docprompt

## Overview

In Docprompt, nodes are fundamental structures used to represent and manage documents and their pages. They provide a way to store state and metadata associated with documents and individual pages, enabling advanced document analysis and processing capabilities.

## Key Concepts

### DocumentNode

A `DocumentNode` represents a single document within the Docprompt system. It serves as a container for document-level metadata and provides access to individual pages through `PageNode` instances.

```python
class DocumentNode(BaseModel, Generic[DocumentNodeMetadata, PageNodeMetadata]):
    document: Document
    page_nodes: List[PageNode[PageNodeMetadata]]
    metadata: Optional[DocumentNodeMetadata]
```

Key features:
- Stores a reference to the underlying `Document` object
- Maintains a list of `PageNode` instances representing individual pages
- Allows for custom document-level metadata
- Provides access to a `DocumentProvenanceLocator` for efficient text search within the document

### PageNode

A `PageNode` represents a single page within a document. It stores page-specific information and provides access to various analysis results, such as OCR data.

```python
class PageNode(BaseModel, Generic[PageNodeMetadata]):
    document: "DocumentNode"
    page_number: PositiveInt
    metadata: Optional[PageNodeMetadata]
    extra: Dict[str, Any]
    ocr_results: ResultContainer[OcrPageResult]
```

Key features:
- References the parent `DocumentNode`
- Stores the page number
- Allows for custom page-level metadata
- Provides a flexible `extra` field for additional data storage
- Stores OCR results in a `ResultContainer`

## Usage

### Creating a DocumentNode

You can create a `DocumentNode` from a `Document` instance:

```python
from docprompt import load_document, DocumentNode

document = load_document("path/to/my.pdf")
document_node = DocumentNode.from_document(document)
```

### Working with OCR Results

After processing a document with an OCR provider, you can access the results through the `DocumentNode` and `PageNode` structures:

```python
from docprompt.tasks.ocr.gcp import GoogleOcrProvider

provider = GoogleOcrProvider.from_service_account_file(
    project_id=my_project_id,
    processor_id=my_processor_id,
    service_account_file=path_to_service_file
)

provider.process_document_node(document_node)

# Access OCR results for a specific page
ocr_result = document_node.page_nodes[0].ocr_results
```

### Using DocumentProvenanceLocator

The `DocumentProvenanceLocator` is a powerful tool for searching text within a document:

```python
# Search for text across the entire document
results = document_node.locator.search("John Doe")

# Search for text on a specific page
page_results = document_node.locator.search("Jane Doe", page_number=4)
```

## Benefits of Using Nodes

1. **Separation of Concerns**: Nodes allow you to separate the core PDF functionality (handled by the `Document` class) from additional metadata and analysis results.

2. **Flexible Metadata**: Both `DocumentNode` and `PageNode` support generic metadata types, allowing you to add custom, type-safe metadata to your documents and pages.

3. **Result Caching**: Nodes provide a convenient way to cache and access results from various analysis tasks, such as OCR.

4. **Efficient Text Search**: The `DocumentProvenanceLocator` enables fast text search capabilities, leveraging OCR results for improved performance.

5. **Extensibility**: The node structure allows for easy integration of new analysis tools and result types in the future.

By using the node structure in Docprompt, you can build powerful document analysis workflows that combine the core PDF functionality with advanced processing and search capabilities.
