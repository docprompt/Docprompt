# Providers in Docprompt

## Overview

Providers in Docprompt are abstract interfaces that define how to add data to document nodes. They encapsulate various tasks such as OCR, classification, and more. The provider system is designed to be extensible, allowing users to create custom providers to add new functionality to Docprompt.

## Key Concepts

### AbstractTaskProvider

The `AbstractTaskProvider` is the base class for all providers in Docprompt. It defines the interface that all task providers must implement.

```python
class AbstractTaskProvider(Generic[PageTaskResult]):
    name: str
    capabilities: List[str]

    def process_document_pages(
        self,
        document: Document,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        **kwargs,
    ) -> Dict[int, PageTaskResult]:
        raise NotImplementedError

    def contribute_to_document_node(
        self,
        document_node: "DocumentNode",
        results: Dict[int, PageTaskResult],
    ) -> None:
        pass

    def process_document_node(
        self,
        document_node: "DocumentNode",
        start: Optional[int] = None,
        stop: Optional[int] = None,
        contribute_to_document: bool = True,
        **kwargs,
    ) -> Dict[int, PageTaskResult]:
        # ... implementation ...
```

Key features:
- Generic type `PageTaskResult` allows for type-safe results
- `capabilities` list defines what the provider can do
- `process_document_pages` method processes pages of a document
- `contribute_to_document_node` method adds results to a `DocumentNode`
- `process_document_node` method combines processing and contributing results

### CAPABILITIES

The `CAPABILITIES` enum defines the various capabilities that a provider can have:

```python
class CAPABILITIES(Enum):
    PAGE_RASTERIZATION = "page-rasterization"
    PAGE_LAYOUT_OCR = "page-layout-ocr"
    PAGE_TEXT_OCR = "page-text-ocr"
    PAGE_CLASSIFICATION = "page-classification"
    PAGE_SEGMENTATION = "page-segmentation"
    PAGE_VQA = "page-vqa"
    PAGE_TABLE_IDENTIFICATION = "page-table-identification"
    PAGE_TABLE_EXTRACTION = "page-table-extraction"
```

### ResultContainer

The `ResultContainer` is a generic class that holds the results of a task:

```python
class ResultContainer(BaseModel, Generic[PageOrDocumentTaskResult]):
    results: Dict[str, PageOrDocumentTaskResult] = Field(
        description="The results of the task, keyed by provider", default_factory=dict
    )

    @property
    def result(self):
        return next(iter(self.results.values()), None)
```

## Creating Custom Providers

To extend Docprompt's functionality, you can create custom providers. Here's an shortened example of a builtin OCR provider from GCP:

```python
from docprompt.tasks.base import AbstractTaskProvider, CAPABILITIES
from docprompt.schema.layout import TextBlock
from pydantic import Field

class OcrPageResult(BasePageResult):
    page_text: str = Field(description="The text for the entire page in reading order")
    word_level_blocks: List[TextBlock] = Field(default_factory=list)
    line_level_blocks: List[TextBlock] = Field(default_factory=list)
    block_level_blocks: List[TextBlock] = Field(default_factory=list)
    raster_image: Optional[bytes] = Field(default=None)

class GoogleOcrProvider(AbstractTaskProvider[OcrPageResult]):
    name = "Google Document AI"
    capabilities = [
        CAPABILITIES.PAGE_TEXT_OCR.value,
        CAPABILITIES.PAGE_LAYOUT_OCR.value,
        CAPABILITIES.PAGE_RASTERIZATION.value,
    ]

    def process_document_pages(
        self,
        document: Document,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        **kwargs,
    ) -> Dict[int, OcrPageResult]:
        # Implement OCR logic here
        pass

    def contribute_to_document_node(
        self,
        document_node: "DocumentNode",
        results: Dict[int, OcrPageResult],
    ) -> None:
        # Add OCR results to document node
        pass
```

## Usage

Here's how you can use a provider in your Docprompt workflow:

```python
from docprompt import load_document, DocumentNode
from docprompt.providers.ocr import GoogleOcrProvider

# Load a document
document = load_document("path/to/my.pdf")
document_node = DocumentNode.from_document(document)

# Create and use the OCR provider
ocr_provider = GoogleOcrProvider(...)
ocr_results = ocr_provider.process_document_node(document_node)

# Access OCR results
for page_number, result in ocr_results.items():
    print(f"Page {page_number} text: {result.page_text[:100]}...")
```

## Benefits of Using Providers

1. **Extensibility**: Easily add new functionality to Docprompt by creating custom providers.
2. **Modularity**: Each provider encapsulates a specific task, making the codebase more organized and maintainable.
3. **Type Safety**: Generic types ensure that providers produce and consume the correct types of results.
4. **Standardized Interface**: All providers follow the same interface, making it easy to switch between different implementations.
5. **Capability-based Design**: Providers declare their capabilities, allowing for dynamic feature discovery and usage.

By leveraging the provider system in Docprompt, you can create flexible and powerful document processing pipelines that can be easily extended and customized to meet your specific needs.
