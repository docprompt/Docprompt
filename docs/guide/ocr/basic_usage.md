# Basic OCR Usage with Docprompt

This guide will walk you through the basics of performing Optical Character Recognition (OCR) using Docprompt. You'll learn how to set up the OCR provider, process a document, and access the results.

## Prerequisites

Before you begin, ensure you have:

1. Installed Docprompt with OCR support: `pip install "docprompt[google]"`
2. A Google Cloud Platform account with Document AI API enabled
3. A GCP service account key file

## Setting Up the OCR Provider

First, let's set up the Google OCR provider:

```python
from docprompt.tasks.factory import GCPTaskProviderFactory

# Initialize the GCP Task Provider Factory
gcp_factory = GCPTaskProviderFactory(
    service_account_file="path/to/your/service_account_key.json"
)

# Create the OCR provider
ocr_provider = gcp_factory.get_page_ocr_provider(
    project_id="your-gcp-project-id",
    processor_id="your-document-ai-processor-id"
)
```

## Loading and Processing a Document

Now, let's load a document and process it using OCR:

```python
from docprompt import load_document, DocumentNode

# Load the document
document = load_document("path/to/your/document.pdf")
document_node = DocumentNode.from_document(document)

# Process the document
ocr_results = ocr_provider.process_document_node(document_node)
```

## Accessing OCR Results

After processing, you can access the OCR results in various ways:

### 1. Page-level Text

To get the full text of a specific page:

```python
page_number = 1  # Pages are 1-indexed
page_text = ocr_results[page_number].page_text
print(f"Text on page {page_number}:\n{page_text[:500]}...")  # Print first 500 characters
```

### 2. Words, Lines, and Blocks

Docprompt provides access to words, lines, and blocks (paragraphs) extracted from the document:

```python
# Get the first page's result
first_page_result = ocr_results[1]

# Print the first 5 words on the page
print("First 5 words:")
for word in first_page_result.word_level_blocks[:5]:
    print(f"Word: {word.text}, Confidence: {word.metadata.confidence}")

# Print the first line on the page
print("\nFirst line:")
if first_page_result.line_level_blocks:
    first_line = first_page_result.line_level_blocks[0]
    print(f"Line: {first_line.text}")

# Print the first block (paragraph) on the page
print("\nFirst block:")
if first_page_result.block_level_blocks:
    first_block = first_page_result.block_level_blocks[0]
    print(f"Block: {first_block.text[:100]}...")  # Print first 100 characters
```

### 3. Bounding Boxes

Each word, line, and block comes with bounding box information:

```python
# Get bounding box for the first word
if first_page_result.word_level_blocks:
    first_word = first_page_result.word_level_blocks[0]
    bbox = first_word.bounding_box
    print(f"\nBounding box for '{first_word.text}':")
    print(f"Top-left: ({bbox.x0}, {bbox.top})")
    print(f"Bottom-right: ({bbox.x1}, {bbox.bottom})")
```

## Conclusion

You've now learned the basics of performing OCR with Docprompt. This includes setting up the OCR provider, processing a document, and accessing the results at different levels of granularity.

For more advanced usage, including customizing OCR settings, using other providers, and leveraging the powerful search capabilities, check out our other guides:

- Customizing OCR Providers and Settings
- Advanced Text Search with Provenance Locators
- Building OCR-based Workflows
