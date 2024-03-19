---
sidebar_position: 1
---

# Docprompt - Getting Started

https://github.com/Page-Leaf/Docprompt

## Supercharged Document Analysis

* Common utilities for interacting with PDFs
  * PDF loading and serialization
  * PDF byte compression using Ghostscript :ghost:
  * Fast rasterization :fire: :rocket:
  * Page splitting, re-export with PDFium
* Support for most OCR providers with batched inference
  * Google :white_check_mark:
  * Azure Document Intelligence :red_circle:
  * Amazon Textract :red_circle:
  * Tesseract :red_circle:



### Installation

Base installation

```bash
pip install docprompt
```

With an OCR provider

```bash
pip install "docprompt[google]
```

## Usage


### Simple Operations
```python
from docprompt import load_document

# Load a document
document = load_document("path/to/my.pdf")

# Rasterize a single page using Ghostscript
page_number = 5
rastered = document.rasterize_page(page_number, dpi=120)

# Split a pdf based on a page range
document_2 = document.split(start=125, stop=130)
```

### Performing OCR
```python
from docprompt import load_document, DocumentNode
from docprompt.tasks.ocr.gcp import GoogleOcrProvider

provider = GoogleOcrProvider.from_service_account_file(
  project_id=my_project_id,
  processor_id=my_processor_id,
  service_account_file=path_to_service_file
)

document = load_document("path/to/my.pdf")

# A container holds derived data for a document, like OCR or classification results
document_node = DocumentNode.from_document(document)

provider.process_document_node(document_node) # Caches results on the document_node

document_node[0].ocr_result # Access OCR results
```


### Document Search

When a large language model returns a result, we might want to highlight that result for our users. However, language models return results as **text**, while what we need to show our users requires a page number and a bounding box.

After extracting text from a PDF, we can support this pattern using `DocumentProvenanceLocator`, which lives on a `DocumentNode`

```
from docprompt import load_document, DocumentNode
from docprompt.tasks.ocr.gcp import GoogleOcrProvider

provider = GoogleOcrProvider.from_service_account_file(
  project_id=my_project_id,
  processor_id=my_processor_id,
  service_account_file=path_to_service_file
)

document = load_document("path/to/my.pdf")

# A container holds derived data for a document, like OCR or classification results
document_node = DocumentNode.from_document(document)

provider.process_document_node(document_node) # Caches results on the document_node

# With OCR results available, we can now instantiate a locator and search through documents.

document_node.locator.search("John Doe") # This will return a list of all terms across the document that contain "John Doe"
document_node.locator.search("Jane Doe", page_number=4) # Just return results a list of matching results from page 4
```

This functionality uses a combination of `rtree` and the Rust library `tantivy`, allowing you to perform thousands of searches in **seconds** :fire: :rocket:
