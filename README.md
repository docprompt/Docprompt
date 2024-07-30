[![pypi](https://img.shields.io/pypi/v/docprompt.svg)](https://pypi.org/project/docprompt/)
[![python](https://img.shields.io/pypi/pyversions/docprompt.svg)](https://pypi.org/project/docprompt/)
[![Build Status](https://github.com/docprompt/Docprompt/actions/workflows/dev.yml/badge.svg)](https://github.com/docprompt/docprompt/actions/workflows/dev.yml)
[![codecov](https://codecov.io/gh/docprompt/Docprompt/branch/main/graphs/badge.svg)](https://codecov.io/github/docprompt/Docprompt)
[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm-project.org)

<br />
<div align="center">
  <a href="https://github.com/docprompt/Docprompt">
    <img src="docs/assets/static/img/logo.png" alt="Logo" width="120" height="120">
  </a>

  <h3 align="center">Docprompt</h3>

  <p align="center">
    Document AI, powered by LLM's
    <br />
    <a href="https://docs.docprompt.io"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    ·
    <a href="https://github.com/docprompt/Docprompt">Report Bug</a>
    ·
    <a href="https://github.com/docprompt/Docprompt">Request Feature</a>
  </p>
</div>

# About

Docprompt is a library for Document AI. It aims to make enterprise-level document analysis easy thanks to the zero-shot capability of large language models.

## Supercharged Document Analysis

* Common utilities for interacting with PDFs
  * PDF loading and serialization
  * PDF byte compression using Ghostscript :ghost:
  * Fast rasterization :fire: :rocket:
  * Page splitting, re-export with PDFium
  * Document Search, powered by Rust :fire:
* Support for most OCR providers with batched inference
  * Google :white_check_mark:
  * Amazon Textract :white_check_mark:
  * Tesseract :white_check_mark:
  * Azure Document Intelligence :red_circle:
* Layout Aware Page Representation
  * Run Document Layout Analysis with text-only LLM's!
* Prompt Garden for common document analysis tasks **zero-shot**, including:
  * Markerization (Pdf2Markdown)
  * Table Extraction
  * Page Classification
  * Key-value extraction (Coming soon)
  * Segmentation (Coming soon)


Documents and large language models


* Documentation: <https://docs.docprompt.io>
* GitHub: <https://github.com/docprompt/docprompt>
* PyPI: <https://pypi.org/project/docprompt/>
* Free software: Apache-2.0


## Features

* Representations for common document layout types - `TextBlock`, `BoundingBox`, etc
* Generic implementations of OCR providers
* Document Search powered by Rust and R-trees :fire:
* Table Extraction, Page Classification, PDF2Markdown

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Docprompt.

```bash
pip install docprompt
```

With an OCR provider

```bash
pip install "docprompt[google]
```

With search support

```bash
pip install "docprompt[search]"
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


### Converting a PDF to markdown

Coverting documents into markdown is a great way to prepare documents for downstream chunking or ingestion into a RAG system.

```python
from docprompt import load_document_node
from docprompt.tasks.markerize import AnthropicMarkerizeProvider

document_node = load_document_node("path/to/my.pdf")
markerize_provider = AnthropicMarkerizeProvider()

markerized_document = markerize_provider.process_document_node(document_node)
```

### Extracting Tables

Extract tables with SOTA speed and accuracy.

```python
from docprompt import load_document_node
from docprompt.tasks.table_extraction import AnthropicTableExtractionProvider

document_node = load_document_node("path/to/my.pdf")
table_extraction_provider = AnthropicTableExtractionProvider()

extracted_tables = table_extraction_provider.process_document_node(document_node)
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

# With OCR results available, we can now instantiate a locator and search through documents.

document_node.locator.search("John Doe") # This will return a list of all terms across the document that contain "John Doe"
document_node.locator.search("Jane Doe", page_number=4) # Just return results a list of matching results from page 4
```

This functionality uses a combination of `rtree` and the Rust library `tantivy`, allowing you to perform thousands of searches in **seconds** :fire: :rocket:

<a href="https://trackgit.com">
<img src="https://us-central1-trackgit-analytics.cloudfunctions.net/token/ping/lw098gfpjhrd7b2ev4rl" alt="trackgit-views" />
</a>
