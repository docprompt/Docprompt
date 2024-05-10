[![pypi](https://img.shields.io/pypi/v/docprompt.svg)](https://pypi.org/project/docprompt/)
[![python](https://img.shields.io/pypi/pyversions/docprompt.svg)](https://pypi.org/project/docprompt/)
[![Build Status](https://github.com/Page-Leaf/Docprompt/actions/workflows/dev.yml/badge.svg)](https://github.com/Page-Leaf/docprompt/actions/workflows/dev.yml)
[![codecov](https://codecov.io/gh/Page-Leaf/Docprompt/branch/main/graphs/badge.svg)](https://codecov.io/github/Page-Leaf/Docprompt)
[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm-project.org)

<br />
<div align="center">
  <a href="https://github.com/Page-Leaf/Docprompt">
    <img src="docs/docprompt/static/img/logo.png" alt="Logo" width="120" height="120">
  </a>

  <h3 align="center">Docprompt</h3>

  <p align="center">
    Document AI, powered by LLM's
    <br />
    <a href="https://docprompt.dev"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    ·
    <a href="https://github.com/Page-Leaf/Docprompt">Report Bug</a>
    ·
    <a href="https://github.com/Page-Leaf/Docprompt">Request Feature</a>
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
  * Azure Document Intelligence :red_circle:
  * Amazon Textract :red_circle:
  * Tesseract :red_circle:
* Prompt Garden for common document analysis tasks **zero-shot**, including:
  * Table Extraction
  * Page Classification
  * Segmentation
  * Key-value extraction


Documents and large language models


* Documentation: <https://docprompt.dev>
* GitHub: <https://github.com/Page-Leaf/docprompt>
* PyPI: <https://pypi.org/project/docprompt/>
* Free software: Apache-2.0


## Features

* Representations for common document layout types - `TextBlock`, `BoundingBox`, etc
* Generic implementations of OCR providers
* Document Search powered by Rust and R-trees :fire:

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
