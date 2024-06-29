# Lightning-Fast Document Search 🔥🚀

Ever wished you could search through OCR-processed documents at the speed of light? Look no further! DocPrompt's Provenance Locator, powered by Rust, offers blazingly fast text search capabilities that will revolutionize your document processing workflows.

## The Power of Rust-Powered Search

DocPrompt's `DocumentProvenanceLocator` is not your average search tool. Implemented in Rust and leveraging the power of `tantivy` and `rtree`, it provides:

- ⚡ Lightning-fast full-text search
- 🎯 Precise text location within documents
- 🧠 Smart granularity refinement (word, line, block)
- 🗺️ Spatial querying capabilities

Let's dive into how you can harness this power!

## Setting Up the Locator

First, let's create a `DocumentProvenanceLocator` from a processed `DocumentNode`:

```python
from docprompt import load_document, DocumentNode
from docprompt.tasks.factory import GCPTaskProviderFactory

# Load and process the document
document = load_document("path/to/your/document.pdf")
document_node = DocumentNode.from_document(document)

# Process with OCR (assuming you've set up the GCP factory)
gcp_factory = GCPTaskProviderFactory(service_account_file="path/to/credentials.json")
ocr_provider = gcp_factory.get_page_ocr_provider(
    project_id="your-project-id",
    processor_id="your-processor-id"
)
ocr_results = ocr_provider.process_document_node(document_node)

# Create the locator
locator = document_node.locator
```

## Searching at the Speed of Rust 🔥

Now that we have our locator, let's see it in action:

```python
# Perform a simple search
results = locator.search("DocPrompt")

for result in results:
    print(f"Found on page {result.page_number}")
    print(f"Text: {result.text}")
    print(f"Bounding box: {result.text_location.merged_source_block.bounding_box}")
    print("---")
```

This search operation happens in milliseconds, even for large documents, thanks to the Rust-powered backend!

## Advanced Search Capabilities

### Refining to Word Level

DocPrompt can automatically refine search results to the word level:

```python
refined_results = locator.search("DocPrompt", refine_to_word=True)
```

This gives you pinpoint accuracy in locating text within your document.

### Page-Specific Search

Need to search on a specific page? No problem:

```python
page_5_results = locator.search("DocPrompt", page_number=5)
```

### Best Match Search

Find the best matches based on different criteria:

```python
best_short_matches = locator.search_n_best("DocPrompt", n=3, mode="shortest_text")

best_long_matches = locator.search_n_best("DocPrompt", n=3, mode="longest_text")

best_overall_matches = locator.search_n_best("DocPrompt", n=3, mode="highest_score")
```

## Spatial Queries: Beyond Text Search 🗺️

DocPrompt's locator isn't just fast—it's spatially aware! You can perform queries based on document layout:

```python
from docprompt.schema.layout import NormBBox

# Get blocks near a specific area on page 1
bbox = NormBBox(x0=0.1, top=0.1, x1=0.2, bottom=0.2)
nearby_blocks = locator.get_k_nearest_blocks(bbox, page_number=1, k=5)

# Get overlapping blocks
overlapping_blocks = locator.get_overlapping_blocks(bbox, page_number=1)
```

This spatial awareness opens up possibilities for advanced document analysis and data extraction!

## Conclusion: Search at the Speed of Thought 🧠💨

DocPrompt's `DocumentProvenanceLocator` brings unprecedented speed and precision to document search and analysis. By leveraging the power of Rust, it offers:

1. Lightning-fast full-text search
2. Precise text location within documents
3. Advanced spatial querying capabilities
4. Scalability for large documents and datasets

Whether you're building a document analysis pipeline, a search system, or any text-based application, DocPrompt's Provenance Locator offers the speed and accuracy you need to stay ahead of the game.
