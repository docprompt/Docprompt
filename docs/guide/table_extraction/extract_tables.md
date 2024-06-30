# Table Extraction with DocPrompt: Invoice Parsing

DocPrompt can be used to extract tables from documents with high accuracy using visual large language models, such as GPT-4 Vision or Anthropic's Claude 3. In this guide, we'll demonstrate how to extract tables from invoices using DocPrompt.

## Setting Up

First, let's import the necessary modules and set up our environment:

```python
from docprompt import load_document_node, DocumentNode
from docprompt.tasks.factory import AnthropicTaskProviderFactory
from docprompt.tasks.table_extraction import TableExtractionInput

# Initialize the Anthropic factory
# Ensure you have set the ANTHROPIC_API_KEY environment variable
factory = AnthropicTaskProviderFactory()

# Create the table extraction provider
table_extraction_provider = factory.get_page_table_extraction_provider()
```

## Preparing the Document

Load a DocumentNode from a path

```python
document_node = load_document_node("path/to/your/invoice.pdf")
```

## Performing Table Extraction

Now, let's run the table extraction task on our invoice:

```python
results = table_extraction_provider.process_document_node(document_node) # Sync

async_results = await table_extraction_provider.aprocess_document_node(document_node)
```

Alternatively, we can do table extraction async as well

## Interpreting Results

Let's examine the extracted tables from a pretend invoice:

```python
for page_number, result in results.items():
    print(f"Tables extracted from Page {page_number}:")
    for i, table in enumerate(result.tables, 1):
        print(f"\nTable {i}:")
        print(f"Title: {table.title}")
        print("Headers:")
        print(", ".join(header.text for header in table.headers))
        print("Rows:")
        for row in table.rows:
            print(", ".join(cell.text for cell in row.cells))
    print('---')
```

This will print the extracted tables, including headers and rows, for each page of the invoice.

## Increasing Accuracy

In Anthropic's case, the default is `"claude-3-haiku-20240307"`. This performs with high accuracy, and is over 5x cheaper than table extraction using Azure Document Intelligence.

In use-cases where accuracy is paramount however, it may be worthwhile to set the provider to a more powerful model.

```python
table_extraction_provider = factory.get_page_table_extraction_provider(
    model_name="claude-3-5-sonnet-20240620"  # setup the task provider with Sonnet 35
)

results = table_extraction_provider.process_document_node(
    document_node,
    table_extraction_input,
    model_name="claude-3-5-sonnet-20240620"  # or declare model name at inference time
)
```

As Large Language Models steadily get cheaper and more capable, your inference costs will drop inevitably. The beauty of progress!


## Resolving Bounding Boxes

**Coming Soon**

In some scenarios, you may want the exact bounding boxes of the various rows, columns, and cells. If you've processed OCR results through Docprompt, this is possible by specifying an additional argument in `process_document_node`

```python
results = table_extraction_provider.process_document_node(
    document_node,
    table_extraction_input,
    model_name="claude-3-5-sonnet-20240620",  # or declare model name at inference time
    resolve_bounding_boxes=True
)
```

If you've collected and stored OCR results on the DocumentNode, this will use word-level bounding boxes coupled with the Docprompt search engine to determine the bounding boxes of the resulting tables, where possible.

## Conclusion

Table extraction with DocPrompt provides a powerful way to automatically parse structured data from any documents containing tabular information in just a few lines of code.

The quality of your results depends on the model and the complexity of the table layouts. Experiment with different configurations and post-processing steps to find what works best for your specific use case.

When combining with other tasks such as classification, layout analysis and markerization, you can build powerful document processing pipelines in just a few steps.
