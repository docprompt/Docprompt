# Advanced Document Analysis:


###  Detecting Potential Conflicts of Interest in 10-K Reports
In this guide, we'll demonstrate how to use DocPrompt's powerful OCR and search capabilities to analyze 10-K reports for potential conflicts of interest. We'll search for mentions of company names and executive names, then identify instances where they appear in close proximity within the document.

## Setup

First, let's set up our environment and process the document:

```python
from docprompt import load_document, DocumentNode
from docprompt.tasks.factory import GCPTaskProviderFactory
from docprompt.schema.layout import NormBBox
from itertools import product

# Load and process the document
document = load_document("path/to/10k_report.pdf")
document_node = DocumentNode.from_document(document)

# Perform OCR
gcp_factory = GCPTaskProviderFactory(service_account_file="path/to/credentials.json")
ocr_provider = gcp_factory.get_page_ocr_provider(project_id="your-project-id", processor_id="your-processor-id")
ocr_provider.process_document_node(document_node)

# Create the locator
locator = document_node.locator

# Define entities to search for
company_names = ["SubsidiaryA", "PartnerB", "CompetitorC"]
executive_names = ["John Doe", "Jane Smith", "Alice Johnson"]
```

## Searching for Entities

Now, let's use DocPrompt's fast search capabilities to find all mentions of companies and executives. By leveraging the speed of the rust powered locator, along with python's builtin comprehension, we can execute our set of queries over the several-hundred page document in a matter of miliseconds.

```python
company_results = {
    company: locator.search(company)
    for company in company_names
}

executive_results = {
    executive: locator.search(executive)
    for executive in executive_names
}
```

## Detecting Proximity

Next, we'll check for instances where company names and executive names appear in close proximity:

```python
def check_proximity(bbox1, bbox2, threshold=0.1):
    left_collision = abs(bbox1.x0 - bbox2.x0) < threshold
    top_collision = abs(bbox1.top - bbox2.top) < threshold

    return left_collision and top_collision

potential_conflicts = []

for company, exec_name in product(company_names, executive_names):
    c_result = company_results[company]
    e_result = exexecutive_results[exec_name]

    # Check if the two results appear on the same page
    if c_result.page_number == e_result.page_number:
        c_bbox = c_result.text_location.merged_source_block.bounding_box
        e_bbox = e_result.text_location.merged_source_block.bounding_box

        # If they do, check if the bounding boxes break our threshold
        if check_proximity(c_bbox, e_bbox):
            potential_conflicts.append({
                'company': company,
                'executive': exec_name,
                'page': c_result.page_number,
                'company_bbox': c_bbox,
                'exec_bbox': e_bbox
            })
```

## Analyzing Results

Finally, let's analyze and display our results:

```python
print(f"Found {len(potential_conflicts)} potential conflicts of interest:")

for conflict in potential_conflicts:
    print(f"\nPotential conflict on page {conflict['page']}:")
    print(f"  Company: {conflict['company']}")
    print(f"  Executive: {conflict['executive']}")

    # Get surrounding context
    context_bbox = NormBBox(
        x0=min(conflict['company_bbox'].x0, conflict['exec_bbox'].x0) - 0.05,
        top=min(conflict['company_bbox'].top, conflict['exec_bbox'].top) - 0.05,
        x1=max(conflict['company_bbox'].x1, conflict['exec_bbox'].x1) + 0.05,
        bottom=max(conflict['company_bbox'].bottom, conflict['exec_bbox'].bottom) + 0.05
    )

    context_blocks = locator.get_overlapping_blocks(context_bbox, conflict['page'])

    print("  Context:")
    for block in context_blocks:
        print(f"    {block.text}")
```

This refined approach demonstrates several key features of DocPrompt:

1. **Fast and Accurate Search**: We use the `DocumentProvenanceLocator` to quickly find all mentions of companies and executives across the entire document.

2. **Spatial Analysis**: By leveraging the bounding box information, we can determine when two entities are mentioned in close proximity on the page.

3. **Contextual Information**: We use spatial queries to extract the surrounding text, providing context for each potential conflict of interest.

4. **Scalability**: This approach can easily handle multiple companies and executives, making it suitable for analyzing large, complex documents.

By combining these capabilities, DocPrompt enables efficient and thorough analysis of 10-K reports, helping to identify potential conflicts of interest that might otherwise be overlooked in manual review processes.
