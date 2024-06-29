# Multi-Label Classification with DocPrompt: Academic Research Papers

In this guide, we'll use DocPrompt to classify academic research papers in a PDF document into multiple relevant fields. We'll use multi-label classification, allowing each page (paper) to be assigned to one or more categories based on its content.

## Setting Up

First, let's import the necessary modules and set up our environment:

```python
from docprompt import load_document, DocumentNode
from docprompt.tasks.factory import AnthropicTaskProviderFactory
from docprompt.tasks.classification import ClassificationInput, ClassificationTypes

# Initialize the Anthropic factory
# Ensure you have set the ANTHROPIC_API_KEY environment variable
factory = AnthropicTaskProviderFactory()

# Create the classification provider
classification_provider = factory.get_page_classification_provider()
```

## Preparing the Document

Load your collection of research papers and create a DocumentNode:

```python
document = load_document("path/to/your/research_papers.pdf")
document_node = DocumentNode.from_document(document)
```

## Configuring the Classification Task

For multi-label classification, we'll create a `ClassificationInput` object that specifies our labels and provides instructions for the model:

```python
classification_input = ClassificationInput(
    type=ClassificationTypes.MULTI_LABEL,
    instructions=(
        "Classify the research paper on this page into one or more relevant fields "
        "based on its title, abstract, methodology, and key findings. A paper may "
        "belong to multiple categories if it spans multiple disciplines."
    ),
    labels=[
        "Machine Learning",
        "Natural Language Processing",
        "Computer Vision",
        "Robotics",
        "Data Mining",
        "Cybersecurity",
        "Bioinformatics",
        "Quantum Computing"
    ],
    descriptions=[
        "Algorithms and statistical models that computer systems use to perform tasks without explicit instructions",
        "Processing and analyzing natural language data",
        "Enabling computers to derive meaningful information from digital images, videos and other visual inputs",
        "Design, construction, operation, and use of robots",
        "Process of discovering patterns in large data sets",
        "Protection of computer systems from theft or damage to their hardware, software, or electronic data",
        "Application of computational techniques to analyze biological data",
        "Computation based on the principles of quantum theory"
    ],
    confidence=True  # Request confidence scores
)
```

Let's break down the `ClassificationInput`:

- `type`: Set to `ClassificationTypes.MULTI_LABEL` for multi-label classification.
- `labels`: List of possible categories for our research papers.
- `instructions`: Clear directions for the model on how to classify the papers, emphasizing that multiple labels can be assigned.
- `descriptions`: Provide additional context for each label to improve classification accuracy.
- `confidence`: Set to `True` to get confidence scores with the results.

## Performing Classification

Now, let's run the classification task on our collection of research papers:

```python
results = classification_provider.process_document_node(
    document_node,
    classification_input
)
```

## Interpreting Results

Let's examine the classification results for each research paper:

```python
for page_number, result in results.items():
    categories = result.labels
    confidence = result.score
    print(f"Research Paper on Page {page_number}:")
    print(f"\tCategories: {', '.join(categories)}  ({confidence})")
    print('---')
```

This will print the assigned categories for each research paper, along with the confidence level.

## Tips for Effective Multi-Label Classification

1. **Comprehensive Label Set**: Ensure your label set covers the main topics in your domain but isn't so large that it becomes unwieldy.

2. **Clear Instructions**: Emphasize in your instructions that multiple labels can and should be assigned when appropriate.

3. **Use Descriptions**: The `descriptions` field helps the model understand the nuances of each category, which is especially important for interdisciplinary papers.

4. **Consider Confidence Scores**: In multi-label classification, confidence scores can indicate how strongly a paper fits into each assigned category.

5. **Analyze Label Co-occurrences**: Look for patterns in which labels frequently appear together to gain insights into interdisciplinary trends.

6. **Handle Outliers**: If a paper doesn't fit well into any category, consider adding a catch-all category like "Other" or "Interdisciplinary" in future iterations.

## Advanced Usage: Increasing the Power

For more control over the classification process, you can specify a beefier model from Anthropic to up the reasoning power. This can be done when setting up the task provider OR at inference time. Allowing for easy, fine-grained control of over provider defaults and runtime overrides.

```python
classification_provider = factory.get_page_classification_provider(
    model_name="claude-3-5-sonnet-20240620"  # setup the task provider with sonnet-3.5
)

results = classification_provider.process_document_node(
    document_node,
    classification_input,
    model_name="claude-3-5-sonnet-20240620" # or you can declare model name at inference time as well
)
```

## Conclusion

Multi-label classification with DocPrompt provides a powerful way to categorize complex documents like research papers that often span multiple disciplines. By carefully crafting your `ClassificationInput` with clear labels, instructions, and descriptions, you can achieve nuanced and informative document analysis.

Remember that the quality of your results depends on the clarity of your instructions, the comprehensiveness of your label set, and the appropriateness of your descriptions. Experiment with different configurations to find what works best for your specific use case.

This approach can be adapted to other multi-label classification tasks, such as categorizing news articles by multiple topics, classifying products by multiple features, or tagging images with multiple attributes.
