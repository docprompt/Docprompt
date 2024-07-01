# Binary Classification with DocPrompt

Binary classification is a fundamental task in document analysis, where you categorize pages into one of two classes. This guide will walk you through performing binary classification using DocPrompt with the Anthropic provider.

## Setting Up

First, let's import the necessary modules and set up our environment:

```python
from docprompt import load_document, DocumentNode
from docprompt.tasks.factory import AnthropicTaskProviderFactory
from docprompt.tasks.classification import ClassificationInput, ClassificationTypes

# Initialize the Anthropic factory
# Make sure you have set the ANTHROPIC_API_KEY environment variable
factory = AnthropicTaskProviderFactory()

# Create the classification provider
classification_provider = factory.get_page_classification_provider()
```

## Preparing the Document

Load your document and create a DocumentNode:

```python
document = load_document("path/to/your/document.pdf")
document_node = DocumentNode.from_document(document)
```

## Configuring the Classification Task

For binary classification, we need to create a `ClassificationInput` object. This object acts as a prompt for the model, guiding its classification decision. Here's how to set it up:

```python
classification_input = ClassificationInput(
    type=ClassificationTypes.BINARY,
    instructions="Determine if the page contains information about financial transactions.",
    confidence=True  # Optional: request confidence scores
)
```

Let's break down the `ClassificationInput`:

- `type`: Set to `ClassificationTypes.BINARY` for binary classification.
- `instructions`: This is crucial for binary classification. Provide clear, specific instructions for what the model should look for.
- `confidence`: Set to `True` if you want confidence scores with the results.

Note: For binary classification, you don't need to specify labels. The model will automatically use "YES" and "NO" as labels.

## Performing Classification

Now, let's run the classification task:

```python
results = classification_provider.process_document_node(
    document_node,
    classification_input
)
```

## Interpreting Results

The `results` dictionary contains the classification output for each page. Let's examine the results:

```python
for page_number, result in results.items():
    label = result.labels
    confidence = result.score
    print(f"Page {page_number}:")
    print(f"\tClassification: {label} ({confidence})")
    print('---')
```

This will print the classification (YES or NO) for each page, along with the confidence level if requested.

## Tips for Effective Binary Classification

1. **Clear Instructions**: The `instructions` field in `ClassificationInput` is crucial. Be specific about what constitutes a "YES" classification.

2. **Consider Page Context**: Remember that the model analyzes each page independently. If your classification requires context from multiple pages, you may need to adjust your approach.

3. **Confidence Scores**: Use the `confidence` option to get an idea of how certain the model is about its classifications. This can be helpful for identifying pages that might need human review.

4. **Iterative Refinement**: If you're not getting the desired results, try refining your instructions. You might need to be more specific or provide examples of what constitutes a positive classification.

## Conclusion

Binary classification with DocPrompt allows you to quickly categorize pages in your documents. By leveraging the Anthropic provider and carefully crafting your `ClassificationInput`, you can achieve accurate and efficient document analysis.

Remember that the quality of your results heavily depends on the clarity and specificity of your instructions. Experiment with different phrasings to find what works best for your specific use case.
