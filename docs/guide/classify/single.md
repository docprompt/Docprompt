# Single-Label Classification with DocPrompt: Recipe Categories

In this guide, we'll use DocPrompt to classify recipes in a PDF document into distinct meal categories. We'll use single-label classification, meaning each page (recipe) will be assigned to one category: Breakfast, Lunch, Dinner, or Dessert.

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

Load your recipe book PDF and create a DocumentNode:

```python
document = load_document("path/to/your/recipe_book.pdf")
document_node = DocumentNode.from_document(document)
```

## Configuring the Classification Task

For single-label classification, we'll create a `ClassificationInput` object that specifies our labels and provides instructions for the model:

```python
classification_input = ClassificationInput(
    type=ClassificationTypes.SINGLE_LABEL,
    labels=["Breakfast", "Lunch", "Dinner", "Dessert"],
    instructions="Classify the recipe on this page into one of the given meal categories based on its ingredients, cooking methods, and typical serving time.",
    descriptions=[
        "Morning meals, often including eggs, cereals, or pastries",
        "Midday meals, typically lighter fare like sandwiches or salads",
        "Evening meals, often the most substantial meal of the day",
        "Sweet treats typically served after a meal or as a snack"
    ],
    confidence=True  # Request confidence scores
)
```

Let's break down the `ClassificationInput`:

- `type`: Set to `ClassificationTypes.SINGLE_LABEL` for single-label classification.
- `labels`: List of possible categories for our recipes.
- `instructions`: Clear directions for the model on how to classify the recipes.
- `descriptions`: (Optional) Provide additional context for each label to improve classification accuracy.
    - _Note that the `descriptions` array must be the same length as the `labels` array._
- `confidence`: Set to `True` to get confidence scores with the results.

## Performing Classification

Now, let's run the classification task on our recipe book:

```python
results = classification_provider.process_document_node(
    document_node,
    classification_input
)
```

## Interpreting Results

Let's examine the classification results for each recipe:

```python
for page_number, result in results.items():
    category = result.labels
    confidence = result.score

    print(f"Recipe on Page {page_number}:")
    print(f"\tCategory: {category} ({confidence})")
    print('---')
```

This will print the assigned category (Breakfast, Lunch, Dinner, or Dessert) for each recipe, along with the confidence level.

## Tips for Effective Single-Label Classification

1. **Comprehensive Labels**: Ensure your label set covers all possible categories without overlap.

2. **Clear Instructions**: Provide specific criteria for each category in your instructions. For recipes, mention considering ingredients, cooking methods, and typical serving times.

3. **Use Descriptions**: The `descriptions` field can help the model understand nuances between categories, especially for edge cases like brunch recipes.

4. **Consider Confidence Scores**: Low confidence scores might indicate recipes that don't clearly fit into one category, such as versatile dishes that could be served at multiple meals.

5. **Handling Edge Cases**: If you encounter many low-confidence classifications, you might need to refine your categories or instructions. For example, you might add an "Anytime" category for versatile recipes.

## Advanced Usage: Customizing the Model

If you need to experiment with different LLM models, based on the complexity of your task, you may control the model_name parameter to the classification provider:

```python
haiku_classification_provider = factory.get_page_classification_provider(
    model_name="claude-3-haiku-20240307"
)

sonnet_classification_provider = factory.get_page_classification_provider(
    model_name="claude-3-5-sonnet-20240620"
)
```

## Conclusion

Single-label classification with DocPrompt provides a powerful way to categorize pages in your documents, such as recipes in a cookbook. By carefully crafting your `ClassificationInput` with clear labels, instructions, and descriptions, you can achieve accurate and efficient document analysis.

Remember that the quality of your results depends on the clarity of your instructions and the appropriateness of your label set. Experiment with different phrasings and label combinations to find what works best for your specific use case.

This approach can be easily adapted to other single-label classification tasks, such as categorizing scientific papers by field, sorting legal documents by type, or classifying news articles by topic.
