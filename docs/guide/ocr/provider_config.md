# Customizing OCR Providers

Docprompt uses a factory pattern to manage credentials and create task providers efficiently. This guide will demonstrate how to configure and customize OCR providers, focusing on Amazon Textract and Google Cloud Platform (GCP) as examples.

## Understanding the Factory Pattern

Docprompt uses task provider factories to manage credentials and create providers for various tasks. This approach allows for:

1. Centralized credential management
2. Easy creation of multiple task providers from a single backend
3. Separation of provider-specific and task-specific configurations

Here's a simplified example of how the factory pattern works:

```python
from docprompt.tasks.factory import GCPTaskProviderFactory, AmazonTaskProviderFactory

# Create a GCP factory
gcp_factory = GCPTaskProviderFactory(
    service_account_file="path/to/service_account.json"
)

# Create an Amazon factory
amazon_factory = AmazonTaskProviderFactory(
    aws_access_key_id="YOUR_ACCESS_KEY",
    aws_secret_access_key="YOUR_SECRET_KEY",
    region_name="us-west-2"
)
```

## Creating OCR Providers

Once you have a factory, you can create OCR providers with task-specific configurations:

```python
# Create a GCP OCR provider
gcp_ocr_provider = gcp_factory.get_page_ocr_provider(
    project_id="YOUR_PROJECT_ID",
    processor_id="YOUR_PROCESSOR_ID",
    max_workers=4,
    return_images=True
)

# Create an Amazon Textract provider
amazon_ocr_provider = amazon_factory.get_page_ocr_provider(
    max_workers=4,
    exclude_bounding_poly=True
)
```

## Understanding Provider Configuration

When configuring OCR providers, you'll encounter two types of parameters:

1. **Docprompt generic parameters**: These are common across different providers and control Docprompt's behavior.
    - `max_workers`: Controls concurrency for processing large documents
    - `exclude_bounding_poly`: Reduces memory usage by excluding detailed polygon data

2. **Provider-specific parameters**: These are unique to each backend and control provider-specific features. For example, if using GCP as an OCR provider, you must specify `project_id`, `processor_id` and you may optionally set `return_image_quality_scores`.

## Provider-Specific Features and Limitations

### Google Cloud Platform (GCP)
- Offers advanced layout analysis and image quality scoring
- Supports returning rasterized images of processed pages
- Requires GCP-specific project and processor IDs

Example configuration:
```python
gcp_ocr_provider = gcp_factory.get_page_ocr_provider(
    project_id="YOUR_PROJECT_ID",
    processor_id="YOUR_PROCESSOR_ID",
    max_workers=4,  # Docprompt generic
    return_images=True,  # GCP-specific
    return_image_quality_scores=True  # GCP-specific
)
```

### Amazon Textract
- Focuses on text extraction and layout analysis
- Provides confidence scores for extracted text
- Does not support returning rasterized images

Example configuration:
```python
amazon_ocr_provider = amazon_factory.get_page_ocr_provider(
    max_workers=4,  # Docprompt generic
    exclude_bounding_poly=True  # Docprompt generic
)
```

## Best Practices

1. **Use factories for credential management**: This centralizes authentication and makes it easier to switch between providers.

2. **Consult provider documentation**: Always refer to the latest documentation from AWS or GCP for the most up-to-date information on their OCR services.

3. **Check Docprompt API reference**: Review Docprompt's API documentation for each provider to understand available configurations.

4. **Optimize for your use case**: Configure providers based on your specific needs, balancing performance and feature requirements.

## Conclusion

Understanding the factory pattern and the distinction between Docprompt generic and provider-specific parameters is key to effectively configuring OCR providers in Docprompt. While this guide provides an overview using Amazon Textract and GCP as examples, the principles apply to other providers as well. Always consult the specific provider's documentation and Docprompt's API reference for the most current and detailed information.
