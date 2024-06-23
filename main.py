"""Test the library."""

from typing import Optional

from pydantic import Field

from docprompt import DocumentNode, load_document
from docprompt.schema.metadata import BaseMetadata


class SamplePageMetadata(BaseMetadata):
    """Must have defaults set."""

    page_number: Optional[int] = Field(None)


class SampleMetadata(BaseMetadata):
    """Must have defaults set."""

    title: Optional[str] = Field(None)


CustomDocumentNode = DocumentNode[SampleMetadata, SamplePageMetadata]


def main():
    # TODO: Fix recursive creating of metadata

    document = load_document("data/example-1.pdf")
    node = CustomDocumentNode.from_document(document)

    node[0].metadata = SamplePageMetadata(page_number=1)
    node.metadata = SampleMetadata(title="Example Document")

    print(node)
    print(node[0])
    print(node[1])
    print(node[2])
    print("---")

    node.persist("s3://docprompt-test-storage/TEST")

    node = DocumentNode.from_storage(
        "s3://docprompt-test-storage/TEST", "191e8a7d232bfdc773858c39a8ff6ac7"
    )

    print(node)
    print(node[0])
    print(node[1])
    print(node[2])
    print("---")

    node.persist("s3://docprompt-test-storage/TEST")

    node = CustomDocumentNode.from_storage(
        "s3://docprompt-test-storage/TEST", "191e8a7d232bfdc773858c39a8ff6ac7"
    )

    print(node)
    print(node[0])
    print(node[1])
    print(node[2])


if __name__ == "__main__":
    main()
