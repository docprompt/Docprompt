"""Test the library."""

from typing import Optional

import fsspec
from pydantic import BaseModel, Field

from docprompt import Document, DocumentNode


class SamplePageMetadata(BaseModel):
    """Must have defaults set."""

    page_number: Optional[int] = Field(None)


class SampleMetadata(BaseModel):
    """Must have defaults set."""

    title: Optional[str] = Field(None)


CustomDocumentNode = DocumentNode[SampleMetadata, SamplePageMetadata]


def main():
    with fsspec.open("data/example-1.pdf", "rb") as f:
        pdf_bytes = f.read()

    _doc = Document.from_bytes(pdf_bytes, name="data/example-1.pdf")

    node = CustomDocumentNode.from_storage(
        "s3://docprompt-test-storage/TEST", "191e8a7d232bfdc773858c39a8ff6ac7"
    )
    print(node)
    print(node[0])
    print(node[1])
    print(node[2])
    print("---")

    node.persist()

    print(node)
    print(node[0])
    print(node[1])
    print(node[2])


if __name__ == "__main__":
    main()
