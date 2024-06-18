"""Simple script for testing various operations."""

import os
import argparse


from pydantic import BaseModel, Field
from docprompt import load_document, DocumentNode

from docprompt.storage import LocalFileSystemStorageProvider


PDF_SAMPLE_DIR = os.path.abspath("data")


def pdf_path():
    """Parse the arguments for the script."""

    # Parse the first argument, which should be a pdf file name
    parser = argparse.ArgumentParser(description="Parse a pdf file.")
    parser.add_argument("pdf_file", type=str, help="The pdf file to parse.")
    args = parser.parse_args()

    pdf_name = args.pdf_file

    # If the pdf_name is a path, then we should get the absolute path of the file
    if os.path.sep in pdf_name:
        pdf_name = os.path.abspath(pdf_name)
    else:
        pdf_name = os.path.join(PDF_SAMPLE_DIR, pdf_name)

    return pdf_name


class TestMetadata(BaseModel):
    title: str = Field(...)


def main():
    fp = pdf_path()
    document = load_document(fp)
    _document_node = DocumentNode.from_document(document)

    _document_node.metadata = TestMetadata(title="Test Title")

    storage_provider = LocalFileSystemStorageProvider(
        document_node_class=DocumentNode, document_metadata_class=TestMetadata
    )

    storage_provider.store(_document_node)

    retrieved_document = storage_provider.retrieve(_document_node.file_hash)

    assert retrieved_document.metadata.title == _document_node.metadata.title


if __name__ == "__main__":
    main()
