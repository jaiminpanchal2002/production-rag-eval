import pytest
from langchain_core.documents import Document

def test_document_chunking(sample_document, text_splitter):
    """
    Test that the text splitter correctly splits documents into manageable chunks.
    """
    chunks = text_splitter.split_documents([sample_document])
    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk.page_content) <= 55
        assert "source" in chunk.metadata
        assert chunk.metadata["source"] == "sample_test.pdf"

def test_document_metadata_retention(sample_document):
    """
    Test that metadata is preserved when working with Document objects.
    """
    assert sample_document.metadata["page"] == 1
    assert sample_document.metadata["source"] == "sample_test.pdf"
