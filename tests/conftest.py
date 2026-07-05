import pytest
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

@pytest.fixture
def sample_document():
    return Document(
        page_content="The Enterprise Retrieval-Augmented Generation (RAG) system integrates advanced semantic search tools. It allows query matching against high dimensional vectors.",
        metadata={"source": "sample_test.pdf", "page": 1}
    )

@pytest.fixture
def text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=5
    )
