import pytest
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def test_vectorstore_flow(sample_document, text_splitter):
    """
    Test vectorstore integration: split documents, build embeddings, add to DB, and retrieve.
    """
    chunks = text_splitter.split_documents([sample_document])
    
    # Use small, fast embeddings model for tests
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create Vector DB in memory (no persist directory is needed for simple memory test)
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    # Verify retrieval
    results = db.similarity_search("semantic search tools", k=1)
    assert len(results) == 1
    assert "RAG" in results[0].page_content or "semantic" in results[0].page_content
