import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

# Simple mock FastAPI application to represent RAG API backend
app = FastAPI()

@app.get("/api/v1/health")
def health_check():
    return {"status": "healthy", "service": "Production RAG Engine"}

@app.post("/api/v1/query")
def process_query(payload: dict):
    query = payload.get("query", "")
    if not query:
        return {"error": "Query cannot be empty"}, 400
    
    # Mock RAG response
    return {
        "query": query,
        "answer": f"This is a mock RAG response for: {query}",
        "sources": [{"file": "sample.pdf", "page": 1}],
        "metrics": {
            "faithfulness": 0.95,
            "answer_relevancy": 0.92,
            "context_precision": 0.89
        }
    }

client = TestClient(app)

def test_api_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_api_query():
    response = client.post("/api/v1/query", json={"query": "What is changing?"})
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "answer" in data
    assert data["metrics"]["faithfulness"] == 0.95
