# 📄 Production RAG Chatbot with Evaluation Harness

A production-grade Retrieval-Augmented Generation (RAG) chatbot built using LangChain, ChromaDB, Groq Llama 3, and Streamlit.

This project allows users to upload PDFs and ask intelligent questions using semantic search and LLM-powered retrieval.

---

# 🚀 Features

- PDF Upload Support
- Semantic Search
- ChromaDB Vector Database
- HuggingFace Embeddings
- Groq Llama 3 Integration
- Streamlit Frontend
- Dynamic Document Retrieval
- Evaluation Harness
- Hallucination Detection
- Retrieval Quality Testing

---

# 🛠️ Tech Stack

- Python
- LangChain
- ChromaDB
- Streamlit
- HuggingFace Embeddings
- Groq API
- Llama 3
- RAG Architecture

---

# 📂 Project Structure

```bash
production-rag-eval/
│
├── backend/
│   ├── app/
│   ├── chroma_db/
│   └── .env
│
├── frontend/
│   └── app.py
│
├── evaluation/
│   ├── evaluate.py
│   └── test_dataset.json
│
├── docs/
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/production-rag-eval.git
```

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Environment

### Windows

```bash
venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file inside `backend/`

```env
GROQ_API_KEY=your_api_key
```

---

# ▶️ Run Application

```bash
cd frontend
streamlit run app.py
```

---

# 📊 Evaluation Metrics

- Faithfulness
- Context Precision
- Hallucination Detection
- Answer Relevancy

---

# 📸 Demo

Upload PDFs and ask intelligent questions using Llama 3.

---

# 👨‍💻 Author

Jaimin Panchal