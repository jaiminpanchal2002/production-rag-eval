
import json

from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)

from langchain_groq import ChatGroq


# ==========================================
# LOAD ENV
# ==========================================

load_dotenv("../backend/.env")


# ==========================================
# EMBEDDINGS
# ==========================================

embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)


# ==========================================
# LOAD VECTOR DB
# ==========================================

vectorstore = Chroma(
    persist_directory="../backend/chroma_db",
    embedding_function=embedding
)


# ==========================================
# LOAD DOCUMENT CHUNKS
# ==========================================

docs = vectorstore.similarity_search(
    "university exchange partnerships",
    k=10
)

context = "\n\n".join([
    doc.page_content
    for doc in docs
])


# ==========================================
# LOAD LLM
# ==========================================

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0
)


# ==========================================
# PROMPT
# ==========================================


prompt = f"""
You are an AI evaluation dataset generator.

Generate 10 high-quality evaluation questions
and answers from the document context.

RULES:
- Questions must come ONLY from context
- Answers must be concise
- No hallucinations
- Return ONLY valid JSON
- No markdown
- No explanation

JSON FORMAT:
[
  {{
    "question": "...",
    "ground_truth": "..."
  }}
]

Context:
{context}
"""




# ==========================================
# GENERATE TESTSET
# ==========================================

response = llm.invoke(prompt)

generated_data = response.content


# ==========================================
# SAVE JSON
# ==========================================

with open("test_dataset.json", "w") as f:
    f.write(generated_data)


print("\n✅ Dynamic test dataset generated!")

