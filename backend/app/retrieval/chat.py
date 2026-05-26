from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq

# LOAD ENV
load_dotenv("backend/.env")

# EMBEDDING MODEL
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# LOAD VECTOR DB
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding
)

# RETRIEVER
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# GROQ MODEL
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0
)

print("✅ Production RAG Chat Ready!")
print("Type 'exit' to quit.\n")

# CHAT LOOP
while True:

    query = input("Ask your question: ")

    if query.lower() == "exit":
        break

    # RETRIEVE DOCS
    docs = retriever.invoke(query)

    # BUILD CONTEXT
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # PROMPT
    prompt = f"""
You are an AI assistant specialized in answering questions from uploaded documents.

Use ONLY the provided context.

If answer is not found in context, say:
"I could not find this information in the document."

Context:
{context}

Question:
{query}
"""

    # GENERATE RESPONSE
    response = llm.invoke(prompt)

    print("\nANSWER:\n")

    print(response.content)

    print("\n" + "=" * 50 + "\n")