import uuid
import streamlit as st

from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq

# LOAD ENV
load_dotenv(r"E:\production-rag-eval\backend\.env")

st.write(os.getenv("GROQ_API_KEY"))
print(os.getenv("GROQ_API_KEY"))

# PAGE SETTINGS
st.set_page_config(
    page_title="Production RAG Chatbot",
    layout="wide"
)

# TITLE
st.title("📄 Production RAG Chatbot")

st.write(
    "Upload a PDF and ask intelligent questions using Llama 3 via Groq."
)

# FILE UPLOAD
uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

# EMBEDDING MODEL
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# PROCESS PDF
if uploaded_file is not None:

    # SAVE TEMP FILE
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ PDF uploaded successfully!")

    # LOAD PDF
    loader = PyPDFLoader("temp.pdf")

    documents = loader.load()

    # SPLIT TEXT
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = text_splitter.split_documents(documents)

    # UNIQUE VECTOR DB
    db_path = f"temp_db_{uuid.uuid4()}"

    # CREATE VECTORSTORE
    vectorstore = Chroma.from_documents(
        docs,
        embedding=embedding,
        persist_directory=db_path
    )

    # RETRIEVER
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )


    # LOAD GROQ MODEL
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    # QUESTION INPUT
    query = st.text_input(
        "Ask a question about the PDF:"
    )

    # ASK BUTTON
    if st.button("Ask"):

        with st.spinner("Generating answer..."):

            # RETRIEVE DOCS
            retrieved_docs = retriever.invoke(query)

            # BUILD CONTEXT
            context = "\n\n".join(
                [doc.page_content for doc in retrieved_docs]
            )

            # PROMPT
            prompt = f"""
You are an intelligent AI assistant specialized in answering questions from uploaded documents.

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

            # ANSWER
            st.subheader("📌 Answer")

            st.write(response.content)

            # RETRIEVED CHUNKS
            st.subheader("📄 Retrieved Chunks")

            for i, doc in enumerate(retrieved_docs):

                with st.expander(f"Chunk {i+1}"):

                    st.write(doc.page_content)