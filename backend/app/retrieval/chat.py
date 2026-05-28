import uuid
import streamlit as st
 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
 
from langchain_google_genai import ChatGoogleGenerativeAI
 
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
 
# =========================================================
# PAGE CONFIG
# =========================================================
 
st.set_page_config(
    page_title="Production Gemini RAG Chatbot",
    layout="wide"
)
 
st.title("📄 Production Gemini RAG Chatbot")
 
st.write("Upload any PDF and ask any question naturally.")
 
# =========================================================
# GEMINI API KEY
# =========================================================
GEMINI_API_KEY = "AIzaSyDgF9tkvXygthFtN6S7rDGOKnJKWQZwEDk"
 
 
# =========================================================
# FILE UPLOAD
# =========================================================
 
uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)
 
# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================
 
@st.cache_resource
def load_embedding():
 
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )
 
embedding = load_embedding()
 
# =========================================================
# LOAD GEMINI MODEL
# =========================================================
 
@st.cache_resource
def load_llm():
 
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0
    )
 
llm = load_llm()
 
# =========================================================
# STRICT PROMPT
# =========================================================
 
prompt_template = """
You are a professional PDF question-answering assistant.
 
Your task:
- Answer ONLY from the provided PDF context
- Give direct and clean answers
- Explain naturally when user asks for summary/explanation
- NEVER say:
  "I do not have access to the document"
  "context missing"
  "based on provided information"
- NEVER mention limitations
- NEVER hallucinate
- NEVER invent information
 
If answer is not found, reply EXACTLY:
"I could not find this information in the document."
 
PDF CONTEXT:
{context}
 
USER QUESTION:
{question}
 
FINAL ANSWER:
"""
 
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)
 
# =========================================================
# MAIN PROCESS
# =========================================================
 
if uploaded_file is not None:
 
    # =====================================================
    # SAVE PDF
    # =====================================================
 
    pdf_path = "temp.pdf"
 
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())
 
    st.success("PDF uploaded successfully!")
 
    # =====================================================
    # LOAD PDF
    # =====================================================
 
    loader = PyPDFLoader(pdf_path)
 
    documents = loader.load()
 
    st.info(f"Loaded {len(documents)} pages.")
 
    # =====================================================
    # SPLIT DOCUMENTS
    # =====================================================
 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )
 
    docs = text_splitter.split_documents(documents)
 
    st.info(f"Created {len(docs)} chunks.")
 
    # =====================================================
    # CREATE VECTOR DATABASE
    # =====================================================
 
    db_path = f"temp_db_{uuid.uuid4()}"
 
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=db_path
    )
 
    # =====================================================
    # CREATE RETRIEVER
    # =====================================================
 
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 12}
    )
 
    # =====================================================
    # CREATE QA CHAIN
    # =====================================================
 
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PROMPT
        }
    )
 
    # =====================================================
    # USER QUESTION
    # =====================================================
 
    query = st.text_input(
        "Ask any question about the PDF:",
        placeholder="Example: Explain this PDF"
    )
 
    # =====================================================
    # ASK BUTTON
    # =====================================================
 
    if st.button("Ask"):
 
        if query.strip() == "":
 
            st.warning("Please enter a question.")
 
        else:
 
            with st.spinner("Analyzing PDF and generating answer..."):
 
                # =============================================
                # GENERATE RESPONSE
                # =============================================
 
                response = qa_chain.invoke(query)
 
                # =============================================
                # SHOW ANSWER
                # =============================================
 
                st.subheader("Answer")
 
                st.write(response["result"].strip())
 
                # =============================================
                # SHOW RETRIEVED CHUNKS
                # =============================================
 
                with st.expander("Retrieved Chunks"):
 
                    for i, doc in enumerate(
                        response["source_documents"]
                    ):
 
                        st.write(f"### Chunk {i+1}")
 
                        st.write(doc.page_content)
 
                        st.divider()