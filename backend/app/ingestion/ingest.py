from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
 
loader = PyPDFLoader("../docs/sample_pdfs/sample.pdf")
documents = loader.load()
 
print(f"Loaded {len(documents)} pages.")
 
# STEP 2: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
 
docs = text_splitter.split_documents(documents)
 
print(f"Created {len(docs)} chunks.")
 
# STEP 3: Create embeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
 
# STEP 4: Store in ChromaDB
vectorstore = Chroma.from_documents(
    docs,
    embedding=embedding,
   persist_directory="./chroma_db"
)
 
 
 
print("Documents embedded successfully!")