
import json
import time

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
# LOAD TEST DATASET
# ==========================================

with open("test_dataset.json", "r") as f:
    test_data = json.load(f)


# ==========================================
# EMBEDDINGS
# ==========================================

embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)


# ==========================================
# LOAD VECTOR DB
# ==========================================

vectorstore = Chroma( persist_directory="E:/production-rag-eval/backend/chroma_db", embedding_function=embedding )


# ==========================================
# RETRIEVER
# ==========================================

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5
    }
)


# ==========================================
# LLM
# ==========================================

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0
)


# ==========================================
# METRICS
# ==========================================

total_questions = 0

successful_retrievals = 0

successful_answers = 0

total_response_time = 0


print("\n🚀 Running Custom RAG Evaluation...\n")


# ==========================================
# EVALUATION LOOP
# ==========================================

for item in test_data:

    question = item["question"]

    ground_truth = item["ground_truth"]

    print("=" * 60)

    print(f"\nQUESTION:\n{question}\n")

    # START TIMER
    start_time = time.time()

    # RETRIEVE DOCS
    docs = retriever.invoke(question)

    # RESPONSE TIME
    retrieval_time = time.time() - start_time

    total_response_time += retrieval_time

    total_questions += 1

    # CHECK RETRIEVAL
    if len(docs) > 0:
        successful_retrievals += 1

    # BUILD CONTEXT
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # PROMPT
    prompt = f"""
You are an AI assistant.

Use ONLY the provided context.

If answer is not found, say:
"I could not find this information."

Context:
{context}

Question:
{question}
"""

    # GENERATE RESPONSE
    response = llm.invoke(prompt)

    answer = response.content

    # CHECK ANSWER
    if (
        "I could not find" not in answer
        and len(answer.strip()) > 10
    ):
        successful_answers += 1

    # PRINT RESULTS
    print(f"GROUND TRUTH:\n{ground_truth}\n")

    print(f"AI ANSWER:\n{answer}\n")

    print(f"RETRIEVED CHUNKS: {len(docs)}")

    print(f"RESPONSE TIME: {retrieval_time:.2f} sec\n")


# ==========================================
# FINAL METRICS
# ==========================================

retrieval_accuracy = (
    successful_retrievals / total_questions
) * 100

answer_success_rate = (
    successful_answers / total_questions
) * 100

average_latency = (
    total_response_time / total_questions
)

print("\n" + "=" * 60)

print("\n✅ FINAL EVALUATION RESULTS\n")

print(f"Total Questions: {total_questions}")

print(
    f"Retrieval Success Rate: "
    f"{retrieval_accuracy:.2f}%"
)

print(
    f"Answer Success Rate: "
    f"{answer_success_rate:.2f}%"
)

print(
    f"Average Retrieval Latency: "
    f"{average_latency:.2f} sec"
)

print("\n🎯 Evaluation Completed Successfully!\n")

