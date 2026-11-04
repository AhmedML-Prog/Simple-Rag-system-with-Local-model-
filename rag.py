import requests
import numpy as np
from pathlib import Path

# LM Studio configuration
BASE_URL = "http://127.0.0.1:1235/v1"

EMBEDDED_MODEL = "text-embedding-nomic-embed-text-v1.5"
CHAT_MODEL = "nvidia/nemotron-3-nano-4b"


# Generate embeddings
def embed_text(text):

    r = requests.post(
        f"{BASE_URL}/embeddings",
        json={
            "model": EMBEDDED_MODEL,
            "input": text
        },
        timeout=120
    )

    r.raise_for_status()

    data = r.json()["data"]
    data.sort(key=lambda x: x["index"])

    return np.array(
        [item["embedding"] for item in data],
        dtype=np.float32
    )


# Ask the local LLM
def ask_llm(prompt):

    r = requests.post(
        f"{BASE_URL}/chat/completions",
        json={
            "model": CHAT_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "stream": False
        },
        timeout=180
    )

    r.raise_for_status()

    return r.json()["choices"][0]["message"]["content"]


# Retrieve relevant chunks using cosine similarity
def retrieve(query, k=3):

    query_vector = embed_text(query)[0]

    similarities = np.dot(
        chunk_vectors,
        query_vector
    ) / (
        np.linalg.norm(chunk_vectors, axis=1)
        * np.linalg.norm(query_vector)
        + 1e-10
    )

    top_k_indices = np.argsort(similarities)[::-1][:k]

    return [chunks[i] for i in top_k_indices]


# Initialize chunks list
chunks = []

docs_path = Path("docs")

if not docs_path.exists():
    raise FileNotFoundError("The docs folder does not exist!")


# Read Markdown files
for file in docs_path.glob("*.md"):

    text_content = file.read_text(encoding="utf-8")

    paragraphs = [
        p.strip()
        for p in text_content.split("\n\n")
        if p.strip()
    ]

    chunks.extend(paragraphs)


# Generate embeddings
if chunks:

    chunk_vectors = embed_text(chunks)

    print(f"Indexed {len(chunks)} chunks")
    print(f"Embedding shape: {chunk_vectors.shape}")

    assert chunk_vectors.shape[0] == len(chunks)

    print("Embeddings generated successfully!")

else:
    raise ValueError("No Markdown files found in docs/")


# Interactive RAG
while True:

    question = input("\nEnter a question (or 'exit' to quit): ")

    if question.lower() == "exit":
        break

    retrieved_chunks = retrieve(question)

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the provided context.
If the answer is not present in the context, say:
"I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

    answer = ask_llm(prompt)
    print("\nAnswer:", answer)