import os
import requests
import numpy as np
import faiss
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

# Load from .npz
data = np.load("still_merged_embeddings.npz", allow_pickle=True)
embeddings = data["embeddings"]
texts = data["texts"]

# Build FAISS index at runtime
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Load model for future compatibility (you already embedded before, so it's optional)
model = SentenceTransformer("all-MiniLM-L6-v2")

# FastAPI setup
app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    image: str = None  # not yet processed

@app.post("/api/")
async def answer(query: QueryRequest):
    # Embed the question
    q_emb = model.encode([query.question])
    D, I = index.search(np.array(q_emb), k=5)

    # Get top matched contexts
    matched_texts = [texts[i] for i in I[0]]
    context = "\n\n".join(matched_texts)

    # Build prompt
    prompt = (
        "You are a helpful TA for the Tools in Data Science course.\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query.question}\n\n"
        "Answer clearly. Include relevant links if they appear in context."
    )

    # Send to AIPipe
    headers = {"Authorization": f"Bearer {os.getenv('AIPIPE_TOKEN')}"}
    payload = {
        "model": "mistral",
        "messages": [
            {"role": "system", "content": "You are a helpful TA."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(os.getenv("AIPIPE_BASE_URL"), json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        ai_answer = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        ai_answer = f"Failed to get response: {str(e)}"

    return {
        "answer": ai_answer,
        "links": []  # optional: parse links from context later
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
