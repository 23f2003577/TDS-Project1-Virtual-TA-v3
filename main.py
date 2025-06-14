import os
import json
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

# Load only once per cold start
npz = np.load("still_merged_embeddings.npz", allow_pickle=True)
embeddings = npz["embeddings"]
texts = npz["texts"]
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
model = SentenceTransformer("all-MiniLM-L6-v2")

def handler(request, response):
    try:
        body = request.get_json()
        question = body["question"]
        q_emb = model.encode([question])
        _, I = index.search(np.array(q_emb), k=5)
        context = "\n\n".join([texts[i] for i in I[0]])

        prompt = (
            "You are a helpful TA.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer clearly. Include links if present."
        )

        aipipe_payload = {
            "model": "mistral",
            "messages": [
                {"role": "system", "content": "You are a helpful TA."},
                {"role": "user", "content": prompt}
            ]
        }

        headers = {"Authorization": f"Bearer {os.environ['AIPIPE_TOKEN']}"}
        resp = requests.post(os.environ['AIPIPE_BASE_URL'], json=aipipe_payload, headers=headers, timeout=15)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]

        return response.status(200).json({
            "answer": content,
            "links": []
        })

    except Exception as e:
        return response.status(500).json({"error": str(e)})
