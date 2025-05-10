import requests
import sys

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
from chromadb import PersistentClient

chroma = PersistentClient(path="./db")
collection = chroma.get_collection(name="repo")

def run_question(question: str):
    query_embedding = model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )

    if results['documents'] == []:
        print("No results found.")
        return
    if results['documents']:
        docs = results['documents'][0]
        context = "\n---\n".join(docs)

        prompt = f"""
        You are an AI assistant that answers questions about the codebase 

        ### Context:
        {context}

        ### Question:
        {question}

        ### Answer:

        """
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "codellama",
                "prompt": prompt,
                "stream": False
            }
        )
        answer = response.json()
        print(f"Answer: {answer['response']}")

if __name__ == "__main__":
    run_question(sys.argv[1])

