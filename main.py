import webbrowser
from sentence_transformers import SentenceTransformer
import threading
import time
import subprocess
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests
import uvicorn
from chromadb import PersistentClient
from pathlib import Path
from tree_sitter_language_pack import get_parser
from tree_sitter import Node
from setup_embeddings import extract_functions_rs  # Make sure to import from the correct location
config = {
    'model': 'codellama',
    'ollama_port': 11434,
    'chroma_port': 8000,
    "web_port": 8001,
}

def launch_chroma():
    try:
        subprocess.Popen(["chroma", "run", "--path", "./db", "--host", "0.0.0.0", "--port", str(config["chroma_port"])])
        print(f"Started ChromaDB on port {config['chroma_port']}")
    except Exception as e:
        print("Failed to launch ChromaDB:", e)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


class Query(BaseModel):
    question: str

class RepoPath(BaseModel):
    path: str

model_name = config['model']
ollama_url = f"http://localhost:{config['ollama_port']}/api/generate"

@app.post("/add-repo")
def add_repo(repo: RepoPath):

    LANGUAGES = {'.rs': 'rust'}
    supported_extensions = ['.rs']
    folders_to_ignore = ['.git', 'node_modules', 'target', '__pycache__', 'dist', 'build', 'venv', 'env', 'out', 'lib', 'libs', 'app', '.next']
    chunks = []
    base_path = Path(repo.path)

    for file in base_path.rglob('*'):
        use_file = file.is_file() and not file.name.startswith('.') and file.suffix.lower() in supported_extensions
        if not use_file:
            continue
        if any(folder in file.parts for folder in folders_to_ignore):
            continue

        text = file.read_text(encoding='utf-8', errors='ignore')
        functions = extract_functions_rs(text, LANGUAGES[file.suffix])
        for function in functions:
            chunks.append({
                "content": function[0],
                "metadata": {"file": str(file), "function": function[1]}
            })

    if not chunks:
        return {"status": "No valid chunks found."}

    texts = [chunk["content"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    client = PersistentClient(path="./db")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    collection = client.get_or_create_collection(name="repo")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32).tolist()
    collection.add(documents=texts, metadatas=metadatas, ids=ids, embeddings=embeddings)

    return {"status": f"Added {len(chunks)} chunks from {repo.path}"}


@app.post("/ask")
def ask(q: Query):
    from sentence_transformers import SentenceTransformer
    from chromadb import HttpClient
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = HttpClient(host=f"http://localhost:{config['chroma_port']}")
    collection = client.get_collection(name="repo")
    query_embedding = model.encode(q.question).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )
    if results['documents'] == []:
        return {"response": "No results found."}
    docs = results['documents'][0]
    context = "\n---\n".join(docs)
    prompt = f"""
    You are an AI assistant that answers questions about the codebase and provides solutions.
    ### Context:
    {context}
    ### Question:
    {q.question}
    ### Answer:
    """
    response = requests.post(
        ollama_url,
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }
    )
    answer = response.json()
    return {"response": answer['response']}

@app.get("/", response_class=HTMLResponse)
def root():
    return open("static/index.html").read()

def launch_browser():
    time.sleep(1)  # Give the server a moment to start
    webbrowser.open(f"http://localhost:{config['web_port']}")

if __name__ == "__main__":
    launch_chroma()
    threading.Thread(target=launch_browser).start()
    uvicorn.run("main:app", host="0.0.0.0", port=config['web_port'], log_level="info", reload=False)
