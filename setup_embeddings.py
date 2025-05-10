import os
import sys
from pathlib import Path
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient 
from chromadb.config import Settings
from tree_sitter_language_pack import SupportedLanguage, get_language, get_parser
from tree_sitter import Node

LANGUAGES = {
  '.py': 'python',
  '.js': 'javascript',
  '.ts': 'typescript',
  '.html': 'html',
  '.json': 'json',
  '.md': 'markdown',
  '.rs': 'rust'
}

def get_function_name(node: Node) -> str:
    for child in node.children:
        if child.type == 'identifier':
            if child.text:
                return child.text.decode('utf-8')
    return '<anonymous>'

def extract_functions_rs(source_code: str, lang: SupportedLanguage) -> List[Tuple[str, str]]:
    parser = get_parser(lang)
    tree = parser.parse(bytes(source_code, 'utf8'))
    root = tree.root_node
    functions = []
    node_types = set()

    def walk(node: Node, context: str| None = None):
        node_types.add(node.type)
        if node.type in ("struct_item", "enum_item", "trait_item"):
            # Extract struct, enum, or trait name
            type_name = None
            for child in node.children:
                if child.type == 'type_identifier':
                    if child.text:
                        type_name = child.text.decode('utf-8')
                        break
            qualified_name = type_name
            code = source_code[node.start_byte:node.end_byte]
            functions.append((code, qualified_name))
        if node.type in ("impl_item", "trait_item"):
            type_name = None
            for child in node.children:
                if child.type == 'type_identifier':
                    if child.text:
                        type_name = child.text.decode('utf-8')
                        break
            context = type_name
        if node.type in ("function_item"):
            # Extract function name
            function_name = get_function_name(node)
            qualified_name = f"{context}::{function_name}" if context else function_name
            code = source_code[node.start_byte:node.end_byte]
            # Add qualified name to the function
            code = qualified_name + "\n" + code
            functions.append((code, qualified_name))

        for child in node.children:
            walk(child, context=context)
    walk(root)
    return functions

def load_repo_text(repo_path):
    #supported_extensions = ['.py', '.md', '.txt', '.rs', '.js', '.ts', '.html', '.json']
    supported_extensions = ['.rs']
    folders_to_ignore = ['.git', 'node_modules', 'target', '__pycache__', 'dist', 'build', 'venv', 'env', 'out', 'lib', 'libs', 'app', '.next']
    chunks = []
    for file in Path(repo_path).rglob('*'):
        use_file = True
        if file.is_file() == False:
            use_file = False
        if file.name.startswith('.'):
            use_file = False
        for folder in folders_to_ignore:
            if folder in file.parts:
                use_file = False
                break
        if use_file == False:
            continue
        if file.suffix.lower() in supported_extensions:
            print(f"Loading {file}")
            text = file.read_text(encoding='utf-8', errors='ignore')
            functions = extract_functions_rs(text, LANGUAGES[file.suffix])
            for function in functions:
                chunks.append({
                    "content": function[0],
                    "metadata": {"file": str(file), "function": function[1] }
                })
    print(f"Loaded {len(chunks)} chunks from {repo_path}")
    return chunks


if __name__ == "__main__":
    chunks = load_repo_text(sys.argv[1])
    texts = [chunk["content"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    # Initialize the ChromaDB Client
    client = PersistentClient(path="./db")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Create a collection
    collection = client.create_collection(name="repo")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32).tolist()
    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings
    )
