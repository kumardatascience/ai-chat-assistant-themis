"""
Day 4 — Document indexing.
Reads files from data/documents, chunks them, embeds them, stores in ChromaDB.
Run this as a script, not imported by main.py yet.
"""

from pathlib import Path
import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "themis_docs"

# Free, local embedding model. Runs on your Mac, no API key needed.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def build_index():
    """Read docs → chunk them → embed them → store in ChromaDB."""

    # 1. Tell LlamaIndex to use our local embedding model (no OpenAI)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    # 2. Load all files in data/documents (PDFs, txt, md, docx, etc.)
    print(f"📂 Reading documents from {DOCUMENTS_DIR}")
    reader = SimpleDirectoryReader(input_dir=str(DOCUMENTS_DIR))
    documents = reader.load_data()
    print(f"   Loaded {len(documents)} document(s)")

    # 3. Connect to ChromaDB (creates the folder if missing)
    print(f"💾 Connecting to ChromaDB at {CHROMA_DIR}")
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    # 4. Wrap ChromaDB so LlamaIndex can talk to it
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 5. The magic step: chunk → embed → store. One line.
    print("🧠 Chunking, embedding, and storing...")
    VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    # 6. Report
    count = chroma_collection.count()
    print(f"✅ Done! {count} chunks stored in collection '{COLLECTION_NAME}'.")


def get_retriever(top_k: int = 3):
    """
    Open the existing ChromaDB and return a retriever object.
    top_k = how many chunks to fetch per query.
    """
    # Use the same embedding model as indexing (must match!)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

    # Open the existing ChromaDB (don't re-index)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Wrap the existing store as a searchable index
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    return index.as_retriever(similarity_top_k=top_k)


def retrieve_context(question: str, top_k: int = 3) -> str:
    """
    Given a user's question, return the most relevant chunks joined as a single string.
    """
    retriever = get_retriever(top_k=top_k)
    nodes = retriever.retrieve(question)

    if not nodes:
        return ""

    # Combine chunks with a separator
    return "\n\n---\n\n".join(node.get_content() for node in nodes)    


if __name__ == "__main__":
    build_index()