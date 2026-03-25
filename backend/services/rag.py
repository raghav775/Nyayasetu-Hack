import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from utils.document_loader import load_all_documents

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DRAFTS_DATA_PATH = Path(os.getenv("DRAFTS_DATA_PATH", BASE_DIR / "data" / "drafts"))
COLLECTION_NAME = "nyayasetu_legal_docs"

print("[RAG] Loading embedding model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("[RAG] Embedding model ready.")

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH,
    settings=Settings(anonymized_telemetry=False)
)


def get_collection():
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 80) -> list:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def ingest_documents():
    collection = get_collection()

    if collection.count() > 0:
        print(f"[RAG] {collection.count()} chunks already stored. Skipping ingestion.")
        print("[RAG] To re-ingest, delete the chroma_db folder and run again.")
        return

    print("DEBUG PATH:", DRAFTS_DATA_PATH)
    documents = load_all_documents(str(DRAFTS_DATA_PATH))
    if not documents:
        print("[RAG] No documents found. Add RTF/DOCX files to data/drafts/")
        return

    all_ids, all_texts, all_metadatas, all_embeddings = [], [], [], []

    doc_id = 0
    for doc in documents:
        for i, chunk in enumerate(chunk_text(doc["text"])):
            all_ids.append(f"doc_{doc_id}_chunk_{i}")
            all_texts.append(chunk)
            all_metadatas.append({
                "filename": doc["metadata"]["filename"],
                "category": doc["metadata"]["category"],
                "chunk_index": i,
            })
            doc_id += 1

    print(f"[RAG] Generating embeddings for {len(all_texts)} chunks...")
    batch_size = 64
    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i:i + batch_size]
        embeddings = embedding_model.encode(batch, show_progress_bar=False).tolist()
        all_embeddings.extend(embeddings)
        done = min(i + batch_size, len(all_texts))
        print(f"[RAG] Embedded {done}/{len(all_texts)}", end="\r")

    print(f"\n[RAG] Storing {len(all_texts)} chunks in ChromaDB...")
    store_batch = 500
    for i in range(0, len(all_texts), store_batch):
        collection.add(
            ids=all_ids[i:i + store_batch],
            embeddings=all_embeddings[i:i + store_batch],
            documents=all_texts[i:i + store_batch],
            metadatas=all_metadatas[i:i + store_batch],
        )

    print(f"[RAG] Done. {collection.count()} chunks stored.")


def search_drafts(query: str, n_results: int = 5, category_filter: str = None) -> list:
    collection = get_collection()

    if collection.count() == 0:
        print("[RAG] Empty collection. Run ingest.py first.")
        return []

    query_embedding = embedding_model.encode([query]).tolist()
    where = {"category": category_filter} if category_filter else None

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
        where=where,
    )

    output = []
    for i in range(len(results["documents"][0])):
        score = 1 - results["distances"][0][i]
        if score < 0.15:
            continue
        output.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": score,
        })

    return output
