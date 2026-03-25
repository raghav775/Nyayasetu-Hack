import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from utils.document_loader import load_all_documents

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
DRAFTS_DATA_PATH = os.getenv("DRAFTS_DATA_PATH", "../data/drafts")
COLLECTION_NAME = "nyayasetu_drafts"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH,
    settings=Settings(anonymized_telemetry=False)
)


def get_collection():
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def ingest_documents():
    collection = get_collection()

    if collection.count() > 0:
        print(f"Collection already has {collection.count()} chunks. Skipping ingestion.")
        return

    documents = load_all_documents(DRAFTS_DATA_PATH)
    if not documents:
        print("No documents found to ingest.")
        return

    all_texts = []
    all_embeddings = []
    all_metadatas = []
    all_ids = []

    doc_id = 0
    for doc in documents:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            all_texts.append(chunk)
            all_metadatas.append({
                "filename": doc["metadata"]["filename"],
                "category": doc["metadata"]["category"],
                "chunk_index": i,
            })
            all_ids.append(f"doc_{doc_id}_chunk_{i}")
            doc_id += 1

    print(f"Generating embeddings for {len(all_texts)} chunks...")
    batch_size = 64
    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i:i + batch_size]
        batch_embeddings = embedding_model.encode(batch_texts).tolist()
        all_embeddings.extend(batch_embeddings)

    print("Storing in ChromaDB...")
    batch_size = 500
    for i in range(0, len(all_texts), batch_size):
        collection.add(
            ids=all_ids[i:i + batch_size],
            embeddings=all_embeddings[i:i + batch_size],
            documents=all_texts[i:i + batch_size],
            metadatas=all_metadatas[i:i + batch_size],
        )

    print(f"Ingestion complete. {collection.count()} chunks stored.")


def search_drafts(query: str, n_results: int = 5) -> list:
    collection = get_collection()

    if collection.count() == 0:
        return []

    query_embedding = embedding_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    output = []
    for i in range(len(results["documents"][0])):
        output.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": 1 - results["distances"][0][i],
        })

    return output