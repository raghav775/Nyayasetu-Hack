from dotenv import load_dotenv
load_dotenv()

from services.rag import ingest_documents

if __name__ == "__main__":
    print("Starting NyayaSetu document ingestion...")
    ingest_documents()
    print("Done.")