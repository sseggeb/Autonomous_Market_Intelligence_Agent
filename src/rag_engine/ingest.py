import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# --- CONFIGURATION ---
DATA_PATH = "data/raw/financial_report.pdf"
DB_PATH = "data/processed/chroma_db"


def ingest_documents():
    """
    Loads a PDF, chunks it, creates embeddings, and saves to a local Vector DB.
    """
    # 1. Check if file exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: File not found at {DATA_PATH}")
        return

    print("--- STARTING INGESTION ---")

    # 2. Load the PDF
    # LangChain handles the messy work of reading binary PDF files
    print(f"Loading {DATA_PATH}...")
    loader = PyPDFLoader(DATA_PATH)
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} pages.")

    # 3. Split Text (The Art of Chunking)
    # We can't feed a 100-page PDF into an LLM at once. We must split it.
    # chunk_size=1000: Roughly 2-3 paragraphs.
    # chunk_overlap=200: Ensures context isn't lost between cuts.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"Split into {len(chunks)} text chunks.")

    # 4. Initialize Embeddings
    # This turns text into numbers (vectors).
    # Requires OPENAI_API_KEY in your environment variables.
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 5. Create and Persist Vector Database
    # We use ChromaDB because it's open-source and file-based (easy for portfolios).
    print("Creating Vector Database...")

    # Optional: Clear old DB if you want a fresh start
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    print(f"--- INGESTION COMPLETE ---")
    print(f"Vector DB saved to: {DB_PATH}")


if __name__ == "__main__":
    # Ensure you have your key set: export OPENAI_API_KEY="sk-..."
    ingest_documents()