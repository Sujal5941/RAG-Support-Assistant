# ingest.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from config import CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_PATH, DATA_PATH

# Name of the ChromaDB collection for this project
COLLECTION_NAME = "shopease_kb"

def ingest_pdf():
    print("📄 Loading ShopEase knowledge base PDF...")
    loader = PyPDFLoader(DATA_PATH)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages from PDF")

    # Split into chunks using values from config.py
    # RecursiveCharacterTextSplitter splits at paragraphs first,
    # then sentences, then words — smarter than fixed-size splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️  Split into {len(chunks)} chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # Tag each chunk with its page number for traceability
    for i, chunk in enumerate(chunks):
        chunk.metadata["source_page"] = chunk.metadata.get("page", i)
        chunk.metadata["chunk_index"] = i

    # Embed using free HuggingFace model — no API key needed
    # all-MiniLM-L6-v2 is fast and works well for support/FAQ text
    print("🔢 Generating embeddings...")
    embedding_model = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Store vectors in ChromaDB under named collection
    print("💾 Storing embeddings in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME
    )
    print(f"✅ ChromaDB ready — {len(chunks)} chunks stored "
          f"in collection '{COLLECTION_NAME}'")
    return vectorstore


def load_vectorstore():
    """Load existing ChromaDB from disk — called by retriever at query time"""
    embedding_model = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME
    )
    return vectorstore


if __name__ == "__main__":
    ingest_pdf()