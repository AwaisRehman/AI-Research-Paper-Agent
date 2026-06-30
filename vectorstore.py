# ============================================================
# vectorstore.py — Vector Database Module
# ============================================================
# Responsibilities:
#   - Initialize and manage the ChromaDB vector database (local, free)
#   - Store embedded chunks with metadata (source paper, chunk index)
#   - Perform semantic similarity search to retrieve relevant chunks
#   - Delete papers from the database when needed
#
# What is ChromaDB?
#   An open-source, local vector database. Stores text + their embeddings
#   and allows fast nearest-neighbor search. No server needed — runs in memory.
#
# Flow:
#   1. Paper uploaded → text extracted → chunked → embedded → stored here
#   2. User asks question → question embedded → nearest chunks retrieved
#   3. Retrieved chunks passed to LLM as context
#
# Author: Research Paper Agent
# ============================================================

import chromadb
from chromadb.config import Settings
from embedder import get_embeddings
import streamlit as st


# ── Client Initialization ─────────────────────────────────────────────────────
# Use Streamlit cache so only one ChromaDB client exists per session.
# EphemeralClient = in-memory only (resets when app restarts).
# For persistent storage across restarts, use PersistentClient instead.

@st.cache_resource(show_spinner=False)
def _get_chroma_client() -> chromadb.Client:
    """
    Initialize and return a singleton ChromaDB client.

    Using EphemeralClient (in-memory) for simplicity.
    Data is lost when the Streamlit app restarts — this is intentional
    for a research session tool. Each session starts fresh.

    Returns:
        chromadb.Client: Initialized ChromaDB client.
    """
    return chromadb.EphemeralClient()


def get_or_create_collection(name: str = "research_papers") -> chromadb.Collection:
    """
    Get an existing ChromaDB collection or create a new one.

    A collection is like a table in a database — it holds all the
    embedded text chunks from all uploaded papers.

    Args:
        name (str): Collection name. Default "research_papers".

    Returns:
        chromadb.Collection: The collection object for storing/querying.
    """
    client = _get_chroma_client()

    # get_or_create is idempotent — safe to call multiple times
    collection = client.get_or_create_collection(
        name=name,
        # cosine distance is best for normalized text embeddings
        metadata={"hnsw:space": "cosine"}
    )

    return collection


def add_paper(
    collection: chromadb.Collection,
    chunks: list[str],
    paper_name: str
) -> int:
    """
    Embed and store all chunks from a research paper into ChromaDB.

    Each chunk gets:
        - A unique ID: "paper_name_chunkindex"
        - An embedding vector (384 floats)
        - Metadata: source filename and chunk position
        - The raw text (stored alongside the embedding)

    Args:
        collection (chromadb.Collection): Target ChromaDB collection.
        chunks (list[str]): List of text chunks from the paper.
        paper_name (str): Original filename, used as source identifier.

    Returns:
        int: Number of chunks successfully stored.
    """
    if not chunks:
        return 0

    # Generate embedding vectors for all chunks in one batch call
    # This is efficient — sentence-transformers batches internally
    embeddings = get_embeddings(chunks)

    # Create unique IDs — format: "filename.pdf_0", "filename.pdf_1", etc.
    # Sanitize paper_name to remove characters invalid in ChromaDB IDs
    safe_name = paper_name.replace(" ", "_").replace("/", "_")
    ids = [f"{safe_name}_{i}" for i in range(len(chunks))]

    # Metadata allows filtering results by source paper later
    metadatas = [
        {
            "source": paper_name,   # original filename
            "chunk_index": i,       # position within the paper
        }
        for i in range(len(chunks))
    ]

    # Store everything in ChromaDB
    collection.add(
        embeddings=embeddings,  # vector representations
        documents=chunks,        # raw text (for display in UI)
        metadatas=metadatas,     # source tracking
        ids=ids                  # unique identifiers
    )

    return len(chunks)


def search(
    collection: chromadb.Collection,
    query: str,
    n_results: int = 5,
    filter_source: str = None
) -> dict:
    """
    Semantic similarity search over all stored paper chunks.

    Embeds the query and finds the n most similar chunks using
    cosine similarity in the vector space.

    Args:
        collection (chromadb.Collection): ChromaDB collection to search.
        query (str): Natural language question or search query.
        n_results (int): Number of top results to return. Default 5.
        filter_source (str): Optional — filter results to a specific paper.
                             Pass the filename to search only that paper.

    Returns:
        dict: ChromaDB query result with keys:
              - 'documents': list of matching text chunks
              - 'metadatas': list of source/chunk metadata
              - 'distances': similarity scores (lower = more similar)
    """
    # Embed the query using the same model as the stored chunks
    # This ensures the vectors are in the same semantic space
    query_embedding = get_embeddings([query])

    # Build optional where-filter for single-paper search
    where_filter = None
    if filter_source:
        where_filter = {"source": {"$eq": filter_source}}

    # Query ChromaDB — returns top n_results by cosine similarity
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(n_results, collection.count()),  # don't exceed stored count
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    return results


def delete_paper(collection: chromadb.Collection, paper_name: str) -> int:
    """
    Remove all chunks belonging to a specific paper from the collection.

    Useful when user wants to remove a paper from the session
    without restarting the whole app.

    Args:
        collection (chromadb.Collection): ChromaDB collection.
        paper_name (str): Filename of the paper to delete.

    Returns:
        int: Number of chunks deleted.
    """
    # Find all IDs belonging to this paper using metadata filter
    results = collection.get(
        where={"source": {"$eq": paper_name}},
        include=["documents"]
    )

    ids_to_delete = results.get("ids", [])

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)

    return len(ids_to_delete)


def get_collection_stats(collection: chromadb.Collection) -> dict:
    """
    Get statistics about the current collection.

    Args:
        collection (chromadb.Collection): ChromaDB collection.

    Returns:
        dict: Stats with total_chunks count.
    """
    return {
        "total_chunks": collection.count()
    }
