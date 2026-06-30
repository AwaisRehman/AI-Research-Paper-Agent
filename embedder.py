# ============================================================
# embedder.py — Text Embedding Module
# ============================================================
# Responsibilities:
#   - Load a pre-trained sentence embedding model (free, local)
#   - Convert text chunks into dense vector representations
#   - These vectors enable semantic similarity search in ChromaDB
#
# Model Used:
#   all-MiniLM-L6-v2 — lightweight, fast, 384-dimensional embeddings
#   Trained on 1B+ sentence pairs. Perfect for academic/research text.
#   Downloads automatically on first run (~90MB).
#
# Why sentence-transformers?
#   - Completely FREE — runs locally, no API cost
#   - Fast inference even on CPU
#   - High quality for semantic similarity tasks
#
# Author: Research Paper Agent
# ============================================================

from sentence_transformers import SentenceTransformer
from typing import Union
import streamlit as st


# ── Model Initialization ──────────────────────────────────────────────────────
# Load the model once at module level (singleton pattern).
# This avoids reloading the model on every function call, which would
# be very slow (~2-3 seconds per load). Streamlit caches this automatically.

@st.cache_resource(show_spinner="Loading embedding model...")
def _load_model() -> SentenceTransformer:
    """
    Load and cache the sentence transformer model.

    Uses Streamlit's @cache_resource decorator so the model is only
    loaded once per session, even if embedder.py functions are called
    many times.

    Returns:
        SentenceTransformer: Loaded model ready for encoding.
    """
    # all-MiniLM-L6-v2: small but powerful — 6 transformer layers, 
    # 384-dim output, ~22M parameters. Downloads on first use.
    return SentenceTransformer('all-MiniLM-L6-v2')


def get_embeddings(texts: Union[list[str], str]) -> list[list[float]]:
    """
    Convert a list of text strings into embedding vectors.

    Each text is encoded into a 384-dimensional float vector that
    captures its semantic meaning. Semantically similar texts will
    have vectors that are close together in this vector space.

    Args:
        texts (list[str] | str): One or more text strings to embed.
                                  Can be a single string or a list.

    Returns:
        list[list[float]]: List of embedding vectors.
                           Each vector has 384 float values.
                           Length matches the number of input texts.

    Example:
        embeddings = get_embeddings(["What is RAG?", "How does attention work?"])
        # Returns: [[0.023, -0.141, ...], [0.089, 0.034, ...]]
        # Each inner list has 384 floats
    """
    # Load the cached model
    model = _load_model()

    # Normalize input — allow single string for convenience
    if isinstance(texts, str):
        texts = [texts]

    # encode() returns a numpy array of shape (n_texts, 384)
    # .tolist() converts to Python list for ChromaDB compatibility
    embeddings = model.encode(
        texts,
        show_progress_bar=False,  # suppress tqdm output in Streamlit
        batch_size=32,            # process in batches for efficiency
        normalize_embeddings=True # L2-normalize for cosine similarity
    )

    return embeddings.tolist()
