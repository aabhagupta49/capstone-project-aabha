# pipeline/embedder.py
# Converts text chunks into vectors (numbers) that capture meaning.
# These vectors are what gets stored in ChromaDB and searched later.

from sentence_transformers import SentenceTransformer
from pipeline.contracts import Chunk
from typing import List, Tuple
import numpy as np

# Global model variable — loaded once, reused every time
_model = None

def _get_model(model_name: str) -> SentenceTransformer:
    """
    Loads the embedding model only once.
    Second call onwards just returns the already-loaded model.
    """
    global _model
    if _model is None:
        print(f"  Loading embedding model: {model_name} (first time only)")
        _model = SentenceTransformer(model_name)
    return _model


def run(chunks: List[Chunk], config: dict) -> Tuple[List[Chunk], List[np.ndarray]]:
    """
    Input:  List of Chunk objects
    Output: Tuple of (same chunks, list of embedding vectors)
            One embedding vector per chunk — same order.
    """
    model_name = config["embedder"]["model"]
    model = _get_model(model_name)

    # Extract just the text from each chunk for batch embedding
    texts = [chunk.text for chunk in chunks]

    print(f"  Embedding {len(texts)} chunks with {model_name}...")

    # embed all texts in one batch — much faster than one by one
    embeddings = model.encode(
        texts,
        batch_size=32,        # process 32 at a time
        show_progress_bar=True,
        normalize_embeddings=True  # normalise for cosine similarity
    )

    return chunks, embeddings