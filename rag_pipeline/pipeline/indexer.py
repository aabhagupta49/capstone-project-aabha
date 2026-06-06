# pipeline/indexer.py
# Stores chunks + their embeddings in ChromaDB (vector database)
# Also builds a BM25 index for keyword search.
# Think of this as building the searchable library before any queries come in.

import chromadb
from rank_bm25 import BM25Okapi
from pipeline.contracts import Chunk
from typing import List, Tuple
import numpy as np
import pickle
import os

# Global clients — loaded once
_chroma_client = None
_collection = None


def _get_collection(config: dict):
    """
    Creates or loads ChromaDB collection.
    If it already exists, loads it (no re-indexing needed).
    """
    global _chroma_client, _collection

    if _collection is None:
        persist_path = config["indexer"]["persist_path"]
        collection_name = config["indexer"]["collection_name"]

        os.makedirs(persist_path, exist_ok=True)

        _chroma_client = chromadb.PersistentClient(path=persist_path)
        _collection = _chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # use cosine similarity
        )

    return _collection


def run(chunks: List[Chunk], embeddings: List[np.ndarray], config: dict):
    """
    Input:  chunks + their embeddings from embedder
    Output: None — just stores everything to disk
            Also saves BM25 index to disk for later use
    """
    collection = _get_collection(config)

    print(f"  Storing {len(chunks)} chunks in ChromaDB...")

    # ChromaDB needs: ids, embeddings, documents, metadatas
    ids = [chunk.chunk_id for chunk in chunks]
    documents = [chunk.text for chunk in chunks]
    metadatas = [{
        "doc_id": chunk.doc_id,
        "domain": chunk.domain,
        **chunk.metadata
    } for chunk in chunks]

    # Convert numpy arrays to lists (ChromaDB requirement)
    embedding_lists = [e.tolist() for e in embeddings]

    # Store in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_end = min(i + batch_size, len(chunks))
        collection.add(
            ids=ids[i:batch_end],
            embeddings=embedding_lists[i:batch_end],
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end]
        )

    print(f"  ChromaDB: {collection.count()} chunks stored")

    # Build BM25 index if enabled
    if config["indexer"].get("use_bm25", True):
        _build_bm25_index(chunks, config)

    return collection


def _build_bm25_index(chunks: List[Chunk], config: dict):
    """
    BM25 works on tokenized text (list of words).
    We save it to disk so retriever can load it.
    """
    print("  Building BM25 index...")

    # Tokenize — split each chunk into words
    tokenized = [chunk.text.lower().split() for chunk in chunks]

    bm25 = BM25Okapi(tokenized)

    # Save BM25 + chunks to disk
    persist_path = config["indexer"]["persist_path"]
    bm25_path = os.path.join(persist_path, "bm25_index.pkl")

    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    print(f"  BM25 index saved to {bm25_path}")