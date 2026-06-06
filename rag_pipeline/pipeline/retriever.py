# pipeline/retriever.py
# Given a query, finds the most relevant chunks.
# Supports three strategies: dense, bm25, hybrid
# Strategy is controlled by config — swap without changing anything else.

import pickle
import os
import numpy as np
from pipeline.contracts import Chunk, RetrievedChunk
from typing import List


def run(query: str, collection, config: dict) -> List[RetrievedChunk]:
    """
    Input:  query string + ChromaDB collection + config
    Output: List of RetrievedChunk (chunk + relevance score)
    """
    strategy = config["retriever"]["strategy"]
    top_k = config["retriever"]["top_k"]

    print(f"  Retrieving with strategy: {strategy}")

    if strategy == "dense":
        return _dense_retrieval(query, collection, top_k, config)
    elif strategy == "bm25":
        return _bm25_retrieval(query, top_k, config)
    elif strategy == "hybrid":
        return _hybrid_retrieval(query, collection, top_k, config)
    else:
        raise ValueError(f"Unknown retrieval strategy: {strategy}")


# ── Strategy 1: Dense Retrieval ──────────────────────────────
def _dense_retrieval(query, collection, top_k, config) -> List[RetrievedChunk]:
    """
    Embeds the query and finds most similar chunks in ChromaDB.
    Pure semantic search — finds meaning, not exact words.
    """
    from sentence_transformers import SentenceTransformer
    model_name = config["embedder"]["model"]
    model = SentenceTransformer(model_name)

    # Embed the query the same way we embedded chunks
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return _chroma_results_to_retrieved_chunks(results)


# ── Strategy 2: BM25 Retrieval ───────────────────────────────
def _bm25_retrieval(query, top_k, config) -> List[RetrievedChunk]:
    """
    Keyword-based search using BM25.
    Fast, great for exact term matching.
    """
    bm25_data = _load_bm25(config)
    bm25 = bm25_data["bm25"]
    chunks = bm25_data["chunks"]

    # Tokenize query same way we tokenized chunks
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Get top_k indices sorted by score
    top_indices = np.argsort(scores)[::-1][:top_k]

    retrieved = []
    for idx in top_indices:
        if scores[idx] > 0:  # only include if there's some match
            retrieved.append(RetrievedChunk(
                chunk=chunks[idx],
                score=float(scores[idx])
            ))

    return retrieved


# ── Strategy 3: Hybrid Retrieval ─────────────────────────────
def _hybrid_retrieval(query, collection, top_k, config) -> List[RetrievedChunk]:
    """
    Combines dense + BM25 using Reciprocal Rank Fusion (RRF).
    Best of both worlds — semantic + keyword.
    """
    bm25_weight = config["retriever"].get("bm25_weight", 0.5)
    dense_weight = config["retriever"].get("dense_weight", 0.5)

    # Get results from both
    dense_results = _dense_retrieval(query, collection, top_k, config)
    bm25_results = _bm25_retrieval(query, top_k, config)

    # Combine using RRF
    return _reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        dense_weight,
        bm25_weight,
        top_k
    )


def _reciprocal_rank_fusion(dense, bm25, dense_weight, bm25_weight, top_k, k=60):
    """
    RRF Formula: score = weight / (k + rank)
    
    Instead of combining raw scores (which are on different scales),
    we combine RANKS. Rank 1 = most relevant.
    
    k=60 is standard — prevents top ranks from dominating too much.
    """
    scores = {}  # chunk_id -> combined score

    # Score dense results
    for rank, result in enumerate(dense):
        chunk_id = result.chunk.chunk_id
        scores[chunk_id] = scores.get(chunk_id, 0)
        scores[chunk_id] += dense_weight / (k + rank + 1)

    # Score BM25 results
    for rank, result in enumerate(bm25):
        chunk_id = result.chunk.chunk_id
        scores[chunk_id] = scores.get(chunk_id, 0)
        scores[chunk_id] += bm25_weight / (k + rank + 1)

    # Merge all results into one dict for lookup
    all_results = {r.chunk.chunk_id: r for r in dense + bm25}

    # Sort by combined RRF score
    sorted_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]

    return [
        RetrievedChunk(
            chunk=all_results[chunk_id].chunk,
            score=scores[chunk_id]
        )
        for chunk_id in sorted_ids
        if chunk_id in all_results
    ]


# ── Helpers ───────────────────────────────────────────────────
def _load_bm25(config):
    """Load BM25 index from disk."""
    persist_path = config["indexer"]["persist_path"]
    bm25_path = os.path.join(persist_path, "bm25_index.pkl")

    if not os.path.exists(bm25_path):
        raise FileNotFoundError(
            f"BM25 index not found at {bm25_path}. Run indexer first."
        )

    with open(bm25_path, "rb") as f:
        return pickle.load(f)


def _chroma_results_to_retrieved_chunks(results) -> List[RetrievedChunk]:
    """Convert ChromaDB result format to our RetrievedChunk contracts."""
    retrieved = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    for doc, meta, dist, chunk_id in zip(documents, metadatas, distances, ids):
        chunk = Chunk(
            text=doc,
            chunk_id=chunk_id,
            doc_id=meta.get("doc_id", ""),
            domain=meta.get("domain", ""),
            metadata=meta
        )
        # ChromaDB returns distance — convert to similarity score
        # cosine distance = 1 - cosine similarity
        score = 1 - dist

        retrieved.append(RetrievedChunk(chunk=chunk, score=score))

    return retrieved