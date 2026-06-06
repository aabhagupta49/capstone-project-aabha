# pipeline/summarizer.py
# Compresses retrieved chunks before passing to LLM.
# Reduces noise — only keeps sentences most relevant to the query.
# Helps when chunks are long and contain irrelevant sentences.

from pipeline.contracts import RetrievedChunk
from typing import List

def run(query: str, chunks: List[RetrievedChunk], config: dict) -> str:
    """
    Input:  query + repacked chunks
    Output: compressed context string ready for LLM
    """
    if not config.get("summarizer"):
        return _concatenate(chunks)

    strategy = config["summarizer"]["strategy"]
    print(f"  Summarizing with strategy: {strategy}")

    if strategy == "none":
        return _concatenate(chunks)
    elif strategy == "extractive":
        return _extractive_summarize(query, chunks, config)
    else:
        raise ValueError(f"Unknown summarizer strategy: {strategy}")


def _concatenate(chunks: List[RetrievedChunk]) -> str:
    """
    Simply joins all chunks together.
    No compression — passes everything to LLM.
    """
    parts = []
    for i, retrieved in enumerate(chunks, 1):
        parts.append(f"[{i}] {retrieved.chunk.text}")
    return "\n\n".join(parts)


def _extractive_summarize(query: str, chunks: List[RetrievedChunk], config: dict) -> str:
    """
    Keeps only the most query-relevant sentences from each chunk.
    Uses sentence embeddings to find which sentences match query best.

    Example:
    Chunk: "SARS infected 8098 people. The weather was nice. Mortality was 9.6%."
    Query: "How many SARS cases?"
    Kept:  "SARS infected 8098 people."   ← most relevant sentence
    """
    from sentence_transformers import SentenceTransformer, util
    import torch

    model = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight model
    max_sentences = config["summarizer"].get("max_sentences", 3)

    query_embedding = model.encode(query, convert_to_tensor=True)

    selected_sentences = []

    for retrieved in chunks:
        # Split chunk into sentences
        sentences = [
            s.strip()
            for s in retrieved.chunk.text.split(".")
            if s.strip()
        ]

        if not sentences:
            continue

        # Embed all sentences
        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

        # Score each sentence against query
        scores = util.cos_sim(query_embedding, sentence_embeddings)[0]

        # Pick top sentences
        top_indices = torch.argsort(scores, descending=True)[:max_sentences]
        top_indices_sorted = sorted(top_indices.tolist())  # restore original order

        for idx in top_indices_sorted:
            selected_sentences.append(sentences[idx])

    return ". ".join(selected_sentences)