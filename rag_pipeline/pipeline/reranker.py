# pipeline/reranker.py
# Takes retrieved chunks and re-scores them more accurately.
# Retriever does fast approximate search (top 20).
# Reranker does slower but more accurate scoring (narrows to top 5).
# Strategy controlled by config.

from pipeline.contracts import RetrievedChunk
from typing import List


def run(query: str, retrieved_chunks: List[RetrievedChunk], config: dict) -> List[RetrievedChunk]:
    """
    Input:  query + retrieved chunks (top 20 from retriever)
    Output: reranked chunks (top 5, more accurately scored)
    """

    # If reranker section missing from config entirely — skip it. Safe fallback.
    if not config.get("reranker"):
        return retrieved_chunks

    strategy = config["reranker"]["strategy"]
    top_n    = config["reranker"].get("top_n", 5)

    print(f"  Reranking {len(retrieved_chunks)} chunks with strategy: {strategy}")

    if strategy == "cross_encoder":
        return _cross_encoder_rerank(query, retrieved_chunks, top_n)
    elif strategy == "monot5":
        return _monot5_rerank(query, retrieved_chunks, top_n)
    elif strategy == "none":
        return retrieved_chunks[:top_n]
    else:
        raise ValueError(f"Unknown reranker strategy: {strategy}")


def _cross_encoder_rerank(query: str, chunks: List[RetrievedChunk], top_n: int) -> List[RetrievedChunk]:
    """
    Cross encoder scores (query, chunk) pairs jointly.
    More accurate than bi-encoder because it sees both together.
    Model: ms-marco-MiniLM-L-6-v2 — fast and accurate.
    """
    from sentence_transformers import CrossEncoder

    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Build pairs — one per chunk
    pairs = [[query, chunk.chunk.text] for chunk in chunks]

    # Score all pairs at once
    scores = model.predict(pairs)

    # Attach new scores and sort
    for chunk, score in zip(chunks, scores):
        chunk.score = float(score)

    reranked = sorted(chunks, key=lambda x: x.score, reverse=True)
    return reranked[:top_n]


def _monot5_rerank(query: str, chunks: List[RetrievedChunk], top_n: int) -> List[RetrievedChunk]:
    """
    monoT5 — best reranker per EMNLP 2024 best practices paper.
    Slower than cross encoder but more accurate.
    Model: castorini/monot5-base-msmarco
    """
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    import torch

    model_name = "castorini/monot5-base-msmarco"
    tokenizer  = T5Tokenizer.from_pretrained(model_name)
    model      = T5ForConditionalGeneration.from_pretrained(model_name)
    model.eval()

    # Token ids for "true" and "false"
    true_id  = tokenizer("true").input_ids[0]
    false_id = tokenizer("false").input_ids[0]

    scored_chunks = []

    for retrieved in chunks:
        # Exact format monoT5 was trained on. Must match — changing this format breaks the model.
        input_text = f"Query: {query} Document: {retrieved.chunk.text} Relevant:"
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=1
            )

        # Get probability of "true" vs "false"
        logits = outputs.scores[0][0]
        score  = torch.softmax(
            torch.tensor([logits[true_id], logits[false_id]]), dim=0
        )[0].item()

        retrieved.score = score
        scored_chunks.append(retrieved)

    reranked = sorted(scored_chunks, key=lambda x: x.score, reverse=True)
    return reranked[:top_n]