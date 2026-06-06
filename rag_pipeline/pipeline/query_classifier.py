# pipeline/query_classifier.py
# Decides if a query needs retrieval or can be answered directly.
# For RAGBench — always needs retrieval.
# But good to have for real-world deployment.

from pipeline.contracts import Chunk
from typing import List


def run(query: str, config: dict) -> bool:
    """
    Input:  query string
    Output: True = needs retrieval, False = answer directly
    """
    if not config.get("query_classifier"):
        return True  # default: always retrieve

    strategy = config["query_classifier"]["strategy"]

    if strategy == "always":
        return True
    elif strategy == "keyword":
        return _keyword_classify(query)
    else:
        return True


def _keyword_classify(query: str) -> bool:
    """
    Simple rule-based classification.
    If query contains retrieval-indicator words → retrieve.
    Otherwise → answer directly (e.g. "hello", "thanks")
    """
    retrieval_keywords = {
        "what", "who", "when", "where", "why", "how",
        "explain", "describe", "tell", "define", "list",
        "compare", "difference", "example", "show"
    }

    query_words = set(query.lower().split())
    needs_retrieval = bool(query_words.intersection(retrieval_keywords))

    print(f"  Query classification: {'retrieve' if needs_retrieval else 'direct answer'}")
    return needs_retrieval