# pipeline/repacker.py
#
# Reorders chunks before passing them to the LLM.
#
# Motivation:
# The "Lost in the Middle" paper found that LLMs often pay more
# attention to information near the beginning and end of long contexts
# than information buried in the middle.
#
# Strategies:
# - forward : keep reranker order (most relevant first)
# - reverse : place most relevant chunks closest to the question
# - sides   : place top-ranked chunks at both context boundaries
# - none    : no reordering

from typing import List
from pipeline.contracts import RetrievedChunk

def run(chunks: List[RetrievedChunk], config: dict) -> List[RetrievedChunk]:
    """
    Input:
        Chunks already sorted by relevance (highest relevance first).

    Output:
        Same chunks reordered to optimize placement within the LLM context.
    """
    if not config.get("repacker"):
        return chunks

    strategy = config["repacker"]["strategy"]
    print(f"  Repacking with strategy: {strategy}")

    if strategy == "forward":
        return chunks

    elif strategy == "reverse":
        # Places the highest-ranked chunk nearest the question.
        # Useful when prompts are formatted as:
        #
        # Context:
        # ...
        # ...
        # Question:
        # ...
        #
        return list(reversed(chunks))

    elif strategy == "sides":
        return _sides_repack(chunks)

    elif strategy == "none":
        return chunks

    else:
        raise ValueError(f"Unknown repacker strategy: {strategy}")


def _sides_repack(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """
    Place the highest-ranked chunk at the start of the context and
    the second-highest-ranked chunk at the end.

    Example:
        Input:
            [1, 2, 3, 4, 5]
            (1 = most relevant)

        Output:
            [1, 3, 4, 5, 2]

    This puts the two strongest chunks at opposite context boundaries,
    which can help mitigate "lost in the middle" effects.
    """
    if len(chunks) <= 2:
        return chunks

    best = chunks[0]
    second_best = chunks[1]
    middle = chunks[2:]

    return [best] + middle + [second_best]