# pipeline/chunker.py
from chonkie import TokenChunker, SentenceChunker, SemanticChunker
from pipeline.contracts import Chunk

def run(documents, config):
    """
    Input:  List of dicts with keys: text, doc_id, domain
    Output: List of Chunk objects
    """
    strategy  = config["chunker"]["strategy"]
    chunk_size = config["chunker"]["chunk_size"]
    overlap    = config["chunker"]["chunk_overlap"]

    # Pick chunker based on config — swap by changing config only
    if strategy == "token":
        chunker = TokenChunker(chunk_size=chunk_size, chunk_overlap=overlap)
    elif strategy == "sentence":
        chunker = SentenceChunker(chunk_size=chunk_size, chunk_overlap=overlap)
    elif strategy == "semantic":
        chunker = SemanticChunker(chunk_size=chunk_size)
    # elif strategy == "sliding":
    #     chunker = SlidingWindowChunker(chunk_size=chunk_size, chunk_overlap=overlap)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    all_chunks = []

    for doc in documents:
        raw_chunks = chunker(doc["text"])  # chonkie does the splitting

        for i, raw in enumerate(raw_chunks):
            all_chunks.append(Chunk(
                text=raw.text,
                chunk_id=f"{doc['doc_id']}_chunk_{i}",
                doc_id=doc["doc_id"],
                domain=doc["domain"],
                metadata={"strategy": strategy}
            ))

    return all_chunks