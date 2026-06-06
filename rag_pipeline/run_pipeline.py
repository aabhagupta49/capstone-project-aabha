# run_pipeline.py
# Main entry point. Ties everything together.
# Run this file to execute the full pipeline end to end.

import yaml
from datasets import load_dataset
from pipeline import (
    chunker, embedder, indexer, retriever, reranker, repacker,
    summarizer, query_classifier, generator, evaluator, results_logger )


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_ragbench(config: dict):
    """
    Loads ALL subsets for a given domain and combines them.
    """
    from datasets import concatenate_datasets

    domain = config["dataset"]["domain"]
    subsets = config["domain_subsets"][domain]
    sample_size = config["dataset"].get("sample_size")

    print(f"\n Loading RAGBench — domain: {domain}")
    print(f"  Subsets: {subsets}")

    all_samples = []

    for subset in subsets:
        print(f"  Loading {subset}...")
        dataset = load_dataset(
            config["dataset"]["name"],
            subset,
            split=config["dataset"]["split"],
            trust_remote_code=True
        )

        samples = list(dataset)

        # Apply sample_size per subset if specified
        if sample_size:
            samples = samples[:sample_size]

        # Tag each sample with its subset
        for s in samples:
            s["subset"] = subset

        all_samples.extend(samples)
        print(f"  {subset}: {len(samples)} samples loaded")

    print(f"\n  Total samples for {domain}: {len(all_samples)}")
    return all_samples


def prepare_documents(samples: list, config: dict) -> list:
    """
    Converts RAGBench samples into document format for chunking.
    Each sample has multiple documents — we treat each as separate.
    """
    documents = []
    domain = config["dataset"]["domain"]

    for i, sample in enumerate(samples):
        for j, doc_text in enumerate(sample["documents"]):
            documents.append({
                "text": doc_text,
                "doc_id": f"sample_{i}_doc_{j}",
                "domain": domain
            })

    print(f"  Prepared {len(documents)} documents for indexing")
    return documents


def run_ingestion(documents: list, config: dict):
    """
    Offline phase — runs once to build the index.
    Chunk → Embed → Index
    """
    print("\n INGESTION PHASE")
    print("─" * 40)

    print("\n[1/3] Chunking documents...")
    chunks = chunker.run(documents, config)
    print(f"  Created {len(chunks)} chunks")

    print("\n[2/3] Embedding chunks...")
    chunks, embeddings = embedder.run(chunks, config)

    print("\n[3/3] Indexing...")
    collection = indexer.run(chunks, embeddings, config)

    return collection, chunks


def run_inference(samples: list, collection, config: dict):
    """
    Online phase — runs for each query.
    Classify → Retrieve → Rerank → Repack → Summarize → Generate → Evaluate
    """
    print("\n INFERENCE PHASE")
    print("─" * 40)

    all_results = []

    for i, sample in enumerate(samples):
        query    = sample["question"]
        expected = sample

        print(f"\n Query {i+1}/{len(samples)}: {query[:80]}...")

        # Step 1 — Query Classification
        needs_retrieval = query_classifier.run(query, config)
        if not needs_retrieval:
            print("  Skipping retrieval — direct answer")
            continue

        # Step 2 — Retrieve
        retrieved = retriever.run(query, collection, config)
        print(f"  Retrieved {len(retrieved)} chunks")

        # Step 3 — Rerank
        reranked = reranker.run(query, retrieved, config)
        print(f"  Reranked to {len(reranked)} chunks")

        # Step 4 — Repack
        repacked = repacker.run(reranked, config)

        # Step 5 — Summarize
        context = summarizer.run(query, repacked, config)

        # Step 6 — Generate
        gen_output = generator.run(query, repacked, config)
        print(f"  Answer: {gen_output.answer[:100]}...")

        # Step 7 — Evaluate
        result = evaluator.run(gen_output, reranked, expected, config)
        all_results.append(result)

    return all_results


def main():
    print("=" * 60)
    print("  RAG PIPELINE")
    print("=" * 60)

    # Load config
    config = load_config("configs/default.yaml")
    print(f"\n Config: {config['config_name']}")
    print(f" Domain: {config['dataset']['domain']}")

    # Load dataset
    samples = load_ragbench(config)

    # Prepare documents
    documents = prepare_documents(samples, config)

    # Ingestion phase
    collection, chunks = run_ingestion(documents, config)

    # Inference phase
    results = run_inference(samples, collection, config)

    # Save results
    print("\n SAVING RESULTS")
    print("─" * 40)
    results_logger.run(results, config)

    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()