# pipeline/results_logger.py
# Saves every experiment run to a single CSV — always appends, never overwrites.
# Each row = one query with our scores, expected scores, gaps, and strategy used.

import csv
import os
from pipeline.contracts import EvaluationResult
from typing import List

LOG_FILE = "results/experiment_log.csv"

HEADERS = [
    # When
    "timestamp",
    "date",
    # What config/strategy was used
    "config_name",
    "domain",
    "strategy_chunker",
    "strategy_retriever",
    "strategy_reranker",
    "strategy_repacker",
    "strategy_summarizer",
    # The query and answers
    "query",
    "our_answer",
    "expected_answer",
    # Our actual scores
    "our_context_relevance",
    "our_context_utilization",
    "our_completeness",
    "our_adherence",
    # RAGBench expected scores
    "expected_context_relevance",
    "expected_context_utilization",
    "expected_completeness",
    "expected_adherence",
    # Gaps (our - expected) — negative = below target
    "gap_relevance",
    "gap_utilization",
    "gap_completeness",
    "gap_adherence",
]


def run(results: List[EvaluationResult], config: dict):
    """
    Input:  List of EvaluationResult + config
    Output: Appends all results to CSV + prints summary
    """
    os.makedirs("results", exist_ok=True)

    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)

        # Write headers only if file is new
        if not file_exists:
            writer.writeheader()

        for result in results:
            writer.writerow({
                # When
                "timestamp": result.timestamp,
                "date": result.timestamp.split(" ")[0],

                # What config/strategy was used
                "config_name": result.config_name,
                "domain": result.domain,
                "strategy_chunker": config["chunker"]["strategy"],
                "strategy_retriever": config["retriever"]["strategy"],
                "strategy_reranker": config.get("reranker", {}).get("strategy", "none"),
                "strategy_repacker": config.get("repacker", {}).get("strategy", "none"),
                "strategy_summarizer": config.get("summarizer", {}).get("strategy", "none"),

                # Query and answers
                "query": result.query,
                "our_answer": result.answer[:300],
                "expected_answer": result.expected_answer[:300],

                # Our actual scores
                "our_context_relevance": result.context_relevance,
                "our_context_utilization": result.context_utilization,
                "our_completeness": result.completeness,
                "our_adherence": result.adherence,

                # RAGBench expected scores
                "expected_context_relevance": result.expected_context_relevance,
                "expected_context_utilization": result.expected_context_utilization,
                "expected_completeness": result.expected_completeness,
                "expected_adherence": result.expected_adherence,

                # Gaps
                "gap_relevance": result.gap_relevance,
                "gap_utilization": result.gap_utilization,
                "gap_completeness": result.gap_completeness,
                "gap_adherence": result.gap_adherence,
            })

    print(f"\n  ✅ Results saved to {LOG_FILE}")
    print(f"  Total rows in log: {_count_rows()}")
    _print_summary(results, config)


def _print_summary(results: List[EvaluationResult], config: dict):
    """Prints a quick comparison table to terminal after each run."""
    if not results:
        return

    # Average our scores
    avg_our_rel  = sum(r.context_relevance for r in results) / len(results)
    avg_our_util = sum(r.context_utilization for r in results) / len(results)
    avg_our_comp = sum(r.completeness for r in results) / len(results)
    avg_our_adh  = sum(r.adherence for r in results) / len(results)

    # Average expected scores
    avg_exp_rel  = sum(r.expected_context_relevance for r in results) / len(results)
    avg_exp_util = sum(r.expected_context_utilization for r in results) / len(results)
    avg_exp_comp = sum(r.expected_completeness for r in results) / len(results)
    avg_exp_adh  = sum(r.expected_adherence for r in results) / len(results)

    print("\n" + "═" * 60)
    print(f"  Run Summary")
    print(f"  Config : {results[0].config_name}")
    print(f"  Domain : {results[0].domain}")
    print(f"  Queries: {len(results)}")
    print("═" * 60)
    print(f"  {'Metric':<25} {'Ours':>8} {'Expected':>10} {'Gap':>8}")
    print("─" * 60)
    print(f"  {'Context Relevance':<25} {avg_our_rel:>8.4f} {avg_exp_rel:>10.4f} {avg_our_rel-avg_exp_rel:>8.4f}")
    print(f"  {'Context Utilization':<25} {avg_our_util:>8.4f} {avg_exp_util:>10.4f} {avg_our_util-avg_exp_util:>8.4f}")
    print(f"  {'Completeness':<25} {avg_our_comp:>8.4f} {avg_exp_comp:>10.4f} {avg_our_comp-avg_exp_comp:>8.4f}")
    print(f"  {'Adherence':<25} {avg_our_adh:>8.4f} {avg_exp_adh:>10.4f} {avg_our_adh-avg_exp_adh:>8.4f}")
    print("═" * 60)


def _count_rows() -> int:
    """Count total experiment rows logged so far."""
    if not os.path.exists(LOG_FILE):
        return 0
    with open(LOG_FILE, "r") as f:
        return sum(1 for _ in f) - 1  # subtract header row