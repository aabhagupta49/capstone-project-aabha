# Real World RAG System
**AIML PGCP Capstone Project — Group 16**

| Name | Trainee ID |
|---|---|
| Aabha Gupta | 2501201 |
| Abhi Sagar Khatri | 2501054 |
| Surya Teja Padala | 2502678 |

**Supervisor:** Dr. Manish Shrivastava  
**Mentors:** Gopichand, Lokesh

---

## What This Project Does

Builds a modular, end-to-end RAG (Retrieval-Augmented Generation) pipeline evaluated on the [RAGBench](https://huggingface.co/datasets/rungalileo/ragbench) benchmark dataset across five industry domains.

The objective is to match or closely approach the benchmark scores already present in the RAGBench dataset using only open-source models and tools.

---

## What is RAG?

Large Language Models hallucinate and go stale. RAG fixes this by:

1. Taking a user query
2. Searching a document database for relevant chunks
3. Passing those chunks as context to the LLM
4. LLM answers using only that context — no hallucination, no stale knowledge

```
User Query
    ↓
[ Query Classifier ] → needs retrieval?
    ↓
[ Retriever ] → finds top-20 relevant chunks
    ↓
[ Reranker ] → narrows to top-5 accurately
    ↓
[ Repacker ] → orders chunks for LLM attention
    ↓
[ Summarizer ] → compresses context (optional)
    ↓
[ Generator ] → LLM produces grounded answer
    ↓
[ Evaluator ] → scores answer vs RAGBench targets
    ↓
[ Results Logger ] → saves to CSV
```

---

## Dataset

**RAGBench** — 100,000 real-world Q&A examples across 5 domains, 12 subsets.

| Domain | Subsets | Key Challenge |
|---|---|---|
| Biomedical | covidqa, pubmedqa | Dense terminology, long docs |
| General Knowledge | hotpotqa, msmarco, hagrid, expertqa | Multi-hop reasoning |
| Legal | cuad | Precise language, hierarchical structure |
| Customer Support | techqa, emanual, delucionqa | Short answers, structured docs |
| Finance | finqa, tatqa | Numerical reasoning, table-heavy |

Each sample contains:
- `question` — the query
- `documents` — 4 pre-retrieved passages
- `response` — GPT-3.5 generated ground truth answer
- `relevance_score` — expected context relevance (our target)
- `utilization_score` — expected context utilization (our target)
- `completeness_score` — expected completeness (our target)
- `adherence_score` — expected adherence/hallucination score (our target)

---

## Project Structure

```
rag_pipeline/
│
├── configs/
│   └── default.yaml          ← swap strategies here, nothing else
│
├── pipeline/
│   ├── contracts.py          ← data shapes passed between modules
│   ├── chunker.py            ← splits documents into chunks
│   ├── embedder.py           ← converts chunks to vectors
│   ├── indexer.py            ← stores vectors in ChromaDB + BM25
│   ├── retriever.py          ← finds relevant chunks for a query
│   ├── reranker.py           ← re-scores retrieved chunks accurately
│   ├── repacker.py           ← reorders chunks for LLM attention
│   ├── summarizer.py         ← compresses context (optional)
│   ├── query_classifier.py   ← decides if retrieval is needed
│   ├── generator.py          ← calls Ollama LLM for final answer
│   ├── evaluator.py          ← computes RAGBench metrics
│   └── results_logger.py     ← saves results to CSV
│
├── data/
│   └── chroma_db/            ← vector database (auto-created)
│
├── results/
│   └── experiment_log.csv    ← all experiment results (appended, never overwritten)
│
└── run_pipeline.py           ← main entry point
```

---

## Design Principles

### 1. Modular
Every pipeline stage is a separate file. Each file has one `run()` function. Modules don't call each other — the pipeline orchestrates them.

### 2. Config-Driven
Everything is controlled by `configs/default.yaml`. Swap chunking strategy, embedding model, retrieval method, reranker — by changing one line. No code changes needed.

```yaml
chunker:
  strategy: "sentence"    # change to "token", "sliding", "semantic"

retriever:
  strategy: "dense"       # change to "bm25", "hybrid"

reranker:
  strategy: "cross_encoder"  # change to "monot5", "none"
```

### 3. Constant Contracts
Every module accepts and returns the same data shapes regardless of which strategy is used. Defined in `pipeline/contracts.py`.

```
chunker    → always returns List[Chunk]
retriever  → always returns List[RetrievedChunk]
generator  → always returns GeneratorOutput
evaluator  → always returns EvaluationResult
```

### 4. Open Source Only
All models and tools are fully open-source. No API keys required after setup.

### 5. Results Tracking
Every experiment run is appended to a single CSV with our scores, expected scores, gaps, and which strategies were used — enabling direct comparison across experiments.

---

## Available Strategies

| Module | Options |
|---|---|
| **Chunker** | `token`, `sentence`, `sliding`, `semantic` |
| **Embedder** | `all-mpnet-base-v2`, `BAAI/bge-base-en-v1.5`, `intfloat/e5-base-v2`, `LLM-Embedder` |
| **Retriever** | `dense`, `bm25`, `hybrid` |
| **Reranker** | `cross_encoder`, `monot5`, `none` |
| **Repacker** | `forward`, `reverse`, `sides`, `none` |
| **Summarizer** | `none`, `extractive` |
| **Query Classifier** | `always`, `keyword` |
| **Generator** | `llama3.2` (via Ollama), any Ollama model |

---

## Evaluation Metrics

### Task 1 — RAGBench Metrics

| Metric | What It Measures |
|---|---|
| Context Relevance | Are retrieved chunks relevant to the query? |
| Context Utilization | Did the LLM use the retrieved chunks? |
| Completeness | Does the answer cover the full expected response? |
| Adherence | Is every sentence grounded in context? (hallucination detector) |

### Task 2 — RGB Robustness Metrics

| Metric | What It Measures |
|---|---|
| Noise Robustness | Correct answer despite irrelevant retrieved docs |
| Negative Rejection | Correctly declines when context is insufficient |
| Information Integration | Synthesises answer from multiple chunks |
| Counterfactual Robustness | Detects and corrects factual errors in retrieved docs |

---

## Setup

### Prerequisites
- macOS (MacBook Air tested)
- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### Installation

```bash
# Clone repo
git clone https://github.com/aabhagupta49/capstone-project-aabha.git
cd capstone-project-aabha

# Create virtual environment
python -m venv rag_env
source rag_env/bin/activate

# Install dependencies
pip install sentence-transformers chromadb rank-bm25 datasets pyyaml tqdm pandas
pip install transformers torch
pip install chonkie
pip install langchain langchain-community

# Pull LLM model
ollama pull llama3.2
```

### Running

```bash
# Terminal 1 — start Ollama
ollama serve

# Terminal 2 — run pipeline
cd rag_pipeline
source ../rag_env/bin/activate
python run_pipeline.py
```

---

## Configuration Guide

Open `configs/default.yaml` to control everything:

```yaml
config_name: "baseline_v1"       # name shown in results CSV

dataset:
  domain: "biomedical"           # which domain to run
  sample_size: 5                 # 5 for testing, null for full dataset
  force_reindex: false           # true = rebuild index, false = reuse existing

chunker:
  strategy: "sentence"           # how to split documents
  chunk_size: 512
  chunk_overlap: 50

retriever:
  strategy: "dense"              # how to search
  top_k: 20                      # how many candidates to retrieve

reranker:
  strategy: "cross_encoder"      # how to rerank
  top_n: 5                       # final chunks passed to LLM

repacker:
  strategy: "reverse"            # how to order chunks

generator:
  model: "llama3.2"              # which Ollama model
  temperature: 0.1               # lower = more deterministic
```

### Swapping Domains

```yaml
dataset:
  domain: "biomedical"           # biomedical | general_knowledge | legal | customer_support | finance
  force_reindex: true            # must rebuild index when changing domain
```

### Running an Experiment

1. Change strategy in `default.yaml`
2. Run `python run_pipeline.py`
3. Results automatically appended to `results/experiment_log.csv`
4. Terminal prints summary table immediately

---

## Results Format

`results/experiment_log.csv` — one row per query, never overwritten:

| Column | Description |
|---|---|
| `timestamp` | When this ran |
| `config_name` | Which config was used |
| `domain` | Which RAGBench domain |
| `strategy_chunker` | Chunking strategy used |
| `strategy_retriever` | Retrieval strategy used |
| `strategy_reranker` | Reranking strategy used |
| `our_context_relevance` | Our computed relevance score |
| `expected_context_relevance` | RAGBench target score |
| `gap_relevance` | Difference (negative = below target) |
| `our_adherence` | Our hallucination score |
| `expected_adherence` | RAGBench target |
| `gap_adherence` | Difference |
| *(same pattern for all 4 metrics)* | |

### Reading Results

```
gap = our_score - expected_score

gap = -0.13  → we're 13% below target → improve this metric
gap = +0.02  → we're above target → good
gap =  0.00  → perfect match
```

---

## Tools & Libraries

| Tool | Purpose | License |
|---|---|---|
| Chonkie | Document chunking | MIT |
| sentence-transformers | Embedding models | Apache 2.0 |
| ChromaDB | Vector database | Apache 2.0 |
| rank-bm25 | BM25 keyword search | Apache 2.0 |
| Ollama + llama3.2 | Local LLM inference | Meta Llama |
| HuggingFace datasets | RAGBench loading | Apache 2.0 |
| CrossEncoder (ms-marco) | Reranking | Apache 2.0 |
| monoT5 | Advanced reranking | Apache 2.0 |
| PyTorch | Model inference | BSD |

All tools are fully open-source. No external API keys required.

---

## References

1. RAGBench: Explainable Benchmark for RAG Systems — https://arxiv.org/pdf/2407.11005
2. RGB: Benchmarking LLMs in RAG — https://arxiv.org/pdf/2309.01431
3. Best Practices in RAG (EMNLP 2024) — https://aclanthology.org/2024.emnlp-main.981/
4. RAGBench Dataset — https://huggingface.co/datasets/rungalileo/ragbench
5. Enterprise RAGBench — https://arxiv.org/pdf/2605.05253

---

## Common Issues

| Error | Fix |
|---|---|
| `ollama: command not found` | Open Ollama app from Applications folder |
| `connection refused` | Run `ollama serve` in separate terminal |
| `model not found` | Run `ollama pull llama3.2` |
| `chonkie import error` | Run `pip install chonkie` |
| `chromadb collection error` | Delete `data/chroma_db/` folder and rerun |
| `dataset not found` | Check internet — HuggingFace download needed first time |
| VS Code shows import errors | Press Cmd+Shift+P → Python: Select Interpreter → choose rag_env |

---

## Project Roadmap

### Task 1 — RAGBench (Weeks 1–6)
- [x] Basic pipeline framework
- [ ] Baseline scores on Customer Support domain
- [ ] Multi-domain evaluation (all 5 domains)
- [ ] Ablation experiments (chunking, embedding, retrieval, reranking)

### Task 2 — RGB Robustness (Weeks 7–12)
- [ ] RGB evaluation setup
- [ ] Noise robustness improvement
- [ ] Negative rejection improvement
- [ ] Final cloud deployment (Heroku/GCP)

### Bonus
- [ ] Semantic chunking
- [ ] Domain-specific embedding fine-tuning
- [ ] Enterprise RAGBench evaluation
