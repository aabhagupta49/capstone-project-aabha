# pipeline/contracts.py
# These are the data shapes passed between every module.
# If you change these, everything else updates automatically.

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Chunk:
    """One piece of a document after splitting."""
    text: str                                           # the actual text
    chunk_id: str                                       # unique ID for this chunk
    doc_id: str                                         # the original document this chunk came from
    domain: str                                         # which RAGBench domain
    metadata: dict = field(default_factory=dict)        # any extra info

@dataclass
class RetrievedChunk:
    """A chunk that was retrieved for a query, with its relevance score."""
    chunk: Chunk                                        # the actual chunk
    score: float                                        # how relevant it is (0 to 1)

@dataclass
class GeneratorOutput:
    """The final answer produced by the LLM."""
    query: str
    answer: str
    source_chunks: List[Chunk]                          # which chunks the LLM used

@dataclass
class EvaluationResult:
    """Our actual scores + RAGBench expected scores for comparison."""
    
    # Query info
    query: str
    answer: str
    expected_answer: str
    
    # Our actual computed scores
    context_relevance: float
    context_utilization: float
    completeness: float
    adherence: float
    
    # RAGBench expected scores (what we're trying to match)
    expected_context_relevance: float = 0.0
    expected_context_utilization: float = 0.0
    expected_completeness: float = 0.0
    expected_adherence: float = 0.0
    
    # Gap (how far we are from target) — computed automatically
    gap_relevance: float = 0.0
    gap_utilization: float = 0.0
    gap_completeness: float = 0.0
    gap_adherence: float = 0.0
    
    # Metadata
    domain: str = ""
    config_name: str = ""
    timestamp: str = ""