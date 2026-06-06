# pipeline/evaluator.py

from pipeline.contracts import GeneratorOutput, EvaluationResult, RetrievedChunk
from typing import List
from datetime import datetime


def run(generator_output: GeneratorOutput, retrieved_chunks: List[RetrievedChunk], expected: dict, config: dict) -> EvaluationResult:

    answer = generator_output.answer
    query  = generator_output.query

    # Our actual scores — always computed
    context_relevance   = _score_context_relevance(retrieved_chunks)
    context_utilization = _score_context_utilization(answer, retrieved_chunks)
    completeness        = _score_completeness(answer, expected)
    adherence           = _score_adherence(answer, retrieved_chunks)

    # RAGBench expected scores — our targets
    exp_relevance   = float(expected.get("relevance_score", 0.0))
    exp_utilization = float(expected.get("utilization_score", 0.0))
    exp_completeness= float(expected.get("completeness_score", 0.0))
    exp_adherence   = float(expected.get("adherence_score", 0.0))

    return EvaluationResult(
        # Query info
        query=query,
        answer=answer,
        expected_answer=expected.get("response", ""),

        # Our actual scores
        context_relevance=context_relevance,
        context_utilization=context_utilization,
        completeness=completeness,
        adherence=adherence,

        # Expected scores from RAGBench
        expected_context_relevance=exp_relevance,
        expected_context_utilization=exp_utilization,
        expected_completeness=exp_completeness,
        expected_adherence=exp_adherence,

        # Gaps — negative means we're below target
        gap_relevance=round(context_relevance - exp_relevance, 4),
        gap_utilization=round(context_utilization - exp_utilization, 4),
        gap_completeness=round(completeness - exp_completeness, 4),
        gap_adherence=round(adherence - exp_adherence, 4),

        # Metadata
        domain=config["dataset"]["domain"],
        config_name=config["config_name"],
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


def _score_context_relevance(retrieved_chunks: List[RetrievedChunk]) -> float:
    """
    Context Relevance: how relevant are our retrieved chunks?

    We compute this from our retrieval similarity scores.
    Higher score = our retriever found more relevant chunks.

    Note: RAGBench's relevance_score is the TARGET we compare against.
    We store both in CSV to measure the gap.
    """
    if not retrieved_chunks:
        return 0.0

    scores = [r.score for r in retrieved_chunks]
    return round(sum(scores) / len(scores), 4)


def _score_context_utilization(answer: str, retrieved_chunks: List[RetrievedChunk]) -> float:
    """
    Context Utilization: did our LLM actually use the retrieved chunks?

    Measure: word overlap between our answer and retrieved context.
    Higher overlap = LLM used the context more.
    """
    if not answer or not retrieved_chunks:
        return 0.0

    context_text  = " ".join([r.chunk.text for r in retrieved_chunks])
    context_words = set(_clean_text(context_text).split())
    answer_words  = set(_clean_text(answer).split())

    # Safety check. Can't measure utilization if either is missing.
    if not answer_words:
        return 0.0

    overlap = answer_words.intersection(context_words)
    return round(len(overlap) / len(answer_words), 4)


def _score_completeness(answer: str, expected: dict) -> float:
    """
    Completeness: does our answer cover the expected response fully?

    Measure: word overlap between our answer and RAGBench reference answer.
    We use the reference answer text here — not the score.
    """
    expected_answer = expected.get("response", "")

    if not expected_answer or not answer:
        return 0.0

    expected_words = set(_clean_text(expected_answer).split())
    answer_words   = set(_clean_text(answer).split())

    if not expected_words:
        return 0.0

    overlap = answer_words.intersection(expected_words)
    return round(len(overlap) / len(expected_words), 4)


def _score_adherence(answer: str, retrieved_chunks: List[RetrievedChunk]) -> float:
    """
    Adherence: is every sentence in our answer grounded in retrieved context?
    This is the hallucination detector. 
    If the LLM says something that has no connection to what we retrieved — that sentence is ungrounded — potential hallucination.

    Measure: fraction of answer sentences that share words with context.
    Ungrounded sentences = potential hallucination.
    """
    if not answer or not retrieved_chunks:
        return 0.0

    context_text  = " ".join([r.chunk.text for r in retrieved_chunks])
    context_words = set(_clean_text(context_text).split())

    sentences = [s.strip() for s in answer.split(".") if s.strip()]

    if not sentences:
        return 0.0

    grounded = 0
    for sentence in sentences:
        sentence_words = set(_clean_text(sentence).split())
        if sentence_words.intersection(context_words):
            grounded += 1

    return round(grounded / len(sentences), 4)


def _clean_text(text: str) -> str:
    """
    Lowercase + remove punctuation + remove stopwords.
    Makes word overlap comparison fair.
    """
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "of", "in", "on", "at",
        "to", "for", "with", "by", "from", "and", "or", "but", "not",
        "this", "that", "it", "its", "as", "if", "than", "then"
    }

    cleaned = ""
    for char in text.lower():
        if char.isalnum() or char == " ":
            cleaned += char

    words = [w for w in cleaned.split() if w not in stopwords]
    return " ".join(words)