# pipeline/generator.py
# Sends retrieved chunks + query to Ollama LLM and gets an answer.
# Ollama runs locally — no API key, no internet needed after setup.

import requests
import json
from pipeline.contracts import RetrievedChunk, GeneratorOutput
from typing import List


def run(query: str, retrieved_chunks: List[RetrievedChunk], config: dict) -> GeneratorOutput:
    """
    Input:  query + retrieved chunks + config
    Output: GeneratorOutput (answer + chunks used)
    """
    model = config["generator"]["model"]
    base_url = config["generator"]["base_url"]
    temperature = config["generator"]["temperature"]
    max_tokens = config["generator"]["max_tokens"]

    # Build context from retrieved chunks
    context = _build_context(retrieved_chunks)

    # Build prompt
    prompt = _build_prompt(query, context)

    print(f"  Calling {model} via Ollama...")

    # Call Ollama
    answer = _call_ollama(prompt, model, base_url, temperature, max_tokens)

    return GeneratorOutput(
        query=query,
        answer=answer,
        source_chunks=[r.chunk for r in retrieved_chunks]
    )


def _build_context(retrieved_chunks: List[RetrievedChunk]) -> str:
    """
    Combines retrieved chunks into one context string.
    Each chunk is numbered so LLM can reference them.
    """
    context_parts = []
    for i, retrieved in enumerate(retrieved_chunks, 1):
        context_parts.append(f"[{i}] {retrieved.chunk.text}")
    return "\n\n".join(context_parts)


def _build_prompt(query: str, context: str) -> str:
    """
    Instruction prompt that tells LLM to:
    1. Only use provided context
    2. Say "I don't know" if context is insufficient
    3. Not hallucinate
    """
    return f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
If the context does not contain enough information to answer, say "I cannot answer this based on the provided context."
Do NOT use any outside knowledge. Do NOT hallucinate.

Context:
{context}

Question: {query}

Answer:"""


def _call_ollama(prompt, model, base_url, temperature, max_tokens) -> str:
    """
    Calls Ollama's REST API.
    Ollama runs as a local server at http://localhost:11434
    """
    url = f"{base_url}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,          # get full response at once
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=120            # 2 min timeout — LLM can be slow
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    except requests.exceptions.ConnectionError:
        return "ERROR: Ollama is not running. Start it with: ollama serve"
    except requests.exceptions.Timeout:
        return "ERROR: Ollama timed out. Try a smaller model or shorter context."
    except Exception as e:
        return f"ERROR: {str(e)}"