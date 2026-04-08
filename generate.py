"""
Generate answers using retrieved context + a local LLM via Ollama.

Requires Ollama running locally (https://ollama.ai).
Falls back gracefully to retrieval-only mode if Ollama is unavailable.

Usage:
    python src/generate.py "What factors influenced the RBA's February 2025 decision?"
    python src/generate.py "Summarise the RBA's view on housing" --model llama3
"""

import argparse
import json
import sys
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).parent.parent))
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, TOP_K
from retrieve import retrieve, format_context


SYSTEM_PROMPT = """You are an AI assistant that answers questions about Reserve Bank of Australia (RBA) monetary policy using ONLY the provided source documents. 

Rules:
- Answer based strictly on the provided context. Do not use outside knowledge.
- Cite which source(s) you are drawing from (e.g., [Source 1]).
- If the context does not contain enough information to answer, say so clearly.
- Be concise and precise.
"""


def check_ollama() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


def generate_answer(query: str, context: str, model: str = OLLAMA_MODEL) -> str:
    """Send query + context to Ollama and return the generated answer."""
    prompt = f"""Context from RBA monetary policy minutes:

{context}

Question: {query}

Answer based only on the context above:"""

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": 0.1,  # low temperature for factual QA
                "num_predict": 512,
            },
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def main():
    parser = argparse.ArgumentParser(description="RAG-powered QA over RBA minutes.")
    parser.add_argument("query", type=str, help="Natural language question.")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL)
    parser.add_argument("--context-only", action="store_true",
                        help="Skip LLM generation; print retrieved context only.")
    args = parser.parse_args()

    # Step 1: Retrieve relevant chunks
    print(f"Query: {args.query}\n")
    results = retrieve(args.query, top_k=args.top_k)
    context = format_context(results)

    print(f"Retrieved {len(results)} chunks (scores: "
          f"{results[0]['score']:.3f} - {results[-1]['score']:.3f})\n")

    # Step 2: Generate answer (or fall back)
    if args.context_only:
        print("--- Retrieved Context ---\n")
        print(context)
        return

    if not check_ollama():
        print("[!] Ollama not detected. Falling back to retrieval-only mode.")
        print("    Install Ollama (https://ollama.ai) and run:")
        print(f"    ollama pull {args.model}\n")
        print("--- Retrieved Context ---\n")
        print(context)
        return

    print(f"Generating answer with {args.model}...\n")
    answer = generate_answer(args.query, context, model=args.model)

    print("--- Answer ---\n")
    print(answer)

    print("\n--- Sources ---\n")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r['title']} (score: {r['score']:.3f})")


if __name__ == "__main__":
    main()
