"""
Retrieve relevant chunks from the RBA vector store given a natural language query.

Usage:
    python src/retrieve.py "What did the RBA say about inflation in February 2025?"
    python src/retrieve.py "How has the labour market changed?" --top-k 10
"""

import argparse
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

sys.path.append(str(Path(__file__).parent.parent))
from config import CHROMA_DIR, EMBEDDING_MODEL, TOP_K


def get_collection() -> chromadb.Collection:
    """Load the persisted ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return client.get_collection(
        name="rba_minutes",
        embedding_function=embedding_fn,
    )


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Query the vector store and return the top-k most relevant chunks.

    Returns:
        List of dicts with keys: text, source, title, score (cosine similarity).
    """
    collection = get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append({
            "text": doc,
            "source": meta["source"],
            "title": meta["title"],
            "chunk_index": meta["chunk_index"],
            "score": 1 - dist,  # ChromaDB returns cosine distance; convert to similarity
        })

    return retrieved


def format_context(results: list[dict]) -> str:
    """Format retrieved chunks into a context string for LLM prompting."""
    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(
            f"[Source {i}: {r['title']}]\n{r['text']}"
        )
    return "\n\n---\n\n".join(context_parts)


def main():
    parser = argparse.ArgumentParser(description="Query the RBA RAG vector store.")
    parser.add_argument("query", type=str, help="Natural language question.")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of chunks to retrieve.")
    args = parser.parse_args()

    print(f"Query: {args.query}\n")
    print(f"Retrieving top {args.top_k} chunks...\n")

    results = retrieve(args.query, top_k=args.top_k)

    for i, r in enumerate(results, 1):
        print(f"{'='*60}")
        print(f"Result {i} | Score: {r['score']:.4f} | Source: {r['source']}")
        print(f"{'='*60}")
        print(r["text"][:500])
        if len(r["text"]) > 500:
            print("... [truncated]")
        print()


if __name__ == "__main__":
    main()
