"""
Evaluate retrieval quality of the RBA RAG pipeline.

Metrics:
    - Mean Reciprocal Rank (MRR): How high does the first relevant result appear?
    - Hit Rate @ K: What fraction of queries have at least one relevant result in top-K?
    - Mean Cosine Similarity: Average similarity score of retrieved chunks.

Uses a hand-curated set of (query, expected_source) pairs as ground truth.
This is a lightweight evaluation — production systems would use LLM-as-judge
or human annotation, but this demonstrates the evaluation mindset.

Usage:
    python src/evaluate.py
    python src/evaluate.py --top-k 10
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import TOP_K
from retrieve import retrieve


# Ground truth: queries paired with expected source documents.
# A retrieved chunk is "relevant" if it comes from the expected source.
EVAL_SET = [
    {
        "query": "What did the RBA decide about the cash rate in February 2025?",
        "expected_sources": ["2025-02-18.html"],
    },
    {
        "query": "What was the RBA's assessment of inflation in late 2025?",
        "expected_sources": ["2025-11-04.html", "2025-12-09.html"],
    },
    {
        "query": "How did the RBA view the labour market in mid 2025?",
        "expected_sources": ["2025-07-08.html", "2025-08-12.html"],
    },
    {
        "query": "What risks to the economic outlook did the Board discuss in April 2025?",
        "expected_sources": ["2025-04-01.html"],
    },
    {
        "query": "What was the Board's view on household consumption and spending?",
        "expected_sources": ["2025-05-20.html", "2025-07-08.html", "2025-08-12.html"],
    },
    {
        "query": "How did global economic conditions affect the RBA's decisions in September 2025?",
        "expected_sources": ["2025-09-30.html"],
    },
    {
        "query": "What did the RBA say about housing prices and supply in 2025?",
        "expected_sources": ["2025-02-18.html", "2025-05-20.html", "2025-08-12.html", "2025-11-04.html"],
    },
    {
        "query": "What was discussed about wages growth in early 2025?",
        "expected_sources": ["2025-02-18.html", "2025-04-01.html"],
    },
]


def is_relevant(result: dict, expected_sources: list[str]) -> bool:
    """Check if a retrieved chunk comes from one of the expected sources."""
    return result["source"] in expected_sources


def evaluate(top_k: int = TOP_K) -> dict:
    """Run evaluation over the ground truth set and return metrics."""
    mrr_scores = []
    hit_rates = []
    mean_similarities = []

    print(f"Evaluating {len(EVAL_SET)} queries @ top-{top_k}\n")

    for item in EVAL_SET:
        query = item["query"]
        expected = item["expected_sources"]
        results = retrieve(query, top_k=top_k)

        # Mean Reciprocal Rank
        rr = 0.0
        for rank, r in enumerate(results, 1):
            if is_relevant(r, expected):
                rr = 1.0 / rank
                break
        mrr_scores.append(rr)

        # Hit Rate (at least one relevant in top-k)
        hit = any(is_relevant(r, expected) for r in results)
        hit_rates.append(float(hit))

        # Mean similarity score
        sim = sum(r["score"] for r in results) / len(results) if results else 0
        mean_similarities.append(sim)

        # Per-query report
        status = "HIT" if hit else "MISS"
        print(f"  [{status}] MRR={rr:.3f} | Sim={sim:.3f} | {query[:60]}...")
        if not hit:
            retrieved_sources = set(r["source"] for r in results)
            print(f"         Expected: {expected}")
            print(f"         Got:      {list(retrieved_sources)}")

    metrics = {
        "num_queries": len(EVAL_SET),
        "top_k": top_k,
        "mrr": sum(mrr_scores) / len(mrr_scores),
        "hit_rate": sum(hit_rates) / len(hit_rates),
        "mean_similarity": sum(mean_similarities) / len(mean_similarities),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate RBA RAG retrieval quality.")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    metrics = evaluate(top_k=args.top_k)

    print(f"\n{'='*50}")
    print(f"  RETRIEVAL EVALUATION RESULTS (top-{metrics['top_k']})")
    print(f"{'='*50}")
    print(f"  Queries evaluated:    {metrics['num_queries']}")
    print(f"  Mean Reciprocal Rank: {metrics['mrr']:.3f}")
    print(f"  Hit Rate @ {metrics['top_k']}:        {metrics['hit_rate']:.3f}")
    print(f"  Mean Cosine Sim:      {metrics['mean_similarity']:.3f}")
    print(f"{'='*50}")

    # Save results
    output_path = Path(__file__).parent / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
