# RBA Monetary Policy RAG Pipeline

A retrieval-augmented generation (RAG) system for querying Reserve Bank of Australia monetary policy board minutes using natural language.

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  RBA Website  │────▶│   Ingest &   │────▶│   ChromaDB   │────▶│   Retrieve   │
│  (HTML pages) │     │   Chunk      │     │ Vector Store │     │   Top-K      │
└──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                      │
                                                                      ▼
                                                               ┌──────────────┐
                                                               │  LLM Answer  │
                                                               │  (Ollama)    │
                                                               └──────────────┘
```

**Components:**
- **Ingestion** (`src/ingest.py`): Downloads RBA board minutes (HTML), extracts text, splits into overlapping chunks, embeds with `all-mpnet-base-v2`, and stores in ChromaDB.
- **Retrieval** (`src/retrieve.py`): Encodes a user query, performs cosine similarity search against the vector store, and returns the top-K most relevant chunks with scores.
- **Generation** (`src/generate.py`): Passes retrieved context to a local LLM (via Ollama) with a grounded system prompt that enforces source attribution. Falls back gracefully to retrieval-only mode if no LLM is available.
- **Evaluation** (`src/evaluate.py`): Measures retrieval quality using Mean Reciprocal Rank (MRR), Hit Rate @ K, and mean cosine similarity against a hand-curated ground truth set.

## Setup

```bash
# Clone and install dependencies
git clone https://github.com/minjun504/rba-rag.git
cd rba-rag
pip install -r requirements.txt

# Ingest RBA documents into the vector store
python src/ingest.py

# (Optional) Install Ollama for LLM generation
# https://ollama.ai
# ollama pull mistral
```

## Usage

### Query with retrieval only
```bash
python src/retrieve.py "What did the RBA say about inflation in February 2025?"
```

### Query with LLM-generated answer (requires Ollama)
```bash
python src/generate.py "What factors influenced the Board's decision on the cash rate?"
```

### Run retrieval evaluation
```bash
python src/evaluate.py --top-k 5
```

## Design Decisions

- **Paragraph-aware chunking**: Splits on paragraph boundaries with word-level overlap to avoid breaking mid-sentence, improving retrieval coherence.
- **Cosine similarity over L2 distance**: Better suited for semantic similarity with normalised sentence-transformer embeddings.
- **Low LLM temperature (0.1)**: Prioritises factual grounding over creativity for policy QA.
- **Evaluation-first approach**: Includes a structured eval harness with ground truth pairs, reflecting the principle that retrieval quality should be measured, not assumed.

## Tech Stack

- **Embeddings**: `sentence-transformers` (all-mpnet-base-v2)
- **Vector Store**: ChromaDB (persistent, local)
- **LLM**: Ollama (optional, local — Mistral/Llama 3)
- **Language**: Python 3.10+
