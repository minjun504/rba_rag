"""
Configuration for the RBA RAG pipeline.
All tuneable parameters in one place.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "vectorstore"

# Embedding model (runs locally via sentence-transformers)
EMBEDDING_MODEL = "all-mpnet-base-v2"

# Chunking parameters
CHUNK_SIZE = 256  # tokens (approx)
CHUNK_OVERLAP = 64

# Retrieval
TOP_K = 5  # number of chunks to retrieve per query

# LLM generation (optional, requires Ollama running locally)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral"  # or "llama3", "phi3", etc.

# RBA document URLs (monetary policy decisions, 2024-2025)
RBA_URLS = [
    "https://www.rba.gov.au/monetary-policy/rba-board-minutes/2025/2025-02-18.html",
    "https://www.rba.gov.au/monetary-policy/rba-board-minutes/2025/2025-04-01.html",
    "https://www.rba.gov.au/monetary-policy/rba-board-minutes/2025/2025-05-20.html",
    "https://www.rba.gov.au/monetary-policy/rba-board-minutes/2025/2025-07-08.html",
    "https://www.rba.gov.au/monetary-policy/rba-board-minutes/2025/2025-08-12.html",
    "https://www.rba.gov.au/monetary-policy/rba-board-minutes/2025/2025-09-30.html",
    "https://www.rba.gov.au/monetary-policy/rba-board-minutes/2025/2025-11-04.html",
    "https://www.rba.gov.au/monetary-policy/rba-board-minutes/2025/2025-12-09.html",
    "https://www.rba.gov.au/monetary-policy/rba-board-minutes/2026/2026-02-03.html",
    "https://www.rba.gov.au/monetary-policy/rba-board-minutes/2026/2026-03-17.html",
]
