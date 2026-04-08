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
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking parameters
CHUNK_SIZE = 512  # tokens (approx)
CHUNK_OVERLAP = 64

# Retrieval
TOP_K = 5  # number of chunks to retrieve per query

# LLM generation (optional, requires Ollama running locally)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral"  # or "llama3", "phi3", etc.

# RBA document URLs (monetary policy decisions, 2024-2025)
RBA_URLS = [
    "https://www.rba.gov.au/monetary-policy/rba-board-minutes/2025/2025-02-17.html",
    "https://www.rba.gov.au/monetary-policy/rba-board-minutes/2025/2025-03-31.html",
    "https://www.rba.gov.au/monetary-policy/rba-board-minutes/2024/2024-12-09.html",
    "https://www.rba.gov.au/monetary-policy/rba-board-minutes/2024/2024-11-04.html",
    "https://www.rba.gov.au/monetary-policy/rba-board-minutes/2024/2024-09-23.html",
    "https://www.rba.gov.au/monetary-policy/rba-board-minutes/2024/2024-08-05.html",
    "https://www.rba.gov.au/monetary-policy/rba-board-minutes/2024/2024-06-17.html",
    "https://www.rba.gov.au/monetary-policy/rba-board-minutes/2024/2024-05-06.html",
    "https://www.rba.gov.au/monetary-policy/rba-board-minutes/2024/2024-03-18.html",
    "https://www.rba.gov.au/monetary-policy/rba-board-minutes/2024/2024-02-05.html",
]
