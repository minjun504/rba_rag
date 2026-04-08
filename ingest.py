"""
Ingest RBA monetary policy documents into a ChromaDB vector store.

Pipeline:
    1. Download HTML pages from the RBA website
    2. Extract and clean text content
    3. Split into overlapping chunks
    4. Embed with sentence-transformers
    5. Store in ChromaDB for retrieval
"""

import re
import sys
import hashlib
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import chromadb
from chromadb.utils import embedding_functions

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DATA_DIR, CHROMA_DIR, EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, RBA_URLS,
)


def download_documents(urls: list[str], output_dir: Path) -> list[Path]:
    """Download RBA HTML pages and save locally."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for url in urls:
        filename = url.rstrip("/").split("/")[-1]
        filepath = output_dir / filename

        if filepath.exists():
            print(f"  [cached] {filename}")
            saved.append(filepath)
            continue

        print(f"  [downloading] {url}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        filepath.write_text(resp.text, encoding="utf-8")
        saved.append(filepath)

    return saved


def extract_text(html_path: Path) -> dict:
    """Extract clean text and metadata from an RBA minutes HTML page."""
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")

    content_div = (
        soup.find("div", id="content")
        or soup.find("article")
        or soup.find("div", class_="rba-content")
        or soup.body
    )

    if content_div is None:
        return {"text": "", "source": html_path.name, "title": "Unknown"}

    # Remove script/style tags
    for tag in content_div.find_all(["script", "style", "nav", "footer"]):
        tag.decompose()

    text = content_div.get_text(separator="\n", strip=True)
    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)

    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else html_path.stem

    return {
        "text": text,
        "source": html_path.name,
        "title": title,
    }


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks by approximate word count.

    Uses paragraph boundaries where possible to avoid splitting mid-sentence.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para.split())

        if para_len > chunk_size:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                sent_len = len(sent.split())
                if current_len + sent_len > chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    overlap_text = " ".join(current_chunk).split()[-overlap:]
                    current_chunk = [" ".join(overlap_text)]
                    current_len = len(overlap_text)
                current_chunk.append(sent)
                current_len += sent_len
        elif current_len + para_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap_text = " ".join(current_chunk).split()[-overlap:]
            current_chunk = [" ".join(overlap_text)]
            current_len = len(overlap_text)
            current_chunk.append(para)
            current_len += para_len
        else:
            current_chunk.append(para)
            current_len += para_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def build_vectorstore(documents: list[dict]) -> chromadb.Collection:
    """Embed document chunks and store in ChromaDB."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    try:
        client.delete_collection("rba_minutes")
    except ValueError:
        pass

    collection = client.get_or_create_collection(
        name="rba_minutes",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks = []
    all_ids = []
    all_metadata = []

    for doc in documents:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{doc['source']}_{i}".encode()).hexdigest()
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadata.append({
                "source": doc["source"],
                "title": doc["title"],
                "chunk_index": i,
            })

    batch_size = 64
    for start in range(0, len(all_chunks), batch_size):
        end = start + batch_size
        collection.add(
            documents=all_chunks[start:end],
            ids=all_ids[start:end],
            metadatas=all_metadata[start:end],
        )

    return collection


def main():
    print("=== RBA RAG Ingestion Pipeline ===\n")

    print("1. Downloading RBA monetary policy minutes...")
    html_files = download_documents(RBA_URLS, DATA_DIR)
    print(f"   {len(html_files)} documents ready.\n")

    print("2. Extracting and chunking text...")
    documents = []
    total_chunks = 0
    for f in html_files:
        doc = extract_text(f)
        if doc["text"]:
            chunks = chunk_text(doc["text"])
            total_chunks += len(chunks)
            documents.append(doc)
            print(f"   {doc['source']}: {len(chunks)} chunks")
    print(f"   Total: {total_chunks} chunks across {len(documents)} documents.\n")

    print("3. Embedding and storing in ChromaDB...")
    collection = build_vectorstore(documents)
    print(f"   Vector store built: {collection.count()} vectors stored.")
    print(f"   Persisted to: {CHROMA_DIR}\n")

    print("Done. Run `python src/retrieve.py 'your question'` to query.")


if __name__ == "__main__":
    main()
