"""RAG pipeline: chunk Markdown documents, embed them, store in Chroma, retrieve with sources."""
import re
from pathlib import Path

import chromadb
from fastembed import TextEmbedding

from .config import EMBEDDING_MODEL, KNOWLEDGE_DIR, runtime_dir

COLLECTION = "real_estate_knowledge"
MAX_CHARS = 800
OVERLAP = 100
# Measured: on-topic queries score 0.76 to 0.83, unrelated ones 0.48 to 0.55.
MIN_SIMILARITY = 0.6


def chunk_markdown(text: str) -> list[tuple[str, str]]:
    """Split by '## ' section, then by size. Returns (section_title, chunk_text) pairs."""
    chunks = []
    for section in re.split(r"(?m)^(?=## )", text):
        section = section.strip()
        if not section:
            continue
        first_line = section.splitlines()[0]
        title = first_line.lstrip("# ").strip()
        start = 0
        while start < len(section):
            piece = section[start : start + MAX_CHARS]
            prefix = "" if start == 0 else f"{title}: "
            chunks.append((title, prefix + piece))
            if start + MAX_CHARS >= len(section):
                break
            start += MAX_CHARS - OVERLAP
    return chunks


class KnowledgeBase:
    def __init__(self, knowledge_dir: Path = KNOWLEDGE_DIR):
        self.knowledge_dir = knowledge_dir
        self._embedder = TextEmbedding(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=str(runtime_dir() / "chroma"))
        self._client = client
        self._collection = client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        if self._collection.count() == 0:
            self.ingest()

    def ingest(self) -> dict:
        """Rebuild the index from every document in the knowledge folder."""
        self._client.delete_collection(COLLECTION)
        self._collection = self._client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        ids, texts, metadatas = [], [], []
        files = sorted(self.knowledge_dir.glob("*.md"))
        for path in files:
            for n, (section, chunk) in enumerate(chunk_markdown(path.read_text(encoding="utf-8"))):
                ids.append(f"{path.name}#{n}")
                texts.append(chunk)
                metadatas.append({"source": path.name, "section": section})
        if not ids:
            raise ValueError(f"No documents found in {self.knowledge_dir}")
        embeddings = [e.tolist() for e in self._embedder.embed(texts)]
        self._collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
        return {"documents": len(files), "chunks": len(ids)}

    def retrieve(self, query: str, k: int = 4) -> list[dict]:
        """Return the top chunks above the relevance floor. Empty means no grounded evidence."""
        vector = next(iter(self._embedder.query_embed(query))).tolist()
        found = self._collection.query(query_embeddings=[vector], n_results=k)
        results = []
        for cid, doc, meta, dist in zip(
            found["ids"][0], found["documents"][0], found["metadatas"][0], found["distances"][0]
        ):
            similarity = 1 - dist
            if similarity >= MIN_SIMILARITY:
                results.append(
                    {
                        "citation": cid,
                        "source": meta["source"],
                        "section": meta["section"],
                        "similarity": round(similarity, 3),
                        "text": doc,
                    }
                )
        return results
