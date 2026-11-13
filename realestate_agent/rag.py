"""RAG pipeline: chunk Markdown documents, embed them, store in Chroma, retrieve with sources.

Three retrievers share one interface (`retrieve(query, k)` returning passages with citations,
and `passage(citation)` returning one chunk by id):

- HybridKnowledgeBase: embeddings and BM25 together, merged by reciprocal rank fusion. The
  default when embeddings are installed. Meaning-based and exact-term matches both count.
- KnowledgeBase: local embeddings (fastembed) in a Chroma index.
- LexicalKnowledgeBase: BM25 keyword ranking in pure Python, for hosts where the embedding
  model and Chroma are too heavy, such as serverless deployments.

Every retriever applies a relevance floor, so an empty result means "the documents do not cover
this" and the agent must say so instead of answering from general knowledge.

Choose with RE_AGENT_RAG=hybrid|embeddings|lexical. Unset, hybrid is used when installed.
"""
import hashlib
import math
import os
import re
from collections import Counter
from pathlib import Path

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
        import chromadb
        from fastembed import TextEmbedding

        self.knowledge_dir = knowledge_dir
        self._embedder = TextEmbedding(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=str(runtime_dir() / "chroma"))
        self._client = client
        self._collection = client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        self._fingerprint_file = runtime_dir() / "chroma" / "knowledge.sha1"
        stored = self._fingerprint_file.read_text() if self._fingerprint_file.exists() else ""
        if self._collection.count() == 0 or stored != self._fingerprint():
            self.ingest()

    def _fingerprint(self) -> str:
        """Hash of every document, so an edited knowledge folder triggers a re-index."""
        digest = hashlib.sha1()
        for path in sorted(self.knowledge_dir.glob("*.md")):
            digest.update(path.name.encode() + path.read_bytes())
        return digest.hexdigest()

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
        self._fingerprint_file.write_text(self._fingerprint())
        return {"documents": len(files), "chunks": len(ids)}

    def passage(self, citation: str) -> dict | None:
        found = self._collection.get(ids=[citation])
        if not found["ids"]:
            return None
        meta = found["metadatas"][0]
        return {"citation": citation, "source": meta["source"], "section": meta["section"],
                "text": found["documents"][0]}

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


STOPWORDS = set("""a an and are as at be by can do does for from has have how i if in is it its me my
of on or should so that the their there this to was what when where which who will with you your
much many about any into than then they them we our us""".split())
LEXICAL_MIN_SCORE = 0.25


def _terms(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 1]


class LexicalKnowledgeBase:
    """BM25 retrieval over the same chunks and citations, with no model download."""

    K1, B = 1.5, 0.75

    def __init__(self, knowledge_dir: Path = KNOWLEDGE_DIR):
        self.knowledge_dir = knowledge_dir
        self.ingest()

    def ingest(self) -> dict:
        self.chunks = []
        files = sorted(self.knowledge_dir.glob("*.md"))
        for path in files:
            for n, (section, chunk) in enumerate(chunk_markdown(path.read_text(encoding="utf-8"))):
                terms = _terms(f"{section} {chunk}")
                self.chunks.append({"id": f"{path.name}#{n}", "source": path.name, "section": section,
                                    "text": chunk, "tf": Counter(terms), "len": len(terms)})
        if not self.chunks:
            raise ValueError(f"No documents found in {self.knowledge_dir}")
        self.avg_len = sum(c["len"] for c in self.chunks) / len(self.chunks)
        df = Counter(t for c in self.chunks for t in c["tf"])
        n = len(self.chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        return {"documents": len(files), "chunks": n}

    def passage(self, citation: str) -> dict | None:
        for c in self.chunks:
            if c["id"] == citation:
                return {"citation": c["id"], "source": c["source"], "section": c["section"],
                        "text": c["text"]}
        return None

    def retrieve(self, query: str, k: int = 4) -> list[dict]:
        q = set(_terms(query))
        if not q:
            return []
        scored = []
        for c in self.chunks:
            s = 0.0
            for term in q & c["tf"].keys():
                tf = c["tf"][term]
                s += self.idf[term] * tf * (self.K1 + 1) / (
                    tf + self.K1 * (1 - self.B + self.B * c["len"] / self.avg_len))
            if s:
                scored.append((s, c))
        if not scored:
            return []
        scored.sort(key=lambda x: -x[0])
        best = scored[0][0]
        results = []
        for s, c in scored[:k]:
            # Relative score in 0..1, so weak matches next to a strong one are dropped.
            similarity = s / best * min(1.0, best / 6)
            if similarity >= LEXICAL_MIN_SCORE:
                results.append({"citation": c["id"], "source": c["source"], "section": c["section"],
                                "similarity": round(similarity, 3), "text": c["text"]})
        return results


RRF_K = 60  # Standard reciprocal rank fusion constant.


class HybridKnowledgeBase:
    """Embeddings plus BM25, fused by rank.

    The semantic retriever decides whether the documents cover the question at all: a shared
    word alone (such as "capital" in "capital of France" and "capital gains") is not enough.
    When the question is covered, BM25 adds exact-term matches and both rankings are fused."""

    def __init__(self, knowledge_dir: Path = KNOWLEDGE_DIR):
        self.dense = KnowledgeBase(knowledge_dir)
        self.lexical = LexicalKnowledgeBase(knowledge_dir)

    def ingest(self) -> dict:
        self.lexical.ingest()
        return self.dense.ingest()

    def passage(self, citation: str) -> dict | None:
        return self.lexical.passage(citation)

    def retrieve(self, query: str, k: int = 4) -> list[dict]:
        dense = self.dense.retrieve(query, k * 2)
        if not dense:
            return []  # Not covered by the documents.
        fused: dict[str, dict] = {}
        for name, hits in (("dense", dense), ("lexical", self.lexical.retrieve(query, k * 2))):
            for rank, hit in enumerate(hits):
                entry = fused.setdefault(hit["citation"], {**hit, "score": 0.0, "matched_by": []})
                entry["score"] += 1 / (RRF_K + rank + 1)
                entry["matched_by"].append(name)
                if name == "dense":
                    entry["similarity"] = hit["similarity"]  # Report the semantic score.
        ranked = sorted(fused.values(), key=lambda h: -h["score"])[:k]
        for h in ranked:
            h["score"] = round(h["score"], 4)
        return ranked


def make_knowledge():
    """Pick the retriever from RE_AGENT_RAG, falling back to BM25 if embeddings are unavailable."""
    mode = os.getenv("RE_AGENT_RAG", "lexical" if os.getenv("VERCEL") else "").strip().lower()
    if mode != "lexical":
        try:
            return KnowledgeBase() if mode == "embeddings" else HybridKnowledgeBase()
        except ImportError:
            if mode in ("embeddings", "hybrid"):
                raise
    return LexicalKnowledgeBase()
