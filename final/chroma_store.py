"""
Chroma backend for long-term memory (期末进阶 #1).

We replace the 自写 TfidfVectorStore with a Chroma-backed store. Because
HuggingFace downloads are blocked in this environment, we implement a custom
EmbeddingFunction that uses our TF-IDF char n-gram vectorizer. The store
gains:
  - persistent SQLite-backed storage (chroma's PersistentClient)
  - production-grade query API (filter by meta, batch operations)
  - drop-in replacement: same query/write API as LongTermMemory expects

The TF-IDF embedding is dimension-padded to a fixed size so Chroma can
build its HNSW/ANN index over a constant-shape vector space.
"""

import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


class TfidfEmbeddingFunction(EmbeddingFunction):
    """
    A Chroma-compatible EmbeddingFunction that uses sklearn TfidfVectorizer.
    Vectors are L2-normalized and padded to `dim` so Chroma sees a fixed shape.
    """

    def __init__(self, vectorizer: TfidfVectorizer, dim: int = 4096):
        self._vec = vectorizer
        self._dim = dim

    def __call__(self, input: Documents) -> Embeddings:
        if not input:
            return []
        mat = self._vec.transform(list(input)).toarray().astype(np.float32)
        # pad/truncate to dim
        if mat.shape[1] < self._dim:
            pad = np.zeros((mat.shape[0], self._dim - mat.shape[1]), dtype=np.float32)
            mat = np.concatenate([mat, pad], axis=1)
        elif mat.shape[1] > self._dim:
            mat = mat[:, :self._dim]
        # L2 normalize
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        mat = mat / norms
        return mat.tolist()


class ChromaVectorStore:
    """
    A Chroma-backed vector store. Behavior parity with TfidfVectorStore so
    LongTermMemory can swap it in transparently.
    """

    def __init__(self, ngram_range=(2, 4), max_features=4096, persist_dir: str = None,
                 collection_name: str = "pj_ag2"):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.dim = max_features
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=ngram_range,
            max_features=max_features, lowercase=True,
        )
        self._fitted = False
        self._next_id = 0
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # client (in-memory by default; use PersistentClient if persist_dir given)
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=persist_dir)
        else:
            self._client = chromadb.EphemeralClient()
        self._collection = None  # created after fit

    def fit(self, corpus: List[str]):
        self.vectorizer.fit(corpus if corpus else [" "])
        self._fitted = True
        # create / get collection now that embedding fn is ready
        embed_fn = TfidfEmbeddingFunction(self.vectorizer, dim=self.dim)
        # purge any old collection of the same name (avoid stale embedding fn)
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = self._client.create_collection(
            name=self.collection_name,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, text: str, meta: Optional[Dict[str, Any]] = None):
        if not self._fitted:
            raise RuntimeError("Vectorizer not fit.")
        doc_id = f"d{self._next_id}"
        self._next_id += 1
        # Chroma rejects None meta values
        clean_meta = {k: (v if v is not None else "") for k, v in (meta or {}).items()} or {"_": ""}
        self._collection.add(documents=[text], ids=[doc_id], metadatas=[clean_meta])

    def query(self, text: str, k: int = 5, threshold: float = 0.0) -> List[Tuple[str, float, Dict[str, Any]]]:
        n_existing = self._collection.count()
        if n_existing == 0:
            return []
        k_eff = min(k, n_existing)
        res = self._collection.query(query_texts=[text], n_results=k_eff)
        docs = res["documents"][0] if res["documents"] else []
        dists = res["distances"][0] if res["distances"] else []
        metas = res["metadatas"][0] if res["metadatas"] else []
        out = []
        for doc, dist, meta in zip(docs, dists, metas):
            sim = 1.0 - float(dist)  # cosine distance -> similarity
            if sim < threshold:
                continue
            out.append((doc, sim, meta or {}))
        return out

    def __len__(self):
        return 0 if self._collection is None else self._collection.count()


if __name__ == "__main__":
    # smoke test
    store = ChromaVectorStore(persist_dir="/tmp/chroma_smoke_test")
    store.fit(["我家wifi密码是Lin8520#家。", "今天天气很好。", "我喜欢素食。"])
    store.add("我家wifi密码是Lin8520#家。", meta={"turn": 1})
    store.add("今天天气很好。", meta={"turn": 3})
    store.add("我喜欢素食。", meta={"turn": 5})
    print(f"Store size: {len(store)}")
    hits = store.query("wifi密码", k=2)
    for txt, score, meta in hits:
        print(f"  sim={score:.3f}  {txt[:40]}  meta={meta}")
