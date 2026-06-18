"""Long-term memory: 自写 TF-IDF 字符 n-gram 向量库 + cosine similarity."""
import os, json, pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfVectorStore:
    def __init__(self, ngram_range=(2, 4), max_features=4096):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=ngram_range,
            max_features=max_features, lowercase=True,
        )
        self._fitted = False
        self.texts: List[str] = []
        self.metas: List[Dict[str, Any]] = []
        self.matrix: Optional[np.ndarray] = None

    def fit(self, corpus: List[str]):
        self.vectorizer.fit(corpus if corpus else [" "])
        self._fitted = True

    def _embed(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Vectorizer not fit.")
        return self.vectorizer.transform(texts).toarray().astype(np.float32)

    def add(self, text: str, meta: Optional[Dict[str, Any]] = None):
        vec = self._embed([text])
        if self.matrix is None:
            self.matrix = vec
        else:
            self.matrix = np.vstack([self.matrix, vec])
        self.texts.append(text)
        self.metas.append(meta or {})

    def query(self, text: str, k: int = 5, threshold: float = 0.0):
        if self.matrix is None or self.matrix.shape[0] == 0:
            return []
        q = self._embed([text])[0]
        sims = self.matrix @ q
        order = np.argsort(-sims)
        out = []
        for idx in order[:k]:
            score = float(sims[idx])
            if score < threshold:
                continue
            out.append((self.texts[idx], score, self.metas[idx]))
        return out

    def __len__(self):
        return 0 if self.matrix is None else int(self.matrix.shape[0])

    def save(self, dirpath: str):
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, "vectorizer.pkl"), "wb") as f:
            pickle.dump(self.vectorizer, f)
        np.savez_compressed(
            os.path.join(dirpath, "vectors.npz"),
            matrix=self.matrix if self.matrix is not None else np.zeros((0, 1), dtype=np.float32),
        )
        with open(os.path.join(dirpath, "docs.jsonl"), "w", encoding="utf-8") as f:
            for t, m in zip(self.texts, self.metas):
                f.write(json.dumps({"text": t, "meta": m}, ensure_ascii=False) + "\n")
        with open(os.path.join(dirpath, "export.jsonl"), "w", encoding="utf-8") as f:
            for i, (t, m) in enumerate(zip(self.texts, self.metas)):
                f.write(json.dumps({"id": i, "text": t, "meta": m}, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, dirpath: str):
        with open(os.path.join(dirpath, "vectorizer.pkl"), "rb") as f:
            vec = pickle.load(f)
        store = cls()
        store.vectorizer = vec
        store._fitted = True
        data = np.load(os.path.join(dirpath, "vectors.npz"))
        m = data["matrix"]
        store.matrix = m if m.shape[0] > 0 else None
        store.texts, store.metas = [], []
        with open(os.path.join(dirpath, "docs.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                store.texts.append(obj["text"])
                store.metas.append(obj["meta"])
        return store


class LongTermMemory:
    def __init__(self, store, top_k: int = 5, threshold: float = 0.10):
        self.store = store
        self.top_k = top_k
        self.threshold = threshold

    def write(self, text: str, meta: Optional[Dict[str, Any]] = None):
        self.store.add(text, meta)

    def retrieve(self, query: str):
        return self.store.query(query, k=self.top_k, threshold=self.threshold)

    def render(self, query: str) -> str:
        hits = self.retrieve(query)
        if not hits:
            return ""
        lines = ["[相关历史记忆，按相关度从高到低]"]
        for txt, score, meta in hits:
            tag = f"(第{meta['turn']}轮) " if "turn" in meta else ""
            lines.append(f"- {tag}{txt}  [sim={score:.3f}]")
        return "\n".join(lines)
