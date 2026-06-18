"""
Hybrid retriever (期末进阶 #2b).

Combines two complementary retrievers via Reciprocal Rank Fusion (RRF):
  - dense:   TF-IDF char-wb n-grams (3-gram of chars) — semantic-ish similarity
  - sparse:  BM25 over char bigrams — term-frequency-aware exact match

RRF formula (Cormack et al., 2009):
    score(d) = sum over each retriever r of  1 / (k_rrf + rank_r(d))
where k_rrf is a small constant (typically 60). RRF requires no score
calibration between retrievers, which is exactly what we need here since
TF-IDF cosine and BM25 are on totally different scales.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from typing import List, Dict, Any, Optional, Tuple
from long_term_memory import TfidfVectorStore
from bm25 import BM25Retriever


class HybridRetriever:
    """Drop-in replacement for TfidfVectorStore (same fit/add/query API)."""

    def __init__(self, ngram_range=(2, 4), max_features=4096, k_rrf: int = 60):
        self.tfidf = TfidfVectorStore(ngram_range=ngram_range, max_features=max_features)
        self.bm25 = BM25Retriever()
        self.k_rrf = k_rrf
        self._fitted = False

    def fit(self, corpus: List[str]):
        self.tfidf.fit(corpus)
        self._fitted = True

    def add(self, text: str, meta: Optional[Dict[str, Any]] = None):
        self.tfidf.add(text, meta)
        self.bm25.add(text, meta)

    @property
    def texts(self):
        return self.tfidf.texts

    @property
    def metas(self):
        return self.tfidf.metas

    def query(self, text: str, k: int = 5, threshold: float = 0.0):
        """Return top-k via RRF fusion of TF-IDF and BM25 rankings."""
        # fetch a wider window from each (top 3k) then fuse
        wide = max(k * 3, 10)
        tfidf_hits = self.tfidf.query(text, k=wide, threshold=0.0)
        bm25_hits = self.bm25.query(text, k=wide, threshold=0.0)

        # build rank maps (1-indexed)
        tfidf_rank: Dict[str, int] = {}
        for r, (t, _, _) in enumerate(tfidf_hits, start=1):
            tfidf_rank[t] = r
        bm25_rank: Dict[str, int] = {}
        for r, (t, _, _) in enumerate(bm25_hits, start=1):
            bm25_rank[t] = r

        # RRF score
        candidates = set(tfidf_rank.keys()) | set(bm25_rank.keys())
        scored = []
        for c in candidates:
            s = 0.0
            if c in tfidf_rank:
                s += 1.0 / (self.k_rrf + tfidf_rank[c])
            if c in bm25_rank:
                s += 1.0 / (self.k_rrf + bm25_rank[c])
            scored.append((c, s))
        scored.sort(key=lambda x: -x[1])

        # locate metas (prefer the TF-IDF store as source of truth)
        text_to_meta = {t: m for t, _, m in tfidf_hits}
        # supplement with BM25 hits
        for t, _, m in bm25_hits:
            text_to_meta.setdefault(t, m)

        out = []
        for txt, score in scored[:k]:
            if score < threshold:
                continue
            out.append((txt, score, text_to_meta.get(txt, {})))
        return out

    def __len__(self):
        return len(self.tfidf)


if __name__ == "__main__":
    h = HybridRetriever()
    corpus = ["我家wifi密码是Lin8520#家。", "今天天气很好。", "我妈妈生日是10月17号。",
              "最近喜欢看科幻电影。", "我女朋友是北京人。"]
    h.fit(corpus)
    for t in corpus:
        h.add(t, meta={"src": "user"})

    for q in ["wifi密码", "妈妈生日"]:
        hits = h.query(q, k=2)
        print(f"\nQ: {q}")
        for txt, s, m in hits:
            print(f"  rrf={s:.4f}  {txt[:30]}  meta={m}")
