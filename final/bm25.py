"""
BM25 retriever (期末进阶 #2a).

A pure-Python Okapi BM25 implementation over character bigrams (works for
Chinese without a word tokenizer). Lower-level companion to the TF-IDF
embedding so we can ensemble them via RRF.
"""

import math
from collections import Counter
from typing import List, Tuple, Dict, Any, Optional


def char_bigrams(text: str) -> List[str]:
    """Character bigrams for Chinese-friendly tokenization."""
    text = text.strip()
    if len(text) < 2:
        return [text] if text else []
    return [text[i:i + 2] for i in range(len(text) - 1)]


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: List[str] = []
        self.metas: List[Dict[str, Any]] = []
        self.tokenized: List[List[str]] = []
        self.df: Counter = Counter()       # document frequency per token
        self.doc_lens: List[int] = []
        self.avg_dl: float = 0.0

    def add(self, text: str, meta: Optional[Dict[str, Any]] = None):
        toks = char_bigrams(text)
        self.docs.append(text)
        self.metas.append(meta or {})
        self.tokenized.append(toks)
        self.doc_lens.append(len(toks))
        seen = set(toks)
        for t in seen:
            self.df[t] += 1
        n = len(self.doc_lens)
        if n:
            self.avg_dl = sum(self.doc_lens) / n

    def _idf(self, tok: str, N: int) -> float:
        # BM25+ smoothing: idf = log((N - df + 0.5) / (df + 0.5) + 1)
        df = self.df.get(tok, 0)
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def query(self, text: str, k: int = 5, threshold: float = 0.0):
        N = len(self.docs)
        if N == 0:
            return []
        q_toks = char_bigrams(text)
        if not q_toks:
            return []
        scores = []
        for i, doc_toks in enumerate(self.tokenized):
            dl = self.doc_lens[i]
            if dl == 0:
                scores.append(0.0)
                continue
            doc_tf = Counter(doc_toks)
            s = 0.0
            for q in q_toks:
                tf = doc_tf.get(q, 0)
                if tf == 0:
                    continue
                idf = self._idf(q, N)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                s += idf * (tf * (self.k1 + 1)) / denom
            scores.append(s)
        # rank
        order = sorted(range(N), key=lambda i: -scores[i])
        out = []
        for i in order[:k]:
            if scores[i] < threshold:
                continue
            out.append((self.docs[i], scores[i], self.metas[i]))
        return out

    def __len__(self):
        return len(self.docs)


if __name__ == "__main__":
    r = BM25Retriever()
    r.add("我家wifi密码是Lin8520#家。", meta={"turn": 1})
    r.add("今天天气很好。", meta={"turn": 3})
    r.add("我妈妈生日是10月17号。", meta={"turn": 5})
    r.add("最近喜欢看科幻电影。", meta={"turn": 7})

    for q in ["wifi密码", "妈妈生日", "电影"]:
        hits = r.query(q, k=2)
        print(f"\nQ: {q}")
        for txt, s, m in hits:
            print(f"  bm25={s:.3f}  {txt[:30]}  meta={m}")
