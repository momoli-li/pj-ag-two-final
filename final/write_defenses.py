"""
Write-side defenses for long-term memory (期末进阶 #3).

Three independent layers, composable:

  1. dedup           — when a new write is near-duplicate of an existing entry
                       (cosine sim ≥ τ_dup), MERGE them: keep the newer text,
                       update the meta to reference the older turn.

  2. time_decay      — entries get a decay weight w_t = exp(-α·Δt) applied at
                       retrieval time, where Δt is "turns since written".
                       Older entries are still retrievable but ranked lower.

  3. source_conf     — entries store a `source` meta ∈ {'user_direct',
                       'agent_inferred', 'polluted', 'summary'}. At retrieval
                       time scores are multiplied by a source weight (e.g.
                       user_direct=1.0, agent_inferred=0.7, polluted=0.3,
                       summary=0.9). Polluted entries thus rank lowest.

Drop-in: wraps a base store (TfidfVectorStore / ChromaVectorStore /
HybridRetriever). Same query/add API.
"""

import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from typing import List, Dict, Any, Optional, Tuple

SOURCE_WEIGHTS = {
    "user_direct":    1.0,
    "agent_inferred": 0.7,
    "summary":        0.9,
    "polluted":       0.3,   # if we later mark an entry as suspicious
    None:             1.0,    # default
}


class DefendedStore:
    """
    Wrap a base store with optional defenses. Each defense can be turned on/off
    independently for ablation.
    """

    def __init__(self, base_store,
                 dedup: bool = False, dup_threshold: float = 0.85,
                 time_decay: bool = False, decay_alpha: float = 0.05,
                 source_conf: bool = False):
        self._base = base_store
        self.dedup = dedup
        self.dup_threshold = dup_threshold
        self.time_decay = time_decay
        self.decay_alpha = decay_alpha
        self.source_conf = source_conf

    @property
    def vectorizer(self):
        return getattr(self._base, "vectorizer", None)

    @property
    def texts(self):
        return self._base.texts

    @property
    def metas(self):
        return self._base.metas

    def fit(self, corpus):
        return self._base.fit(corpus)

    def _peek_top_sim(self, text: str) -> Tuple[float, int]:
        """Return (max_sim, idx) of the most similar existing doc, or (0, -1)."""
        if len(self._base) == 0:
            return 0.0, -1
        hits = self._base.query(text, k=1, threshold=0.0)
        if not hits:
            return 0.0, -1
        top_txt, sim, _ = hits[0]
        for i, t in enumerate(self._base.texts):
            if t == top_txt:
                return sim, i
        return sim, -1

    def add(self, text: str, meta: Optional[Dict[str, Any]] = None):
        meta = meta or {}
        if self.dedup and len(self._base) > 0:
            sim, idx = self._peek_top_sim(text)
            if sim >= self.dup_threshold and idx >= 0:
                # MERGE: overwrite the older entry's text with newer one,
                # bump its turn meta to the latest, mark merged=True
                old_meta = dict(self._base.metas[idx])
                old_meta.update(meta)
                old_meta["merged"] = True
                old_meta["merged_from_turn"] = old_meta.get("turn")
                self._base.texts[idx] = text
                self._base.metas[idx] = old_meta
                # rebuild the embedding row for this position
                if hasattr(self._base, "vectorizer") and self._base.vectorizer is not None:
                    new_vec = self._base.vectorizer.transform([text]).toarray().astype(np.float32)
                    if hasattr(self._base, "matrix") and self._base.matrix is not None:
                        self._base.matrix[idx] = new_vec[0]
                return
        # otherwise normal add
        self._base.add(text, meta)

    def _apply_post_weights(self, hits, current_turn: Optional[int]):
        """Apply time decay & source confidence to each hit's score."""
        out = []
        for txt, score, meta in hits:
            w = 1.0
            if self.time_decay and current_turn is not None and "turn" in (meta or {}):
                try:
                    dt = max(0, current_turn - int(meta["turn"]))
                except (TypeError, ValueError):
                    dt = 0
                w *= math.exp(-self.decay_alpha * dt)
            if self.source_conf:
                src = (meta or {}).get("source")
                if isinstance(src, str):
                    w *= SOURCE_WEIGHTS.get(src, 1.0)
            out.append((txt, float(score) * w, meta))
        return sorted(out, key=lambda x: -x[1])

    def query(self, text: str, k: int = 5, threshold: float = 0.0,
              current_turn: Optional[int] = None):
        """If any defense reweights, fetch wider then re-rank+truncate."""
        wider = k * 3 if (self.time_decay or self.source_conf) else k
        hits = self._base.query(text, k=wider, threshold=0.0)
        if self.time_decay or self.source_conf:
            hits = self._apply_post_weights(hits, current_turn)
        # apply threshold
        out = [(t, s, m) for (t, s, m) in hits if s >= threshold]
        return out[:k]

    def __len__(self):
        return len(self._base)
