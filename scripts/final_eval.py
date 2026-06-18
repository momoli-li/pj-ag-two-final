"""
Final unified evaluator (期末).

Supports all期末 enhancements via a flag dict:
  backend     ∈ {'tfidf', 'chroma', 'hybrid'}
  defenses    : dict {dedup, time_decay, source_conf}  -> bool
  conflict    : bool  — if True, agent triggers clarification on conflict
  slot_summary: bool  — if True, periodically inject slot-summary
"""

import os, sys, json, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "final"))

import numpy as np
from typing import List, Dict, Any
from config import DEFAULT_CONFIG
from short_term_memory import ShortTermMemory
from long_term_memory import TfidfVectorStore, LongTermMemory
from llm import DeterministicLLM
from agent import Agent

from chroma_store import ChromaVectorStore
from hybrid_retriever import HybridRetriever
from write_defenses import DefendedStore
from conflict_detector import detect_conflicts, format_clarification
from slot_summary import slot_summary

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "conversation_tests.jsonl"
RESULTS = ROOT / "final_results"
RESULTS.mkdir(exist_ok=True)


def load_tests():
    with open(TESTS, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def keyword_hit(response: str, keywords: List[str]) -> bool:
    if not keywords:
        return False
    r = response.lower()
    return any(kw.lower() in r for kw in keywords if kw)


def build_base_store(backend: str, ngram_range, max_features, corpus_seed):
    if backend == "tfidf":
        s = TfidfVectorStore(ngram_range=ngram_range, max_features=max_features)
    elif backend == "hybrid":
        s = HybridRetriever(ngram_range=ngram_range, max_features=max_features)
    elif backend == "chroma":
        s = ChromaVectorStore(ngram_range=ngram_range, max_features=max_features)
    else:
        raise ValueError(f"unknown backend {backend}")
    s.fit(corpus_seed)
    return s


def wrap_with_defenses(store, defenses: Dict[str, bool]):
    if not any(defenses.values()):
        return store
    return DefendedStore(
        store,
        dedup=defenses.get("dedup", False),
        time_decay=defenses.get("time_decay", False),
        source_conf=defenses.get("source_conf", False),
    )


class EnhancedAgent(Agent):
    """Agent that supports conflict detection + slot-summary periodic inject."""

    def __init__(self, mode, llm, short_term=None, long_term=None,
                 enable_conflict=False, enable_slot_summary=False,
                 summary_every: int = 8):
        super().__init__(mode, llm, short_term, long_term)
        self.enable_conflict = enable_conflict
        self.enable_slot_summary = enable_slot_summary
        self.summary_every = summary_every
        self._user_turn_count = 0
        self._user_messages = []

    def observe_user(self, text: str):
        super().observe_user(text)
        self._user_turn_count += 1
        self._user_messages.append(text)
        if self.enable_slot_summary and self.long_term is not None:
            if self._user_turn_count % self.summary_every == 0 and self._user_turn_count >= self.summary_every:
                summ = slot_summary(self._user_messages)
                if summ:
                    self.long_term.write(summ, meta={"source": "summary",
                                                     "turn": self.turn_index})

    def respond(self, question: str, ground_truth_keywords=None):
        if self.mode != "short_long" or self.long_term is None:
            return super().respond(question, ground_truth_keywords)

        # retrieve and possibly run conflict detection BEFORE LLM call
        hits = self.long_term.retrieve(question)
        long_ctx = self.long_term.render(question)
        if self.enable_conflict and hits:
            confs = detect_conflicts(hits)
            if confs:
                # the agent's response is a clarification — does NOT directly use keywords
                # for grading purposes we still also include the keyword extraction the
                # mock LLM would have done, so a downstream check can decide.
                clarification = format_clarification(confs[0])
                # combine: mention clarification, then attempt extraction
                short_ctx = self.short_term.render() if self.short_term else ""
                resp = self.llm.generate(question=question,
                                          short_term_context=short_ctx,
                                          long_term_context=long_ctx,
                                          ground_truth_keywords=ground_truth_keywords)
                resp = clarification + " " + resp
                return {"response": resp, "short_ctx": short_ctx, "long_ctx": long_ctx,
                        "conflict": confs}

        short_ctx = self.short_term.render() if self.short_term else ""
        resp = self.llm.generate(question=question,
                                  short_term_context=short_ctx,
                                  long_term_context=long_ctx,
                                  ground_truth_keywords=ground_truth_keywords)
        return {"response": resp, "short_ctx": short_ctx, "long_ctx": long_ctx}


def run_one_config(cfg: Dict, scripts: List[Dict], seed: int = 42) -> List[Dict]:
    """Returns aggregate summary across 3 modes for this config."""
    backend     = cfg.get("backend", "tfidf")
    defenses    = cfg.get("defenses", {})
    enable_conf = cfg.get("conflict", False)
    enable_slot = cfg.get("slot_summary", False)
    extract_noise = cfg.get("extract_noise", DEFAULT_CONFIG["llm_extract_noise"])

    base_cfg = DEFAULT_CONFIG
    all_rows = {}

    for mode in ("no_memory", "short_only", "short_long"):
        bucket_cons, bucket_forg = [], []
        # ONE llm per mode so its RNG advances continuously across scripts
        # (otherwise the same seed produces the same first random() value for
        #  every script, never exercising the noise model)
        llm = DeterministicLLM(seed=seed,
                               extract_noise=extract_noise,
                               no_context_correct=base_cfg["llm_no_context_correct"])
        for s in scripts:
            short_term = ShortTermMemory(window_n=base_cfg["short_window_n"]) if mode != "no_memory" else None
            long_term = None
            if mode == "short_long":
                base = build_base_store(backend, base_cfg["tfidf_ngram_range"],
                                        base_cfg["tfidf_max_features"],
                                        [t["text"] for t in s["turns"]])
                base = wrap_with_defenses(base, defenses)
                threshold = 0.0 if backend == "hybrid" else base_cfg["retrieval_threshold"]
                long_term = LongTermMemory(base, top_k=base_cfg["top_k"],
                                           threshold=threshold)

            agent = EnhancedAgent(mode, llm, short_term, long_term,
                                   enable_conflict=enable_conf,
                                   enable_slot_summary=enable_slot)
            for i, turn in enumerate(s["turns"], start=1):
                if turn["role"] == "user":
                    agent.observe_user(turn["text"])
                else:
                    agent.observe_agent(turn["text"])
                for j, probe in enumerate(s["probes"]):
                    if probe["at_turn"] == i:
                        out = agent.respond(question=probe["question"],
                                            ground_truth_keywords=probe["ground_truth_keywords"])
                        is_hit = keyword_hit(out["response"], probe["ground_truth_keywords"])
                        row = {
                            "script_id": s["id"], "scenario": s["scenario"],
                            "probe_type": probe["type"], "response": out["response"],
                            "judgment": int(is_hit if probe["type"] == "consistency" else (not is_hit)),
                        }
                        if probe["type"] == "consistency":
                            bucket_cons.append(row)
                        else:
                            bucket_forg.append(row)
        all_rows[mode] = {"consistency": bucket_cons, "forgetting": bucket_forg}

    summary = []
    for mode, b in all_rows.items():
        cons, forg = b["consistency"], b["forgetting"]
        per_script = {}
        for r in cons:
            per_script.setdefault(r["script_id"], []).append(r["judgment"])
        consistency = (sum(sum(v)/len(v) for v in per_script.values()) / len(per_script)) if per_script else float("nan")
        forgetting = (sum(r["judgment"] for r in forg) / len(forg)) if forg else float("nan")
        summary.append({"mode": mode, "consistency": consistency, "forgetting": forgetting})
    return summary


def headline_table():
    """Compare baseline (midterm) vs each期末 enhancement individually, then 全开."""
    scripts = load_tests()
    configs = [
        ("M0 中期 baseline (TF-IDF)",          {"backend": "tfidf"}),
        ("E1 Chroma 后端",                      {"backend": "chroma"}),
        ("E2 Hybrid (TF-IDF+BM25 RRF)",        {"backend": "hybrid"}),
        ("E3 防御: dedup",                      {"backend": "tfidf", "defenses": {"dedup": True}}),
        ("E4 防御: time_decay",                  {"backend": "tfidf", "defenses": {"time_decay": True}}),
        ("E5 防御: source_conf",                 {"backend": "tfidf", "defenses": {"source_conf": True}}),
        ("E6 防御: 全开 (dedup+decay+conf)",     {"backend": "tfidf",
                                                "defenses": {"dedup": True, "time_decay": True, "source_conf": True}}),
        ("E7 槽位摘要注入",                       {"backend": "tfidf", "slot_summary": True}),
        ("E8 Hybrid + 全防御 + 槽位摘要 [全开]", {"backend": "hybrid",
                                                "defenses": {"dedup": True, "time_decay": True, "source_conf": True},
                                                "slot_summary": True}),
    ]

    rows = []
    print(f"\n{'config':<48} {'cons':>6} {'forg':>6}  (short_long only)")
    for name, cfg in configs:
        summary = run_one_config(cfg, scripts, seed=42)
        sl = next(s for s in summary if s["mode"] == "short_long")
        rows.append({"config": name,
                     "consistency_short_long": round(sl["consistency"], 4),
                     "forgetting_short_long":  round(sl["forgetting"], 4)})
        print(f"{name:<48} {sl['consistency']:>6.3f} {sl['forgetting']:>6.3f}")

    with open(RESULTS / "headline_enhancements.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "consistency_short_long", "forgetting_short_long"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWritten to {RESULTS / 'headline_enhancements.csv'}")
    return rows


if __name__ == "__main__":
    headline_table()
