"""
Variance / noise / latency analyses (期末).
  A) variance:    run baseline + best config across 5 seeds, report mean ± std
  B) noise:       sweep LLM extract_noise ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.5}
  C) latency:     wall-clock per-query for TF-IDF / Chroma / Hybrid backends
"""

import os, sys, csv, time, statistics, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "final"))

from typing import List, Dict
from config import DEFAULT_CONFIG
from short_term_memory import ShortTermMemory
from long_term_memory import TfidfVectorStore, LongTermMemory
from llm import DeterministicLLM
from agent import Agent
from chroma_store import ChromaVectorStore
from hybrid_retriever import HybridRetriever

from scripts.final_eval import run_one_config, load_tests

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "final_results"
RESULTS.mkdir(exist_ok=True)


# --------- A) variance analysis ---------
def variance_analysis():
    print("\n[A] Variance over 5 seeds")
    scripts = load_tests()
    configs = [
        ("baseline (TF-IDF, no defenses)", {"backend": "tfidf"}),
        ("Hybrid only",                     {"backend": "hybrid"}),
        ("Hybrid + 全防御 + 槽位摘要",       {"backend": "hybrid",
                                            "defenses": {"dedup": True, "time_decay": True, "source_conf": True},
                                            "slot_summary": True}),
    ]
    seeds = [42, 1337, 2024, 9, 21]

    rows = []
    print(f"{'config':<40} {'cons_mean±std':>20} {'forg_mean±std':>20}")
    for name, cfg in configs:
        cons_vals, forg_vals = [], []
        for seed in seeds:
            summary = run_one_config(cfg, scripts, seed=seed)
            sl = next(s for s in summary if s["mode"] == "short_long")
            cons_vals.append(sl["consistency"])
            forg_vals.append(sl["forgetting"])
        cm, cs = statistics.mean(cons_vals), statistics.stdev(cons_vals) if len(cons_vals) > 1 else 0
        fm, fs = statistics.mean(forg_vals), statistics.stdev(forg_vals) if len(forg_vals) > 1 else 0
        rows.append({"config": name,
                     "consistency_mean": round(cm, 4), "consistency_std": round(cs, 4),
                     "forgetting_mean":  round(fm, 4), "forgetting_std":  round(fs, 4),
                     "n_seeds": len(seeds)})
        print(f"{name:<40} {cm:.3f}±{cs:.3f}      {fm:.3f}±{fs:.3f}")

    with open(RESULTS / "variance.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "consistency_mean", "consistency_std",
                                          "forgetting_mean", "forgetting_std", "n_seeds"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  → {RESULTS / 'variance.csv'}")


# --------- B) LLM noise sensitivity ---------
def noise_sensitivity():
    print("\n[B] LLM extract_noise sensitivity")
    scripts = load_tests()
    rows = []
    print(f"{'noise':>6} {'baseline_forg':>14} {'hybrid_forg':>13} {'full_forg':>11}")
    for noise in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
        b_summary = run_one_config({"backend": "tfidf", "extract_noise": noise}, scripts, seed=42)
        h_summary = run_one_config({"backend": "hybrid", "extract_noise": noise}, scripts, seed=42)
        f_summary = run_one_config({"backend": "hybrid", "extract_noise": noise,
                                     "defenses": {"dedup": True, "time_decay": True, "source_conf": True},
                                     "slot_summary": True}, scripts, seed=42)
        b_f = next(s for s in b_summary if s["mode"] == "short_long")["forgetting"]
        h_f = next(s for s in h_summary if s["mode"] == "short_long")["forgetting"]
        f_f = next(s for s in f_summary if s["mode"] == "short_long")["forgetting"]
        rows.append({"extract_noise": noise,
                     "baseline_tfidf_forgetting": round(b_f, 4),
                     "hybrid_forgetting":          round(h_f, 4),
                     "full_forgetting":            round(f_f, 4)})
        print(f"{noise:>6.2f} {b_f:>14.3f} {h_f:>13.3f} {f_f:>11.3f}")

    with open(RESULTS / "noise_sensitivity.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["extract_noise", "baseline_tfidf_forgetting",
                                          "hybrid_forgetting", "full_forgetting"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  → {RESULTS / 'noise_sensitivity.csv'}")


# --------- C) latency benchmark ---------
def latency_benchmark():
    print("\n[C] Latency benchmark (ms / query)")
    docs = [f"用户陈述：今天我做了第{i}件事，相关的数字是{i*7+13}。" for i in range(200)]
    queries = [f"我之前提到的第{i*5}件事是什么？" for i in range(20)]

    backends = {
        "TF-IDF (自写)":  lambda: TfidfVectorStore(),
        "Hybrid (TF-IDF + BM25 + RRF)": lambda: HybridRetriever(),
    }
    # Chroma is slow on first query (HNSW init), benchmark separately
    backends["Chroma"] = lambda: ChromaVectorStore()

    rows = []
    print(f"{'backend':<40} {'n_docs':>8} {'fit+add(ms)':>13} {'query(ms,avg)':>16}")
    for name, ctor in backends.items():
        store = ctor()
        t0 = time.perf_counter()
        store.fit(docs)
        for d in docs:
            store.add(d, meta={"src": "user"})
        t_fit = (time.perf_counter() - t0) * 1000

        # warm up
        store.query(queries[0], k=5)
        t1 = time.perf_counter()
        for q in queries:
            store.query(q, k=5)
        t_q = (time.perf_counter() - t1) * 1000 / len(queries)

        rows.append({"backend": name, "n_docs": len(docs),
                     "fit_add_ms": round(t_fit, 1),
                     "query_ms_avg": round(t_q, 3)})
        print(f"{name:<40} {len(docs):>8} {t_fit:>11.1f}   {t_q:>14.3f}")
        del store
        gc.collect()

    with open(RESULTS / "latency.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["backend", "n_docs", "fit_add_ms", "query_ms_avg"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  → {RESULTS / 'latency.csv'}")


if __name__ == "__main__":
    variance_analysis()
    noise_sensitivity()
    latency_benchmark()
