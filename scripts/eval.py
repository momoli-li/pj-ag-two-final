"""Main evaluation."""
import os, sys, json, csv
from pathlib import Path
from typing import List, Dict
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_CONFIG
from short_term_memory import ShortTermMemory
from long_term_memory import TfidfVectorStore, LongTermMemory
from llm import DeterministicLLM
from agent import Agent

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "conversation_tests.jsonl"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def load_tests():
    with open(TESTS, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def keyword_hit(response, keywords):
    if not keywords: return False
    r = response.lower()
    return any(kw.lower() in r for kw in keywords if kw)


def run_script(script, mode, cfg, llm, long_term=None):
    short_term = ShortTermMemory(window_n=cfg["short_window_n"]) if mode != "no_memory" else None
    if mode == "short_long":
        store = TfidfVectorStore(ngram_range=cfg["tfidf_ngram_range"], max_features=cfg["tfidf_max_features"])
        store.fit([t["text"] for t in script["turns"]])
        long_term = LongTermMemory(store, top_k=cfg["top_k"], threshold=cfg["retrieval_threshold"])
    else:
        long_term = None

    agent = Agent(mode=mode, llm=llm, short_term=short_term, long_term=long_term)
    llm.reset()

    consistency_rows, forgetting_rows = [], []

    for i, turn in enumerate(script["turns"], start=1):
        if turn["role"] == "user":
            agent.observe_user(turn["text"])
        else:
            agent.observe_agent(turn["text"])
        for j, probe in enumerate(script["probes"]):
            if probe["at_turn"] == i:
                out = agent.respond(question=probe["question"],
                                    ground_truth_keywords=probe["ground_truth_keywords"])
                is_hit = keyword_hit(out["response"], probe["ground_truth_keywords"])
                row = {
                    "script_id": script["id"], "scenario": script["scenario"],
                    "probe_idx": j, "probe_type": probe["type"],
                    "question": probe["question"], "ground_truth": probe["ground_truth"],
                    "response": out["response"],
                    "judgment": int(is_hit if probe["type"] == "consistency" else (not is_hit)),
                }
                if probe["type"] == "consistency":
                    consistency_rows.append(row)
                else:
                    forgetting_rows.append(row)

    return {"consistency": consistency_rows, "forgetting": forgetting_rows}


def aggregate(all_rows):
    summary = []
    for mode, buckets in all_rows.items():
        cons, forg = buckets["consistency"], buckets["forgetting"]
        per_script = {}
        for r in cons:
            per_script.setdefault(r["script_id"], []).append(r["judgment"])
        if per_script:
            sm = [sum(v) / len(v) for v in per_script.values()]
            consistency = sum(sm) / len(sm)
        else:
            consistency = float("nan")
        forgetting = sum(r["judgment"] for r in forg) / len(forg) if forg else float("nan")
        summary.append({
            "setting": mode, "consistency": consistency, "forgetting": forgetting,
            "n_consistency_probes": len(cons), "n_forgetting_probes": len(forg),
            "n_scripts_with_consistency": len(per_script),
        })
    return summary


def main(cfg=None):
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    scripts = load_tests()
    llm = DeterministicLLM(seed=cfg["llm_seed"],
        extract_noise=cfg["llm_extract_noise"], no_context_correct=cfg["llm_no_context_correct"])

    all_rows = {}
    for mode in ("no_memory", "short_only", "short_long"):
        bc, bf = [], []
        for s in scripts:
            res = run_script(s, mode, cfg, llm)
            bc.extend(res["consistency"]); bf.extend(res["forgetting"])
        all_rows[mode] = {"consistency": bc, "forgetting": bf}

    cons_fields = ["setting","script_id","scenario","probe_idx","question","ground_truth","response","judgment"]
    forg_fields = cons_fields

    with open(RESULTS/"consistency_detail.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cons_fields); w.writeheader()
        for mode, b in all_rows.items():
            for r in b["consistency"]:
                w.writerow({"setting": mode, **{k: r[k] for k in cons_fields if k != "setting"}})
    with open(RESULTS/"forgetting_detail.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=forg_fields); w.writeheader()
        for mode, b in all_rows.items():
            for r in b["forgetting"]:
                w.writerow({"setting": mode, **{k: r[k] for k in forg_fields if k != "setting"}})

    annot_fields = ["setting","script_id","scenario","probe_type","question","ground_truth","response","判定"]
    with open(RESULTS/"annotation_table.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=annot_fields); w.writeheader()
        for mode, b in all_rows.items():
            for r in b["consistency"]:
                w.writerow({"setting": mode, "script_id": r["script_id"], "scenario": r["scenario"],
                    "probe_type": "consistency", "question": r["question"],
                    "ground_truth": r["ground_truth"], "response": r["response"], "判定": r["judgment"]})
            for r in b["forgetting"]:
                w.writerow({"setting": mode, "script_id": r["script_id"], "scenario": r["scenario"],
                    "probe_type": "forgetting", "question": r["question"],
                    "ground_truth": r["ground_truth"], "response": r["response"], "判定": 1 - r["judgment"]})

    summary = aggregate(all_rows)
    with open(RESULTS/"preliminary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["setting","consistency","forgetting","n_consistency_probes","n_forgetting_probes","n_scripts_with_consistency"])
        w.writeheader()
        for row in summary: w.writerow(row)

    print(f"{'setting':<14} {'consistency':>12} {'forgetting':>12}")
    for row in summary:
        print(f"{row['setting']:<14} {row['consistency']:>11.3f}  {row['forgetting']:>11.3f}")
    return summary


if __name__ == "__main__":
    main()
