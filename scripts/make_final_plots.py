"""Generate final report plots."""
import os, sys, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "final_results"
VIS = ROOT / "vis"
VIS.mkdir(exist_ok=True)

# Find a Chinese-capable font
for fname in ["Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]:
    try:
        font_manager.findfont(fname, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [fname]
        print(f"Using font: {fname}")
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --- Fig 1: headline enhancements bar chart ---
headline = read_csv(RES / "headline_enhancements.csv")
configs_short = ["M0\nbaseline", "E1\nChroma", "E2\nHybrid", "E3\ndedup",
                 "E4\ntime_decay", "E5\nsource_conf", "E6\n全防御",
                 "E7\nslot_summary", "E8\n全开"]
forgs = [float(r["forgetting_short_long"]) for r in headline]
colors = ["#888888", "#888888", "#4CAF50", "#888888", "#888888", "#888888",
          "#888888", "#888888", "#4CAF50"]

fig, ax = plt.subplots(figsize=(11, 4.5))
bars = ax.bar(range(len(configs_short)), forgs, color=colors,
              edgecolor="#333", linewidth=0.5)
for i, v in enumerate(forgs):
    ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(range(len(configs_short)))
ax.set_xticklabels(configs_short, fontsize=9)
ax.set_ylabel("信息遗忘率 ↓ (越低越好)", fontsize=11)
ax.set_title("期末进阶模块 vs 中期 baseline (32 scripts × 26 forgetting probes)",
             fontsize=12, pad=15)
ax.axhline(y=forgs[0], color="#888888", linestyle="--", alpha=0.5,
           label=f"中期 baseline ({forgs[0]:.3f})")
ax.set_ylim(0, max(forgs) * 1.25)
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="upper right", fontsize=10)
fig.tight_layout()
fig.savefig(VIS / "fig1_headline.png", dpi=150)
plt.close(fig)
print("  fig1_headline.png")

# --- Fig 2: LLM noise sensitivity ---
noise = read_csv(RES / "noise_sensitivity.csv")
xs = [float(r["extract_noise"]) for r in noise]
baseline_y = [float(r["baseline_tfidf_forgetting"]) for r in noise]
hybrid_y = [float(r["hybrid_forgetting"]) for r in noise]
full_y = [float(r["full_forgetting"]) for r in noise]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(xs, baseline_y, marker="o", linewidth=2, markersize=8,
        label="中期 baseline (TF-IDF)", color="#DD8452")
ax.plot(xs, hybrid_y, marker="s", linewidth=2, markersize=8,
        label="E2: Hybrid (TF-IDF + BM25 + RRF)", color="#4CAF50")
ax.plot(xs, full_y, marker="^", linewidth=2, markersize=8,
        label="E8: 全开 (Hybrid + 防御 + 摘要)", color="#3F51B5", linestyle="--")
ax.set_xlabel("LLM extract_noise (信息抽取失败率)", fontsize=11)
ax.set_ylabel("信息遗忘率 ↓", fontsize=11)
ax.set_title("LLM 噪声敏感性：Hybrid 在所有噪声水平下都优于 baseline", fontsize=12, pad=15)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=10)
ax.set_ylim(-0.02, 0.55)
fig.tight_layout()
fig.savefig(VIS / "fig2_noise_sensitivity.png", dpi=150)
plt.close(fig)
print("  fig2_noise_sensitivity.png")

# --- Fig 3: pollution defense stacked bars ---
poll = read_csv(RES / "pollution_defense.csv")
configs = [r["config"] for r in poll]
verdicts = ["robust", "polluted", "no_recall", "ambiguous", "clarified"]
colors_v = {
    "robust":    "#4CAF50",
    "polluted":  "#C44E52",
    "no_recall": "#937860",
    "ambiguous": "#8172B2",
    "clarified": "#5DADE2",
}

fig, ax = plt.subplots(figsize=(11, 5.5))
xs = list(range(len(configs)))
bottom = [0] * len(configs)
for v in verdicts:
    vals = [int(r[v]) for r in poll]
    ax.bar(xs, vals, bottom=bottom, label=v, color=colors_v[v],
           edgecolor="white", linewidth=0.5)
    for i, val in enumerate(vals):
        if val > 0:
            ax.text(xs[i], bottom[i] + val/2, str(val), ha="center",
                    va="center", color="white", fontsize=10, fontweight="bold")
    bottom = [b + v for b, v in zip(bottom, vals)]

short_labels = ["P0\n无防御", "P1\n+source_conf", "P2\n+dedup", "P3\n+time_decay",
                "P4\n全防御", "P5\n+冲突检测", "P6\n+Hybrid"]
ax.set_xticks(xs)
ax.set_xticklabels(short_labels, fontsize=9)
ax.set_ylabel("案例数 (共 12)", fontsize=11)
ax.set_title("污染防御对照实验 (harsh 设置：真实事实仅在短期 N=1)",
             fontsize=12, pad=15)
ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
ax.set_ylim(0, 14)
fig.tight_layout()
fig.savefig(VIS / "fig3_pollution_defense.png", dpi=150)
plt.close(fig)
print("  fig3_pollution_defense.png")

# --- Fig 4: latency comparison ---
lat = read_csv(RES / "latency.csv")
names = [r["backend"] for r in lat]
short_names = ["TF-IDF\n(自写)", "Hybrid\n(TF-IDF + BM25)", "Chroma"]
fit_ms = [float(r["fit_add_ms"]) for r in lat]
query_ms = [float(r["query_ms_avg"]) for r in lat]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ax = axes[0]
bars = ax.bar(range(len(short_names)), fit_ms, color=["#4C72B0", "#4CAF50", "#DD8452"],
              edgecolor="#333", linewidth=0.5)
for i, v in enumerate(fit_ms):
    ax.text(i, v + max(fit_ms)*0.02, f"{v:.0f}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(range(len(short_names)))
ax.set_xticklabels(short_names, fontsize=10)
ax.set_ylabel("时间 (ms)", fontsize=11)
ax.set_title("索引构建 + 200 文档写入耗时", fontsize=11, pad=10)
ax.set_ylim(0, max(fit_ms) * 1.18)
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
bars = ax.bar(range(len(short_names)), query_ms, color=["#4C72B0", "#4CAF50", "#DD8452"],
              edgecolor="#333", linewidth=0.5)
for i, v in enumerate(query_ms):
    ax.text(i, v + max(query_ms)*0.02, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(range(len(short_names)))
ax.set_xticklabels(short_names, fontsize=10)
ax.set_ylabel("时间 (ms)", fontsize=11)
ax.set_title("单次查询平均耗时 (20 次平均)", fontsize=11, pad=10)
ax.set_ylim(0, max(query_ms) * 1.18)
ax.grid(axis="y", alpha=0.3)

fig.suptitle("延迟基准：Hybrid 是 TF-IDF 的 ~3.6×，Chroma 的 ~1.6×", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(VIS / "fig4_latency.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  fig4_latency.png")

# --- Fig 5: variance error bars ---
var = read_csv(RES / "variance.csv")
labels = ["baseline\n(TF-IDF)", "Hybrid only", "Hybrid +\n全防御 + 摘要"]
forg_means = [float(r["forgetting_mean"]) for r in var]
forg_stds = [float(r["forgetting_std"]) for r in var]

fig, ax = plt.subplots(figsize=(7.5, 4.5))
xs = range(len(labels))
bars = ax.bar(xs, forg_means, yerr=forg_stds, capsize=8,
              color=["#DD8452", "#4CAF50", "#3F51B5"],
              edgecolor="#333", linewidth=0.5,
              error_kw={"linewidth": 1.5, "ecolor": "#333"})
for i, (m, s) in enumerate(zip(forg_means, forg_stds)):
    ax.text(i, m + s + 0.015, f"{m:.3f}±{s:.3f}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(list(xs))
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("信息遗忘率 ↓", fontsize=11)
ax.set_title("方差分析 (5 seeds: 42, 1337, 2024, 9, 21)", fontsize=12, pad=10)
ax.set_ylim(0, max(forg_means) + max(forg_stds) + 0.05)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(VIS / "fig5_variance.png", dpi=150)
plt.close(fig)
print("  fig5_variance.png")

print(f"\nAll plots written to {VIS}")
