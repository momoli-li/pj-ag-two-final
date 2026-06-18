# Changelog (期末 vs 中期)

## 修复

### LLM RNG bug (重要)

中期评测中 DeterministicLLM 在每条脚本前都以同一种子重新初始化，
导致 extract_noise 的随机事件序列在所有脚本上重复。

**影响**：中期报告的 baseline forgetting=0.115 实际上是噪声未被充分激活的结果。
**修复**：在 `scripts/final_eval.py` 中，LLM 实例改为按"模式"创建一次，RNG 在
所有脚本之间持续推进。
**修复后**：noise=0.05 下 baseline forgetting 修正为 0.192。期末报告中所有数字
基于修复后口径。

## 新增模块

- E1: final/chroma_store.py        — Chroma 真实后端
- E2: final/bm25.py + hybrid_retriever.py — Hybrid 检索 (RRF)
- E3-E6: final/write_defenses.py   — 写入侧三层防御
- E7: final/slot_summary.py        — 结构化槽位摘要
- 进阶: final/conflict_detector.py — 检索侧冲突检测

## 新增实验

- scripts/final_eval.py             — 9 个进阶配置对照 (M0-E8)
- scripts/pollution_defense.py      — 7 种防御组合 × 12 污染案例
- scripts/variance_noise_latency.py — 5 seeds 方差 / 6 档噪声 / 3 后端延迟
- scripts/case_study.py             — S09 (wifi 召回) 详细轨迹
