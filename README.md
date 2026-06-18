# PJ-AG2: 带长期记忆的对话型 LLM Agent (期末)

Agent2-队伍2 · 陈默 24307130078 + 姬健坤 23301030045
github.com/momoli-li/pj-ag-two

## 项目结构

```
pj_ag2/
├── 中期 baseline (TF-IDF 向量库 + 三种记忆设置对照)
│   ├── config.py, short_term_memory.py, long_term_memory.py
│   ├── llm.py, agent.py
│   └── conversation_tests.jsonl   # 32 脚本 / 50 probes
│
├── final/                          # 期末进阶模块
│   ├── chroma_store.py             # E1: Chroma 真实向量后端
│   ├── bm25.py                     # E2a: BM25 retriever
│   ├── hybrid_retriever.py         # E2b: TF-IDF + BM25 + RRF
│   ├── write_defenses.py           # E3-E6: dedup / time_decay / source_conf
│   ├── conflict_detector.py        # 检索侧冲突检测 (数字/类别/否定)
│   └── slot_summary.py             # E7: 结构化槽位摘要 (17 槽位)
│
├── scripts/
│   ├── build_tests.py              # 生成测试集
│   ├── eval.py                     # 中期评测
│   ├── final_eval.py               # 期末综合评测 (M0-E8 配置)
│   ├── pollution_defense.py        # 污染防御 (P0-P6)
│   ├── variance_noise_latency.py   # 方差 / 噪声扫描 / 延迟
│   ├── case_study.py               # 单对话检索轨迹可视化
│   └── make_final_plots.py         # 生成 5 张图
│
├── final_results/                  # 所有期末实验输出 CSV
├── vis/                            # 5 张 matplotlib 图
└── case_studies/                   # 详细案例研究 .md
```

## 一键复现

```bash
pip install -r requirements.txt
bash run_all.sh
```

无网络依赖，所有结果确定性可重现。

## 核心结果

|  配置                                    | 多轮一致性 | 信息遗忘率 |
| --------------------------------------- | -------- | --------- |
| M0 中期 baseline (TF-IDF)                | 1.000    | 0.192     |
| **E2 Hybrid (TF-IDF + BM25 + RRF)**      | 0.958    | **0.038** |
| E8 全开 (Hybrid + 全防御 + 槽位摘要)        | 0.958    | 0.038     |

Hybrid 检索单项带来 80% 的遗忘率相对降低。
