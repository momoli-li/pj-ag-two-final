#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "[1/7] 生成对话测试集"
python3 scripts/build_tests.py

echo "[2/7] 中期 baseline 评测"
python3 scripts/eval.py

echo "[3/7] 期末综合评测 (9 个配置: M0-E8)"
python3 scripts/final_eval.py

echo "[4/7] 污染防御实验 (P0-P6)"
python3 scripts/pollution_defense.py

echo "[5/7] 方差 + 噪声 + 延迟基准"
python3 scripts/variance_noise_latency.py

echo "[6/7] 单对话案例研究"
python3 scripts/case_study.py

echo "[7/7] 生成可视化"
python3 scripts/make_final_plots.py

echo "完成。结果在 final_results/、vis/、case_studies/。"
