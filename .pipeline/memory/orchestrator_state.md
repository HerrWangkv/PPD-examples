# Orchestrator State
_最后同步：2026-06-13_

## 全局进度看板

| 阶段 | 状态 | 备注 |
|------|------|------|
| Survey / Ideation | ✅ done | |
| Experiment | ✅ done | v7 全量正式评估完成；三 benchmark 全指标定稿 |
| Results/Slides | ✅ done | Results.md 定稿；ss26 已推送 (48c6fdc)；eval_video 已推送 (3910ba4) |
| Publication 正文 | 🚀 待启动 | 理论实验 E1–E5 + Method/Experiment 写作 |

## 当前活跃任务

| 任务 | 状态 |
|------|------|
| 主仓库 commit（gitignore + 评估脚本 + 日志 + Results.md + figures + submodule 指针）| 🔄 进行中（被 sync 打断）|
| 理论实验 E1 子带域判别性 / E2 信息预算重分析 | ⏳ P0 |
| Method/Experiment 正文写作 | ⏳ P0 |

## 最近完成任务（最近5条）

| Task | Date | Key result |
|------|------|-----------|
| v7 全量正式评估 + per-variant 日志体系 | 06-13 | 54 logs；summarize_{vkitti,nucarla}_eval.py 一键复现 |
| nuCarla paper/full table 定稿 | 06-13 | ours 六列全冠（CLIP+ADE×4+FDE）；MS 10/10 补齐 |
| Ablation 1 三观察 + 4-panel 图 | 06-13 | KID 混杂论证（r24 反例 + 单调性）vs CLIP 解耦 |
| Slides 全面更新推送 | 06-13 | S11–S18 + ψ-PD 命名（ss26 48c6fdc）|
| eval_video 推送 | 06-13 | sFID/sKID + --tag + MS 结果（3910ba4）|

## 决策点

无待决 — v7 口径、表格结构、slides 命名均已拍板。

## 下一步建议

1. 完成主仓库 commit（继续被打断的操作）
2. 启动 E1（CPU）+ Method 骨架写作 — 唯一剩余主线
