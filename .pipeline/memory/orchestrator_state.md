# Orchestrator State
_最后同步：2026-06-14_

## 全局进度看板

| 阶段 | 状态 | 备注 |
|------|------|------|
| Survey / Ideation | ✅ done | |
| Experiment | ✅ done | v7 全量正式评估完成；三 benchmark 全指标定稿 |
| Results / Slides | ✅ done | Results.md 定稿；ss26 ea88f82；PPD-examples c60f5b3 |
| Publication 正文 | 🚀 待启动 | 唯一剩余主线 |

## 当前活跃任务

| 任务 ID | 标题 | 状态 |
|--------|------|------|
| publication | 论文正文写作（Method + Experiments） | 🔄 in-progress（未实质启动）|
| wan_video_training | Wan video drop_ll LoRA 训练 | 🔄 状态未确认 |

## 最近完成任务（最近 5 条）

| 任务 | 完成日期 | 关键产出 |
|------|----------|---------|
| slides 收尾（Application + Conclusion + CARLA） | 06-14 | ss26 ea88f82；portal lightbox；before/after 比较滑块 |
| 主仓库 commit & push | 06-14 | PPD-examples c60f5b3（wavelet branch）|
| 理论实验 E1–E5 + 天气实验关闭 | 06-14 | tasks.json 全部标 closed |
| v7 全量正式评估 + per-variant 日志体系 | 06-13 | 54 logs；summarize_{vkitti,nucarla}_eval.py |
| Ablation 1 三观察 + 4-panel 图 + slides S11–S18 | 06-13 | KID 混杂论证；ss26 48c6fdc→ea88f82 |

## 决策点

无待决。所有指标口径、表格结构、slides、commit 均已完成。

## 下一步建议

**启动正文写作**，推荐顺序：
1. **Method 节**：ψ-PD 架构（subband decomposition → structured noise → drop_ll）+ flow matching 连接（why linear operator commutes with rectified flow）
2. **Experiments 节**：vKITTI → KITTI benchmark → nuCarla video benchmark → Ablation 1
3. **Related Work 节**：参照 slides Slide04_RelatedWork 五类方法

内容已完备，Results.md + slides 可直接作为写作原材料。
