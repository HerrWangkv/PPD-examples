# Orchestrator State
_最后同步：2026-05-31_

## 全局进度看板

| 阶段 | 状态 | 备注 |
|------|------|------|
| Survey | ✅ done | baselines 确定：FlowEdit/DNAEdit/Cosmos/Kontext |
| Ideation | ✅ done | WPD J=4 r12 主 operating point，J sweep 验证完毕 |
| Experiment | 🔄 active | vKITTI + 消融 A 完成，Hypersim + 消融 B 待做 |
| Publication | ⏳ pending | paper table 就绪，待写作 |

## 当前活跃任务

无正在运行的 GPU 任务（Hypersim 被 kill，待重启）

## 最近完成任务

| 任务 | 完成时间 | 关键结果 |
|------|----------|---------|
| drop_ll 消融 A（推理贡献）| 2026-05-31 | 推理时 LL zeroing 贡献 ~15pt FID，训练 ~1pt |
| J sweep at r12 (J=3/5) | 2026-05-30 | J=4 FID 最优，J=5 结构更好 |
| vKITTI benchmark 全量 | 2026-05-29 | WPD J=4 r12 优于 Cosmos depth+edge |
| Ablation 1 & 2 plots | 2026-05-29/30 | 两张消融图已生成 |
| Pipeline docs sync | 2026-05-30 | project_truth/orchestrator/execution 更新 |

## 决策点

1. **是否跑 Ablation Variant B**（step-6000 lora 无 drop_ll flag）— 建议：是，完整 2×2 矩阵
2. **Hypersim 是否优先于 Ablation B** — 当前建议先跑 Ablation B（更快，用 vKITTI）
3. **写作时机** — Ablation B + Hypersim 结束后立即开始

## 下一步建议（优先级）

1. **重启 Hypersim dropll_J5_r24**（GPU 0-3）
2. **同时/之后跑 Ablation Variant B**（step-6000 lora 无 drop_ll flag，vKITTI r=12）
3. **重构 Hypersim eval 脚本**（`--gen_folder` 接口）
4. **开始论文 Results section**
