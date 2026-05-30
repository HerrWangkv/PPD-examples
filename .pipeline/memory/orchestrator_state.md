# Orchestrator State
_最后同步：2026-05-30_

## 全局进度看板

| 阶段 | 状态 | 备注 |
|------|------|------|
| Survey | ✅ done | baselines 确定：FlowEdit/DNAEdit/Cosmos/Kontext |
| Ideation | ✅ done | WPD J=4 r12 为主 operating point，J sweep 验证完毕 |
| Experiment | 🔄 active | vKITTI 完成，Hypersim 运行中，ablation 排队 |
| Publication | ⏳ pending | paper table 就绪，待写作 |

## 当前活跃任务

| 任务 | GPU | 进度 | ETA |
|------|-----|------|-----|
| Hypersim dropll_J5_r24 翻译 | 0-3 | ~1552/7402 (21%) | ~9h |
| drop_ll ablation (training vs inference) | 等待 GPU | 排队中 | Hypersim 完成后 |

## 已完成任务（最近完成）

- [2026-05-30] J sweep at r12 eval (J=3/5)：J=4 FID 最优，J=5 结构更好
- [2026-05-30] Ablation 2 图（J sweep 3-panel FID/mIoU/DepSSIM vs J）
- [2026-05-29] vKITTI FLUX-Kontext baseline eval（排除出 paper）
- [2026-05-29] WPD baseline r8/r12/r20/r24 eval 完成
- [2026-05-29] Paper table + Ablation 1 plot（PPD vs WPD baseline）
- [2026-05-28] vKITTI 全量 benchmark：PPD/WPD/drop_ll/Cosmos/FlowEdit/DNAEdit

## 决策点

1. **CUT baseline** — reviewer 必问，尚未实施。低优先级但需要在写作前决定
2. **Hypersim eval 脚本重构** — `calc_fid_hypersim.py` 等硬编码路径，需适配 `--gen_folder`
3. **写作时机** — benchmark 完整性 vs 早开始写作的 trade-off

## 下一步建议（优先级）

1. **等 Hypersim 完成** → 立刻跑 eval（FID vs ScanNet，depth metrics）
2. **启动 drop_ll ablation**（2 变体 × vKITTI）
3. **重构 Hypersim eval 脚本** → 适配 `--gen_folder`
4. **开始论文写作**（Results section + 消融分析）
