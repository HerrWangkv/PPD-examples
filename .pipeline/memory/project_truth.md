# Project Truth
_最后同步：2026-05-31_

## 研究主题

**WPD / ψ-PD — Wavelet Phase-Preserving Diffusion for Sim-to-Real Translation**

DTCWT-based structured noise injection：保留 sim 输入的相位结构（几何/语义），通过 drop_ll 去除低频全局光照偏差，实现无额外 conditioning 的 sim-to-real 图像翻译。

**论文状态**: 已投稿，两轮评审（2/4/4 分），post-rebuttal，准备改进后重投

## 当前阶段

**实验执行** — vKITTI benchmark + 消融实验完成，Hypersim benchmark 待重启

## 已确认决策

- [2026-05-21] drop_ll 推理用 J=4（J=3 过激进，J=4 平衡全局光照修正与细节保留）
- [2026-05-21] Prompt 去除 "dashboard" 措辞（导致 car interior hallucination）
- [2026-05-29] FID/KID 作为论文主要指标（CLIP-IQA 奖励锐度，不适合本文故事）
- [2026-05-29] WPD J=4 r12 为 paper 主要 operating point
- [2026-05-29] FLUX-Kontext 不列入 paper（mIoU 虚高）
- [2026-05-29] WPD baseline 变体保留用于消融，不列入 paper 主表
- [2026-05-30] Hypersim→ScanNet 作为 lighting claim 第二 benchmark
- [2026-05-31] 推理时 LL zeroing 是 FID 提升主因（~15pt），drop_ll 训练贡献边际（~1pt）

## 阶段进展摘要

### Experiment

**vKITTI→KITTI benchmark（完成，2026-05-29）**
- 2126 clone frames，5 scenes，配对 KITTI tracking 序列
- 指标：FID / KID / mIoU / DepSSIM / AbsRel / LPIPS
- 核心结论：WPD J=4 r12 同时优于 Cosmos depth+edge（FID 75.72 vs 76.09，mIoU 43.50 vs 39.36），无任何 conditioning
- Ablation 1：PPD vs WPD baseline plot（`plot_ablation_baseline_vs_ppd.py`）
- Ablation 2：J sweep at r12 & r16（`plot_ablation_J_sweep.py`）

**drop_ll 消融——推理 vs 训练贡献（完成，2026-05-31）**
- 变体 A：`flux.safetensors` + `--flux_drop_ll J=4` at inference，radius sweep r8/12/20/24
- 结论（at r12）：WPD baseline FID 89.61 → infer-only 74.54 → full WPD J=4 75.72
- 推理时 LL zeroing 贡献 ~15pt FID，训练贡献 ~1pt 额外 FID + mIoU 恢复（42.97→43.50）
- 变体 B（step-6000 lora 无 drop_ll flag）尚未跑——待做以完整 2×2 矩阵

**Hypersim→ScanNet benchmark（进行中）**
- 7402 帧 indoor，ScanNet test 32k 帧 FID reference
- 现有：wpd_r20 / flowedit；待做：dropll_J5_r24（需重启，之前被 kill）
- eval 脚本需重构 `--gen_folder` 接口

### Publication
- Results.md：完整结果表 + paper table + ablation 图引用
- Paper table（one variant per method）已就绪

## 当前最佳实验结果（vKITTI→KITTI）

| 指标 | 最佳方法 | 值 |
|------|----------|-----|
| FID↓ | WPD J=4 r8 | 68.56 |
| KID↓ | WPD J=4 r8 | 0.0388 |
| mIoU↑ | WPD baseline r24 | 48.46 |
| DepSSIM↑ | WPD baseline r24 | 0.8814 |
| **综合推荐** | **WPD J=4 r12** | **FID 75.72 / mIoU 43.50** |

## 方向调整记录

- Benchmark 主战场：Synthia → vKITTI→KITTI（更严格配对评估）
- 不再报告 Synthia 结果于论文
- Primary metric：CLIP-IQA → FID/KID

## 风险 / 阻塞项

- **Ablation Variant B 缺失**：step-6000 lora 无 drop_ll flag，完成 2×2 矩阵需此对照
- **Hypersim translation 未完成**：需重启 `run_hypersim_dropll.sh --J 5 --radius 24`
- **Hypersim eval 脚本需重构**：硬编码路径，需 `--gen_folder` 接口
- **CUT/CycleGAN baseline 缺失**：reviewer 必问

## 命名约定

- `PPD<r>`：FLUX-only，旧 lora（flux1-dev_phipd_lora_302000）
- `WPD baseline r<N>`：FLUX-only，`flux.safetensors`（step-20000），无 drop_ll
- `WPD J=<J> r<N>`：FLUX-only，`step-6000` drop_ll lora，drop_ll J=J
- `Ablation: infer drop_ll r<N>`：`flux.safetensors` + `--flux_drop_ll J=4`（无 drop_ll 训练）
