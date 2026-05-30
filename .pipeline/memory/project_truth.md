# Project Truth
_最后同步：2026-05-30_

## 研究主题

**WPD / ψ-PD — Wavelet Phase-Preserving Diffusion for Sim-to-Real Translation**

DTCWT-based structured noise injection：保留 sim 输入的相位结构（几何/语义），通过 drop_ll 去除低频全局光照偏差，实现无额外 conditioning 的 sim-to-real 图像翻译。

**论文状态**: 已投稿，两轮评审（2/4/4 分），post-rebuttal，准备改进后重投

## 当前阶段

**实验执行** — vKITTI benchmark 完成，消融实验进行中，Hypersim benchmark 运行中

## 已确认决策

- [2026-05-21] drop_ll 推理用 J=4（J=3 过激进，J=4 平衡全局光照修正与细节保留）
- [2026-05-21] Prompt 去除 "dashboard" 措辞（导致 car interior hallucination）
- [2026-05-29] FID/KID 作为论文主要指标（CLIP-IQA 奖励锐度而非真实感，不适合本文故事）
- [2026-05-29] WPD J=4 r12 为 paper 主要 operating point
- [2026-05-29] FLUX-Kontext 不列入 paper（mIoU 虚高：图像外观接近 Cityscapes 训练域）
- [2026-05-29] WPD baseline 变体保留用于消融，不列入 paper 主表
- [2026-05-30] Hypersim→ScanNet 作为 lighting claim 第二 benchmark

## 阶段进展摘要

### Experiment

**vKITTI→KITTI benchmark（完成）**
- 2126 clone frames，5 scenes，配对 KITTI tracking 序列
- 指标：FID / KID / mIoU / DepSSIM / AbsRel / LPIPS
- **核心结论**：WPD J=4 r12 同时优于 Cosmos depth+edge（FID 75.72 vs 76.09，mIoU 43.50 vs 39.36），无任何 conditioning
- Ablation 1：PPD vs WPD baseline（plot: `plot_ablation_baseline_vs_ppd.py`）
- Ablation 2：J sweep at r12 & r16（plot: `plot_ablation_J_sweep.py`）

**Hypersim→ScanNet benchmark（进行中）**
- 7402 帧 indoor，ScanNet test 32k 帧作 FID reference
- 现有：wpd_r20 / flowedit；运行中：dropll_J5_r24（GPU 0-3，~1552/7402）
- 目的：验证 drop_ll 在强光照差距场景中效果更显著

### Publication
- Results.md：完整结果表 + paper table + ablation 图引用
- Paper table（one variant per method，无 LPIPS/CLIP-IQA）已就绪

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

- **CUT/CycleGAN baseline 缺失**：reviewer 必问
- **drop_ll training vs inference ablation 待做**：GPU 被 Hypersim 占用，排队中
- **Hypersim eval 脚本未适配 `--gen_folder`**：需重构才能跑 drop_ll 变体
- **step-6000 drop_ll lora 训练不足**：vs baseline lora 302k steps，mIoU 损约 4 点

## 命名约定

- `PPD<r>`: FLUX-only，cutoff radius=r，旧 lora（flux1-dev_phipd_lora_302000）
- `WPD baseline r<N>`: FLUX-only，新 lora（step-20000），无 drop_ll，radius=N
- `WPD J=<J> r<N>`: FLUX-only，新 drop_ll lora（step-6000），drop_ll J=J，radius=N
- 推理约束：无额外 conditioning，无配对数据

## 核心方法文件

- `wavelet_noise.py`: DTCWT 分解，structured noise，drop_ll
- `diffsynth/pipelines/flux_image_new.py`: PPD-modified FLUX pipeline
- `batch_sim2real_image_wavelet.py`: 批量推理（多 GPU）
- `summarize_vkitti_eval.py`: 结果汇总表
