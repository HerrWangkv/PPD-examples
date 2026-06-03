# Project Truth
_最后同步：2026-06-03 (sync v2)_

## 研究主题

**WPD / ψ-PD — Wavelet Phase-Preserving Diffusion for Sim-to-Real Translation**

DTCWT-based structured noise injection：保留 sim 输入的相位结构，通过 drop_ll 去除低频全局光照偏差，实现无额外 conditioning 的 sim-to-real 图像翻译。

**论文状态**: ECCV 2026 rejected (2/4/4). Submitting to new venue (TBD).

## 当前阶段

**Publication（论文写作 + 新实验补强）** — 11/13 任务完成

## 已确认决策

- [2026-05-21] drop_ll 推理用 J=4（J=3 过激进）
- [2026-05-21] Prompt 去除 "dashboard" 措辞
- [2026-05-29] FID/KID 作为主要指标（CleanFID mode）
- [2026-05-29] WPD J=4 r12 为 vKITTI paper 主要 operating point
- [2026-05-29] FLUX-Kontext 不列入 paper（mIoU 虚高）
- [2026-05-31] 推理时 LL zeroing 是 FID 提升主因（~13pt at r12）
- [2026-06-01] 切换至 CleanFID（clean mode）
- [2026-06-01] PPD 使用 FFT 脚本；旧 DTCWT 结果作废
- [2026-06-01] Paper table PPD 代表：vKITTI=r12，Hypersim=r20
- [2026-06-01] Two-track strategy: Track A multi-view AD sim2real, Track B video fallback
- [2026-06-01] WPD J=5 r24 (drop_ll) 为 Hypersim paper operating point
- [2026-06-02] nuCarla 6-camera data pipeline 完成（CARLA 0.9.16 安装，40帧/scene，depth+rgb+calibration）
- [2026-06-02] 3D noise projection (wavelet_noise_mv.py) + nuCarla utils 实现；forward warp 几何验证通过
- [2026-06-02] Multi-view training pipeline 设计完成：DA3 depth precompute + NuScenes finetune
- [2026-06-02] nuScenes finetune 需在真实数据训练（不是 nuCarla），用 DA3NESTED-GIANT-LARGE-1.1 提供 metric depth
- [2026-06-02] Hypersim dropll_J5_r20 eval 完成；Results.md 重构为 Paper Table / Full Table / Key Findings
- [2026-06-03] WPD baseline r10 vKITTI eval 完成（FID 74.80 / KID 0.0455）；Ablation 1 曲线 gap 填补
- [2026-06-03] DA3 precompute 完成；nuScenes finetune 启动（step-500 checkpoint 存在）
- [2026-06-03] Hypersim PPD r24 在 HPC 运行中
- [2026-06-03] train_nuscenes_mv.py 加入 cross-view consistency loss（--lambda_consistency，x̂_0 warp L1）
- [2026-06-03] Track A 多视角一致性训练方向已关闭（v1–v5 全部失败）
- [2026-06-03] 根本原因确认：drop_ll 使每相机 LL/全局色调独立随机；nuScenes FOV 重叠仅 ~12%；噪声相关性无法解决
- [2026-06-03] 结论：需要推理时同步去噪（synchronized denoising）或后处理颜色对齐
- [2026-06-03] Hypersim PPD r24 eval 完成（FID 67.64 / KID 0.0457 / CLIP-IQA 0.6804 / mIoU 0.3014）；PPD r24 赢得 Hypersim FID
- [2026-06-03] Results.md Hypersim paper table 更新：PPD r24 替换 PPD r20；bold/italic 重新标注
- [2026-06-03] Wan low LoRA 训练启动（train_wan_low_dropll.sh, GPU 0-3）；Wan high LoRA 待启动

## 阶段进展摘要

### Survey
- Baselines: FlowEdit / DNAEdit / Cosmos depth+edge / PPD r20 (FFT)
- 12 papers OCR'd for sim2real-baselines corpus

### Ideation
- WPD J=4 r12 (vKITTI) + J=5 r24 (Hypersim) as paper operating points
- **Two-track**: Track A = AD multi-view sim2real (nuCarla + nuScenes finetune); Track B = video (fallback)

### Experiment

**vKITTI→KITTI (complete)**
- Paper table: Input / FlowEdit / DNAEdit / Cosmos / PPD r12 / WPD J=4 r12 (ours)
- Ablation 1 (3-curve), Ablation 2 (J sweep), Ablation A (LL zeroing) all done

**Hypersim→ScanNet (complete)**
- Paper table: FlowEdit / DNAEdit / Cosmos / PPD r20 / WPD J=5 r24 (drop_ll, ours)
- dropll_J5_r20 eval added to full table: CLIP-IQA 0.7643 / FID 70.53 / KID 0.0481 / mIoU 0.2985
- PPD r24 eval complete: FID **67.64** (best on Hypersim) / KID 0.0457 / CLIP-IQA 0.6804 / mIoU 0.3014 / DepSSIM 0.8978 / AbsRel 0.3994
- Paper table updated: PPD r24 replaces PPD r20

**Track B: Video translation (in-progress)**
- Wan low LoRA training running (GPU 0-3, train_wan_low_dropll.sh, from wan2.2-14b-low-step-12400)
- Wan high LoRA training: ready to launch (train_wan_high_dropll.sh)
- Both scripts use HF streaming dataset + drop_ll prob=0.8 in train.py

**Track A: Multi-view (concluded 2026-06-03)**
- Infrastructure complete: nuCarla pipeline, wavelet_noise_mv.py, sim2real_nucarla_mv.py
- 5 training variants (v1–v5) all failed — see project_multiview_consistency.md in auto-memory
- Root cause: LL global color independently random per camera; 12% FOV overlap insufficient
- **Next step**: synchronized denoising (sim2real_nucarla_mv_sync.py, no training needed)
  - warp_and_blend_latents() already in wavelet_noise_mv.py
  - Plan documented in .claude/plans/consistency-loss-artifacts-consistent-glistening-falcon.md

### Publication
- Results.md complete; two paper tables; ablation figures paper-ready
- **Writing not started**

## 当前最佳実験結果

### vKITTI→KITTI (CleanFID)
| Metric | Best | Value | Paper entry |
|--------|------|-------|------------|
| FID↓ | WPD J=4 r8 | 67.53 | WPD J=4 r12: 73.84 |
| KID↓ | WPD J=4 r8 | 0.0361 | WPD J=4 r12: 0.0441 |
| mIoU↑ | WPD baseline r24 | 48.46 | WPD J=4 r12: 43.50 |

### Hypersim→ScanNet (CleanFID)
| Metric | Best | Value | Paper entry |
|--------|------|-------|------------|
| FID↓ | PPD r24 | **67.64** | WPD J=5 r24: 68.31 |
| KID↓ | DNAEdit / WPD r24 | 0.0448 | WPD J=5 r24: 0.0448 |
| mIoU↑ | WPD J=5 r24 | 0.3772 | WPD J=5 r24: 0.3772 |

## 方向調整記録

- Benchmark: Synthia → vKITTI + Hypersim
- Primary metric: CLIP-IQA → FID/KID
- PPD: DTCWT → FFT
- New contribution: Track A multi-view AD sim2real (AD motivation + nuScenes finetune for consistency)

## 風险 / 阻塞項

- **Writing not started**: 所有实验数据就绪（vKITTI + Hypersim 两 benchmark 完整），需立即开始
- **Wan low LoRA**: 运行中（GPU 0-3）；Wan high LoRA 待启动（GPU 4-7）
- **Track A**: synchronized denoising 待实现（sim2real_nucarla_mv_sync.py）
- **PPD r12 vKITTI**: paper table 用 r12，但 Hypersim paper table 用 PPD r24（已更新）
