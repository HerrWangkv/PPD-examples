# Project Truth
_最后同步：2026-06-01_

## 研究主题

**WPD / ψ-PD — Wavelet Phase-Preserving Diffusion for Sim-to-Real Translation**

DTCWT-based structured noise injection：保留 sim 输入的相位结构，通过 drop_ll 去除低频全局光照偏差，实现无额外 conditioning 的 sim-to-real 图像翻译。

**论文状态**: ECCV 2026 rejected (2/4/4). Preparing strengthened resubmission.

## 当前阶段

**Publication（论文写作 + 新实验补强）** — 6/10 任务完成

## 已确认决策

- [2026-05-21] drop_ll 推理用 J=4（J=3 过激进）
- [2026-05-21] Prompt 去除 "dashboard" 措辞
- [2026-05-29] FID/KID 作为主要指标（CleanFID mode）
- [2026-05-29] WPD J=4 r12 为 paper 主要 operating point
- [2026-05-29] FLUX-Kontext 不列入 paper（mIoU 虚高）
- [2026-05-31] 推理时 LL zeroing 是 FID 提升主因（~13pt at r12）
- [2026-06-01] 切换至 CleanFID（clean mode）
- [2026-06-01] Hypersim→ScanNet 作为 lighting claim 第二 benchmark
- [2026-06-01] PPD 使用 FFT 脚本（batch_sim2real_image_ppd.py），旧 DTCWT 结果作废
- [2026-06-01] Paper table PPD 代表：r20（FID 76.29，PPD 系列最佳 FID）
- [2026-06-01] Ablation 1 图：KID x 轴，3 曲线（PPD / WPD baseline / WPD J=4 drop_ll）
- [2026-06-01] Hypersim bold 规则：排除 input；FID best = WPD baseline r20（68.22）
- [2026-06-01] ECCV rejection confirmed; resubmission target TBD
- [2026-06-01] 3D Noise Field Projection idea (multi-view sim2real via depth-projected noise): Novelty 5/5 — save for next paper or resubmission extension

## 阶段进展摘要

### Survey
- Original baselines confirmed: FlowEdit / DNAEdit / Cosmos depth+edge / PPD r20 (FFT)
- New 2025 literature surveyed (sim2real-baselines corpus, 12 papers OCR'd):
  - **Driving with DINO** (2602.06159) — direct competitor, VFM features as sim2real bridge
  - **CACTI** (2505.16360) — GTA5→Cityscapes style transfer baseline, has code
  - **VACE** (2503.07598) — video editing baseline (reviewer wxAN requested)
  - **DITTO** (2510.15742) — instruction-based video editing (reviewer wxAN requested)
  - **RL3DEdit** (2603.03143) — multi-view consistent 3D editing via RL, related work
  - **3D-Consistent MV Editing** (2511.22228) — training-free multi-view consistency, related work
  - **Antithetic Noise** (2506.06185) — structured noise, related work to WPD noise design

### Ideation
- WPD J=4 r12 as main operating point; drop_ll + DTCWT phase as core novelty
- **3D Noise Field Projection**: user's novel idea — project Gaussian noise to 3D via sim depth, render per-view, use as WPD noise. Novelty 5/5, Feasibility 3.5/5. Decided to save for next paper (too large for current revision scope).
- **V1 Flow-Warped Noise**: warp wavelet noise across frames using sim optical flow. Feasibility 5/5, good for video section. Pending decision.

### Experiment

**vKITTI→KITTI benchmark (complete)**
- PPD r8–r24 (FFT), WPD baseline/drop_ll full series, FlowEdit/DNAEdit/Cosmos all evaluated
- Paper table: Input / FlowEdit / DNAEdit / Cosmos depth+edge / PPD r20 / WPD J=4 r12
- Ablation 1 (3 curves KID x-axis): PPD < WPD baseline < WPD drop_ll
- Ablation 2 (J sweep): J=4 best FID, J=5 best structure
- Ablation A (infer-only drop_ll): LL zeroing contributes ~13pt FID
- PPD r32: in-progress (sbatch submitted)

**Hypersim→ScanNet benchmark (complete, DNAEdit row pending)**
- 5 variants: input / FlowEdit / Cosmos / WPD baseline r20 / WPD J=5 r24 drop_ll
- FID: WPD baseline 68.22 (best); KID: drop_ll 0.0448 (best)
- DepSSIM: Cosmos 0.9259 (best); mIoU: drop_ll 0.3772 (best)
- AbsRel: drop_ll 0.3459 (best); CLIP-IQA: FlowEdit 0.7563 (best)
- DNAEdit row: in-progress (~43% translated)

### Publication
- Results.md: two paper tables (vKITTI + Hypersim) at top
- Summarize scripts: best/2nd-best bold/italic markup for both benchmarks
- Ablation figures: paper-ready (plot_ablation_baseline_vs_ppd.py reads from logs)
- **Writing not started yet**

## 当前最佳实验结果

### vKITTI→KITTI (CleanFID, translation methods only)

| Metric | Best | Value | 2nd best |
|--------|------|-------|---------|
| FID↓ | WPD J=4 r8 | 67.53 | WPD J=3 r16 (73.61) |
| KID↓ | WPD J=4 r8 | 0.0361 | WPD J=3 r12 (0.0419) |
| mIoU↑ | WPD baseline r24 | 48.46 | WPD baseline r20 (48.32) |
| **Paper operating point** | **WPD J=4 r12** | **FID 73.84 / KID 0.0441 / mIoU 43.50** | |
| **PPD best** | **PPD r20** | **FID 76.29 / KID 0.0474 / mIoU 43.75** | |

### Hypersim→ScanNet (CleanFID, excl. input)

| Metric | Best | Value | 2nd best |
|--------|------|-------|---------|
| FID↓ | WPD baseline r20 | 68.22 | WPD drop_ll (68.31) |
| KID↓ | WPD drop_ll | 0.0448 | WPD baseline (0.0451) |
| mIoU↑ | WPD drop_ll | 0.3772 | Cosmos (0.3236) |
| DepSSIM↑ | Cosmos | 0.9259 | WPD drop_ll (0.9190) |
| AbsRel↓ | WPD drop_ll | 0.3459 | Cosmos (0.3568) |
| CLIP-IQA↑ | FlowEdit | 0.7563 | WPD drop_ll (0.7412) |

## 方向调整记录

- Benchmark: Synthia → vKITTI→KITTI (primary) + Hypersim→ScanNet (lighting claim)
- Primary metric: CLIP-IQA → FID/KID (CleanFID)
- PPD baseline: DTCWT results obsolete → all use FFT
- Paper status: ECCV reject → resubmission with strengthened baselines + video section

## 风险 / 阻塞项

- **Writing not started**: data complete, can begin immediately
- **ECCV reviewer requests (pending)**:
  - wxAN: VACE/DITTO video baselines + FVD metric
  - ZZKc: downstream metrics (segmentation/depth) — partially addressed by mIoU + DepSSIM
  - FfwS: temporal coherence specification for video
- **New baselines to run**: CACTI (has code), Driving with DINO (check code), VACE, DITTO
- **Hypersim DNAEdit row**: ~43% translated, in-progress
- **Hypersim PPD r20**: HPC complete, rsync pending
- **PPD r32 vKITTI**: sbatch in-progress (extends ablation curve)
