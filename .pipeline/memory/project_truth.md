# Project Truth
_最后同步：2026-06-01_

## 研究主题

**WPD / ψ-PD — Wavelet Phase-Preserving Diffusion for Sim-to-Real Translation**

DTCWT-based structured noise injection：保留 sim 输入的相位结构，通过 drop_ll 去除低频全局光照偏差，实现无额外 conditioning 的 sim-to-real 图像翻译。

**论文状态**: ECCV 2026 rejected (2/4/4). Submitting to new venue (TBD).

## 当前阶段

**Publication（论文写作 + 新实验补强）** — 8/10 任务完成

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
- [2026-06-01] Paper table PPD 代表：r12（direct comparison to WPD J=4 r12 at same radius）
- [2026-06-01] Ablation 1 图：KID x 轴，3 曲线（PPD / WPD baseline / WPD J=4 drop_ll）
- [2026-06-01] ECCV rejection confirmed; new venue submission
- [2026-06-01] Two-track strategy: Track A multi-view image sim2real (3D noise projection), Track B video fallback
- [2026-06-01] 3D Noise Field Projection: Novelty 5/5 — save for next paper if Track A infeasible
- [2026-06-01] dropll_step6000_J4 = dropll_J4_r16 (key alias confirmed)
- [2026-06-01] J=5 curve removed from Ablation 1 (sits below PPD, not informative)
- [2026-06-01] WPD baseline r10 identified as next ablation point (fills KID gap r8→r12)

## 阶段进展摘要

### Survey
- Baselines confirmed: FlowEdit / DNAEdit / Cosmos depth+edge / PPD r20 (FFT)
- sim2real-baselines corpus (12 papers OCR'd):
  - DwD (2602.06159), Control-DINO (2604.01761), CACTI (2505.16360), VACE (2503.07598),
    DITTO (2510.15742), RL3DEdit (2603.03143), 3D-Consistent MV Editing (2511.22228),
    Antithetic Noise (2506.06185), SSB (2602.16664), ViewMask (2512.14099),
    3D-Fixup (2505.10566), Pro3D-Editor (2506.00512)

### Ideation
- WPD J=4 r12 as main operating point; drop_ll + DTCWT phase as core novelty
- **Two-track strategy for new venue**:
  - Track A: multi-view image sim2real via 3D Noise Field Projection (inference-time, no retraining)
  - Track B: video translation fallback using existing Wan2.2 pipeline

### Experiment

**vKITTI→KITTI benchmark (complete)**
- PPD r8–r32 (FFT), WPD baseline/drop_ll full series, FlowEdit/DNAEdit/Cosmos all evaluated
- Paper table: Input / FlowEdit / DNAEdit / Cosmos depth+edge / PPD r20 / WPD J=4 r12
- Ablation 1 figure: 3 curves KID x-axis, J=4 r16 fixed, PPD r32 included
- Ablation 2 figure: J sweep at r12/r16
- Ablation A: LL zeroing ~13pt FID contribution

**Hypersim→ScanNet benchmark (complete, DNAEdit row pending)**
- 6 variants: input / FlowEdit / Cosmos / PPD r20 / WPD baseline r20 / WPD J=5 r24 drop_ll
- PPD r20: FID 70.53 / KID 0.0484 — competitive with Cosmos without conditioning
- FID best: WPD baseline r20 (68.22); KID/mIoU/AbsRel best: drop_ll
- DNAEdit: in-progress on GPUs 0,2,3 (~43% → ~57% remaining)

### Publication
- Results.md: two paper tables (vKITTI + Hypersim) at top, PPD r32 row added
- Summarize scripts: best/**/ + 2nd/*/ markup for both benchmarks
- Ablation figures: paper-ready, 3-curve clean
- **Writing not started yet**

## 当前最佳実験結果

### vKITTI→KITTI (CleanFID, translation methods only)

| Metric | Best | Value | 2nd best |
|--------|------|-------|---------|
| FID↓ | WPD J=4 r8 | 67.53 | WPD J=3 r16 (73.61) |
| KID↓ | WPD J=4 r8 | 0.0361 | WPD J=3 r12 (0.0419) |
| mIoU↑ | WPD baseline r24 | 48.46 | WPD baseline r20 (48.32) |
| **Paper point** | **WPD J=4 r12** | **FID 73.84 / KID 0.0441 / mIoU 43.50** | |
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

## 方向調整記録

- Benchmark: Synthia → vKITTI→KITTI + Hypersim→ScanNet
- Primary metric: CLIP-IQA → FID/KID (CleanFID)
- PPD baseline: DTCWT obsolete → FFT
- Paper direction: ECCV reject → new venue; add Track A (multi-view) or Track B (video)

## 風险 / 阻塞項

- **Writing not started**: data complete, begin immediately
- **Track A feasibility unknown**: need multi-camera sim data + 3D noise projection implementation
- **New baselines TBD**: CACTI / DwD (check code), VACE / DITTO (for Track B)
- **DNAEdit Hypersim**: ~57% remaining, in-progress
- **WPD baseline r10**: inference not yet run (next ablation point)
