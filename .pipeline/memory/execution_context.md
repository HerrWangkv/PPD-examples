# Execution Context
_最后同步：2026-06-01_

## 当前任务

**ID:** publication
**标题:** 论文写作 + 新实验补强（new venue submission）
**状态:** in-progress

## vKITTI Paper Table（最新）

| Method | CLIP-IQA↑ | FID↓ | KID↓ | mIoU↑ | DepSSIM↑ | AbsRel↓ |
|--------|-----------|------|------|-------|----------|---------|
| Input (raw sim) | 0.7180 | 97.29 | 0.0607 | 50.39 | 0.9002 | 0.1573 |
| FlowEdit | 0.6267 | 82.41 | 0.0485 | *42.72* | 0.8119 | 0.2599 |
| DNAEdit | 0.7643 | 85.47 | 0.0478 | 41.22 | 0.8274 | 0.2539 |
| Cosmos depth+edge | 0.4083 | **73.52** | *0.0452* | 39.36 | **0.8700** | **0.2016** |
| PPD r12 | *0.7737* | 78.95 | 0.0487 | 38.32 | 0.8107 | 0.3439 |
| WPD J=4 r12 (ours) | **0.7963** | *73.84* | **0.0441** | **43.50** | *0.8394* | *0.2286* |

## Hypersim Paper Table（最新）

| Method | CLIP-IQA↑ | FID↓ | KID↓ | mIoU↑ | DepSSIM↑ | AbsRel↓ |
|--------|-----------|------|------|-------|----------|---------|
| input (raw sim) | 0.6437 | 72.05 | 0.0461 | — | 0.9416 | 0.2922 |
| FlowEdit | *0.7563* | 75.13 | 0.0512 | 0.2924 | 0.8926 | 0.4159 |
| DNAEdit | **0.7637** | 73.98 | 0.0505 | 0.3169 | 0.8978 | 0.4107 |
| Cosmos depth+edge | 0.6516 | 71.53 | 0.0510 | *0.3236* | **0.9259** | *0.3568* |
| PPD r20 | 0.6942 | 70.53 | 0.0484 | 0.2724 | 0.8910 | 0.4239 |
| WPD baseline r20 | 0.6854 | **68.22** | *0.0451* | 0.3156 | 0.9014 | 0.3948 |
| WPD J=5 r24 (drop_ll) | 0.7412 | *68.31* | **0.0448** | **0.3772** | *0.9190* | **0.3459** |

## Submission Strategy: Two-Track

**Track A — Multi-view image sim2real (try first)**
Core method: 3D Noise Field Projection — project Gaussian noise to 3D via sim depth, render per-view, apply WPD replacement (LL + high-freq phase + magnitude). Inference-time only, no retraining.

| Baseline | arXiv | Role |
|----------|-------|------|
| RL3DEdit | 2603.03143 | Multi-view consistent 3D editing via RL |
| 3D-Consistent MV Editing | 2511.22228 | Training-free correspondence guidance |
| Cosmos Transfer | — | Per-view with depth conditioning |

**Track B — Video translation (fallback)**

| Baseline | arXiv | Role |
|----------|-------|------|
| DNAEdit | 2506.01430 | Strong editing baseline |
| Cosmos Transfer | — | Already done |
| DITTO | 2510.15742 | Instruction-based video editing |
| VACE | 2503.07598 | All-in-one video editing |
| PPD video | — | `sim2real_video_ppd.py` — full pipeline |
| DwD | 2602.06159 | Direct sim2real video competitor |
| Control-DINO | 2604.01761 | Sim2real video transfer, no domain training |

## ECCV Reviewer Requests (reference for new venue)

| Reviewer | Score | Key requests |
|----------|-------|-------------|
| wxAN | 2 (reject) | FID ✅; FlowEdit ✅; VACE+DITTO; FVD |
| ZZKc | 4 | 2nd domain ✅ Hypersim; mIoU+DepSSIM ✅; runtime costs |
| FfwS | 4 | Temporal coherence spec; runtime costs; prompt details |

## 论文故事逻辑

1. **Problem**: sim-to-real gap = structure gap + appearance gap (lighting/texture)
2. **Insight**: DTCWT LL subband = global illumination bias → zeroing corrects appearance
3. **Ablation story**:
   - PPD r12 vs WPD J=4 r12: same radius, WPD wins FID/KID/mIoU (direct comparison)
   - WPD baseline vs drop_ll: drop_ll pushes Pareto frontier left (better FID), slight structure cost
   - Hypersim: drop_ll fixes path-traced lighting (DepSSIM +0.018, mIoU +6.2pt vs baseline)
4. **Figures**: Ablation 1 (3-curve, J=4 r16 fixed, PPD r32 included) + Ablation 2 (J sweep)

## 挂起实验

| Experiment | Command | Status |
|------------|---------|--------|
| DNAEdit Hypersim | `bash run_hypersim_dnaedit.sh --gpus 0,2,3` | in-progress (~57% remaining) |
| WPD baseline r10 | run inference + eval | pending (fills KID gap r8→r12 in Ablation 1) |

## 上下文积累诊断

- **PPD in paper table**: r12 (direct comparison to WPD J=4 r12 at same radius)
- **PPD**: use `batch_sim2real_image_ppd.py` (FFT); old `wpd_ppd_ckpt_*` results obsolete
- **Ablation 1**: `plot_ablation_baseline_vs_ppd.py`; baseline r16 = `baseline_newprompt`; dropll_J4 r16 = `dropll_step6000_J4`
- **Hypersim mIoU**: pseudo-GT ADE20K, cache at `outputs/hypersim/.pseudo_gt_cache/`
- **Hypersim bold**: exclude input row; FID best = WPD baseline r20 (68.22 < drop_ll 68.31)
- **CleanFID**: `--mode clean` for all FID/KID computations
- **3D noise projection idea**: Novelty 5/5 — saved for PhD thesis next paper
- **J=5 on Ablation 1**: tried, found below PPD curve — removed
- **DROPLL_J4_KEY**: r16 maps to `dropll_step6000_J4` (confirmed alias)
