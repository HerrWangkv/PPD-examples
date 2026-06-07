# Project Truth
_最后同步：2026-06-07_

## 研究主题

**WPD / ψ-PD — Wavelet Phase-Preserving Diffusion for Sim-to-Real Translation**

DTCWT-based structured noise injection：保留 sim 输入的相位结构，通过 drop_ll 去除低频全局光照偏差，实现无额外 conditioning 的 sim-to-real 图像/视频翻译。

**论文状态**: ECCV 2026 rejected (2/4/4). Submitting to new venue (TBD).

## 当前阶段

**Publication（论文写作 + 新实验补强）** — 14/15 任务完成

## 已确认决策

- [2026-05-21] drop_ll 推理用 J=4（J=3 过激进）
- [2026-05-21] Prompt 去除 "dashboard" 措辞
- [2026-05-29] FID/KID 作为主要指标（CleanFID mode）
- [2026-05-29] WPD J=4 r12 为 vKITTI paper 主要 operating point
- [2026-05-29] FLUX-Kontext 不列入 paper（mIoU 虚高）
- [2026-05-31] 推理时 LL zeroing 是 FID 提升主因（~13pt at r12）
- [2026-06-01] 切换至 CleanFID（clean mode）
- [2026-06-01] PPD 使用 FFT 脚本；旧 DTCWT 结果作废
- [2026-06-01] Paper table PPD 代表：vKITTI=r12，Hypersim=r24
- [2026-06-01] Two-track strategy: Track A multi-view AD sim2real, Track B video fallback
- [2026-06-01] WPD J=5 r24 (drop_ll) 为 Hypersim paper operating point
- [2026-06-03] Track A 多视角一致性训练方向已关闭（v1–v5 全部失败）
- [2026-06-03] Hypersim PPD r24 eval 完成（FID 67.64 / KID 0.0457）
- [2026-06-07] **LightEMMA E2E eval**: WPD baseline r30 最佳（ADE_avg −5.46% vs raw sim）；所有非 WPD 方法使规划精度下降
- [2026-06-07] nuCarla Cosmos depth+edge 翻译：50/60 完成，运行中

## 阶段进展摘要

### Survey
- Baselines: FlowEdit / DNAEdit / Cosmos depth+edge / PPD r20 (FFT)
- 12 papers OCR'd for sim2real-baselines corpus

### Ideation
- WPD J=4 r12 (vKITTI) + J=5 r24 (Hypersim) as paper operating points
- **Two-track**: Track A = AD multi-view sim2real (concluded failed); Track B = video (fallback)

### Experiment

**vKITTI→KITTI (complete)**
- Paper table: Input / FlowEdit / DNAEdit / Cosmos / PPD r12 / WPD J=4 r12 (ours)
- Ablation 1 (3-curve), Ablation 2 (J sweep), Ablation A (LL zeroing) all done

**Hypersim→ScanNet (complete)**
- Paper table: FlowEdit / DNAEdit / Cosmos / PPD r24 / WPD J=5 r24 (drop_ll, ours)
- PPD r24 best FID (67.64); WPD J=5 r24 best mIoU (0.3772) / KID tied with DNAEdit

**nuCarla → LightEMMA E2E (complete for 5 methods, cosmos pending)**
- WPD baseline r30: ADE_avg=2.2097 (−5.46%), FDE=4.9527 (−5.18%) — best
- WPD drop_ll r30 J=5: ADE_avg=2.2336 (−4.44%) — 2nd best
- PPD/Cosmos d+e+s/Ditto all degrade planning vs raw sim (+2–6%)
- cosmos_depth_edge (no seg) eval: pending (50/60 translated, running on container confident_pascal)
- extension_dropll_r30_J5 (scenes 0060–0159) eval: pending (100 scenes done, needs Gemini key)

**Track B: Video translation**
- extension_dropll_r30_J5: 100 scenes (scene_0060–0159) complete
- Wan low/high LoRA: status unclear

**Track A: Multi-view (concluded 2026-06-03)**
- 5 training variants (v1–v5) all failed
- Root cause: LL global color independently random per camera; 12% FOV overlap insufficient
- mv modifications stashed: `stash@{0}` = debug/test scripts, `stash@{1}` = core mv modifications

### Publication
- Results.md: vKITTI + Hypersim + nuCarla LightEMMA sections complete
- Writing not started

## 当前最佳实验结果

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

### nuCarla → LightEMMA (60 scenes, 5 methods evaluated)
| Method | ADE_avg↓ | FDE↓ | vs raw sim |
|--------|---------|------|-----------|
| carla (raw sim) | 2.3372 | 5.2234 | baseline |
| WPD baseline r30 | **2.2097** | **4.9527** | **−5.46% / −5.18%** |
| WPD drop_ll r30 J=5 | 2.2336 | 5.0061 | −4.44% / −4.16% |
| PPD r30 | 2.4133 | 5.3885 | +3.25% / +3.16% |
| Ditto | 2.3838 | 5.4236 | +1.99% / +3.83% |
| Cosmos d+e+s | 2.4665 | 5.5520 | +5.53% / +6.29% |

## 方向調整記録

- Benchmark: Synthia → vKITTI + Hypersim + nuCarla (LightEMMA E2E)
- Primary metric: CLIP-IQA → FID/KID; added E2E planning metric (ADE/FDE)
- PPD: DTCWT → FFT
- Track A multi-view: closed (training approach failed)

## 風险 / 阻塞項

- **Writing not started**: 所有实验数据就绪，需立即开始
- **Cosmos depth_edge LightEMMA eval**: 等 translation 完成（50→60）
- **Extension LightEMMA eval**: 100 scenes done，等 Gemini key 更新后运行
- **Gemini API key**: 旧 key 暴露在 GitHub，需在 Google Cloud Console 撤销并更新 `LightEMMA/config.yaml`
