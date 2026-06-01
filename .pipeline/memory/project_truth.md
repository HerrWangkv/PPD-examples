# Project Truth
_最后同步：2026-06-01_

## 研究主题

**WPD / ψ-PD — Wavelet Phase-Preserving Diffusion for Sim-to-Real Translation**

DTCWT-based structured noise injection：保留 sim 输入的相位结构，通过 drop_ll 去除低频全局光照偏差，实现无额外 conditioning 的 sim-to-real 图像翻译。

**论文状态**: 已投稿，两轮评审（2/4/4 分），post-rebuttal，准备改进后重投

## 当前阶段

**Publication（论文写作）** — 6/10 任务完成，写作阶段正式开始

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
- [2026-06-01] Ablation 1 图：KID x 轴，3 曲线（PPD / WPD baseline / WPD J=4 drop_ll），从 logs 读取
- [2026-06-01] Hypersim bold 规则：排除 input；FID best = WPD baseline r20（68.22）；FlowEdit best CLIP-IQA

## 阶段进展摘要

### Survey
baselines 确定：FlowEdit / DNAEdit / Cosmos depth+edge / PPD r20（正确 FFT）

### Ideation
WPD J=4 r12 为主 operating point；drop_ll 去除 LL 是核心机制；DTCWT 相位保留是 novelty

### Experiment

**vKITTI→KITTI benchmark（完成）**
- PPD r8–r24（FFT）、WPD baseline/drop_ll 全系列、FlowEdit/DNAEdit/Cosmos 全部评估
- Paper table：Input / FlowEdit / DNAEdit / Cosmos depth+edge / PPD r20 / WPD J=4 r12
- Ablation 1 图（3曲线 KID x轴）：PPD < WPD baseline（同 KID structure 更好）< WPD drop_ll（最佳 KID）
- Ablation 2 图（J sweep）：J=4 最佳 FID，J=5 最佳 structure
- Ablation A（infer-only drop_ll）：LL zeroing 贡献 ~13pt FID，训练贡献 ~1pt

**Hypersim→ScanNet benchmark（完成）**
- 5 variants: input / FlowEdit / Cosmos depth+edge / WPD baseline r20 / WPD J=5 r24 drop_ll
- FID: WPD baseline 68.22（最佳），KID: drop_ll 0.0448（最佳）
- DepSSIM: Cosmos 0.9259（最佳，depth conditioning），WPD drop_ll *0.9190*（2nd）
- mIoU: drop_ll 0.3772（最佳），CLIP-IQA: FlowEdit 0.7563（最佳）
- AbsRel: drop_ll **0.3459**（最佳），Cosmos *0.3568*（2nd）
- summarize_hypersim_eval.py 已支持 bold/italic 标记

### Publication
- Results.md：两个 paper table（vKITTI + Hypersim）优先于详细表格
- summarize 脚本：vKITTI + Hypersim 均支持 best/2nd-best 标记
- Ablation 图：paper-ready（plot_ablation_baseline_vs_ppd.py 从 logs 读取）
- 写作尚未开始

## 当前最佳实验结果

### vKITTI→KITTI（CleanFID，translation methods only）

| 指标 | 最佳 | 值 | 2nd best |
|------|------|----|---------|
| FID↓ | WPD J=4 r8 | 67.53 | WPD J=3 r16 (73.61) |
| KID↓ | WPD J=4 r8 | 0.0361 | WPD J=3 r12 (0.0419) |
| mIoU↑ | WPD baseline r24 | 48.46 | WPD baseline r20 (48.32) |
| **Paper 推荐** | **WPD J=4 r12** | **73.84 / 0.0441 / 43.50** | |
| **PPD 最佳** | **PPD r20** | **76.29 / 0.0474 / 43.75** | |

### Hypersim→ScanNet（CleanFID，excl. input）

| 指标 | 最佳 | 值 | 2nd best |
|------|------|----|---------|
| FID↓ | WPD baseline r20 | 68.22 | WPD drop_ll (68.31) |
| KID↓ | WPD drop_ll | 0.0448 | WPD baseline (0.0451) |
| mIoU↑ | WPD drop_ll | 0.3772 | Cosmos (0.3236) |
| DepSSIM↑ | Cosmos | 0.9259 | WPD drop_ll (0.9190) |
| AbsRel↓ | WPD drop_ll | 0.3459 | Cosmos (0.3568) |
| CLIP-IQA↑ | FlowEdit | 0.7563 | WPD drop_ll (0.7412) |

## 方向调整记录

- Benchmark: Synthia → vKITTI→KITTI
- Primary metric: CLIP-IQA → FID/KID（CleanFID）
- Hypersim→ScanNet 作为 lighting claim 证据
- PPD baseline: DTCWT 结果作废 → 全部改用 FFT

## 风险 / 阻塞项

- **写作未开始**：数据充分，可立即进入
- **PPD r32 待跑**：延伸 Ablation 1 曲线（低优先级，sbatch 已准备）
- **DNAEdit Hypersim 未完成**：~43%，Hypersim 表格暂缺该行
- **Hypersim PPD r20 未 rsync**：HPC 完成但未同步
- **CUT/CycleGAN baseline 缺失**：reviewer 必问（revision 阶段补）
