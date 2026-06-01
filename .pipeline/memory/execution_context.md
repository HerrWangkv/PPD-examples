# Execution Context
_最后同步：2026-06-01_

## 当前任务

**ID:** publication
**标题:** 论文写作
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
| FlowEdit | **0.7563** | 75.13 | 0.0512 | 0.2924 | 0.8926 | 0.4159 |
| Cosmos depth+edge | 0.6516 | 71.53 | 0.0510 | *0.3236* | **0.9259** | *0.3568* |
| WPD baseline r20 | 0.6854 | **68.22** | *0.0451* | 0.3156 | 0.9014 | 0.3948 |
| WPD J=5 r24 (drop_ll) | *0.7412* | *68.31* | **0.0448** | **0.3772** | *0.9190* | **0.3459** |

## 论文故事逻辑

1. **Problem**: sim-to-real gap = structure gap + appearance gap (lighting/texture)
2. **Insight**: DTCWT LL subband = global illumination bias → zeroing corrects appearance
3. **Ablation story**:
   - PPD (FFT) vs WPD baseline: 同 KID 下 WPD baseline structure 更好（Pareto 优势）
   - WPD baseline vs drop_ll: drop_ll 将 frontier 向左推（更好 KID），轻微 structure 代价
   - Hypersim: drop_ll 修正路径追踪光照（DepSSIM +0.018，mIoU +6.2pt vs baseline）
4. **Figures**: Ablation 1（3 曲线 KID x 轴）+ Ablation 2（J sweep）均 paper-ready

## 挂起实验

| 实验 | 命令 | 优先级 |
|------|------|--------|
| PPD r32 vKITTI | `sbatch sbatch_inference_vkitti_ppd_r32.sh` | 中 |
| DNAEdit Hypersim | `bash run_hypersim_dnaedit.sh --gpus 0,1,2,3` | 低 |
| Hypersim PPD r20 | rsync HPC→mrtstorage | 低 |

## 上下文积累诊断

- **PPD**: `batch_sim2real_image_ppd.py`（FFT）；旧 `wpd_ppd_ckpt_*` 作废
- **Ablation 1 图**: `plot_ablation_baseline_vs_ppd.py` 从 logs 读取；baseline r16 = `baseline_newprompt`
- **Hypersim mIoU**: pseudo-GT ADE20K，cache 在 `outputs/hypersim/.pseudo_gt_cache/`
- **Hypersim bold**: excl. input；FID best = WPD baseline（68.22 < 68.31）
- **CleanFID**: `--mode clean` for all FID/KID
- **summarize scripts**: 均支持 `**best**` / `*2nd*` 标记
