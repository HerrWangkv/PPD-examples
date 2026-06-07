# Execution Context
_最后同步：2026-06-07_

## 当前任务

**ID:** nucarla_dnaedit_translate + cosmos_lightemma
**标题:** DNAEdit nuCarla 60-scene 翻译（HPC）+ cosmos LightEMMA 收尾
**状态:** 双线并行

## DNAEdit HPC 部署步骤

```bash
# 1. rsync 至 HPC（调整 USER/HOST/HPCDIR）
rsync -av --progress wpd-dnaedit.sif USER@HPC_HOST:HPCDIR/
rsync -av --progress batch_dnaedit_nucarla.py sbatch_dnaedit_nucarla.sh USER@HPC_HOST:HPCDIR/

# 2. 在 HPC 上更新 sbatch_dnaedit_nucarla.sh 的两个路径变量：
#    REPO_DIR=<HPC 上 repo 根目录>
#    DATA_DIR=<HPC 上 nuCarla rgb 视频目录>

# 3. 提交
sbatch sbatch_dnaedit_nucarla.sh
```

**关键修复（本次会话）：**
- `guide_scale`: 5.0 → **1.0**（官方默认；5.0 导致输出不真实）
- frame sampling: 已确认 nuCarla 视频 49 帧 @10fps，逐帧读取 = 均匀采样，无问题

## cosmos LightEMMA 收尾

```bash
# 等 tmux scene_0059 完成后：
cd LightEMMA && python calculate_metrics.py
# 结果自动写入 output/cosmos_depth_edge/
```

## LightEMMA 当前评估结果（nuCarla 60 scenes）

| Method | ADE_avg↓ | FDE↓ | vs carla |
|--------|---------|-----|---------|
| carla (raw sim) | 2.3372 | 5.2234 | baseline |
| Cosmos d+e+s | 2.4665 | 5.5520 | +5.53% / +6.29% |
| Ditto | 2.3838 | 5.4236 | +1.99% / +3.83% |
| PPD r30 | 2.4133 | 5.3885 | +3.25% / +3.16% |
| WPD drop_ll r30 J=5 | 2.2336 | 5.0061 | −4.44% / −4.16% |
| WPD baseline r30 | **2.2097** | **4.9527** | **−5.46% / −5.18%** |
| Cosmos depth+edge | TBD (running) | — | — |
| DNAEdit | TBD (HPC) | — | — |

## nuCarla 翻译状态

| Variant | Scenes | Status |
|---------|--------|--------|
| carla (input) | 0000–0059 | ✅ done |
| ppd r30 | 0000–0059 | ✅ done |
| wavelet (WPD baseline r30) | 0000–0059 | ✅ done |
| extension (WPD drop_ll r30 J5) | 0000–0059 | ✅ done |
| ditto | 0000–0059 | ✅ done |
| cosmos d+e+s | 0000–0059 | ✅ done |
| cosmos depth+edge | 0000–0059 | ✅ done (60/60) |
| dnaedit | 0000–0059 | ⏳ pending HPC |

## 上下文积累诊断

**DNAEdit 参数：**
- guide_scale=1.0 (source CFG), tgt_guide_scale=5.0, jmp=12, seed=42
- nuCarla 视频：49 帧, 10fps, 1280×704 — 逐帧连续读取正确
- FSDP 4-GPU；dist.barrier() 在 save 之后；/usr/bin/ffmpeg (libx264)
- 已验证：skip_denoise roundtrip 保存正常；完整去噪后 scene_0000/0001/0002 已正常保存（但用的 guide_scale=5.0，需重跑）

**LightEMMA：**
- predict_and_eval_carla.py: 每 2 帧 (5Hz)，3 Gemini calls/frame；已有 JSON 则跳过
- calculate_metrics.py: intersection of scenes across methods；macro-avg（场景数相同则等价于 micro-avg）
- Config: LightEMMA/config.yaml → api_keys.gemini

**Apptainer：**
- wpd-dnaedit.sif: 8.3 GB，位于 repo 根目录
- 从 docker-daemon://wpd-dnaedit:latest 构建
- SLURM 脚本: sbatch_dnaedit_nucarla.sh（需更新 REPO_DIR / DATA_DIR）
