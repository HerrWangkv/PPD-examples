# Execution Context
_最后同步：2026-06-07_

## 当前任务

**ID:** publication + pending_lightemma_evals
**标题:** 论文写作 + cosmos/extension LightEMMA eval
**状态:** in-progress

## LightEMMA 评估结果（nuCarla 60 scenes, 5 methods）

Run: `cd LightEMMA && python calculate_metrics.py`

| Method | ADE_1s↓ | ADE_2s↓ | ADE_3s↓ | ADE_avg↓ | FDE↓ |
|--------|---------|---------|---------|---------|-----|
| carla (raw sim) | 0.5200 | 2.0076 | 4.4841 | 2.3372 | 5.2234 |
| Cosmos d+e+s | 0.5197 | 2.1143 | 4.7657 | 2.4665 (+5.53%) | 5.5520 (+6.29%) |
| Ditto | 0.4951 | 2.0218 | 4.6345 | 2.3838 (+1.99%) | 5.4236 (+3.83%) |
| PPD r30 | 0.5315 | 2.0785 | 4.6298 | 2.4133 (+3.25%) | 5.3885 (+3.16%) |
| WPD drop_ll r30 J=5 | 0.4927 | 1.9165 | 4.2916 | 2.2336 (−4.44%) | 5.0061 (−4.16%) |
| WPD baseline r30 | **0.4851** | **1.8972** | **4.2468** | **2.2097 (−5.46%)** | **4.9527 (−5.18%)** |

Results.md: section "nuCarla → LightEMMA" already added.

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
| DNAEdit | **0.7637** | *67.85* | **0.0448** | 0.3169 | 0.8978 | 0.4107 |
| Cosmos depth+edge | 0.6516 | 71.53 | 0.0510 | *0.3236* | **0.9259** | *0.3568* |
| PPD r24 | 0.6804 | **67.64** | *0.0457* | 0.3014 | 0.8978 | 0.3994 |
| WPD J=5 r24 (drop_ll, ours) | 0.7412 | 68.31 | **0.0448** | **0.3772** | *0.9190* | **0.3459** |

## nuCarla 翻译状态

| Variant | Scenes | Status |
|---------|--------|--------|
| carla (input) | 0000–0059 | ✅ done |
| ppd r30 | 0000–0059 | ✅ done |
| wavelet (WPD baseline r30) | 0000–0059 | ✅ done |
| extension (WPD drop_ll r30 J5) | 0000–0059 | ✅ done |
| ditto | 0000–0059 | ✅ done |
| cosmos d+e+s | 0000–0059 | ✅ done |
| extension_dropll_r30_J5 | **0060–0159** (100 scenes) | ✅ done |
| cosmos_depth_edge (no seg) | 0000–0059 | 🔄 50/60 done; container running |

## 待处理命令

```bash
# 1. Cosmos depth+edge LightEMMA (once 60 scenes done):
bash run_lightemma_eval.sh outputs/nucarla/cosmos_depth_edge cosmos_depth_edge
# Note: filter _control_*.mp4 — predict_and_eval_carla.py only processes scene_XXXX.mp4

# 2. DNAEdit translation (after cosmos frees GPUs 0-3):
# First fix imageio[ffmpeg] in run_nucarla_dnaedit.sh (add pip install before torchrun)
bash run_nucarla_dnaedit.sh

# 3. DNAEdit LightEMMA:
bash run_lightemma_eval.sh outputs/nucarla/dnaedit dnaedit

# 4. eval_video new variants (patch-sFID + motion_smoothness):
cd eval_video
python eval_patch_sfid.py --real_dir /tmp/nuscenes --fake_dir ../outputs/nucarla/cosmos_depth_edge > cosmos_depth_edge.log
vbench evaluate --dimension motion_smoothness --videos_path ../outputs/nucarla/cosmos_depth_edge --mode=custom_input >> cosmos_depth_edge.log
# (repeat for dnaedit)

# 5. After all evals, rerun metrics:
cd LightEMMA && python calculate_metrics.py
```

## 上下文积累诊断

**LightEMMA**:
- `predict_and_eval_carla.py`: processes every 2nd frame (5Hz), 3 Gemini calls/frame; resumes from existing scene JSONs
- `calculate_metrics.py`: intersection of scenes across all output dirs; fair comparison
- Scene JSON format: `frame["metrics"][m]` for ADE_1s/2s/3s/avg, FDE
- Config: `LightEMMA/config.yaml` with `api_key:` — gitignored to prevent exposure
- Old Gemini key `AIzaSyC6I2igYAcvR9Uhyhz4wKURh0Su5lUJr9I` was exposed; must revoke at Google Cloud Console

**Cosmos translation**:
- Skip logic: checks `$OUT_DIR/${scene}.mp4` before adding to REMAINING
- Output: 3 files per scene (translated + control_depth + control_edge)
- Container name: confident_pascal; GPUs 0-3; NVIDIA_VISIBLE_DEVICES=all

**wavelet_noise.py hf_only**:
- Added for mv pipeline; no effect on single-view (all existing call sites use default `hf_only=False`)

**mv stashes** (git stash list):
- `stash@{0}`: debug/test scripts (debug_noise_warp_vis.py, debug_warp_*.py, etc.)
- `stash@{1}`: core mv modifications (wavelet_noise_mv.py, nucarla_utils.py, sim2real_nucarla_mv.py, etc.)

## eval_video 现有结果（motion_smoothness）

| Variant | motion_smoothness↑ | patch_sfid | 路径 |
|---------|-------------------|-----------|------|
| carla (raw sim) | 0.9858 | missing log | `/mrtstorage/users/kwang/nucarla_videos/rgb/` |
| cosmos (d+e+s) | 0.9881 | missing log | `/mrtstorage/users/kwang/nucarla_videos_cosmos/` |
| ditto | 0.9801 | missing log | `/mrtstorage/users/kwang/nucarla_videos_ditto/` |
| ppd | 0.9830 | missing log | `/mrtstorage/users/kwang/nucarla_videos_ppd/flux_30_wan_30/` |
| wavelet (WPD baseline) | 0.9861 | missing log | `/mrtstorage/users/kwang/nucarla_videos_wavelet/flux_30_30_1_wan_30_30_1/` |

eval_video submodule at `eval_video/`. New variants need eval: cosmos_depth_edge (filter `_control_*.mp4`), dnaedit.

## 待处理实验 / 任务

| 实验 | 优先级 | 状态 |
|------|--------|------|
| Gemini key revoke + update | critical | ⚠️ 阻塞 cosmos/dnaedit LightEMMA eval |
| cosmos_depth_edge LightEMMA eval | high | ⏳ 等 translation 完成 (50/60) |
| DNAEdit translate (fix imageio[ffmpeg]) | high | ⏳ 等 cosmos 释放 GPU 0-3 |
| DNAEdit LightEMMA eval | high | ⏳ 等 translation |
| eval_video 新 variants | medium | ⏳ 等 translation 完成 |
| 论文写作 Method + Experiment | high | ⏳ 待开始 |
