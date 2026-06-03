# Execution Context
_最后同步：2026-06-03 (sync v2)_

## 当前任务

**ID:** publication + wan_video_training
**标题:** 论文写作 + Wan video LoRA 训练
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
| DNAEdit | **0.7637** | *67.85* | **0.0448** | 0.3169 | 0.8978 | 0.4107 |
| Cosmos depth+edge | 0.6516 | 71.53 | 0.0510 | *0.3236* | **0.9259** | *0.3568* |
| PPD r24 | 0.6804 | **67.64** | *0.0457* | 0.3014 | 0.8978 | 0.3994 |
| WPD J=5 r24 (drop_ll, ours) | 0.7412 | 68.31 | **0.0448** | **0.3772** | *0.9190* | **0.3459** |

## Track A: Multi-view 关键文件

| 文件 | 用途 |
|------|------|
| `wavelet_noise_mv.py` | 3D 噪声投影核心；接受 K/T/depth，不依赖 nuCarla |
| `nucarla_utils.py` | nuCarla 标定解析 → K, T (4×4)，depth 加载，scene/frame 迭代 |
| `sim2real_nucarla_mv.py` | 单帧 6-cam 推理（含 --independent baseline flag） |
| `batch_sim2real_nucarla_mv.py` | 多场景分布式推理 |
| `precompute_da3_nuscenes.py` | DA3 depth 预计算（GPU 0-3 运行中，~34k samples） |
| `examples/flux/model_training/nuscenes_mv_dataset.py` | nuScenes 多视角对数据集（随机采样6相邻pair） |
| `examples/flux/model_training/train_nuscenes_mv.py` | nuScenes finetune 训练脚本（Option A/B OOM自适应） |
| `test_da3_nuscenes.py` | DA3 深度验证 + forward warp 测试脚本 |

## Track A: 训练启动命令（DA3 precompute 完成后）

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --multi_gpu --num_processes 4 \
  examples/flux/model_training/train_nuscenes_mv.py \
  --nuscenes_root /tmp/nuscenes \
  --depth_cache_dir /mrtstorage/users/kwang/nuscenes_da3_depth \
  --model_id_with_origin_paths "black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/,black-forest-labs/FLUX.1-dev:ae.safetensors" \
  --lora_checkpoint models/train/FLUX.1-dev_lora_wpd_dropll/step-6000.safetensors \
  --output_path models/train/FLUX.1-dev_lora_wpd_mv_nuscenes \
  --learning_rate 1e-5 --save_steps 500 \
  --lora_base_model dit --lora_rank 32 \
  --use_gradient_checkpointing
```

## 上下文积累诊断

**vKITTI paper table**:
- PPD = r12（与 WPD J=4 r12 同 radius 直接对比）
- WPD = J=4 r12（paper operating point）
- Ablation 1: `plot_ablation_baseline_vs_ppd.py`；baseline r16 = `baseline_newprompt`；dropll J=4 r16 = `dropll_step6000_J4`

**Hypersim paper table**:
- PPD = r20；WPD = J=5 r24 drop_ll；WPD baseline r20 移除出 paper table
- DNAEdit FID = 67.85（注意：paper 中用 67.85，不是 73.98）
- mIoU: pseudo-GT ADE20K，cache at `outputs/hypersim/.pseudo_gt_cache/`

**Track A 关键约束**:
- wavelet_noise_mv.py 传播的是原始高斯噪声（pre-DTCWT），DTCWT 在最后每相机独立做
- nuCarla extrinsic: q.rotation_matrix（不加 .T，不加 yaw/roll correction）
- DA3 model: `DA3NESTED-GIANT-LARGE-1.1`（-1.1 修复了 street scene bug）
- nuScenes depth cache: `/mrtstorage/users/kwang/nuscenes_da3_depth/<sample_token>/<cam>.npy` (float16)

**通用**:
- CleanFID: `--mode clean` for all FID/KID
- GPU 0-3: DA3 precompute 运行中；GPU 4-7: 其他任务

## 待处理实验 / 任务

| 实验 | 优先级 | 状态 |
|------|--------|------|
| Wan low LoRA 训练 | high | 🔄 运行中（GPU 0-3） |
| Wan high LoRA 训练 | high | ⏳ 待启动（GPU 4-7，train_wan_high_dropll.sh） |
| 论文写作 Method + Experiment | high | ⏳ 待开始（所有数据就绪） |
| Track A synchronized denoising | medium | ⏳ 待实现（sim2real_nucarla_mv_sync.py） |
| Hypersim PPD r24 eval | done | ✅ FID 67.64 / KID 0.0457 / mIoU 0.3014 |
