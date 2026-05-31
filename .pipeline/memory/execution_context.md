# Execution Context
_最后同步：2026-05-31_

## 当前任务

**ID:** 待分配
**状态:** GPU 空闲，等待用户决策

## 排队任务

### 优先 1：Ablation Variant B（完成 2×2 矩阵）

| | No drop_ll flag | drop_ll flag J=4 |
|--|--|--|
| flux.safetensors | WPD baseline r12 ✅ | Ablation infer r12 ✅ |
| step-6000 lora | **Variant B 待做** | WPD J=4 r12 ✅ |

- 变体：step-6000 lora + no `--flux_drop_ll`，radius=12，vKITTI neutral_flat
- 命令参考：`batch_sim2real_image_wavelet.py --flux_lora models/train/FLUX.1-dev_lora_wpd_dropll/step-6000.safetensors --flux_cutoff_radius 12`（无 `--flux_drop_ll` 标志）
- 注意：`flux.safetensors` = `models/train/FLUX.1-dev_lora_wpd/step-20000.safetensors`（WPD baseline lora）

### 优先 2：Hypersim dropll_J5_r24 重启

- 命令：`bash run_hypersim_dropll.sh --gpus 0,1,2,3 --J 5 --radius 24`
- 输出：`/mrtstorage/users/kwang/hypersim_dropll_J5_r24/`（已有 ~1552/7402）
- 注意：output dir 权限 chmod 777 已设置

### 优先 3：Hypersim eval 脚本重构

- 文件：`calc_fid_hypersim.py`, `calc_depth_metrics_hypersim.py`, `calc_as_hypersim.py`
- 问题：硬编码实验路径，需添加 `--gen_folder` 接口（参考 calc_fid_vkitti.py 模式）
- 输出目录：`logs/hypersim_eval/`

## 2×2 消融结论（at r12）

| | FID | mIoU |
|--|--|--|
| WPD baseline (flux lora, no drop_ll) | 89.61 | 47.22 |
| Ablation A (flux lora + infer drop_ll) | 74.54 | 42.97 |
| Ablation B (drop_ll lora, no infer drop_ll) | **待跑** | **待跑** |
| WPD J=4 (drop_ll lora + infer drop_ll) | 75.72 | 43.50 |

**预期 Variant B**：FID ≈ 89 (无 inference-time LL zeroing → FID 不改善)，mIoU ≈ 46-47（lora 训练本身对 mIoU 的影响）

## 上下文积累诊断

- **mrtstorage 权限**：Docker 写入需先 `chmod 777`（host 创建目录后 Docker root 被 root-squash 拒绝）
- **clone_only**：`batch_sim2real_image_kontext.py` 有此 flag；`batch_sim2real_image_wavelet.py` 无（全量处理）
- **flux.safetensors ≠ 旧 PPD lora**：flux.safetensors = step-20000 WPD baseline lora
- **GPUs 4-7**：被历史进程占用，当前只有 0-3 可用
- **Hypersim torchrun**：用 `--nproc_per_node=4`，需 `--gpus '"device=0,1,2,3"'`
