# Execution Context
_最后同步：2026-05-30_

## 当前任务

**ID:** hypersim_dropll_J5_r24
**标题:** Hypersim drop_ll J=5 r=24 翻译（支持 lighting claim）
**状态:** in-progress（GPU 0-3）
**详细说明:**
- 输入：`/mrtstorage/datasets_tmp/hypersim/` (7402 images)
- 输出：`/mrtstorage/users/kwang/hypersim_dropll_J5_r24/`（symlink: `outputs/hypersim/dropll_J5_r24`）
- 脚本：`bash run_hypersim_dropll.sh --gpus 0,1,2,3 --J 5 --radius 24`
- Lora：`models/train/FLUX.1-dev_lora_wpd_dropll/step-6000.safetensors`
- 进度：~1552/7402，ETA ~9h

## 排队任务

**drop_ll ablation（training vs inference 贡献分离）**

完整 2×2 消融矩阵（以 r=12 为基准）：

| | No drop_ll flag | drop_ll flag (J=4) |
|--|--|--|
| **flux.safetensors** (WPD baseline lora) | WPD baseline r12 ✅ done | 变体 A 待做 |
| **step-6000** (drop_ll lora) | 变体 B 待做 | WPD J=4 r12 ✅ done |

- 变体 A：`ablation_wpd_lora_dropll_J4_r12` — `flux.safetensors` + `--flux_drop_ll --flux_J 4`（推理时 drop_ll 贡献）
- 变体 B：`ablation_dropll_lora_nodropll_r12` — `step-6000` lora，无 `--flux_drop_ll`（训练时 drop_ll 贡献）
- 输入：`/mrtstorage/datasets_tmp/vkitti/neutral_flat/`
- GPU：0-3（Hypersim 完成后）
- lora 路径：`flux.safetensors` = `models/train/FLUX.1-dev_lora_wpd/step-20000.safetensors`（即 WPD baseline lora）

## 评估配置（vKITTI，已完成）

```bash
python calc_fid_vkitti.py --gen_folder outputs/vkitti/<variant> --clone_only
python calc_clipiqa_vkitti.py --gen_folder outputs/vkitti/<variant> --clone_only
python calc_miou_vkitti.py --gen_folder outputs/vkitti/<variant> --clone_only
python calc_depth_metrics_vkitti.py --gen_folder outputs/vkitti/<variant> --clone_only
python calc_lpips_vkitti.py --gen_folder outputs/vkitti/<variant>
python summarize_vkitti_eval.py
```

## 上下文积累诊断

- **mrtstorage 权限**：Docker 写入需先 `chmod 777`（host 创建后 Docker root 被 root-squash 拒绝）
- **torchrun**：Hypersim 用 `nproc_per_node=4`；vKITTI 变体用独立 Docker + RANK/WORLD_SIZE
- **flux.safetensors**：WPD baseline lora（step-20000）≠ 旧 PPD lora（flux1-dev_phipd_lora_302000）
- **GPUs 4-7**：被其他进程占用，当前只有 0-3 可用

## 待处理 Agent 反馈

- Hypersim eval 脚本硬编码路径，需重构为 `--gen_folder` 接口后才能评估 drop_ll 变体
- `run_vkitti_ablation_dropll.sh` 尚未创建
