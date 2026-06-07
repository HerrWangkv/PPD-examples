# Experiment Ledger

## 已完成实验

### WPD baseline（全局 radius，无 drop_ll）
- 方法: `flux.safetensors` LoRA + Wan LoRA，wavelet noise r=30，无 drop_ll
- 结果: Synthia mIoU WPD20_20=28.46，WPD30_30=29.18，WPD40_40=32.27
- 状态: ✅ 完成，作为 baseline

### FLUX drop_ll 训练
- 训练脚本: `examples/flux/model_training/lora/WPD-FLUX.1-dev-dropll.sh`
- GPU: 0-3，warm-start from `flux.safetensors`，LR=1e-5，rank=32
- 噪声分布: `r = 4 + Exp(16)`，drop_ll prob=0.8，J~Uniform[max(auto_J,3), auto_J+4]
- 收敛情况: ~step 1000-2000 视觉稳定，step 6462 停止训练
- 最优 checkpoint: **step-6000**（J=4 效果优于 J=3）
- 保存路径: `models/train/FLUX.1-dev_lora_wpd_dropll/step-{1000..6000}.safetensors`
- 状态: ✅ 完成

### Batch inference scene_0000–0059（nuCarla）
- GPU 4: WPD baseline (flux.safetensors, r=30, new prompt) → `outputs/batch_scene0000_0059/baseline_newprompt/`
- GPU 5: FLUX drop_ll step-6000 J=4 r=30 new prompt → `outputs/batch_scene0000_0059/dropll_step6000_J4_newprompt/`
- 进度 (2026-05-21): baseline=16/60, dropll=17/60，预计~1.5小时完成
- 状态: 🔄 运行中

### Wan low LoRA drop_ll 训练
- 脚本: `train_wan_low_dropll.sh`
- GPU: 0-3，warm-start from `models/ppd/wan2.2-14b-low-step-12400.safetensors`
- 进度 (2026-05-21 07:15): step ~2，~3.5 min/step，first ckpt at step-100 ≈ 6h
- 输出: `models/train/Wan2.2-I2V-A14B_low_lora_wpd_dropll/`
- 状态: 🔄 运行中

### Wan high LoRA drop_ll 训练
- 脚本: `train_wan_high_dropll.sh`（已准备）
- 状态: ⏳ 待启动

### nuCarla → LightEMMA E2E eval (2026-06-07)
- **方法**: Gemini-2.5-flash 从前视频预测 waypoints；ADE/FDE at 1s/2s/3s
- **数据**: 60 scenes (scene_0000–0059)，每帧 3 Gemini calls
- **结果**: WPD baseline r30 best: ADE_avg=2.2097 (−5.46%); WPD drop_ll r30 J=5: ADE_avg=2.2336 (−4.44%); PPD/Ditto/Cosmos 均使规划精度下降
- **关键发现**: WPD 是唯一改善规划精度的方法
- **状态**: ✅ 完成；Results.md 已更新

### extension_dropll_r30_J5 翻译 + cosmos_depth_edge 翻译 (2026-06-07)
- extension (scenes 0060–0159): ✅ 100 scenes done
- cosmos_depth_edge (scenes 0000–0059): 🔄 50/60 done, running
- LightEMMA eval on both: ⏳ 待 Gemini key 更新 + cosmos 完成

## 待做实验

### 定量评估 drop_ll 效果
- 在 Synthia 上跑 mIoU（FLUX drop_ll step-6000 vs WPD baseline）
- 在 Hypersim 上跑 depth SSIM
- 估计时间: batch inference 完成后 ~1 天

### Adaptive Radius Map（见 decision_log）
- 无需新训练，inference-time 修改
- 需要: 实现 `compute_adaptive_radius_map`，可视化 radius map，对比 AS/SSIM
