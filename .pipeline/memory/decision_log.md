# Decision Log

## 2026-05-21 | Adaptive Radius Map

**想法**: 用 sim latent 的局部高频能量自动计算 per-pixel cutoff radius，替代全局固定 radius。

**设计逻辑**:
- 高梯度区域（建筑轮廓、边缘）= sim 几何可靠 → 大 radius（保留到高频）
- 低梯度区域（天空、路面）= sim 外观不可信 → 小 radius（让模型自由替换）

**实现方式** (inference-time only，无需重训练):
```python
def compute_adaptive_radius_map(latent, r_min=5.0, r_max=50.0, smooth_kernel=9):
    x = latent.mean(dim=1, keepdim=True).float()
    blurred = F.avg_pool2d(x, kernel_size=smooth_kernel, stride=1, padding=smooth_kernel//2)
    hf_energy = (x - blurred).abs()
    energy_smooth = F.avg_pool2d(hf_energy, kernel_size=smooth_kernel, stride=1, padding=smooth_kernel//2)
    N = energy_smooth.shape[0]
    emin = energy_smooth.view(N, -1).min(dim=1)[0].view(N, 1, 1, 1)
    emax = energy_smooth.view(N, -1).max(dim=1)[0].view(N, 1, 1, 1)
    energy_norm = (energy_smooth - emin) / (emax - emin + 1e-8)
    return r_min + (r_max - r_min) * energy_norm
```

**兼容性**: `radius_map` 参数已支持 `(N,1,H,W)` 张量，基础设施已就位。

**超参数**: `r_min`, `r_max`，等价于控制全局 radius 范围。

**验证方法**: 
1. 可视化 radius map（天空应低，建筑边缘应高）
2. 与固定 r=30 对比 AS/SSIM on Synthia

**约束满足**: 无 conditioning，无新训练，纯信号处理。

---

## 2026-05-21 | drop_ll J 选择

**决策**: 推理时用 J=4（非 J=3）。

**原因**: J=3 过于激进（全局灰调），J=4 在去除 lens flare 的同时保留自然光照变化。

**LL 空间大小**: J=4 时 latent 88×160 → LL = 11×10px，仅包含全局亮度/色调信息。

---

## 2026-05-21 | Sim-to-real prompt 修改

**旧 prompt**: 包含 "view from a car dashboard" → 导致 FLUX 生成可见仪表板遮挡（hallucination）

**新 prompt**: "A photorealistic photograph taken from a forward-facing vehicle-mounted camera. Natural outdoor lighting, authentic surface textures, real-world colors."

**新 negative prompt**: 增加 "dashboard, steering wheel, windshield frame, car interior, lens artifacts"

---

## 2026-05-21 | 确认 Baseline Set（FID benchmark on vKITTI→KITTI）

**确认的 baseline 方法集合**：

| 方法 | 类别 | Backbone | 年份 | 代码状态 |
|------|------|----------|------|---------|
| SDEdit | 扩散噪声编辑（经典） | FLUX | 2022 | ✅ 已有脚本 |
| FlowEdit | RF flow 编辑 | FLUX | 2024 | ✅ 已有脚本 (`batch_sim2real_image_flowedit.py`) |
| DNAEdit | SOTA RF inversion 编辑 | FLUX | 2025.06 | ❓ 待接入 (arxiv: 2506.01430) |
| Cosmos-Transfer2.5 | Sim-to-real 专用世界模型 | Cosmos | 2025 | ❓ 待接入 |
| **WPD (ours)** | 结构化 wavelet 噪声注入 | FLUX | — | ✅ 运行中 |

**实现注意事项**：
- FlowEdit 使用 diffusers FluxPipeline (0.30.3)，`encode_prompt` 无 `negative_prompt` 参数，无法添加负向提示词
- WPD 使用 diffsynth fork，支持负向提示词（抑制仪表盘/车内遮挡等 artifact）
- 论文中以脚注说明：FLUX guidance-distillation 架构下负向提示词效果有限，差异可接受（option 1）

**排除的方法及原因**：
- Step1X-Edit：MLLM 架构，定位不同（instruction-following），与 WPD 架构差异过大
- DirectEdit (arxiv: 2605.02417)：与 DNAEdit 同类（RF inversion 误差修正），选 DNAEdit 作代表即可
- ControlNet：违反"无 conditioning"约束（需 depth/canny 输入）
- ICEdit：偏 general editing，不专注 sim-to-real

**论文故事逻辑**：
- SDEdit → FlowEdit → DNAEdit：展示 WPD 对同 backbone 不同 noise 策略的优势
- Cosmos：WPD 对专用 sim-to-real 方法的竞争力

---

## 2026-05-21 | 其他改进方向（论文约束内）

以下方向均满足"无 conditioning + 只用真实训练数据"约束：

1. **训练分布对齐**: 对 r≈30, J=4 附近加重要性采样权重，减少 train-test gap
2. **频域加权 loss**: 低频 band 加大训练 loss 权重，更专注修正光照/色调
3. **时空 3D wavelet**: Wan 阶段在时空联合域注入噪声，改善帧间一致性

已排除方向：
- ~~LL 分布非高斯替换~~: FLUX 本身从 N(0,1) 去噪，高斯替换是正确的
- ~~magnitude 替换~~: 当前 magnitude 已完全来自噪声（leak=0），无需修改

---

## 2026-06-01 | New Venue Submission Strategy: Multi-view First, Video Fallback

**Decision**: For the new venue submission (post-ECCV-rejection), adopt a two-track strategy:

**Track A — Multi-view image sim2real (try first)**
- Core idea: 3D Noise Field Projection — use sim depth to project Gaussian noise into 3D, render per-view, apply WPD noise replacement (LL + high-freq phase + magnitude). Inference-time only, no retraining.
- Baselines:
  - RL3DEdit (2603.03143) — multi-view consistent 3D editing via RL
  - 3D-Consistent MV Editing (2511.22228) — training-free correspondence guidance
  - Cosmos Transfer — per-view with depth conditioning
- Removed: DwD (single-view temporal video only); FlowEdit and CACTI (single-image, no multi-view consistency); Control-DINO (video domain transfer, not multi-view image)

**Track B — Video translation (fallback if Track A fails)**
- Baselines:
  - DNAEdit — strong image/video editing baseline
  - Cosmos Transfer — conditioning-based sim2real (already done)
  - DITTO (2510.15742) — instruction-based video editing
  - VACE (2503.07598) — all-in-one video editing
  - PPD video (sim2real_video_ppd.py / batch_sim2real_video_ppd.py) — full video pipeline, not per-frame
  - DwD (2602.06159) — direct sim2real video competitor
  - Control-DINO (2604.01761) — sim2real video transfer without domain-specific training
- Note: FlowEdit is single-image only — not a video baseline.
- Use existing Wan2.2 pipeline; no new training needed
- Weakness: competitive against DwD without driving-specific video training

**Why multi-view first**: DwD trained on driving-specific video data makes video track hard to win; multi-view sim2real is an open field with no dominant trained competitor; 3D noise projection fits WPD's "no conditioning, inference-time" philosophy.

**Data needed for Track A**: multi-camera synchronized frames (nuCarla multi-cam? vKITTI stereo?), sim depth maps (available in both), camera calibration matrices.
