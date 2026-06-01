# Gap Matrix — sim2real baselines survey
_生成: 2026-06-01_

## 已调研论文（sim2real-baselines corpus）

| arXiv | 标题 | 方向 | 任务类型 | 可作 baseline? | OCR |
|-------|------|------|---------|---------------|-----|
| 2602.06159 | Driving with DINO | Foundation model sim2real | 视频生成（sim→real video） | ✅ 直接对比（同一问题定义） | ✓ |
| 2604.01761 | Control-DINO | DINO feature conditioning | I2V diffusion conditioning | ⚠️ 更偏 novel view synthesis | ✓ |
| 2505.16360 | CACTI (Style Transfer for Sim2Real) | 图片 style transfer | GTA5→Cityscapes image translation | ✅ 直接 baseline，有代码 | ✓ |
| 2503.07598 | VACE | 视频编辑 | All-in-one video creation/editing | ✅ reviewer 点名 | ✓ |
| 2510.15742 | DITTO | 视频编辑 | Instruction-based video editing | ✅ reviewer 点名 | ✓ |
| 2602.16664 | SSB | Unpaired I2I translation | Self-supervised semantic bridge | ⚠️ 主要测 medical imaging | ✓ |
| 2506.06185 | Antithetic Noise | 噪声设计 | Structured noise for diffusion | ❌ 非 sim2real，相关工作 | ✓ |
| 2512.14099 | ViewMask-1-to-3 | 多视角一致性 | Multi-view consistent image generation | ⚠️ 不是 sim2real，多视角 | ✓ |

## 研究空白分析

### Gap 1：Foundation model 特征 vs. Wavelet 相位 作为 sim2real bridge
- **Driving with DINO** 用 DINO VFM 特征解决 Consistency-Realism Dilemma
- WPD 用 wavelet 相位解决同一问题
- 两者是**直接竞争**：feature-space bridge vs. frequency-space bridge
- **WPD 优势**：无需 VFM inference overhead，纯信号处理，无需额外模型

### Gap 2：图片 sim2real 没有 2025 年的强 baseline
- CACTI (2505.16360) 是唯一直接做 GTA5→Cityscapes 的 2025 方法，有代码
- 其他方法（SSB）主要测医学图像，迁移性未验证

### Gap 3：视频 sim2real 方法定义不统一
- VACE/DITTO 是通用视频编辑，不是专门的 sim2real
- Driving with DINO 是 sim2real 但偏向 video generation，不是 translation
- WPD 是唯一结构保留的 sim2real 视频翻译方法

### Gap 4：多视角一致性 sim2real 基本空白
- ViewMask-1-to-3 做多视角一致性生成，但不是 sim2real domain transfer
- 没有找到专门做多视角 sim2real 的 2025 方法
