# Project Truth

**项目名称**: WPD / ψ-PD — Wavelet Phase-Preserving Diffusion for Sim-to-Real Translation  
**论文状态**: 已投稿，两轮评审（2/4/4 分），post-rebuttal，准备改进后重投  
**核心贡献**: DTCWT-based structured noise injection，在扩散去噪过程中保留 sim 输入的相位结构（几何/边缘），实现 sim-to-real 翻译

## 核心方法

- **wavelet_noise.py**: DTCWT 分解 → phase from sim, magnitude from Gaussian noise → 重建结构化噪声
- **drop_ll**: 去除 LL subband（全局光照）用噪声替换，消除 sim 全局亮度偏差
- **Flux stage**: FLUX.1-dev + PPD LoRA，单帧 sim → real
- **Wan stage**: Wan2.2-I2V-A14B + WPD LoRA（high/low），视频时序一致性

## 命名约定

- `PPD<r>`: 只用 FLUX，cutoff radius=r
- `WPD<flux_r>_<wan_r>`: 两阶段，Flux + Wan
- `WPD<r>_dropll_J<j>`: 带 drop_ll 的版本

## 约束（不可突破）

- 训练数据：**只用真实数据**（无配对 sim-real）
- 推理：**无额外 conditioning**（无 depth、seg 等）
- 方法论必须在频域噪声设计层面创新
