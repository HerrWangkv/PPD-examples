# Project Truth
_最后同步：2026-06-14_

## 研究主题

**ψ-PD — Phase-Preserving Diffusion for Sim-to-Real Translation**

DT-ℂWPT-based structured noise injection：保留 sim 输入的相位结构，通过 drop_ll（LL 子带随机化）去除低频全局光照偏差，实现无额外 conditioning 的 sim-to-real 图像/视频翻译。

**论文状态**: ECCV 2026 rejected (2/4/4)。新 venue 重投准备中。对外命名统一为 **ψ-PD**。

## 当前阶段

**Publication — 实验 + 指标 + slides 全部定稿；正文写作待启动**

## 已确认决策

- [2026-05-21] drop_ll 推理 J=4；prompt 去 "dashboard"
- [2026-05-29] WPD J=4 r12 = vKITTI operating point；PPD=FFT；Hypersim = J=5 r24
- [2026-06-03] Track A 多视角训练关闭
- [2026-06-11] sFID/sKID 命名对标 DwD（EPE 协议）；reference-based 指标 sensor-anchored 结论
- [2026-06-12] 单调性 = 指标纯度判据；prompt 合规标准（负面必须锚定合成属性）
- [2026-06-13] v7 最终确认并全量正式评估；KID 混杂反例论证（baseline r24 KID 0.0759 差于 raw input 0.0606）
- [2026-06-13] nuCarla paper table 口径：carla / DNAEdit / VACE / Cosmos d+s+v+e / PPD / dropll(ours)；WPD baseline 留 ablation；Ditto 排除
- [2026-06-13] 表格列序统一 KID → FID → CLIP；slides 中指标名 CLIP-IQA（+ prompt 脚注）
- [2026-06-14] 理论实验 E1–E5（线性算子框架）正式关闭，不入论文
- [2026-06-14] 天气/光照可控性实验关闭，定性有趣但不入论文
- [2026-06-14] Slides 收尾完成：Application slide 新增、线性算子理论 slide 删除、Conclusion 重写、CARLA results 精炼

## 阶段进展摘要

### Experiment（全部收口）

**CLIP-Residual v7（最终指标）** = 5 对合成锚定 prompt（plastic 材质×3 + render blur + rendered bloom）：
- 验证：40/41 联合约束；唯一例外 dropll r8/r12 平局（−0.0007）
- 正式评估：54 per-variant 日志（logs/vkitti_eval/ ×44 含 REAL_KITTI；logs/nucarla_eval/ ×10）
- vKITTI：ours 0.4481 第一；nuCarla：dropll 0.4340 第一（ditto 0.5045 排除在 paper table）

**nuCarla 全指标定稿**（summarize_nucarla_eval.py，12 列）：
- Motion smoothness 10/10；dropll 0.9840（翻译组第 2）
- sFID/sKID（EPE/DwD 协议）：baseline sKID 0.0133 全场第一
- LightEMMA：dropll ADE_avg −4.44%、baseline −5.46%（唯二改善规划精度）

### Publication

- **Results.md 定稿**：vKITTI/nuCarla paper + full 表，Ablation 1 三观察
- **Figures**：ablation1_4panel.png，ablation_clip_residual_v2–v7 系列
- **Slides（ss26 ea88f82，PPD-examples c60f5b3）**：
  - Slide_Application：instance-level style transfer + before/after 比较滑块（portal lightbox，scroll-to-zoom）
  - 删除 Slide16b_Theory（线性算子）
  - Slide16_LightEMMAResults → "CARLA: Quantitative Results"，3 chip key findings
  - Slide17_Conclusion：两列布局，去除 Future Directions，字号增大
  - 全 deck ψ-PD 命名
- **正文写作：未启动**

## 当前最佳实验结果（paper 入口）

- **vKITTI** ours (ψ-PD J=4 r12)：KID 0.0441 / FID 73.84 / CLIP-Res 0.4481 / mIoU 43.50
- **nuCarla** ours (dropll r30 J=5)：CLIP-Res 0.4340 + ADE 全 horizon 最优 + FDE 最优（paper table）；MS 0.9840 第 2
- **nuCarla** baseline r30：sKID 0.0133 全场第一、LightEMMA −5.46%（full table 留档）

## 已关闭方向

- [2026-06-14] 天气/光照可控性实验 — 定性有趣但不入论文
- [2026-06-14] 理论实验 E1–E5（线性算子框架）— 不入论文
- [2026-06-03] Track A 多视角训练 — 5 次尝试均失败（LL 独立性 + FOV 重叠不足）

## 风险 / 阻塞项

- **正文写作未启动** — 唯一剩余主线（Method + Experiments 节）
- nuCarla dropll vs ppd/wavelet 逐 scene 不显著 — 写作措辞已在 Results.md 固定
- Wan video drop_ll LoRA 训练状态长期未确认
