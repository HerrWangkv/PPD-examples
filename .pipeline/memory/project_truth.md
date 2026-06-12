# Project Truth
_最后同步：2026-06-13_

## 研究主题

**WPD / ψ-PD — Wavelet Phase-Preserving Diffusion for Sim-to-Real Translation**

DTCWT-based structured noise injection：保留 sim 输入的相位结构，通过 drop_ll（coarse-band randomization）去除低频全局光照偏差，实现无额外 conditioning 的 sim-to-real 图像/视频翻译。

**论文状态**: ECCV 2026 rejected (2/4/4)。新 venue 重投准备中。对外命名统一为 **ψ-PD**（slides 已全部替换 WPD）。

## 当前阶段

**Publication — 指标体系与全部结果定稿；slides 已更新；待理论实验 E1–E5 与正文写作**

## 已确认决策

- [2026-05-21] drop_ll 推理 J=4；prompt 去 "dashboard"
- [2026-05-29] WPD J=4 r12 = vKITTI operating point；[06-01] PPD=FFT；Hypersim = J=5 r24
- [2026-06-03] Track A 多视角训练关闭
- [2026-06-11] sFID/sKID 命名对标 DwD（EPE 协议）；reference-based 指标 sensor-anchored 结论
- [2026-06-12] 单调性 = 指标纯度判据；prompt 合规标准（负面必须锚定合成属性）
- [2026-06-13] **v7 最终确认并全量正式评估**（用户拍板）；KID 混杂的反例论证（baseline r24 KID 0.0759 差于 raw input 0.0606，而 input 是全场最不真实图像）
- [2026-06-13] nuCarla paper table 口径：carla / DNAEdit / VACE / Cosmos d+s+v+e（最佳 cosmos）/ PPD / dropll(ours)；WPD baseline 留 ablation；Ditto 排除（指令式编辑非 sim2real，full table 保留）
- [2026-06-13] 表格列序统一 KID → FID → CLIP；slides 中指标名沿用 CLIP-IQA（框架正确引用）+ prompt 脚注

## 阶段进展摘要

### Experiment（全部收口）

**CLIP-Residual v7（最终指标）**= 5 对合成锚定 prompt（plastic 材质×3 + render blur + rendered bloom）：
- 验证：40/41 联合约束（nuCarla 10 变体排序、vKITTI paper 排序、radius 单调性、matched-radius、frontier 支配）；唯一例外 dropll r8/r12 平局（−0.0007，同 FID 低结构端行为）
- 正式评估：54 个 per-variant 日志（logs/vkitti_eval/<v>_clipres.log ×44 含 REAL_KITTI=0.2570；logs/nucarla_eval/<v>_clipres.log ×10）
- vKITTI：ours 0.4481 第一（vs ppd p=4e-115）；全表最优 = WPD J=3 r12 0.4633
- nuCarla：ditto 0.5045 > dropll 0.4340 > wavelet 0.4329 > ppd 0.4321；逐 scene n.s.（写作只述排序）

**nuCarla 全指标定稿**（summarize_nucarla_eval.py 一键复现，12 列）：
- Motion smoothness 10/10（VBench 官方命令；dropll 0.9840、vace 最佳 0.9847 翻译组）
- sFID/sKID（EPE/DwD）：baseline sKID 0.0133 全场第一超 Ditto；LightEMMA：baseline −5.46% / dropll −4.44% 唯二大幅改善

### Publication

- **Results.md 定稿**：vKITTI paper/full 表（v7 + KID/FID/CLIP 列序）；nuCarla paper table（6 行，ours CLIP+LightEMMA 六列全冠、MS 第2）+ 12 列 full table；Ablation 1 三条观察（结构保持 / 无幻觉真实感 / KID 混杂 vs CLIP 解耦 + r24 反例）
- **图**：figures/ablation1_4panel.png（KID 排 + CLIP 排，paper 候选）；ablation_clip_residual_v2–v7 系列
- **Slides（ss26 commit 48c6fdc 已推送）**：S11 CLIP-IQA 重定义 + photorealistic/simulated 示意图；S12 v7 表；S13/14 4-panel + baseline r24 + key findings 重写；S17(=Slide15) 三评估维度；S18(=Slide16) paper+Ditto 结果表（page-12 配色标准）；全 deck ψ-PD
- **正文写作未启动**

## 当前最佳实验结果（paper 入口）

- vKITTI ours (ψ-PD J=4 r12)：KID **0.0441** / FID *73.84* / CLIP-Res **0.4481** / mIoU **43.50**（四列第一或第二）
- nuCarla ours (dropll r30 J=5)：CLIP-Res 0.4340 + ADE 全 horizon + FDE 六列全冠（paper table 口径）；MS 0.9840 第 2
- nuCarla baseline r30：sKID 0.0133 全场第一、LightEMMA −5.46% 全场第一（full table 留档）

## 风险 / 阻塞项

- **理论实验 E1–E5 与正文写作仍未启动**（指标战役 + slides 共耗三天）
- nuCarla dropll vs ppd/wavelet 逐 scene 不显著 — 写作措辞已在 Results.md 固定（排序 + 多证据联合）
- 主仓库 commit 进行中（gitignore 已更新；ss26/eval_video 指针待入库）
- Wan video drop_ll LoRA 训练状态长期未确认
