# Execution Context
_最后同步：2026-06-14_

## 当前任务

**ID:** publication
**标题:** 论文正文写作（Method + Experiments 节）
**状态:** 待启动

**详细说明:**
所有实验数据、指标、图表均已就绪。写作目标为新 venue 投稿版本（post-ECCV-2026 rejection）。

## 写作原材料入口

**数据**
- `Results.md` — vKITTI + nuCarla paper/full 表，Ablation 1 三观察
- `python summarize_vkitti_eval.py` — vKITTI 一键复现（KID/FID/CLIP-Res/mIoU/DepSSIM/AbsRel/LPIPS）
- `python summarize_nucarla_eval.py` — nuCarla 一键复现（12 列）

**图表**
- `figures/ablation1_4panel.png` — KID 排 + CLIP 排（paper-ready）
- `figures/ablation_clip_residual_v7.png` — CLIP-Res 指标说明图

**Slides 作为写作大纲**
- `ss26/src/slides/Slide09_Method.tsx` — 方法流程图
- `ss26/src/slides/Slide10_VkittiResults.tsx` — vKITTI 结果
- `ss26/src/slides/Slide12_MethodComparison.tsx` — 方法对比
- `ss26/src/slides/Slide13_Ablation*.tsx` — Ablation 1
- `ss26/src/slides/Slide16_LightEMMAResults.tsx` — CARLA 结果

**论文目录**（ECCV-2026-WPD/）
- `sections/` — 现有各节草稿（需基于新结果全面更新）

## 指标体系（定稿，勿再改动）

**CLIP-Residual v7**（5 对合成锚定 prompt）：
- plastic 材质×3 + render blur + rendered bloom
- 显示名 = CLIP-IQA（+ synthetic-residual prompts 脚注）

**vKITTI paper table 口径**：PPD r20 / WPD baseline r12 / WPD drop_ll J4 r12 / FlowEdit / DNAEdit / Cosmos d+e / REAL_KITTI
**nuCarla paper table 口径**：carla / DNAEdit / VACE / Cosmos d+s+v+e / PPD / dropll(ours)

## 上下文积累诊断（踩坑记录）

- FID/KID 在 image editing 任务中已被社区放弃（paired task 需 conditional metrics）；ψ-PD 是 distribution-level 任务故继续使用
- KID 有混杂（baseline r24 KID 0.0759 > raw input 0.0606，而 input 是全场最不真实图像）；写作需同时报 CLIP-Res 解耦
- nuCarla dropll vs ppd/wavelet 逐 scene Wilcoxon n.s.，写作措辞固定为"一致排序 + 多证据联合"
- 理论框架（线性算子 + flow matching commutation）已关闭，不在正文展开

## 待处理事项

- Wan video drop_ll LoRA 训练状态待确认（长期 in-progress，实际进度未知）
- 论文 venue 选择未定（CVPR 2027 / ICCV 2027 / TPAMI）
