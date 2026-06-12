# Execution Context
_最后同步：2026-06-13_

## 当前任务

**ID:** repo_commit → theory_e1 + writing
**标题:** 主仓库 commit 收尾，随后理论实验 + 写作
**状态:** commit 进行中

## 指标体系（定稿，勿再改动）

**CLIP-Residual v7**（`calc_clipiqa_prompts_nucarla.py::RESIDUAL_V7_PROMPTS`，`--prompt_set residual_v7`）：
plastic 材质×3 + render blur + rendered bloom。Slides/论文显示名 = CLIP-IQA（+synthetic-residual prompts 脚注）。

**一键复现入口**：
- `python summarize_vkitti_eval.py`（列序 KID/FID/CLIP-Res/CLIP-IQA/mIoU/DepSSIM/AbsRel/LPIPS）
- `python summarize_nucarla_eval.py [--pct]`（12 列：KID/FID/sKID/sFID/CMMD/CLIP-Res/MS/ADE×4/FDE）
- 图：`python plot_ablation1_4panel.py --label-radii`；`plot_ablation_clip_residual.py --set v2..v7`

**日志地图**：
- logs/vkitti_eval/<v>_clipres.log（44 含 REAL_KITTI）；logs/nucarla_eval/<v>_{clipres,sfid}.log（各10）
- logs/kid_nucarla.log（解析时跳过 RESULTS 汇总段）；logs/cmmd_nucarla.log
- eval_video/evaluation_results/*_eval_results.json（MS，10/10，scene<60 过滤）
- LightEMMA/output/<m>/scene_*.json（逐帧→scene→macro 均值）

## 待办：主仓库 commit 内容

- .gitignore（已更新：根 npy/npz、dinov2_emb、models/ppd|ufd、tmp_*带tag、nucarla_eval 白名单）
- 新评估脚本 ~20 个（calc_*、summarize_nucarla_eval.py、plot_ablation1_4panel.py 等）
- Results.md、figures/（7 张新图）、logs/{vkitti,nucarla}_eval 新日志
- submodule 指针：ss26 (48c6fdc)、eval_video (3910ba4)
- 未决：fft-vs-dtcwpt/ 目录是否入库（用户定）

## 理论实验（commit 后的 P0）

E1 子带域判别性（CPU 1天，per-subband AUC → drop_ll 原理图）；E2 信息预算重分析（半天）。
设计细节见 tasks.json theory_e1/e2 与 06-11 冲刺计划。

## 上下文积累诊断（踩坑记录）

- conda run 缓冲 log 到进程结束；.venv 直跑可实时
- pgrep -f 自匹配（门控脚本）；vbench 用官方原始命令、按 cwd 隔离输出
- KITTI 真实帧混合分辨率 → batch 按尺寸分桶
- kid log 末尾 RESULTS 汇总段会污染逐段解析
- 指标教训与 prompt 合规标准见 project_truth 决策记录
