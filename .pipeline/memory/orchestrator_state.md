# Orchestrator State
_最后同步：2026-06-03 (sync v2)_

## 全局进度看板

| Stage | Status | Notes |
|-------|--------|-------|
| Survey | ✅ done | 12 papers OCR'd; baselines confirmed |
| Ideation | ✅ done | Two-track strategy; vKITTI+Hypersim operating points fixed |
| Experiment | 🚀 in-progress | vKITTI+Hypersim complete; Track A pipeline built, training pending |
| Publication | 🚀 in-progress | Data ready; writing not started |

## 当前活跃任务

| ID | Title | Status | Notes |
|----|-------|--------|-------|
| publication | 论文写作 | in-progress | 所有数据就绪，需立即开始 |
| wan_video_training | Wan low/high LoRA 训练 | in-progress | Low: GPU 0-3 运行中；High: 待启动 |
| track_a_sync_denoising | Track A: synchronized denoising | pending | warp_and_blend_latents() 已实现；需写 sim2real_nucarla_mv_sync.py |

## 最近完成任务（最近5条）

| Task | Date | Key result |
|------|------|-----------|
| Hypersim PPD r24 eval | 2026-06-03 | FID **67.64** (best) / KID 0.0457 / CLIP-IQA 0.6804 / mIoU 0.3014 / DepSSIM 0.8978 |
| Results.md PPD r24 更新 | 2026-06-03 | Hypersim paper table: PPD r24 替换 r20；bold/italic 重新标注 |
| Wan low LoRA 启动 | 2026-06-03 | train_wan_low_dropll.sh，GPU 0-3，from wan2.2-14b-low-step-12400 |
| Track A 训练方向关闭 | 2026-06-03 | v1–v5 全部失败；根本原因：LL 独立随机 + 12% FOV 重叠不足 |
| baseline_r10 vKITTI eval | 2026-06-03 | FID 74.80 / KID 0.0455 / CLIP-IQA 0.7827 / mIoU 45.01 |

## 决策点

1. **开始写论文**: 所有实验数据就绪（vKITTI + Hypersim）；Track A 训练方向已关闭
2. **Track A 下一步**: synchronized denoising（无需训练）或 post-hoc 颜色匹配
3. **Hypersim PPD r24**: HPC 运行中，完成后补全 paper table

## 下一步建议

1. **最高优先级**: 开始论文写作 Method + Experiment 章节
2. **Track A**: 实现 sim2real_nucarla_mv_sync.py（synchronized denoising，已有 warp_and_blend_latents）
3. **快速验证**: post-hoc 颜色匹配——直接修复 seam 色调，无需训练
</content>