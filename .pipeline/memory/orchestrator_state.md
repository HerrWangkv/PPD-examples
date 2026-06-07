# Orchestrator State
_最后同步：2026-06-07_

## 全局进度看板

| Stage | Status | Notes |
|-------|--------|-------|
| Survey | ✅ done | 12 papers OCR'd; baselines confirmed |
| Ideation | ✅ done | Two-track strategy; vKITTI+Hypersim operating points fixed |
| Experiment | ✅ done | vKITTI+Hypersim+nuCarla LightEMMA complete; 2 pending evals (cosmos, extension) |
| Publication | 🚀 in-progress | Data ready; writing not started |

## 当前活跃任务

| ID | Title | Status | Notes |
|----|-------|--------|-------|
| nucarla_lightemma_cosmos_depth_edge | Cosmos d+e LightEMMA eval | pending | 50/60 translated; confident_pascal running; run eval after done (filter _control_*.mp4) |
| nucarla_dnaedit_translate | DNAEdit 60-scene translation | pending | Fix imageio[ffmpeg] in run_nucarla_dnaedit.sh; needs GPUs 0-3 after cosmos |
| nucarla_dnaedit_lightemma | DNAEdit LightEMMA eval | pending | After translation |
| nucarla_eval_video | eval_video (patch-sFID + motion_smoothness) new variants | pending | eval_video submodule added; update eval.sh; variants: cosmos_depth_edge, dnaedit |
| publication | 论文写作 | in-progress | 所有数据就绪，需立即开始 |

## 最近完成任务（最近5条）

| Task | Date | Key result |
|------|------|-----------|
| nuCarla LightEMMA eval (5 methods) | 2026-06-07 | WPD baseline r30 best: ADE_avg −5.46%, FDE −5.18% vs raw sim |
| extension_dropll_r30_J5 translation | 2026-06-07 | 100 scenes (0060–0159) complete |
| Cosmos depth+edge translation (partial) | 2026-06-07 | 50/60 done, running |
| wavelet_noise.py hf_only mode | 2026-06-07 | Added for mv pipeline; no effect on single-view |
| Hypersim PPD r24 eval | 2026-06-03 | FID **67.64** (best on Hypersim) / KID 0.0457 / mIoU 0.3014 |

## 决策点

1. **Gemini key**: 旧 key `AIzaSyC6I2igYAcvR9Uhyhz4wKURh0Su5lUJr9I` 暴露在 GitHub — 需立即撤销并在 `LightEMMA/config.yaml` 更新新 key
2. **开始写论文**: 所有 benchmark 数据就绪（vKITTI + Hypersim + LightEMMA E2E）
3. **Track A**: closed; synchronized denoising 仍是可选项但非必须

## 下一步建议

1. **最高优先级**: 撤销并更新 Gemini key → 运行 extension + cosmos LightEMMA eval
2. **论文写作**: Method + Experiment 章节，三个 benchmark 数据全部就绪
3. **可选**: synchronized denoising (sim2real_nucarla_mv_sync.py) — 如果需要 Track A 展示
