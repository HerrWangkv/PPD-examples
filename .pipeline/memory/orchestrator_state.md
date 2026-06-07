# Orchestrator State
_最后同步：2026-06-07_

## 全局进度看板

| Stage | Status | Notes |
|-------|--------|-------|
| Survey | ✅ done | 12 papers OCR'd; baselines confirmed |
| Ideation | ✅ done | Two-track strategy; vKITTI+Hypersim operating points fixed |
| Experiment | 🔄 in-progress | vKITTI+Hypersim+LightEMMA 5-method done; cosmos d+e eval running; dnaedit pending HPC |
| Publication | 🚀 in-progress | Data nearly complete; writing not started |

## 当前活跃任务

| ID | Title | Status | Notes |
|----|-------|--------|-------|
| nucarla_lightemma_cosmos_depth_edge | Cosmos depth+edge LightEMMA eval | 🔄 running | tmux session; ~88% done (scene_0053/60) |
| nucarla_dnaedit_translate | DNAEdit 60-scene translation | ⏳ blocked | guide_scale fixed to 1.0; wpd-dnaedit.sif ready; needs HPC transfer + sbatch submit |
| nucarla_dnaedit_lightemma | DNAEdit LightEMMA eval | ⏳ pending | After translation |
| nucarla_eval_video | eval_video (patch-sFID + motion_smoothness) new variants | ⏳ pending | cosmos_depth_edge + dnaedit |
| publication | 论文写作 | 🚀 in-progress | 所有数据就绪，需立即开始 |

## 最近完成任务（最近5条）

| Task | Date | Key result |
|------|------|-----------|
| DNAEdit guide_scale fix + Apptainer 打包 | 2026-06-07 | guide_scale 5.0→1.0；wpd-dnaedit.sif (8.3 GB) 构建完成 |
| nuCarla LightEMMA eval (5 methods) | 2026-06-07 | WPD baseline r30 best: ADE_avg −5.46%, FDE −5.18% vs raw sim |
| extension_dropll_r30_J5 translation | 2026-06-07 | 100 scenes (0060–0159) complete |
| Cosmos depth+edge translation | 2026-06-07 | 60/60 done |
| Hypersim PPD r24 eval | 2026-06-03 | FID **67.64** (best on Hypersim) / KID 0.0457 / mIoU 0.3014 |

## 决策点

1. **DNAEdit HPC**: 将 wpd-dnaedit.sif + repo rsync 至 HPC，更新 sbatch_dnaedit_nucarla.sh 中 REPO_DIR/DATA_DIR，提交 sbatch
2. **开始写论文**: vKITTI + Hypersim benchmark 数据全部就绪；nuCarla LightEMMA 5/7 methods done（cosmos d+e + dnaedit 待补）
3. **Gemini key**: 旧 key `AIzaSyC6I2igYAcvR9Uhyhz4wKURh0Su5lUJr9I` 暴露 — 需撤销

## 下一步建议

1. **立即**: rsync `wpd-dnaedit.sif` + repo → HPC，更新路径，`sbatch sbatch_dnaedit_nucarla.sh`
2. **cosmos LightEMMA 完成后**: 运行 `cd LightEMMA && python calculate_metrics.py` 更新结果表
3. **论文写作**: Method + Experiment 章节，三个 benchmark 数据全部就绪
