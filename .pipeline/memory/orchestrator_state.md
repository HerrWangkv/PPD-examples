# Orchestrator State
_最后同步：2026-06-01_

## 全局进度看板

| Stage | Status | Notes |
|-------|--------|-------|
| Survey | ✅ done | sim2real-baselines corpus added (12 papers OCR'd) |
| Ideation | ✅ done | WPD J=4 r12 main op point; 3D noise projection idea saved for next paper |
| Experiment | ✅ done (main) | vKITTI full + Hypersim 5 variants + ablation figs; 3 runs in-progress |
| Publication | 🚀 in-progress | Writing not started; ECCV rejected, resubmission prep |

## 当前活跃任务

| ID | Title | Status |
|----|-------|--------|
| publication | 论文写作 | in-progress |
| ppd_r32_vkitti | PPD r32 vKITTI | in-progress (HPC) |
| hypersim_dnaedit | DNAEdit Hypersim translation | in-progress (~43%) |
| hypersim_ppd_r20 | Hypersim PPD r20 rsync + eval | in-progress |

## 最近完成任务

| Task | Date | Key result |
|------|------|-----------|
| sim2real-baselines literature survey | 2026-06-01 | 12 papers downloaded + OCR'd; gap matrix written |
| Hypersim bold/italic fix | 2026-06-01 | summarize_hypersim_eval.py supports **best**/*2nd* |
| vKITTI PPD r8–r24 eval | 2026-06-01 | PPD r20 best FID 76.29; paper table updated |
| Hypersim Cosmos + mIoU + CLIP-IQA | 2026-06-01 | drop_ll wins KID/mIoU/AbsRel; Cosmos wins DepSSIM |
| Ablation 1 figure update | 2026-06-01 | 3 curves KID x-axis, reads from logs |

## 待处理决策点

1. **Start writing**: highest priority — all data ready
2. **New baselines to run for resubmission**:
   - CACTI (2505.16360) — has GitHub code, direct sim2real image baseline
   - Driving with DINO (2602.06159) — direct competitor, check inference code
   - VACE (2503.07598) + DITTO (2510.15742) — video baselines, reviewer wxAN requested
3. **Video section**: Add video results with existing Wan pipeline + VACE/DITTO comparison
4. **FVD metric**: needed for video section per reviewer wxAN
5. **Runtime/memory numbers**: easy to add, multiple reviewers requested

## 下一步建议

Priority order for resubmission:
1. **Write paper** — start with Results section (data complete)
2. **Run CACTI + Driving with DINO** as new image baselines (check code availability)
3. **Video section**: Wan pipeline results on nuCarla + VACE/DITTO comparison
4. **Update related work**: cite Driving with DINO, RL3DEdit, 3D-Consistent MV Editing, CACTI
