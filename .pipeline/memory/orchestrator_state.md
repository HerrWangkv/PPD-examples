# Orchestrator State
_最后同步：2026-06-01_

## 全局进度看板

| Stage | Status | Notes |
|-------|--------|-------|
| Survey | ✅ done | sim2real-baselines corpus (12 papers OCR'd) |
| Ideation | ✅ done | Two-track strategy decided; 3D noise projection saved for next paper |
| Experiment | ✅ done (main) | vKITTI full (incl. r32) + Hypersim 6 variants; DNAEdit in-progress |
| Publication | 🚀 in-progress | Writing not started; new venue submission |

## 当前活跃任务

| ID | Title | Status |
|----|-------|--------|
| publication | 论文写作 + 新实验 | in-progress |
| hypersim_dnaedit | DNAEdit Hypersim translation | in-progress (~57% remaining, GPUs 0,2,3) |

## 最近完成任务

| Task | Date | Key result |
|------|------|-----------|
| ppd_r32_vkitti | 2026-06-01 | FID 88.09 / KID 0.0620 / mIoU 46.40 — extends Ablation 1 into KID~0.062 |
| hypersim_ppd_r20 | 2026-06-01 | FID 70.53 / KID 0.0484 — competitive with Cosmos without conditioning |
| ablation figure fix | 2026-06-01 | J=4 r16 fixed, J=5 removed, PPD r32 added, clean 3-curve |
| sim2real-baselines survey | 2026-06-01 | 12 papers OCR'd; gap matrix; two-track baseline lists |
| vKITTI PPD r8–r24 eval | 2026-06-01 | PPD r20 best FID 76.29 |

## 待处理决策点

1. **Track A feasibility check**: do we have multi-camera synchronized sim data? (nuCarla multi-cam?)
2. **WPD baseline r10**: run inference + eval to fill KID gap (r8→r12) in Ablation 1
3. **Start writing**: Results section — all image data complete
4. **New baselines**: CACTI code available (github.com/echigot/cactif); DwD code TBD

## 下一步建议

1. **Write paper** — Results + ablation sections first (all data ready)
2. **Check Track A data**: confirm nuCarla multi-camera availability
3. **Run WPD baseline r10**: quick inference + eval, sharpens Ablation 1
