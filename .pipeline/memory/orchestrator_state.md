# Orchestrator State
_最后同步：2026-06-01_

## 全局进度看板

| 阶段 | 状态 | 备注 |
|------|------|------|
| Survey | ✅ done | baselines 确定（含正确 PPD FFT） |
| Ideation | ✅ done | WPD J=4 r12 主 operating point |
| Experiment | ✅ done (主体) | vKITTI 全量 + Hypersim 5 variants + ablation 图 |
| Publication | 🚀 in-progress | 数据就绪，写作未开始 |

## 当前活跃任务

| ID | 标题 | 状态 |
|----|------|------|
| publication | 论文写作 | in-progress |

## 后台进行中

| 实验 | 进度 | 备注 |
|------|------|------|
| DNAEdit Hypersim 翻译 | ~43% (3185/7402) | 需重启 Docker |

## 最近完成任务

| 任务 | 完成时间 | 关键结果 |
|------|----------|---------|
| Hypersim bold/italic 修正 | 2026-06-01 | summarize_hypersim_eval.py 支持 **best** / *2nd* |
| Results.md 结构优化 | 2026-06-01 | vKITTI + Hypersim paper table 置顶 |
| vKITTI PPD r8–r24 正确评估 | 2026-06-01 | PPD r20 最佳 FID 76.29，paper table 更新 |
| Hypersim Cosmos + mIoU + CLIP-IQA | 2026-06-01 | drop_ll wins KID/mIoU/AbsRel；Cosmos wins DepSSIM |
| Ablation 1 图更新 | 2026-06-01 | 3 曲线 KID x 轴，从 logs 读取 |

## 待处理决策点

1. **开始写作**：最高优先级，数据全部就绪
2. **PPD r32**：sbatch 脚本已准备（sbatch_inference_vkitti_ppd_r32.sh），等待 HPC 提交
3. **DNAEdit 重启**：`bash run_hypersim_dnaedit.sh --gpus 0,1,2,3`

## 下一步建议

**立即开始论文写作** — `/omp:write`
- vKITTI Results section（paper table + ablation）
- Hypersim lighting claim 段落
- Related work（DNAEdit 2506.01430, Cosmos-Transfer2.5）
