# Orchestrator State

_最后更新: 2026-05-21_

## 当前阶段
**实验执行** — 核心训练/推理任务并行运行中

## 运行中任务
- `wan_low_training`: GPU 0-3，step ~2，~6h 到第一个 checkpoint
- `batch_inference_60scenes`: GPU 4/5，~16-17/60 完成，~1h 到完成

## 待解决
- Wan high training 还未启动（GPU 0-3 被 Wan low 占用）
- batch inference 完成后需要立刻跑定量评估

## 下一个里程碑
1. batch inference 完成 → 跑 Synthia mIoU eval（今日）
2. Wan low 第一个 ckpt (step-100) 出现 → 检查 loss curve（~6h 后）
3. 评估完成 → 决定是否需要继续训练或调整超参数

## 近期决策记录
- drop_ll J=4（不是 J=3）用于推理
- prompt 修改去除 dashboard artifact
- Adaptive radius map 作为下一个待实现的方法改进（见 decision_log.md）
