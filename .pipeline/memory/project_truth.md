# Project Truth

## 研究主题
DinoPD — Learned structure-preserving diffusion via DINOv2 features. Replaces hand-designed frequency-domain noise structure (PPD/WPD) with a learned training objective that embeds input structure into the flow-matching trajectory through DINO-preserving noise seeds.

## 已确认决策
- `z1*` (DINO-preserving noise endpoint) must remain in training — structure must enter through `z_t` since DiT never sees a DINO target at inference.
- v2 aux `L_dino` was vacuous because training input `z_t = z_t*` made `x0_hat ≈ z0` automatic; rejected as a standalone fix.
- v3→v4 showed clamping the rollout sigma gap was a real fix (not just loss-scale cosmetic); max loss dropped ~80×, validation final DINO distance dropped ~0.13.
- v5 randomizes both rollout gap `[0.05, 0.3]` and substep count `[1, 15]` per sample to cover drift regimes the model encounters at inference.
- No CFG in training rollout (`cfg_scale=1, embedded_guidance=1`), no `L_dino` — rejected for v5 (compute cost + vacuous-signal risk).
- v4 step-1000 outperforms step-2000 → warm-start basin overfit; best v4 checkpoint is step-1000.
- Evaluation prompt matters: `test1.txt` vs generic prompt shifts validation DINO curve by ~0.2 — always use task-matched prompt.

## v5 implementation details (from `train_dino_pd.py`)
- **Two independent safety floors** beyond the clamped gap (both are v5-new, not in v4):
  - `sigma_target_min=0.1` (default raised from 0.05 on 2026-04-23 before first v5 launch) — floor on `σ_target`. Rectified target `v = (z_target - z0)/σ_target` amplifies drift by `1/σ_target`; at 0.05 the gain is 20×, at 0.1 it's 10× — cuts outlier-gradient mass roughly in half without losing meaningful signal.
  - `min_substep_sigma=0.02` — caps `K` by realized gap so substep granularity never exceeds FLUX-50 inference (~0.02 σ/step mid-sigma). `K_eff = min(K_sampled, realized_gap/0.02, span_in_native_timesteps)`.
- **Noise field refreshed every substep**: during the detached K-step Euler rollout, `inputs["noise"]` is recomputed as `(z_curr - (1-σ_k)·z0)/σ_k` before each model_fn call. The `noise` field is conditioning for model_fn, not just a seed — stale noise would feed the DiT an inconsistent view of "what the endpoint looks like at this σ".
- **DINO model**: `dinov2_vitl14_reg` (ViT-L/14 with registers) by default. Loaded once, moved to accelerator device in `set_accelerator`, kept out of the `nn.Module` registry (`object.__setattr__`) to prevent DDP from touching it.
- **Training data**: `bghira/photo-concept-bucket` HF dataset, `cogvlm_caption` as prompt field — *not* a sim-to-real domain dataset. Domain adaptation for Synthia/sim2real happens at inference, not training.
- **Dataset loader**: `HuggingFaceURLImageDataset` with `num_workers=2` hardcoded at `launch_training_task` call; arg `--dataset_num_workers` is ignored for this param only.

## v5 code-correctness audit (resolved, 2026-04-23)
Verified `train_dino_pd.py` logic end-to-end. No bugs. Non-obvious findings to avoid re-investigating:
- `model_fn_flux_image` in `diffsynth/pipelines/flux_image_new.py` does **not** declare `noise` — kwargs are swallowed by `**kwargs` and never reach the DiT. The per-substep `inputs["noise"]` refresh in the K-step rollout loop is dead weight (harmless). `inputs["noise"]` is only load-bearing at `scheduler.training_target(z0, noise, t)` on the final loss step. This means the rollout cannot leak z0 into the DiT via the noise field — no train/inference mismatch there.
- `FlowMatchScheduler.training_weight` reads from pre-tabulated `linear_timesteps_weights[timestep_id]` — no 1/σ or σ scaling. Small `σ_target` does not blow up the weight.
- `FlowMatchScheduler.training_target` returns `noise - sample` = `z1_target - z0`, matching the rectified-flow MSE target.
- **Tuning lever, not bug**: rectified target `v = (z_target - z0)/σ_target` still has 20× gain at `σ_target_min=0.05`. If TB `loss/max` shows outlier flares in v5, raise `--sigma_target_min` to 0.1. This is the same failure mode v4 fixed via gap clamping — the new amplification source is the target reparameterization, not the gap.
- Cosmetic-only issues noted: dead `inputs["noise"] = z1_star` at line 198 (overwritten on first rollout iter); `sampled_max_gap` can be marginally exceeded near top-σ when `sigmas[i+1] < sigma_floor` already.
