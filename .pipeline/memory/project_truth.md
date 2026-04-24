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
- **v5 step-1000 did NOT improve on v4 step-1000** (validated 2026-04-24 on `test1.jpg`+`test1.txt`, cfg=2.0, 50 steps): final DINO distance 0.656 vs v4's 0.62; peak 0.722 @ step 46 vs v4's 0.66 @ 45. v5 warm-started from v4 step-1000 and drifted *away* from v4's basin. Run stopped at step ~1020 to avoid wasting 8h of compute.
- **v4 step-1000 remains the best-known DinoPD checkpoint.** Do not resume v5 from its step-1000.
- **v5 mechanism did not stress what it was designed to stress.** Realized `sigma_gap` distribution: median 0.076, max 0.16 — far below the 0.3 sampled ceiling. The gap is capped by the realized timestep-grid spacing near `sigma_start`, not by the sampled ceiling. So v5's "expose model to coarse drift" regime almost never fires; effective training distribution is v4-like with K∈{1..7} substep variation on top. Training loss floor 0.70 (v4 was 0.59) — different objective surface, not strictly improvement.
- **v6 implementation (DinoPD-v6, 2026-04-24)**:
  - **Fixed sigma_start=1.0**: Training rollout always starts from sigma=1.0 using Adam-optimized z1*.
  - **Rationale**: v5 sampled sigma_start and used fresh z_t* seeds, but at inference, sigma=0.5 latents are reached via Euler rollout from sigma=1.0. This created a distribution mismatch (fresh seeds vs drifted latents). v6 ensures training integration path matches inference integration path from the same noise distribution.
  - **Rollout steps K ∈ [5, 15]**: Randomized to simulate accumulated drift. K=1 is excluded to ensure the model always sees off-manifold inputs.
  - **Safety floor sigma_target_min=0.1**: Retained to prevent rectified target blow-up (1/σ amplification). Mid-sigma (0.4–0.7) structural failure band is fully covered.
  - **Deprecated**: `max_sigma_gap` and `min_substep_sigma` are removed/ignored as v6 covers the full trajectory.
  - **Output path**: `./models/train/FLUX.1-dev_lora_dino_pd_v6`.
  - **Baseline**: Warm-started from v4 step-1000 checkpoint.
  - **sigma_target distribution (per-sample, not TB mean)**: `timestep_id_target ~ Uniform(1, valid_upper≈968)` in id-space. FLUX shift=3 maps this nonlinearly to σ-space — per-sample σ_target range is **[0.1, 0.95]** with median ~0.7, and **~14% of samples fall in σ_target ∈ [0.1, 0.4]**. Earlier claim that "[0.1, 0.4] is unsampled / 90% in [0.6, 1.0]" was a misreading of TB 8-GPU-mean aggregates; the per-sample distribution is broader. Real implication: low-σ regime gets proportionally less supervision due to shift=3 skew, not zero supervision. If desired, v6.1 can sample σ_target uniform in σ-space and reverse-look the closest id.
  - **Failure position unchanged across versions**: val/dino_peak_step = 45 (v4), 46 (v5), 42–47 (v6). v6 lowers final dino distance but does NOT move the mid-σ peak location. Framing as "failure pushed to end of trajectory" was incorrect — the peak is where it always was.
  - **v6 progress (live, step 100→200→300 on test1)**: val/dino_final = 0.648 → 0.59 → **0.58**. v6 step-300 already lower than v5 step-1000 (0.656) and below v4 step-1000's 0.62 (but v4's 0.62 was under old 0.5·cls+0.5·patch metric — not apples-to-apples). val/dino_step_00 ≈ 0.045 confirms z_1* optimization converges to DINO-patch floor at σ=1.
  - **Loss scale 10× larger than v5 is structural, not a bug**: training loss median 2.25, max 13.94 (v5 max was 1.35). Full-rollout gap up to σ≈0.6, combined with `sigma_target_min=0.1` → rectified target `(z_target-z0)/σ_target` can amplify 10× at σ_target=0.1 and has 7× larger raw magnitude than v5's clamped gap of 0.08. MSE scales quadratically. So far val improvement confirms training is stable at this scale, but if val regresses, lower `sigma_target_min` ceiling or reduce gap.
  - **TODO**: rerun `validate_dino_trajectory.py` on v4 step-1000 under patch-only metric to establish apples-to-apples baseline. Current v4 baseline (0.62) was measured under the old 0.5·cls+0.5·patch mix. ✅ Done 2026-04-24: v4 step-1000 @ (cfg=2, emb=3.5) = **0.683 final**. Still need v4 @ (cfg=1, emb=3.5) for v7-α-matched baseline.

- **v7-α decision (2026-04-24)** — abandon CFG at inference; collapse train/infer mismatch to a single scalar:
  - **Rationale**: 2×2 ablation showed CFG dominates structure destruction (+0.12 alone) and the prompt-dropout workaround (v7-ε) is theoretically disproven (CFG is not a linear structure-preserving operation even with both branches trained — see prior block). Rather than complicate training to defend against CFG (v7-β, 2× cost), commit to cfg=1 at inference. FLUX-dev's `embedded_guidance=3.5` provides internal distilled-guidance-based sharpness; we don't need runtime CFG for prompt amplification (DinoPD doesn't care about prompt adherence — it cares about structure).
  - **Implementation**: ONE scalar change `train_dino_pd.py:95` `embedded_guidance: 1` → `3.5`. Zero extra training cost vs v6.
  - **Inference config committed**: `cfg_scale=1.0, embedded_guidance=3.5`. Updated as default in both `validate_dino_trajectory.py` and embedded `run_validation`.
  - **Warm-start**: from **v4 step-1000** (not v6 — v6 was trained at embed=1 and would bias the LoRA toward that regime).
  - **Launcher**: `DinoPD-FLUX.1-dev.sh` updated in-place (output_path = `FLUX.1-dev_lora_dino_pd_v7a`).
  - **Expected gain**: v6@A1 config (train-match) gave 0.471 but blurry (FLUX at embed=1 is blurry). v7-α's config is embed=3.5 everywhere → LoRA learns structure preservation within FLUX's detailed-generation regime → should give ≈A1 structure + A2-or-better visuals.
  - **Paper narrative simplification**: no "CFG-aware training" chapter needed; DinoPD's inference is single-branch model_fn, identical to training. Cleaner story for reviewers.
  - **Tradeoff accepted**: runtime prompt-adherence knob (cfg) is gone. For sim2real, this is acceptable — structure preservation is the central goal, not prompt following.
  - **CFG / embedded_guidance ablation (2×2 + v4 baseline, 2026-04-24) — RESULTS**:

    Ran v6 step-300 under 4 inference configs + v4 step-1000 under current default. Test1 prompt, patch-only DINO metric.

    | Config | cfg | embed | Final DINO | Peak | Visual quality (user-observed) |
    |---|---|---|---|---|---|
    | A1 (train-match) | 1.0 | 1.0 | **0.471** | 0.471 | **Very blurry** — structure preserved but detail lost |
    | A2 (embed-mismatch only) | 1.0 | 3.5 | 0.495 | 0.537 | Visually OK (may improve with more training), **structure insufficient** |
    | A3 (cfg-mismatch only) | 2.0 | 1.0 | 0.594 | 0.594 | not inspected |
    | A4 (current val default) | 2.0 | 3.5 | **0.505** | 0.519 | Sharpest of the v6 set |
    | B (v4 step-1000 @ A4) | 2.0 | 3.5 | **0.683** | 0.683 | v4 baseline under patch-only metric |

    **Real v6 → v4 improvement under apples-to-apples (A4): 0.683 → 0.505 = −0.18.** This is the headline paper number, not the old 0.62→0.58 mixed-metric comparison.

  - **Key finding: DINO distance alone is NOT sufficient metric**. A1 has lowest DINO (0.471) but is blurry — VGGT would fail on its output. Must combine DINO with VGGT confidence / image sharpness in final evaluation. A1's low DINO reflects "blurred-away both noise and detail" not "structure preserved with detail".

  - **Contribution decomposition (relative to A1 = train-match baseline)**:
    - embed=1→3.5 alone adds +0.024 to DINO distance (structure cost of enabling FLUX's internal distilled guidance)
    - cfg=1→2 alone adds +0.123 to DINO distance (CFG is the larger structure-destroying term)
    - Both together (A4) adds +0.034 — **negative interaction**: embed=3.5 partially compensates for cfg=2's damage. embed=3.5 fixes DiT's conditioning to high-confidence state; CFG's v_uncond subtraction is less destabilizing there.

  - **Theoretical constraint on "simple" CFG training fixes**: A naive proposal was to train both v_posi and v_uncond branches with DinoPD objective via prompt dropout (so both push toward z0). **This does NOT guarantee v_cfg = 2·v_posi − v_uncond preserves DINO.** Three reasons:
    1. DINO-preservation at z_t is not a linear subspace but a local nonlinear constraint (tangent space of `DINO(decode(·)) ≈ DINO(decode(z0))`). Linear combinations like CFG may push off this manifold.
    2. Prompt effect (v_posi − v_uncond) is NOT orthogonal to the DINO-preservation direction because prompt conditioning alters texture/material via attention, which DINO patch features encode.
    3. If both branches trained strictly to push to z0 → v_posi ≈ v_uncond → CFG becomes identity → degrades to A1 regime (blurry).
    There is a genuine tension: sharpness requires the v_posi − v_uncond delta to be non-trivial; structure-preservation requires it to be orthogonal to DINO-structure directions; FLUX's prompt conditioning does not satisfy orthogonality.

  - **Therefore the only way to guarantee v_cfg preserves DINO is direct supervision of v_cfg**, not of the branches separately. This is v7-β's design.

  - **Caveat on fixed-cfg training**: v7-β (or -lite) hardcodes a specific cfg_scale (e.g. 2.0) into training rollout. At inference, users may prefer different cfg values. Training to one cfg_scale may generalize poorly to others. Options: (a) randomize cfg_scale in training to learn a CFG-robust LoRA, (b) train at cfg=2 and accept that's the only officially-supported inference cfg, (c) train at cfg=1 and rely on FLUX-base's CFG-aware velocities to compose cleanly (unclear if valid).

  - **v7 options ranked (pending experimental validation)**:
    | Version | Changes vs v6 | Cost | Theoretical guarantee | Preferred tried order |
    |---|---|---|---|---|
    | v7-α | `embedded_guidance` 1→3.5 in training (single scalar change) | 1× v6 | Resolves embed mismatch only; cfg mismatch remains | 1st (zero extra cost, fast null check) |
    | v7-β-lite | v7-α + at σ_target loss, compute v_posi and v_nega, supervise `v_cfg = v_nega + cfg·(v_posi−v_nega)` against rectified target. Rollout still single-branch. | 1.3× v6 | Partial: loss aligns v_cfg, but rollout still uses single-branch velocities → residual train/inference drift | 2nd (if α insufficient) |
    | v7-β | v7-β-lite + rollout uses `v_cfg` per substep (2× forward everywhere) | 2× v6 | Full: both rollout and loss are CFG-aware | 3rd (if β-lite insufficient) |
    | ~~v7-ε (prompt dropout)~~ | Rejected — does not compose through CFG, see theoretical analysis above | — | None | — |

  - **Historical note on CFG in training (originally suspected root cause of stuck mid-σ peak across v4/v5/v6)**: v4/v5/v6 all showed val/dino_peak at step 45-47 regardless of training-side changes. Version invariance suggested the failure was at inference guidance, not training rollout coverage. The 2×2 ablation above confirmed this: A1 (train-match) eliminates the peak (DINO stays low across trajectory). **CFG+embed mismatch IS the root cause of the stuck peak.** Two independent mismatches between training and inference:

    | Mechanism | Training value | Inference default | Effect |
    |---|---|---|---|
    | `cfg_scale` (classic CFG) | 1.0 (no v_uncond computed at all) | 2.0 | At inference: `v = v_nega + 2·(v_posi − v_nega) = 2v_posi − v_nega`. LoRA never saw v_nega during training; v_nega points toward unconditional data manifold, not z0 → may push mid-σ latent AWAY from z0. |
    | `embedded_guidance` (FLUX-dev distilled guidance scalar, fed as a token via `FluxImageUnit_GuidanceEmbedder`) | 1.0 (weak) | 3.5 (typical) | The guidance token is an input to the DiT; different values change the internal conditioning before the velocity is computed. Independent of cfg_scale. |

    Training forward only computes `v_train = model_fn(z_t, prompt, embed_g=1)` — single forward, weak built-in guidance, no v_uncond. Inference uses `v_infer = 2·model_fn(z_t, prompt, embed_g=3.5) − model_fn(z_t, null, embed_g=3.5)`. The LoRA is queried under a velocity field it was never trained on.

    - **2×2 ablation (~1 hour total, 1 GPU, no training interruption)** using `validate_dino_trajectory.py` on v6 step-N:

      | cfg_scale | embedded_guidance | Meaning |
      |---|---|---|
      | 1.0 | 1.0 | Matches training exactly — lower bound on DinoPD drift |
      | 1.0 | 3.5 | Isolates embedded_guidance contribution |
      | 2.0 | 1.0 | Isolates CFG contribution |
      | 2.0 | 3.5 | Current validation config (upper bound) |

      Needs CLI flag for `--embedded_guidance` in `validate_dino_trajectory.py` (currently hardcoded to pipe default 3.5).

    - ~~Decision rules after ablation~~: **superseded by actual 2×2 results above**. Summary: CFG dominates (+0.12 alone), embed secondary (+0.024 alone), negative interaction at A4 (+0.034 combined). Conclusions recorded in "CFG / embedded_guidance ablation" block above.

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
