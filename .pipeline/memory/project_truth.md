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

  - **v7-α result (failure mode = blurry-but-structure-preserving, 2026-04-24)**: val/dino_final 0.464 (step 100) → 0.460 (step 300) → **0.429 (step 1000)** — DINO monotonically improves, but visual output progressively blurs. Step 30 (σ=0.66) latent decode is sharp/structured; steps 40 (σ=0.41) and 49 (σ=0.009) progressively smooth detail away. Reproduces A1's "low DINO + blurry" pathology at fewer training steps. embed=3.5 in training did NOT solve A1's blur — the `embed-mismatch` branch of the 2×2 was a misdiagnosis of the blur root cause.

  - **Real root cause of blur (diagnosed 2026-04-24): sigma sampling skew × 1/σ amplification**.
    - `timestep_id_target ~ Uniform(1, valid_upper)` is uniform in id-space. FLUX shift=3 maps this to a heavily right-skewed σ distribution: per-sample buckets are 25.9% in [0.9,1.0], 32.4% in [0.7,0.9), 26.5% in [0.4,0.7), 10.9% in [0.2,0.4), **only 4.3% in [0.1,0.2)**.
    - Rectified target `(z_target - z0)/σ_target` has gain `1/σ_target`; MSE contribution `1/σ_target²`. At σ=0.1 a sample contributes 100× the gradient mass of a σ=1 sample.
    - The 4% of low-σ samples thus dominate the total loss (training-loss spikes to 14 are these). Their gradient direction is "drag z toward z0". z0 is a smooth low-freq VAE-decoded image and DINO patch loss further biases low-freq → at σ=0.1 (latent already mostly clean) this means "smooth high-freq detail toward z0" = blur. Mid/high-σ majority samples train trivial-denoising, not detail preservation.
    - Net: LoRA learns aggressive low-σ pull-to-z0 from a small number of huge-gradient samples. Structure (low-freq) preserved; detail (high-freq) erased.

  - **v7-b (sigma_uniform sampling, embed=3.5 in training, warm-start v7a/step-200)** — *blur unchanged, falsifies "sample count was the bottleneck"*.
    - `--sigma_sampling sigma_uniform`: inverse-CDF `σ ~ Uniform(0.1, sigmas[1]≈1.0)`, snap to nearest timestep_id. Realized distribution: ~11% per 0.1-band → low-σ [0.1,0.4] now 33.7% of samples (was 15.2%). Single CLI flag added (`--sigma_sampling`) defaulting to `id_uniform` for backwards compat.
    - **TB metrics (1179 steps, run 20260424_220006)**: `rollout/sigma_target` median ~0.5–0.7, `rollout/sigma_gap` ~0.4 (sampling working). Loss noisy 2.8–8 with no downward trend.
    - **Validation (test1, cfg=1, embed=3.5)**: val/dino_final 0.616 (s100) → 0.482 (s600) → **0.446 (s900)** → 0.514 (s1100, regress). val/cls_final dropped to 0.265 at s900. **val/dino_peak_step shifted from 45–47 (v4/v5/v6) to 49 (last step)** — failure now localized to the final 5–10 low-σ steps.
    - **Visual (s900)**: σ=0.66 (step 30) sharp+structured, σ=0.41 (step 40) structure+detail intact, **σ=0.009 (step 49 / output_final.png) heavy blur emerges in the last few steps**. Best v7b ckpt: step-900.
    - **Read**: sigma_uniform's rebalanced sample count made the blur **worse, not better** — the additional low-σ samples each carry 1/σ² gradient mass and aggregate pressure now even more dominant than v6. Pure intervention (A) was insufficient AND counterproductive on its own.

  - **Refined blur diagnosis (post-v7b)**: rectified MSE target `v_pred = (z_target − z0)/σ_target` **fully specifies the velocity** at supervised timesteps. The LoRA has zero degrees of freedom left to allow embed_g's prompt-aligned amplification to flow through — whatever embed_g contributes at training time, the LoRA learns to subtract to hit the target. Three regimes, all bad:
    - train=1, infer=1 (A1): structure ✓, blur ✗ — no embed_g detail at all
    - train=1, infer=3.5 (v6@A4): embed_g adds detail ✓, structure drifts (+0.024) ✗
    - train=3.5, infer=3.5 (v7-α/b): structure ✓, LoRA cancels embed_g detail ✗ → blur

  - **v7-c (sigma_uniform + train embed=1, warm-start v6/step-300, 2026-04-25 → 04-26)** — *blur resolved, structural fidelity sacrificed.*
    - Setup: `embedded_guidance` 3.5 → 1 in `train_dino_pd.py:98` (revert to v6 setup); `--sigma_sampling sigma_uniform` kept; warm-start `models/train/FLUX.1-dev_lora_dino_pd_v6/step-300.safetensors`; output `FLUX.1-dev_lora_dino_pd_v7c`.
    - **TB metrics (2591 steps, run 20260425_085128)**: loss median flat 4.3–4.8 across all bins (no downward trend, same plateau as v7-α/b). val/dino_final wobbles 0.46–0.58, val/cls_final 0.24–0.62, **val/dino_peak_step shifted back to 44–49** (mid-trajectory) — the "blur generated in last 5 steps" pathology of v7-b is gone.
    - **Visual (test1, cfg=1, embed=3.5)**: outputs are sharp, photorealistic across all checkpoints. Skin/hair/fabric/stone all crisp. Embed-cancellation hypothesis confirmed: training at embed=1 lets FLUX-base's distilled-guidance contribution flow through unmolested at inference embed=3.5.
    - **Best ckpt: step-600** (dino_final 0.458, cls_final 0.238, most input-faithful background). Higher steps drift further from warm-start basin as training continues on non-sim2real `bghira/photo-concept-bucket` data.
    - **DINO numbers are now noisy and non-monotonic** → checkpoint selection is visual, not metric-driven. Loss plateau at 4.5 mirrors v7-α/b regardless of embed/sampling settings — rectified-flow MSE has hit its noise floor.
    - **v7-d (σ²-reweighting) cancelled**: the blur was embed=3.5-cancellation, NOT 1/σ² gradient mass amplification. The σ²-reweight hypothesis is falsified by v7c.

  - **DINO trajectory comparison vs PPD/WPD/Gaussian (2026-04-26, `outputs/dino_traj_compare/`)** — *the linearity diagnosis is empirically validated and DinoPD is dominated by linear-invariant baselines on its own metric.*

    Setup: same input (`models/ppd/test1.jpg`), same prompt (`models/ppd/test1.txt`), same seed (42), same scheduler (50 steps), patch-only DINO metric. Each method at its native cfg (PPD/WPD: cfg=2 published; DinoPD: cfg=1 v7c-native), embed=3.5 throughout.

    | Method | LoRA | step 0 | final | shape |
    |---|---|---|---|---|
    | Gaussian (cfg=2) | none | 0.793 | 0.788 | flat ~0.79 — no structure preservation |
    | **PPD r=20 (cfg=2)** | `models/ppd/flux1-dev_phipd_lora_302000.safetensors` | 0.732 | **0.388** | **monotone descent** |
    | **WPD r=20 (cfg=2)** | `models/train/FLUX.1-dev_lora_wpd/step-20000.safetensors` | 0.674 | **0.360** | **monotone descent** |
    | DinoPD v7c step-600 (cfg=1) | `FLUX.1-dev_lora_dino_pd_v7c/step-600.safetensors` | **0.037** | 0.521 | **monotone ascent** |

    **Trajectory shapes (smoking gun)**:
    - **PPD/WPD start moderately mismatched (≈0.7) and monotonically descend toward z0** — exactly the behavior we want, despite *neither method ever being trained on DINO*.
    - **DinoPD starts at the DINO-optimal z1\* (0.037) and monotonically ascends away from it** — the LoRA's nonlinear constraint is enforced exactly only at the optimization point and falls apart under composition with subsequent denoising steps.
    - Gaussian is flat at ~0.79 — confirms ~0.8 is the no-structure-preservation reference.

    **Validates the linearity argument**: a structure constraint that composes cleanly with CFG / embed_g / flow-matching automatically enforces convergence on any downstream feature correlated with the linear invariant (DINO clearly is). DinoPD's nonlinear constraint cannot.

    **Visuals**: PPD r=20 and WPD r=20 outputs preserve Lara's pose, top, belt, gloves, and cave geometry, with FLUX-quality detail added. DinoPD output is a completely different woman in a different room — high visual quality, zero structural fidelity. Low step-0 DINO score is a red herring; the trajectory throws away every gain.

    **Caveat**: DinoPD ran at cfg=1 (favorable v7c regime). At cfg=2 (matching PPD/WPD), expected DinoPD final DINO ≈ 0.65 (+0.12 from 2×2 ablation). Gap would widen, not close.

    **Wrong-LoRA pitfall noted**: initial run used `flux1-dev_lora_color_step=266000_biased.safetensors` for both PPD and WPD modes — that's the published PPD LoRA, not WPD. Real WPD requires the WPD-trained LoRA (`FLUX.1-dev_lora_wpd/step-20000.safetensors`). Default in `validate_dino_trajectory.py` now correctly routes per noise mode.

    **Implication for paper direction**: DinoPD as currently formulated cannot beat WPD on DINO distance — the paper story "we use DINO supervision to get better structure preservation" is empirically false. Two viable rescue directions remain: (a) **tangent-space supervision** of DINO loss (project to locally-DINO-preserving tangent of `decode(z)` so embed_g/CFG/prompt detail flows through the orthogonal complement); (b) **hybrid** — use WPD wavelet noise as the enforced invariant, add a low-weight DINO reward as tie-breaker on `x0_hat`.

  - **Multi-layer DINO noise opt + z1\* variance audit (2026-04-27, `outputs/dino_std_check/`)** — *clean inference-time experiment, no training. Tests whether richer DINO loss (more ViT blocks) gives stronger noise structure, and whether Adam-on-z_t blows up the variance prior FLUX expects.*

    Setup: `validate_dino_trajectory.py` extended with `--dino_loss_layers` flag; `find_dino_preserving_noise` extended with `loss_layers` param that averages `(1 - cos)` across multiple ViT block depths via `dino.get_intermediate_layers()`. Patch tokens only, same spatial position alignment, equal-weighted layer sum. All runs: 300 Adam steps, lr=1e-2, 50-step denoising, cfg=1 (v7c-native), `models/ppd/test1.jpg`, seed=0.

    **z1\* statistics (variance is NOT the bug)**:

    | Mode | mean | std (overall) | std (per-channel) | z_t std drift over 300 steps |
    |---|---|---|---|---|
    | gaussian | -0.003 | **0.999** | 0.999 | — |
    | ppd r=20 | +0.000 | 0.974 | 0.974 | — |
    | wpd r=20 | -0.002 | 0.955 | 0.955 | — |
    | dino baseline (final layer) | -0.000 | **1.024** | 1.024 | 1.011 → 1.022 → 1.024 (slow up-drift) |
    | dino layer-4 only | -0.039 | 0.953 | **0.912** ⚠ | 0.975 → 0.960 → 0.953 (down-drift) |
    | dino [4,11,23] mix | -0.019 | 0.975 | 0.969 | 0.999 → 0.988 → 0.975 (mild down) |

    All modes within 5% of N(0,1) — **no catastrophic variance blowup. Std penalty / renormalization is unnecessary.** The earlier hypothesis "Adam-on-z_t with no prior could push z1\* far from N(0,1)" is falsified for 300 Adam steps at lr=1e-2 on this image.

    **One real concern surfaced**: `dino_layer4` shows per-channel std 0.912 vs aggregate 0.953 → some FLUX latent channels get pushed harder than others. FLUX's 16 latent channels carry asymmetric semantic content; DiT's normalization expects channel-isotropic noise. Layer-4-only's channel anisotropy is the most plausible reason it underperforms multi-layer. Multi-layer [4,11,23] keeps per-channel std nearly aggregate (0.969 vs 0.975) — better behaved.

    **Final DINO trajectory comparison (50 denoising steps)**:

    | Mode | step 0 | final | max @ step | shape |
    |---|---|---|---|---|
    | gaussian (cfg=1) | 0.799 | 0.780 | 0.827 @ 19 | flat, no preservation |
    | **PPD r=20 (cfg=1)** | 0.718 | **0.405** | 0.718 @ 0 | **monotone descent** ✓ |
    | **WPD r=20 (cfg=1)** | 0.673 | **0.375** | 0.673 @ 2 | **monotone descent** ✓ |
    | DinoPD baseline | **0.038** | 0.512 | 0.520 @ 47 | monotone ascent ✗ |
    | DinoPD layer-4 | 0.185 | 0.496 | 0.496 @ 49 | monotone ascent ✗ |
    | **DinoPD [4,11,23]** | **0.019** | **0.425** | 0.425 @ 49 | monotone ascent, but lowest endpoint among DINO modes |

    **Reads:**
    - Multi-layer [4,11,23] is the **best DINO variant**: lowest step-0 (0.019, 2× better than baseline 0.037) AND lowest final (0.425, 17% better than baseline 0.512). Adding early-block constraints both shrinks the optimization basin (early ViT layers carry more spatial info, harder to game) and produces less-anisotropic noise.
    - **Layer-4-only is the worst DINO variant** despite richest spatial signal at step 0 — channel anisotropy + single-layer redundancy lets Adam exploit a narrow direction, hurting downstream LoRA usability.
    - **Multi-layer DINO still cannot match WPD** (final 0.425 vs WPD 0.375). The trajectory shape (monotone ascent) is unchanged — the failure mode is architectural, not loss-strength. Even the DINO-optimal noise diverges away from z0 once denoising composes embed_g + cfg + per-step velocities.
    - **Step-0 DINO distance is misleading**: WPD's step-0 is **35× worse** than DinoPD [4,11,23]'s step-0, but WPD's final is 12% **better**. The figure of merit is trajectory endpoint, not initialization quality.

    **Inspiration for v8**: Multi-layer noise is a clean win at the *noise-generation* stage. The current v7c LoRA was trained on baseline (final-layer) z1\*; retraining the LoRA with `--dino_loss_layers 4 11 23` may compound the gain (LoRA learns to decode multi-layer-DINO-preserving noise, not single-layer). But: the trajectory-shape problem (ascent vs descent) is unsolved — multi-layer is unlikely to invert it on its own. Recommend launching v8 only as a confirmation, not as the headline experiment.

    **Files**: `dino_noise.py:88-112` (`latent_to_dino_multilayer`), `dino_noise.py:131-206` (loss_layers branch in `find_dino_preserving_noise`), `validate_dino_trajectory.py:34-66` (`--dino_loss_layers` arg), `std_check.sh` (driver). Per-run logs in `outputs/dino_std_check/{gaussian,ppd_r20,wpd_r20,dino_baseline,dino_layer4,dino_4_11_23}/log.txt`.

  - **v8** (`models/train/FLUX.1-dev_lora_dino_pd_v8/`, launched 2026-04-27) — *retrains v7c LoRA with multi-layer DINO loss [4,11,23] in noise opt*. Same algorithm as v7c (rectified flow rollout, sigma_uniform sampling, embed=1, cfg=1, σ_target_min=0.1, K∈[5,15]); only the `find_dino_preserving_noise` call now uses `loss_layers=[4,11,23]` so z1\* is supervised against three ViT-L/14 block depths simultaneously.

    **Hypothesis**: validate_dino_trajectory's multi-layer experiment showed [4,11,23] gives lowest step-0 (0.019) AND lowest final (0.425) DINO distance among DINO variants — but the LoRA was v7c, trained on baseline (final-layer) z1\*. If v8 LoRA is trained directly on multi-layer-DINO-preserving noise, the gap may close further; or the trajectory shape may invert.

    **Setup**: `--dino_loss_layers 4 11 23`, warm-start `v7c/step-600.safetensors`, all other knobs identical to v7c. Plumbing: `train_dino_pd.py` constructor + CLI flag + both training and validation `find_dino_preserving_noise` calls thread `loss_layers` through.

    **Watchpoints**:
    - `dino/distance_start` interpretation changes: now reports averaged `(1−cos)` across 3 layers, not single layer. Lower-bounded ~0.02 (vs v7c's 0.04) per std_check baseline.
    - Loss scale may shift: 3-layer loss has different gradient magnitude. If `loss/max` blows up, may need to scale lr down or revert sigma_uniform.
    - True success criterion is val DINO trajectory final + visual sharpness on `val/step-N/output_final.png`. If trajectory still monotone-ascends, multi-layer loss is not the lever — confirms architectural-not-loss-strength diagnosis from std_check.

    **Stop conditions**: (a) val/dino_final < 0.40 at any checkpoint (would beat WPD's 0.375 on this image), (b) visual sharpness regression vs v7c at step 600 mark, or (c) flat training loss at v7c-equivalent floor for 1000+ steps with no val improvement.

  - **v7-c plan (2026-04-25) — superseded by v7-c results above. Original plan text retained for context**: accept the smaller evil: structural drift over blur.
    - **`embedded_guidance` 3.5 → 1** in `train_dino_pd.py:98` (revert to v6 setup).
    - **Keep** `--sigma_sampling sigma_uniform` from v7b.
    - **Warm-start**: `models/train/FLUX.1-dev_lora_dino_pd_v6/step-300.safetensors` (v7a/b weights have already absorbed embed=3.5-cancellation; not a usable seed).
    - **Output path**: `./models/train/FLUX.1-dev_lora_dino_pd_v7c`.
    - **Hypothesis under test**: sigma_uniform's low-σ supervision rebalance, applied in the v6 (embed=1) velocity field, lets high-freq detail flow through embed_g at inference instead of being cancelled. Trades A2's +0.024 structural drift for sharpness.
    - **Watchpoints**: val/dino_final landing in v6's ~0.5 range (HIGHER than v7a/b's 0.43–0.45 is the *positive* sign — those low numbers were blurred-toward-z0 artifacts). Visual sharpness on `val/step-N/output_final.png` is the actual success criterion. If still blurry at ~step 500, falsifies embed-cancellation hypothesis and v7d should layer on σ²-reweighting on the rectified-flow MSE.

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
