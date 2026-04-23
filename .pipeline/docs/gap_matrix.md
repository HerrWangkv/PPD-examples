# Gap Matrix — DinoPD vs. prior art (2026-04-23)

Corpus: 28 papers OCR'd under `.pipeline/literature/dinopd-structure-preserving/`.

## 1. Competitive map

The landscape splits along **two axes**:
- *Where structure enters the model*: (A) via the **noise/corruption** process, (B) via an **auxiliary conditioning branch**, (C) via an **auxiliary feature loss**, (D) via a **rectified/few-step trajectory**.
- *How train/inference mismatch is handled*: (i) ignored, (ii) rollout-during-training (self-forcing family), (iii) rectified targets.

| Paper | Axis | Train-test gap | Sim2real? | Notes |
|---|---|---|---|---|
| **NeuralRemaster / ϕ-PD** (2512.05106, *user's own*) | A (Fourier phase) | (i) ignored | ✓ CARLA | Gold standard baseline. Zero-parameter, model-agnostic. Fourier bases have **global support** → ringing artifacts on sharp boundaries (WPD §1). |
| **WPD / ψ-PD** (*user's own, ECCV 2026 submission*) | A (wavelet phase, DT-CWPT) | (i) ignored | ✓ | Fixes ϕ-PD's ringing via localized wavelet packets. Still hand-designed noise prior. |
| **DinoPD** (this project) | A (DINO-preserving noise via `z_t*` search) + D (rectified flow) | (iii) rectified rollout (v3–v5) | *intended* | Replaces hand-designed priors with a **learned** structure-preservation objective. Status: v5 still shows mid-σ drift. |
| **Control-DINO** (2604.01761) | B (DINO feats as ControlNet) | (i) | ✓ VKITTI → real | DINOv3 features as explicit side input. Decouples appearance. Adds parameters. |
| **Driving-with-DINO (DwD)** (2602.06159) | B (DINO feats + Principal Subspace Projection + Causal Temporal Aggregator) | (i) | ✓ driving | State-of-the-art driving sim2real via DINO conditioning. Explicitly argues low-level edge-based controls "bake in" synthetic artifacts. Adds parameters. |
| **U-REPA** (2503.18414), **DiT-REPA** (2505.02831) | C (DINO-feature alignment loss in DiT) | (i) | ✗ | Auxiliary alignment loss speeds convergence. No structure-preservation goal — used for unconditional quality. |
| **Self Forcing** (2506.08009) | — | (ii) AR rollout + DMD/SiD/GAN dist. match | ✗ | Video AR, post-training, few-step. Distribution-level holistic loss. |
| **Diffusion Forcing** (2407.01392) | — | partial (per-frame noise levels) | ✗ | Sequence modeling; Self-Forcing critiques as TF-like exposure bias. |
| **End-to-End AR Self-Resampling** (2512.15702) | — | (ii) | ✗ | Contemporary self-forcing variant. |
| **Relax Forcing** (2603.21366) | — | (ii) | ✗ | Relaxed KV memory for long video. |
| **InstaFlow** (2309.06380) | D (ReFlow straightening) | (iii) | ✗ | One-step via rectified flow. |
| **Variational RFM** (2502.09616) | D (variational RFM) | (iii) | ✗ | Reduces RFM's path multimodality. Orthogonal to v5. |
| **Antithetic Noise** (2506.06185) | — | — | ✗ | Variance reduction. |
| REPA-Survey (2407.00783) | C (survey) | — | — | Background. |

## 2. Gaps vs. DinoPD

**G1. No prior work combines a learned structure-preserving noise (axis A) with rollout-aware training (axis ii).**
ϕ-PD and ψ-PD use hand-designed noise and ignore the train-test gap. Self-Forcing targets the gap but assumes standard Gaussian noise. DinoPD v5 is the only point in the literature that attempts both, but does so with *bounded K-step rollouts from a seeded `z_t*`*, not full `σ=1 → σ_target` rollouts. Self-Forcing's unrolled paradigm has never been combined with a structure-preserving input distribution. **→ v6 candidate: pure-Gaussian σ=1 rollout + DINO-distance loss applied only on the final decoded image (distribution-level supervision, à la Self-Forcing's holistic loss).**

**G2. REPA-style alignment has never been used for structure preservation.**
U-REPA and DiT-REPA align intermediate DiT features with DINO features to speed convergence on *unconditional* FID/IS. Neither uses the alignment signal to preserve *input image* structure. DinoPD v2 tried pixel-space DINO distance and failed (vacuous because `z_t = z_t*`). REPA-style *intermediate-feature* alignment, conditioned on the input image's DINO features, is untried. **→ v6 candidate: auxiliary loss aligning DiT intermediate feature maps with DINOv2 patch embeddings of the *input image*, gated like v2's `L_dino` but on off-manifold `z_target` from the rollout (where v2 was vacuous by construction).**

**G3. Axis-A (corruption-only) and axis-B (conditioning-branch) have never been compared head-to-head on the same sim2real benchmark.**
DwD (2602.06159) cites NeuralRemaster only for the CLIP-Real metric, not as a compared baseline. Control-DINO doesn't compare against ϕ-PD at all. The only head-to-head evidence in the corpus is **inside ϕ-PD's own paper**: ϕ-PD matches or beats ControlNet-Tile and SDEdit on CARLA→Waymo at equal noise level, with zero added parameters. So the published evidence *favors* axis-A on ϕ-PD's chosen benchmark. **→ DinoPD's real competitive question is narrower: does a learned structure-preserving noise beat hand-designed phase/wavelet priors on the same CARLA→Waymo protocol?** Beating DwD on a different benchmark would be a weaker claim.

**G4. Rectified-flow methods all assume target distribution is unconditional.**
InstaFlow, Variational RFM, Rectified Point Flow all reflow from Gaussian noise to the data manifold. DinoPD's rectified target `(z_target - z0)/σ_target` reflows to a *specific input image*'s latent — this is a structural departure not covered in the rectified-flow literature. The 1/σ amplification concern (v5's `sigma_target_min=0.05` lever) is a direct consequence of this per-sample target formulation and not addressed by prior rectified-flow work.

**G5. Driving-sim2real (DwD) uses Principal Subspace Projection to drop high-frequency DINO components that "bake in" synthetic artifacts.**
DinoPD's DINO distance is on CLS + all patches of DINOv2. No frequency/subspace filtering of the DINO signal. DwD's finding implies raw DINO distance may over-penalize stylistic drift that should be allowed. **→ v6 candidate: project DINO patch tokens onto a low-rank subspace (PCA of sim features) before computing distance.**

## 3. Non-obvious prior-art findings

- **ϕ-PD (user's own) already benchmarks sim2real on CARLA planner transfer, not just mIoU/FID.** If DinoPD doesn't match or beat CARLA planner scores, it fails the strongest motivating application of the paper line.
- **WPD's §1 explicitly pins ϕ-PD's failure mode to "global support of Fourier bases."** This gives DinoPD a clean narrative: *learned* structure preservation sidesteps *any* fixed-basis choice (Fourier or wavelet).
- **Self-Forcing works at post-training scale (few hundred steps) with a few-step distilled backbone.** DinoPD currently trains from scratch LoRA for multi-thousand steps. A post-training, few-step-distilled variant would be a natural efficiency story.
- **DwD applies a Causal Temporal Aggregator to propagate DINO features across frames** — this is a video-specific design DinoPD will eventually need if the sim2real *video* (Wan2.2) application is claimed.
