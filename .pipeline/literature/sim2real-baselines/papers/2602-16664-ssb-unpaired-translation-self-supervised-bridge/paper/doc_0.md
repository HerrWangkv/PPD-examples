6
2
0
2

b
e
F
8
1

]

V
C
.
s
c
[

1
v
4
6
6
6
1
.
2
0
6
2
:
v
i
X
r
a

UNPAIRED IMAGE-TO-IMAGE TRANSLATION VIA A
SELF-SUPERVISED SEMANTIC BRIDGE

Jiaming Liu1 ∗ Felix Petersen1 Yunhe Gao1 Yabin Zhang1

Hyojin Kim2 Akshay S. Chaudhari1 ∗ Yu Sun3 ∗ Stefano Ermon1

Sergios Gatidis1

1Stanford University

2LLNL

3Johns Hopkins University

ABSTRACT

Adversarial diffusion and diffusion-inversion methods have advanced unpaired image-to-image
translation, but each faces key limitations. Adversarial approaches require target-domain adversarial
loss during training, which can limit generalization to unseen data, while diffusion-inversion methods
often produce low-fidelity translations due to imperfect inversion into noise-latent representations.
In this work, we propose the Self-Supervised Semantic Bridge (SSB), a versatile framework that
integrates external semantic priors into diffusion bridge models to enable spatially faithful translation
without cross-domain supervision. Our key idea is to leverage self-supervised visual encoders to learn
representations that are invariant to appearance changes but capture geometric structure, forming
a shared latent space that conditions the diffusion bridges. Extensive experiments show that SSB
outperforms strong prior methods for challenging medical image synthesis in both in-domain and
out-of-domain settings, and extends easily to high-quality text-guided editing. Code and models are
publicly available in here.

1

Introduction

Unpaired image-to-image (I2I) translation poses a critical challenge in unsupervised representation learning: how to
disentangle and transfer semantic content across distinct domains without explicit correspondence [1]. This capability
is fundamental to computer vision and medical imaging, enabling tasks such as medical image synthesis [2, 3, 4, 5, 6,
7, 8, 9] and general image editing [10, 11, 12, 13, 14, 15, 16]. Despite significant progress, existing approaches still
face fundamental challenges in balancing distributional robustness and structural preservation under limited paired
supervision.

Early GAN-based methods [17, 15, 16, 18, 19] optimize adversarial objectives—often combined with cycle-consistency
constraints—to learn cross-domain mappings and achieve promising translation results. Building on this, recent
approaches incorporate diffusion models [20, 21] for conditional image translation [22, 23, 24, 25, 2]. These methods
improve translation realism through explicit cross-domain regularization on unpaired data during diffusion model
training. However, both families depend on explicit coupling between source and target domains, limiting their
scalability and generalization across diverse distributions. For example, in MRI→CT synthesis, variations in MRI
contrast often lie beyond the training distribution, leading to reduced performance to out-of-domain (OOD) data. This
motivates decoupling translation from cross-domain training by introducing a shared semantic interface that connects
domains without explicit alignment objectives

Alternatively, inversion-based approaches [26, 27, 28, 29, 30] translate images by inverting them into the latent noise
space of a pretrained diffusion model and re-synthesizing under target-domain conditioning. In practice, inversion
is approximate and errors propagate through sampling, often resulting in structural drift from the source. Recent
methods mitigate this by injecting intermediate features—e.g., reusing attention maps recorded during inversion [31,
32, 33, 34]—but these interventions are usually tied to specific architectures or sampling procedures, which limits
transferability across methods.

∗Corresponding authors. Questions to: jiamliu@stanford.edu

 
 
 
 
 
 
A PREPRINT -

Figure 1: Overview of our Self-Supervised Semantic-Bridge (SSB) framework for unpaired image translation and
editing. SSB trains without paired data or adversarial objectives, relying on a shared latent-space assumption to connect
domains via a common representation; × denotes no cross-domain supervision. The top rows showcase image-to-image
translation results using text-free guidance, while the bottom rows demonstrate text-guided editing based on Stable
Diffusion 3-Medium [52]. This single-domain training enables unpaired translation across medical and natural images
with improved structural consistency and OOD robustness (e.g., unseen MRI contrasts at test-time).

Bridge models [35, 36, 37, 38] and stochastic interpolants [39] learn stochastic or deterministic paths between arbitrary
distributions and achieve high-fidelity translation when paired supervision is available [40, 37, 41]. A related line
of work targets unpaired translation via distribution alignment, including optimal transport [42, 43, 44], Schrödinger
bridges [45, 46, 47], and bridge distillation [48]. While promising, scaling these unpaired approaches to complex
high-dimensional regimes, such as self-supervised medical synthesis and text-guided editing, remains underexplored.

In this work, we propose a new, fully self-supervised solution for unpaired image-to-image translation, called the
Self-supervised Semantic Bridge (SSB). SSB addresses practical settings where paired data are scarce and test-time
distribution shifts are pervasive, aiming to strictly preserve source fidelity while enabling high-quality cross-domain
translation. Our key insight is to perform unpaired translation through a shared semantic manifold derived from
self-supervised visual encoders’ patch embeddings (e.g., DINOs [49, 50, 51]). These embeddings provide a geometry-
consistent semantic interface between domains, enabling independent self-supervised training of domain-specific
generative models without cross-domain alignment or adversarial loss. As a result, extending SSB to additional domains
requires training only one new single-domain model, yielding linear scaling in the number of domains rather than the
pairwise cost of domain-coupled approaches. We summarize the main contributions as follows:

• We introduce SSB, a simple but effective framework for unpaired image-to-image translation that connects do-
mains through a self-supervised shared semantic latent space with independent per-domain training, supported
by theoretical justification.

• We develop a geometry-aware MRI–CT representation via DINOv2 pre-training, enabling SSB to achieve
strong unpaired MRI→CT translation in both in-domain and out-of-domain settings, with performance
comparable to supervised approaches.

• We extend SSB to natural-image translation and text-guided editing, achieving competitive performance on

both scene transfer and object-level editing.

2 Related Work

Unpaired I2I Methods. Unpaired image translation methods commonly rely on cycle-consistency, adversarial learning,
or their combination [17, 15, 16, 18, 19, 1, 53, 25, 2, 22]. Unlike these approaches that align domains through direct
coupling or discriminator feedback at training, USBM constructs a shared semantic latent space via self-supervised

2

A PREPRINT -

representation learning and learns domain-specific bridges without explicit cross-domain supervision. Alternatively,
inversion-based diffusion and flow models [26, 54, 27, 29] leverage powerful pre-trained generative models by mapping
images to noise latents and re-synthesizing them under new conditions, later extended to text-guided and conditional
tasks via feature injection ( e.g., cross-attention) [31, 32, 34]. Recent inversion-free variants [55, 56] further improve
translation consistency using advanced flow backbones such as Flux [57] and SD3 [52]. Although conceptually related in
their goal of injecting source structures during sampling, USBM departs from these methods by aligning self-supervised
and diffusion representations to achieve structure-preserving translation within a unified bridge formulation.

Diffusion Models with External Representations.Recent studies have explored leveraging external representations to
improve the controllability, efficiency, and performance of diffusion models across image-related tasks. One line of
work uses structured conditions—such as edges, depth—as explicit geometric guidance for controllable generation
and editing [13, 58]. A complementary line of work uses learned embeddings from pretrained vision encoders to
provide semantic priors, benefiting image synthesis [59, 60, 61], image matching and I2I editing [62, 63], and depth
estimation [64]. REPA [61] aligns diffusion feature similarities between diffusion transformers and DINOv2 for
efficient training, while SD-DINO [62] fuses DINOv2 features into diffusion models at test time for appearance-level
I2I translation. In contrast to these methods, SSB introduces a unified bridge formulation that leverages self-supervised
representations as a geometry-consistent semantic interface to learn domain transitions from unpaired data, enabling
faithful structure preservation during translation and editing without cross-domain alignment or adversarial objectives.

3 Preliminaries

Diffusion Bridge Models. Bridge models [35, 65, 45] and Stochastic interpolants (SI) [66, 39] unify denoising diffusion
models and rectified flows by extending beyond Gaussian priors. They interpolate between arbitrary data x0 ∼ qdata and
prior samples xT ∼ pprior. In the linear–Gaussian case, diffusion bridges reduce to SIs, sharing the same conditional
marginals p(xt | x0, xT ) and reverse dynamics [39, 36],

xt = I(t, x0, xT ) + γtϵ,

ϵ ∼ N (0, I), t ∈ [0, T ].

(1)

Without loss of generality, we set T = 1 by time reparameterization, ensuring γ0 = γT = 0 and boundary consistency
I(0) = x0, I(T ) = xT . A typical choice is the linear interpolant I(t) = αtx0 + βtxT , which yields the stochastic
velocity v(t, xt) = ∂tI(t, x0, xT ) + ˙γtϵ. The probability–flow velocity removes noise via conditioning, v⋆(t, xt) =
E[v | xt], which is intractable and thus approximated by a neural net vθ(t, xt). Sampling then follows the probability
flow ODE (PF-ODE),

dxt = vθ(t, xt)dt.
(2)
When γt ≡ 0 and xT ∼ N (0, I), this reduces to rectified flows [67, 68]. Ideally, integrating the forward field vθ(t)
transports an image x0 to its terminal latent representation xT , while reverse integration (sampling) reconstructs x0,
forming a deterministic encoder–decoder pair [54].

Self-Supervised Visual Encoders. The DINO [49, 50] family employs self-distillation to learn spatially-aware patch
representations. Given an input x, the encoder Tϕ produces a [CLS] token T [CLS]
(x).
Predictions are then obtained from the [CLS] token via a softmax over K learned prototypes. During training, DINOs
sample two independent augmentations A1, A2 of the same image. A teacher network Tϕ′ (an EMA of Tϕ) provides
the target distribution, and the student Tϕ is optimized to match it through the following cross-entropy loss function

(x) and patch tokens T patch

ϕ

ϕ

(cid:104)
L(ϕ; ϕ′) = E

− pϕ′(A2(x))⊤ log pϕ(A1(x))

(cid:105)
.

(3)

This objective encourages invariance to local perturbations, yielding robust semantic representations that we leverage to
construct our shared latent space in the next section.

4 Proposed Method

4.1 Shared Latent Space Assumption

Inspired by I2I translation [16, 18, 27], we assume that multi-domain observations (x(1), . . . , x(M )) ∼ qdata share a
common latent representation. Although the true joint distribution qdata is unknown, we posit a shared latent variable
y ∼ pprior that captures semantic content aligned across domains, yielding,

p(z(1), . . . , z(M ), y) = p(y)

M
(cid:89)

i=1

p(i)(z(i) | y),

(4)

3

A PREPRINT -

Figure 2: Unlike inversion-based methods that invert toward an unstructured Gaussian noise, SSB defines a unified
semantic latent endpoint y = Eϕ(x) using a self-supervised visual encoder and trains domain-specific bridges
independently to connect each domain to this shared endpoint, enabling reliable translation by composing source-to-T
inversion with target-bridge generation.

where z = Eφ(x) are VAE [69] latents and p(i)(· | y) denotes the conditional distribution for domain i. Conditioned
on y, the domains are independent. To formalize “shared semantics,” it is convenient to introduce an oracle encoder
E∗ such that for any semantically corresponding images (x(i), x(j)), E∗(x(i)) ≈ E∗(x(j)) ≈ y. In practice, we
approximate E∗ with a fixed pretrained network Eϕ. Importantly, our theoretical error analysis and empirical results on
medical and natural images below demonstrate that our assumption in (4) remains effective despite the domain gap in
the pre-trained feature space.

Cross-domain Translation using Bridge Models. Translation from domain j to i is then expressed as an integral over
the shared latent. Since this posterior is intractable, we approximate it by a Dirac mass at the encoder output E∗,

p(z(i) | z(j)) ≈ p(i)
θ

(cid:0)z(i) | y = E∗(x(j))(cid:1).

(5)

where p(i)
θ
enables a clear constructive inference: encoding a source image x(j) into y, sampling from ¯z(i) ∼ p(i)
decoding to the target image ¯x(i) = Dφ(¯z(i)).

is a diffusion bridge that maps the shared latent y to the domain-i VAE latents. Its reverse process thus
θ (· | y), and

4.2 Shared Latent Space Encoders

The key ingredient of our method is a practical proxy for the semantic encoder E∗, built from DINO-ViT features.
Specifically, we factor augmentations as A = G ◦ C, where C ∈ C applies appearance transforms (e.g., color, contrast,
etc) and G ∈ G applies geometric transforms (e.g., crop, scale, etc). Near convergence, minimizing the student-teacher
consistency loss in (3) biases the learned predictor to be approximately insensitive to appearance change,

pϕ(G ◦ C(x)) ≈ pϕ(G(x)),

∀C ∈ C, G ∈ G,

for almost every input image x. This appearance insensitivity makes DINO features suitable for settings with strong
structure correspondence but large appearance gaps (e.g., MRI–CT). Moreover, since the [CLS] prediction is computed
via self-attention over patch tokens, the same training signal propagates to patch-level representations, which retain richer
local spatial structure than the global token. Accordingly, we define the shared latent encoder as Eϕ(x) = P (T patch
(x)),
where P is a linear PCA projection used for dimensional alignment. In practice, the number of retained components B
(e.g., 8 or 16) is chosen to match the KL-VAE latent dimensionality, so that the resulting components align with VAE
channels while emphasizing geometry-consistent factors and suppressing residual appearance noise.

ϕ

4

A PREPRINT -

Figure 3: Cross-modal semantic alignment and improvement in the shared latent space. Left: PCA visualizations
of intermediate feature maps for paired CT/MRI inputs. Our ViT-B/8 encoder yields smoother and more anatomi-
cally consistent representations across modalities. Middle: Quantitative alignment on paired MRI–CT validation
samples, comparing global alignment (cosine similarity) and local structural consistency (CKNNA). Right: Semantic
improvement ∆cosim measures the net gain in feature alignment with the ground-truth CT (gtCT) achieved by our
generated translation (genCT) over the original input (MRI). Formally, ∆cosim = cosim(Eϕ(genCT), Eϕ(gtCT)) −
cosim(Eϕ(MRI), Eϕ(gtCT)). We plot this gain against the initial input alignment x. The dashed line y = 1 − x
represents the theoretical ceiling, corresponding to perfect recovery of the ground-truth features (similarity = 1). The
improvements confirm that our method effectively minimizes semantic divergence, largely mitigating the impact of the
initial domain gap.

4.3 Latent Bridge Models as Conditional Decoders

In principle, a bridge trajectory can be constructed directly from the shared embedding zT = Eϕ(x(j)) to the target
latent z(i)
0 . However, this rigid mapping often fails under strong appearance ambiguities. To address this, we introduce
T ∼ N (cid:0)Eϕ(x(i)), b2I(cid:1), where b controls
a unified endpoint formulation that adapts to the domain characteristics: z(i)
endpoint uncertainty. For geometry-dominated tasks (e.g., MRI→CT) where the encoder accurately captures the shared
structure, we set b = 0, reducing the model to a simple deterministic map that strictly preserves fidelity. Conversely,
for appearance-ambiguous tasks (e.g., natural images), we employ a stochastic endpoint (b > 0), which enables the
PF-ODE to refine the noisy state toward the geometric center while synthesizing valid domain details. Thus, while our
framework is general enough to handle diverse modalities, it naturally simplifies to the minimal necessary complexity
for each specific task.

For clarity, we omit the domain index in the following sections when it is clear. With pinned endpoints (z0, zT ), we
have the latent transition kernel with normalized time step T = 1 for a single domain-i as

t I(cid:1) ,
where I(t, z0, zT ) := αtz0 + βtzT ,
t ∈ [0, 1]. This corresponds to the latent stochastic interpolants zt = αtz0 +
βtzT + γtϵ, ϵ ∼ N (0, I), which induces a PF-ODE ˙zt = v(t, zt, zT ), with v(t, z, zT ) = ˙αtE[z0|zt = z, zT ] +
˙βtzT + ˙γtE[ϵ|zt = z, zT ]. Without loss of generality, we train a network vθ to approximate the v as

p(zt | z0, zT ) = N (cid:0)zt; I(t, z0, zT ), γ2

(6)

Lθ = Et,z0,zT

(cid:104)(cid:13)
(cid:13)vθ(t, zt, zT ) − (∂tI(t, z0, zT ) + ˙γtϵ)||2(cid:105)
,

(7)

where t ∈ [ε, 1 − ε] with a small ε, ensuring numerical stability of ˙γt near the endpoints. Alternatively, one may
approximate v by first learning a network to predict the conditional mean E[z0|zt, zT ] [35, 36], and subsequently
computing the full velocity field in the closed form.

Reverse ODE for Conditional Translation. The cross domain translation p(z(i)|z(j)) in Eq. 5 can then be formulated
0 )(cid:1), where the shared latent y can be obtained deterministically
as a conditional bridge p(i)
θ
y = Eϕ(x(j)
T (x(j)
0 ) is obtained by integrating the forward
ODE dz(j)
)dt from t = 0 to t = T . As a result, the reverse ODE of the bridge process for target
domain-i,

(cid:0)z(i)
0
0 ) or through PF-ODE inversion y = zinv

t = v(j)(t, z(j)

| zT = y(x(j)

0 ), where zinv

T (x(j)

t

dt, zT = y(x(j)

0 ),

(8)

t = v(i) (cid:16)
is solved backward in time, starting from t = T .

dz(i)

(cid:17)

t, z(i)
t

, zT

5

Table 1: Quantitative comparison on MRI→CT (in-domain and out-of-domain (OOD)). (*) denotes comparison
to pre-registered paired CT images. Best and second-best values are highlighted among methods trained without
ground-truth supervision. The I2SB and SelfRDB are supervised baselines.

A PREPRINT -

Method

CycleGAN [15]
UNIT [16]
SDEdit [26]
DDIB [27]
Syndiff [2]
DDBM [35]
I2SB [37]
SelfRDB [41]
SSB(Ours)

MRI→CT

MRI→CT (OOD)
MS-SSIM* ↑ PSNR* ↑ FID ↓ MS-SSIM ↑

0.657
0.712
0.628
0.625
0.805
0.768
0.776
0.818
0.810

18.86
20.88
18.87
18.45
22.65
22.09
22.74
23.01
23.21

127.22
115.84
50.17
48.15
79.56
42.75
85.19
91.57
30.15

0.514
0.458
0.385
0.411
0.405
0.540
0.462
0.393
0.585

Table 2: Quantitative comparison on Natural I2I translations.

Method

CycleGAN [15]
CUT [1]
SDEdit [26]
DDIB [27]
CycleNet [25]
ControlNet [13]
SSB (Ours)

Horse→Zebra / Apple→Orange (256×256)
CLIP-T ↑ LPIPS ↓ SSIM ↑
0.630
0.416
0.607
0.425
0.611
0.335
0.595
0.387
0.719
0.301
0.426
0.475
0.794
0.285

PSNR ↑
15.57
15.39
17.84
16.51
19.75
14.43
20.20

0.225
0.227
0.280
0.263
0.305
0.327
0.322

4.4 Translation Error Analysis

In this section, we present a theoretical analysis of translation error arising from inaccuracies in the shared latent
approximation introduced in Sec. 4.2 and examine how such errors propagate through the ODE flow during translation.
Our goal is to quantify the deviation between the ground-truth target x0 and the predicted translation ¯x0. For clarity,
we focus on the ODE formulation of linear bridge models under Euler discretization, which serves as the foundation for
many first-order solvers used in diffusion-based sampling. We make basic but necessary assumptions throughout the
analysis below.
Theorem 4.1. Given target and source images that share a latent code under a learned vision encoder Eϕ. Assume the
learned vector field vθ and decoder Dφ are Lipschitz-continuous, and both the encoder and decoder introduce bounded
reconstruction errors. If the bridge ODE is solved using an Euler scheme with step size τ = T /N , then under mild
smoothness and boundedness conditions on the latent dynamics, the translation error between the predicted and target
images is bounded, with probability at least 1 − δ, by

∥x0 − ¯x0∥2 ≤ W (T )∥∆(T )∥2 +

(cid:113)

WE

δ + Cτ 2 + εDφ,

(9)

where each term corresponds respectively to the encoder alignment, vector-field approximation, discretization, and
decoder reconstruction errors.
Note that W , E, and C are constants and detailed derivation of them are provided in the Appendix. Theorem 4.1
explicitly incorporates the encoder alignment error rather than presuming it to be negligible. We minimize this error
via structure-preserving fine-tuning and empirically validate the bound on challenging MRI-to-CT translation tasks in
Section 5.2 (Fig. 3), showing that our method significantly reduces the gap compared to baselines and maintains robust
translation performance across a broad range of initial semantic misalignment due to encoder imperfections.

4.5 Practical Design Choices

We employ the following simple yet effective design choices to enhance controllability in structure-preserving translation
and editing. We interpolate between the source and target drifts, v(j) and v(i), using a time-varying coefficient ηt > 0 as
t + ηt(dz(j)
This combination defines a smooth interpolating drift field that enables continuous domain transition within a shared
latent manifold. Similar to [26, 55, 29], we parameterize ηt = (1 − t)I[t > tend], which provides a gradual trade-off

t − dz(i)
t ).

(i) = dz(i)

d ˜zt

(10)

6

A PREPRINT -

Figure 4: Our SSB ensures anatomically consistent MRI→CT translation across in-domain and out-of-domain scenarios.
Segmentation masks are overlaid on MRI source images in OOD settings to provide structural reference, as paired
CT ground truth is unavailable. They are not used during training or inference and serve only to illustrate anatomical
fidelity without segmentation supervision.

between structural preservation and appearance adaptation. During early reverse steps, smaller ηt promotes semantic
consistency under high noise, while the cutoff tend relaxes this constraint—allowing flexible appearance modulation for
translations with large domain or structural gaps, as illustrated in Fig. 7. See Appendix for more details.

5 Experiments

In this section, we present experimental comparisons of our method. For fairness, all baselines use their official
implementations when publicly available, and detailed configurations are provided in the Appendix due to space
limitations.

5.1 Experimental Setup

Medical MRI–CT Datasets. We conduct experiments on axial-view MRI→CT translation with all images resized to
256 × 256 for efficiency. For in-domain evaluation, we use SynthRAD2023 [70] and SynthRAD2025 [71], comprising
∼ 950 pre-registered MRI–CT volumetric pairs. Each data set is divided into training sets and test sets without subject
overlap, yielding 6,596 2D test slices from 50 subjects and the rest for training. For training our own shared-latent
encoder and bridge models, we additionally aggregate unpaired MRI/CT data from AMOS2022 [72], TotalSegmentator-
MRI/CT [73], IXI [74], pelvic MRI–CT [75], PSMA-FDG-PET-CT [76], and a subset of CT-RATE [77], augmented
with sagittal and coronal views. This yields about 1.05M MRI and 2.2M CT slices in total. For out-of-domain
evaluation, we sample 5,000 fat- and water-contrast slices from 100 UKBB-MRI [78] subjects, respectively, differing
in contrast and resolution from the MRI training datasets.

General Domain Datasets. We evaluate our method on both class-label and text-guided I2I tasks. For class-label
translation, we adopt Horse→Zebra and Apple→Orange [15], using 30 images per domain (60 total). Each sample is
paired with a GPT-5–generated target prompt to evaluate textual authenticity with respect to the target domain. For text-
guided I2I editing, following [55, 12], we curate a diverse dataset for scene-style transfer (e.g., summer→fall/winter)
and object-level editing. Images are sourced from Flickr [79], DIV2K [80], and Pexels [81], each paired with GPT-
5–generated source captions and target prompts. In total, we collect 35 natural scene images, synthesize 15 additional

7

A PREPRINT -

Figure 5: Visual comparison with recent diffusion editing methods using SD3-M base model. Our approach generates
visually consistent results with the source image, preserving structural integrity while convincingly applying appearance
changes across diverse scenarios.

Figure 6: Numerical results based on SD3-M model. Our method achieves a favorable balance between text adherence
(CLIP-T) and structural preservation (DINO and PSNR) across both scene-editing and object-change tasks. Scene
editing requires substantial global modifications, whereas object change focuses on localized adjustments. Connected
markers represent different hyperparameter settings.

scenes using FLUX-dev. [57], and gather 60 real object images, yielding over 100 scene-style and 200 object-editing
text–image pairs.

Evaluation Metrics. For medical image translation (MRI→CT), we evaluate image quality using normalized
PSNR [82], multi-scale SSIM (MS-SSIM) [83], and FID [84]. For natural I2I translation, which requires balancing
semantic and structural consistency, we report CLIP-Text similarity (CLIP-T) [85, 29, 55, 25], DINO similarity [49, 55],
CLIP-Image similarity (CLIP-I), LPIPS [86], MS-SSIM, and PSNR. Similarly, for text-guided image editing, we
evaluate semantic alignment with CLIP-T and structural fidelity with DINO, CLIP-I, LPIPS, SSIM [87], and PSNR.

5.2 Appearance-Invariant Encoder Validation

We begin by validating the encoder alignment assumption from Theorem 4.1 via feature analysis on the MRI–CT dataset.
Using CKNNA [88] for local structure and Cosine Similarity for global alignment, we find that standard DINOv2
exhibits a trade-off: acceptable alignment but high-frequency structural noise (see Fig. 3), indicating a gap from the ideal
geometry-preserving manifold.To minimize the encoder alignment error (∥∆(T )∥2), we fine-tune a DINOv2-ViT-B/8
on the MRI–CT training set. Crucially, we integrate a retina-inspired filter [89] to suppress modality-specific appearance
(e.g., contrast) and force the model to rely on structure representations. As shown in Fig. 3 (Left), our encoder achieves
the optimal balance of metrics, significantly reducing the deviation from the ideal manifold compared to baselines like
BiomedCLIP [90], MedSigCLIP [91], DINOv2-B [50], and MedVAE [92]. Qualitative visualizations (Right) confirm
that our method eliminates patch-grid artifacts, yielding the smooth, anatomically consistent feature space required to

8

satisfy our theoretical error bound. To the best of our knowledge, this is the first adaptation of DINOv2 specifically for
MRI–CT image translation.

5.3 Unpaired Image Translation

A PREPRINT -

Medical I2I. Given the pre-trained DINOv2 ViT-B/8 encoder described in Sec. 5.2, we construct the latent bridge using
a U-Net backbone [93] between the DINOv2 encoder and a KL-regularized VAE [69], producing latent feature maps
with a spatial resolution of 64 × 64 and 8 channels. Since our focus is MRI→CT synthesis, we train a single latent
bridge model on the CT training datasets. Following [94, 35, 36], we parameterize the interpolant weights defined in
t /σ2
Eq. 6 in terms of the signal-to-noise ratio (SNR), defined as SNRt = a2
t with (at, σt) the standard DDPM [21]
(cid:16)
(cid:16)
1 − SNRT
schedules. The weights are αt = at
. At inference, we adapt the
SNRt
DDBM [35] hybrid sampler which is based on the predictor-corrector sampler introduced in [20].

1 − SNRT
SNRt

, βt = at
aT

t = σ2
t

SNRT
SNRt

, γ2

(cid:17)

(cid:17)

We compare our method against representative baselines widely used in medical I2I translation, including GAN-based
models CycleGAN [15] and UNIT [16], zero-shot diffusion approaches SDEdit [26] and DDIB [27], and the hybrid
CycleGAN–diffusion framework SynDiff [95]. Since edge-based representations, such as Canny edges, are commonly
used in MRI–CT translation—where diffusion models learn to reconstruct images from edge maps in a self-supervised
manner [58]—we additionally implement a DDBM variant using Canny edges. This comparison demonstrates that
DINO-based embeddings capture richer geometric and semantic information than handcrafted edge filters. Fig. 4 and
Table 1 present qualitative and quantitative results on in-domain and OOD MRI→CT translation. Since no paired CT
is available, we compute FID against CT scans from the training set and MS-SSIM between the input MRI and the
synthesized CT as a proxy for structural similarity. We also provide qualitative results and segmentation overlays, with
additional visuals in the Appendix. Our method shows stronger robustness to new MRI contrasts and achieves more
accurate translations than SynDiff and DDBM, preserving geometry and modality realism.

General Domain I2I Translation. We compare our method against established baselines on the Horse → Zebra
and Apple → Orange benchmarks. For this task, we fine-tune the flow-based SiT transformer [96, 61], pre-trained
on ImageNet1K, at 256×256 image resolution. Since recently released DINOv3 [51] provides higher-quality dense
features than official DINOv2, we adopt the pretrained DINOv3 ViT-L/16 as our backbone for general-domain tasks.
As discussed in Sec. 4.5, we inject PCA-compressed DINOv3 features (first-B=16) into the SiT backbone via a
zero-initialized linear projection layer with positional embeddings. The projected features are added in parallel to the
SiT input layer for feature fusion, enabling smooth initialization when introducing new conditioning inputs, similar to
the zero-start strategy in [13]. For endpoint zT , the projected features are channel-wise averaged to match the latent
dimensionality of the KL-VAE endpoint z0. We adopt the linear weights for the endpoints and a bridge variance
max t(1 − t), with γmax = 0.1. We set b = 1, resulting zT ∼ N (cid:0)Eϕ(x0), I). The
schedule αt = 1 − t, βt = t, γ2
entire SiT is then finetuned on the same ImageNet1k-256 dataset used for pretraining. Table 2 presents the average
numerical comparisons with baselines including CycleGAN [15], CUT [1], SDEdit [26], DDIB [27], CycleNet [25],
and ControlNet [13]. Overall, our method achieves the best combination of authenticity to the target text (CLIP-T) and
structural consistency with the source.

t = γ2

5.4 Text-to-Image Editing

In this section, we present our experimental results on text-guided I2I editing. We finetune the pre-trained text-image
SD3 [52] medium (SD3-M) model. Similar to SiT fine-tuning, we inject first-B=32 PCA-compressed DINOv3 features
into SD3’s hidden layers via zero-initialized projections, and construct zT by channel-wise averaging across the PCA
components. We use the same endpoint weighting and bridge variance schedules for the unpaired I2I translation present
before. We collect a small yet high-resolution subset of the LAION-5B [97] dataset to fine-tune our SD3-M model.
This subset contains approximately 1.2 M text–image pairs, with an average resolution of about 1200 × 1400 pixels.
For computational efficiency, we update only the attention layers of the model while keeping all original linear layers
frozen (see additional implementation details in the Appendix). We compare our method with recent baselines officially
implemented on SD3-M, including SDEdit [26], DDIB [27], iRFDS [98], FlowEdit [55], and ControlNet [13]. Because
semantic alignment (CLIP-T) and structural fidelity often trade off [55], we evaluate each method across a range of
hyperparameters to reveal their operating tradeoff curves in Fig. 6. The left panel shows scene-style editing, which
involves substantial global appearance changes. Accordingly, structural fidelity for scene-style editing metrics (SSIM,
PSNR, LPIPS) are computed on the luminance (YCbCr) channel to better reflect perceptual structure consistency.
Fig. 6 (Right) shows object-level editing with localized modifications. Our method achieves comparable or superior
performance to state-of-the-art approaches such as FlowEdit across both semantic and structural metrics, particularly in
complex scene edits that demand significant style shifts. Fig. 8 presents visual comparisons with ControlNet.

9

A PREPRINT -

Figure 7: Balancing structure and appearance via vector-field control. Interpolating between domain flows ( e.g.,
Left: “lighthouse” → “clock tower” and Right: “water” → “milk”) enables controllable trade-offs between structural
consistency and appearance fidelity.

Figure 8: More results comparing with FlowEdit and ControlNet.

6 Conclusion and Limitations

We introduce the SSB, a diffusion-based framework for unpaired image translation and editing built upon a shared
latent-space formulation. By constructing a unified latent space across domains through self-supervised, appearance-
invariant encoders, SSB enables structure-preserving and semantically aligned translation without paired supervision
or adversarial training. Experiments across both medical and natural domains demonstrate that this unified latent
representation facilitates robust cross-domain generalization, yielding anatomically consistent MRI→CT synthesis and
controllable appearance manipulation in natural images. While SSB’s strong structure preservation enables high-fidelity
appearance changes, it becomes less effective when transformations require fundamentally altering object geometry.
Edits that imply a shift in object category violate the geometry-aware prior underlying our semantic manifold. We
discuss and illustrate these failure cases in the Appendix.

Acknowledgments

This work was supported by the Stanford Center for AI in Medicine and Imaging (AIMI) and the Stanford Institute for
Human Centered AI (HAI).

References

[1] Taesung Park, Alexei A Efros, Richard Zhang, and Jun-Yan Zhu. Contrastive learning for unpaired image-to-

image translation. In European conference on computer vision, pages 319–345. Springer, 2020.

[2] Muzaffer Özbey, Onat Dalmaz, Salman UH Dar, Hasan A Bedel, ¸Saban Özturk, Alper Güngör, and Tolga Cukur.
Unsupervised medical image translation with adversarial diffusion models. IEEE Transactions on Medical
Imaging, 42(12):3524–3539, 2023.

[3] Chenlu Zhan, Yu Lin, Gaoang Wang, Hongwei Wang, and Jian Wu. Medm2g: Unifying medical multi-modal
generation via cross-guided diffusion with visual invariant. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, pages 11502–11512, 2024.

[4] Zhifeng Wang, Renjiao Yi, Xin Wen, Chenyang Zhu, and Kai Xu. Vastsd: Learning 3d vascular tree-state space
diffusion model for angiography synthesis. In Proceedings of the Computer Vision and Pattern Recognition
Conference, pages 15693–15702, 2025.

[5] Qing Lyu and Ge Wang. Conversion between ct and mri images using diffusion and score-matching models.

arXiv preprint arXiv:2209.12104, 2022.

10

A PREPRINT -

[6] Yanwu Xu, Li Sun, Wei Peng, Shuyue Jia, Katelyn Morrison, Adam Perer, Afrooz Zandifar, Shyam Visweswaran,
Motahhare Eslami, and Kayhan Batmanghelich. Medsyn: text-guided anatomy-aware synthesis of high-fidelity
3-d ct images. IEEE Transactions on Medical Imaging, 43(10):3648–3660, 2024.

[7] Lan Jiang, Ye Mao, Xiangfeng Wang, Xi Chen, and Chao Li. Cola-diff: Conditional latent diffusion model for
multi-modal mri synthesis. In International Conference on Medical Image Computing and Computer-Assisted
Intervention, pages 398–408. Springer, 2023.

[8] Jinzhuo Wang, Kai Wang, Yunfang Yu, Yuxing Lu, Wenchao Xiao, Zhuo Sun, Fei Liu, Zixing Zou, Yuanxu Gao,
Lei Yang, et al. Self-improving generative foundation model for synthetic medical image generation and clinical
applications. Nature Medicine, 31(2):609–617, 2025.

[9] Florinel-Alin Croitoru, Vlad Hondru, Radu Tudor Ionescu, and Mubarak Shah. Diffusion models in vision: A

survey. IEEE transactions on pattern analysis and machine intelligence, 45(9):10850–10869, 2023.

[10] Leon A Gatys, Alexander S Ecker, and Matthias Bethge. Image style transfer using convolutional neural networks.

In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2414–2423, 2016.

[11] Huan Ling, Karsten Kreis, Daiqing Li, Seung Wook Kim, Antonio Torralba, and Sanja Fidler. Editgan: High-
precision semantic image editing. Advances in Neural Information Processing Systems, 34:16331–16345,
2021.

[12] Narek Tumanyan, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Splicing vit features for semantic appearance
transfer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages
10748–10757, 2022.

[13] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models.
In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023.

[14] Youssef Alami Mejjati, Christian Richardt, James Tompkin, Darren Cosker, and Kwang In Kim. Unsupervised
attention-guided image-to-image translation. Advances in neural information processing systems, 31, 2018.

[15] Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation using
cycle-consistent adversarial networks. In Proceedings of the IEEE international conference on computer vision,
pages 2223–2232, 2017.

[16] Ming-Yu Liu, Thomas Breuel, and Jan Kautz. Unsupervised image-to-image translation networks. Advances in

neural information processing systems, 30, 2017.

[17] Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron
Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems,
27, 2014.

[18] Xun Huang, Ming-Yu Liu, Serge Belongie, and Jan Kautz. Multimodal unsupervised image-to-image translation.

In ECCV, 2018.

[19] Yunjey Choi, Minje Choi, Munyoung Kim, Jung-Woo Ha, Sunghun Kim, and Jaegul Choo. Stargan: Unified
generative adversarial networks for multi-domain image-to-image translation. In Proceedings of the IEEE
conference on computer vision and pattern recognition, pages 8789–8797, 2018.

[20] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole.
Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456,
2020.

[21] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural

information processing systems, 33:6840–6851, 2020.

[22] Beomsu Kim, Gihyun Kwon, Kwanyoung Kim, and Jong Chul Ye. Unpaired image-to-image translation via
neural schrödinger bridge. In The Twelfth International Conference on Learning Representations, 2024.

[23] Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Yuhta Takida, Naoki Murata, Toshimitsu Uesaka, Yuki
Mitsufuji, and Stefano Ermon. Pagoda: Progressive growing of a one-step generator from a low-resolution
diffusion teacher. Advances in Neural Information Processing Systems, 37:19167–19208, 2024.

[24] Yipin Zhang, Ziqi Yu, Xiange Zhang, Shengjie Zhang, Xiang Chen, Haibo Yang, and Xiao-Yong Zhang. Disdiff:
Disentanglement diffusion network for mr imaging translation. In International Conference on Medical Image
Computing and Computer-Assisted Intervention, pages 153–163. Springer, 2025.

[25] Sihan Xu, Ziqiao Ma, Yidong Huang, Honglak Lee, and Joyce Chai. Cyclenet: Rethinking cycle consistent in
text-guided diffusion for image manipulation. In Advances in Neural Information Processing Systems (NeurIPS),
2023.

11

A PREPRINT -

[26] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit:
Guided image synthesis and editing with stochastic differential equations. In International Conference on
Learning Representations, 2022.

[27] Xuan Su, Jiaming Song, Chenlin Meng, and Stefano Ermon. Dual diffusion implicit bridges for image-to-image

translation. In International Conference on Learning Representations, 2023.

[28] Inbar Huberman-Spiegelglas, Vladimir Kulikov, and Tomer Michaeli. An edit friendly ddpm noise space:
Inversion and manipulations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 12469–12478, 2024.

[29] Litu Rout, Yujia Chen, Nataniel Ruiz, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Semantic
image inversion and editing using rectified stochastic differential equations. In The Thirteenth International
Conference on Learning Representations, 2025.

[30] Chen Henry Wu and Fernando De la Torre. A latent space of stochastic diffusion models for zero-shot image
editing and guidance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages
7378–7387, 2023.

[31] Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu. Zero-shot

image-to-image translation. In ACM SIGGRAPH 2023 conference proceedings, pages 1–11, 2023.

[32] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven
In Proceedings of the IEEE/CVF conference on computer vision and pattern

image-to-image translation.
recognition, pages 1921–1930, 2023.

[33] Jiwoo Chung, Sangeek Hyun, and Jae-Pil Heo. Style injection in diffusion: A training-free approach for adapting
large-scale diffusion models for style transfer. In Proceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 8795–8805, 2024.

[34] Omri Avrahami, Or Patashnik, Ohad Fried, Egor Nemchinov, Kfir Aberman, Dani Lischinski, and Daniel
Cohen-Or. Stable flow: Vital layers for training-free image editing. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), pages 7877–7888, June 2025.

[35] Linqi Zhou, Aaron Lou, Samar Khanna, and Stefano Ermon. Denoising diffusion bridge models. In The Twelfth

International Conference on Learning Representations, 2024.

[36] Shaorong Zhang, Yuanbin Cheng, Xianghao Kong, and Greg Ver Steeg. Exploring the design space of diffusion

bridge models via stochasticity control. arXiv preprint arXiv:2410.21553, 2024.

[37] Guan-Horng Liu, Arash Vahdat, De-An Huang, Evangelos Theodorou, Weili Nie, and Anima Anandkumar. I2SB:
Image-to-image schrödinger bridge. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt,
Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine
Learning, volume 202 of Proceedings of Machine Learning Research, pages 22042–22062. PMLR, 23–29 Jul
2023.

[38] Bohan Xiao, Peiyong Wang, Qisheng He, and Ming Dong. Deterministic image-to-image translation via
denoising brownian bridge models with dual approximators. In Proceedings of the Computer Vision and Pattern
Recognition Conference, pages 28232–28241, 2025.

[39] Michael S Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework

for flows and diffusions. arXiv preprint arXiv:2303.08797, 2023.

[40] Clément Chadebec, Onur Tasar, Sanjeev Sreetharan, and Benjamin Aubin. Lbm: Latent bridge matching for fast

image-to-image translation. arXiv preprint arXiv:2503.07535, 2025.

[41] Fuat Arslan, Bilal Kabas, Onat Dalmaz, Muzaffer Ozbey, and Tolga Çukur. Self-consistent recursive diffusion

bridge for medical image translation. Medical Image Analysis, 106:103747, 2025.

[42] Alexander Korotin, Daniil Selikhanovych, and Evgeny Burnaev. Neural optimal transport. In The Eleventh

International Conference on Learning Representations, 2023.

[43] Petr Mokrov, Alexander Korotin, Alexander Kolesov, Nikita Gushchin, and Evgeny Burnaev. Energy-guided
entropic neural optimal transport. In The Twelfth International Conference on Learning Representations, 2024.
Improving neural optimal transport via displacement

[44] Jaemoo Choi, Yongxin Chen, and Jaewoong Choi.

interpolation. In The Thirteenth International Conference on Learning Representations, 2025.

[45] Yuyang Shi, Valentin De Bortoli, Andrew Campbell, and Arnaud Doucet. Diffusion schrödinger bridge matching.

Advances in Neural Information Processing Systems, 36:62183–62223, 2023.

[46] Nikita Gushchin, Daniil Selikhanovych, Sergei Kholkin, Evgeny Burnaev, and Aleksandr Korotin. Adversarial
schrödinger bridge matching. Advances in Neural Information Processing Systems, 37:89612–89651, 2024.

12

A PREPRINT -

[47] Grigoriy Ksenofontov and Alexander Korotin. Categorical schr\" odinger bridge matching. arXiv preprint

arXiv:2502.01416, 2025.

[48] Suhyeon Lee, Kwanyoung Kim, and Jong Chul Ye. Single-step bidirectional unpaired image translation using

implicit bridge consistency distillation. arXiv preprint arXiv:2503.15056, 2025.

[49] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin.
Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international
conference on computer vision, pages 9650–9660, 2021.

[50] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre
Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech
Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel
Synnaeve, Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2:
Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. Featured
Certification.

[51] Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov,
Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.

[52] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik
Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis.
In Forty-first international conference on machine learning, 2024.

[53] Chuanxia Zheng, Tat-Jen Cham, and Jianfei Cai. The spatially-correlative loss for various image translation tasks.
In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16407–16417,
2021.

[54] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International

Conference on Learning Representations, 2021.

[55] Vladimir Kulikov, Matan Kleiner, Inbar Huberman-Spiegelglas, and Tomer Michaeli. Flowedit: Inversion-free

text-based editing using pre-trained flow models. arXiv preprint arXiv:2412.08629, 2024.

[56] Jeongsol Kim, Yeobin Hong, Jonghyun Park, and Jong Chul Ye. Flowalign: Trajectory-regularized, inversion-free

flow-based image editing. arXiv preprint arXiv:2505.23145, 2025.

[57] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim
Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik
Lorenz, Jonas Müller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext:
Flow matching for in-context image generation and editing in latent space, 2025.

[58] Yuwen Chen, Nicholas Konz, Hanxue Gu, Haoyu Dong, Yaqian Chen, Lin Li, Jisoo Lee, and Maciej A
Mazurowski. Contourdiff: Unpaired image-to-image translation with structural consistency for medical imaging.
arXiv preprint arXiv:2403.10786, 2024.

[59] Pablo Pernias, Dominic Rampas, Mats Leon Richter, Christopher Pal, and Marc Aubreville. Würstchen: An
efficient architecture for large-scale text-to-image diffusion models. In The Twelfth International Conference on
Learning Representations, 2024.

[60] Tianhong Li, Dina Katabi, and Kaiming He. Return of unconditional generation: A self-supervised representation

generation method. Advances in Neural Information Processing Systems, 37:125441–125468, 2024.

[61] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie.
Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint
arXiv:2410.06940, 2024.

[62] Junyi Zhang, Charles Herrmann, Junhwa Hur, Luisa Polania Cabrera, Varun Jampani, Deqing Sun, and Ming-
Hsuan Yang. A tale of two features: Stable diffusion complements dino for zero-shot semantic correspondence.
Advances in Neural Information Processing Systems, 36:45533–45547, 2023.

[63] Fei Xue, Sven Elflein, Laura Leal-Taixé, and Qunjie Zhou. Matcha: Towards matching anything. In Proceedings

of the Computer Vision and Pattern Recognition Conference, pages 27081–27091, 2025.

[64] Yunpeng Bai and Qixing Huang. Fiffdepth: Feed-forward transformation of diffusion-based generators for
detailed depth estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages
6023–6033, 2025.

[65] Kaiwen Zheng, Guande He, Jianfei Chen, Fan Bao, and Jun Zhu. Diffusion bridge implicit models. In The

Thirteenth International Conference on Learning Representations, 2025.

13

A PREPRINT -

[66] Michael Samuel Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In

The Eleventh International Conference on Learning Representations, 2023.

[67] Xingchao Liu, Chengyue Gong, and qiang liu. Flow straight and fast: Learning to generate and transfer data

with rectified flow. In The Eleventh International Conference on Learning Representations, 2023.

[68] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for

generative modeling. In The Eleventh International Conference on Learning Representations, 2023.

[69] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image
synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 10684–10695, 2022.

[70] Adrian Thummerer, Erik Van der Bijl, Arthur Galapon Jr, Joost JC Verhoeff, Johannes A Langendijk, Stefan
Both, Cornelis (Nico) AT van den Berg, and Matteo Maspero. Synthrad2023 grand challenge dataset: Generating
synthetic ct for radiotherapy. Medical physics, 50(7):4664–4674, 2023.

[71] Adrian Thummerer, Erik van der Bijl, Arthur Jr Galapon, Florian Kamp, Mark Savenije, Christina Muijs, Shafak
Aluwini, Roel JHM Steenbakkers, Stephanie Beuel, Martijn PW Intven, et al. Synthrad2025 grand challenge
dataset: Generating synthetic cts for radiotherapy from head to abdomen. Medical physics, 52(7):e17981, 2025.
[72] Yuanfeng Ji, Haotian Bai, Chongjian Ge, Jie Yang, Ye Zhu, Ruimao Zhang, Zhen Li, Lingyan Zhanng, Wanling
Ma, Xiang Wan, et al. Amos: A large-scale abdominal multi-organ benchmark for versatile medical image
segmentation. Advances in neural information processing systems, 35:36722–36732, 2022.

[73] Jakob Wasserthal, Hanns-Christian Breit, Manfred T Meyer, Maurice Pradella, Daniel Hinck, Alexander W
Sauter, Tobias Heye, Daniel T Boll, Joshy Cyriac, Shan Yang, et al. Totalsegmentator: robust segmentation of
104 anatomic structures in ct images. Radiology: Artificial Intelligence, 5(5):e230024, 2023.

[74] IXI dataset: Brain development imaging. https://brain-development.org/ixi-dataset/. Accessed:

2025-08-05.

[75] Tufve Nyholm, Stina Svensson, Sebastian Andersson, Joakim Jonsson, Maja Sohlin, Christian Gustafsson,
Elisabeth Kjellén, Karin Söderström, Per Albertsson, Lennart Blomqvist, et al. Mr and ct data with multiobserver
delineations of organs in the pelvic area—part of the gold atlas project. Medical physics, 45(3):1295–1300, 2018.
[76] Sergios Gatidis, Tobias Hepp, Marcel Früh, Christian La Fougère, Konstantin Nikolaou, Christina Pfannenberg,
Bernhard Schölkopf, Thomas Küstner, Clemens Cyran, and Daniel Rubin. A whole-body fdg-pet/ct dataset with
manually annotated tumor lesions. Scientific Data, 9(1):601, 2022.

[77] Ibrahim Ethem Hamamci, Sezgin Er, Chenyu Wang, Furkan Almas, Ayse Gulnihan Simsek, Sevval Nil Esirgun,
Irem Doga, Omer Faruk Durugol, Weicheng Dai, Murong Xu, et al. Developing generalist foundation models
from a multimodal dataset for 3d computed tomography. arXiv preprint arXiv:2403.17834, 2024.

[78] Thomas J Littlejohns, Jo Holliday, Lorna M Gibson, Steve Garratt, Niels Oesingmann, Fidel Alfaro-Almagro,
Jimmy D Bell, Chris Boultwood, Rory Collins, Megan C Conroy, et al. The uk biobank imaging enhancement
of 100,000 participants: rationale, data collection, management and future directions. Nature communications,
11(1):2624, 2020.

[79] Flickr image dataset. https://flickr.com. Accessed: 2025-09-14.
[80] Eirikur Agustsson and Radu Timofte. Ntire 2017 challenge on single image super-resolution: Dataset and study.
In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pages 126–135,
2017.

[81] Pexels. https://www.pexels.com/. Accessed: 2025-09-14.
[82] Harshit Gupta, Kyong Hwan Jin, Ha Q Nguyen, Michael T McCann, and Michael Unser. Cnn-based projected
gradient descent for consistent ct image reconstruction. IEEE transactions on medical imaging, 37(6):1440–1453,
2018.

[83] Zhou Wang, Eero P Simoncelli, and Alan C Bovik. Multiscale structural similarity for image quality assessment.
In The thrity-seventh asilomar conference on signals, systems & computers, 2003, volume 2, pages 1398–1402.
Ieee, 2003.

[84] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by
a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing
systems, 30, 2017.

[85] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language
supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

14

A PREPRINT -

[86] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness

of deep features as a perceptual metric. In CVPR, 2018.

[87] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error

visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004.

[88] Minyoung Huh, Brian Cheung, Tongzhou Wang, and Phillip Isola. Position: The platonic representation

hypothesis. In Forty-first International Conference on Machine Learning, 2024.

[89] Effrosyni Doutsi, Lionel Fillatre, Marc Antonini, and Julien Gaulmin. Retina-inspired filter. IEEE Transactions

on Image Processing, 27(7):3484–3499, 2018.

[90] Sheng Zhang, Yanbo Xu, Naoto Usuyama, Hanwen Xu, Jaspreet Bagga, Robert Tinn, Sam Preston, Rajesh Rao,
Mu Wei, Naveen Valluri, et al. Biomedclip: a multimodal biomedical foundation model pretrained from fifteen
million scientific image-text pairs. arXiv preprint arXiv:2303.00915, 2023.

[91] Andrew Sellergren, Sahar Kazemzadeh, Tiam Jaroensri, Atilla Kiraly, Madeleine Traverse, Timo Kohlberger,
Shawn Xu, Fayaz Jamil, Cían Hughes, Charles Lau, et al. Medgemma technical report. arXiv preprint
arXiv:2507.05201, 2025.

[92] Maya Varma, Ashwin Kumar, Rogier Van der Sluijs, Sophie Ostmeier, Louis Blankemeier, Pierre Joseph Marcel
Chambon, Christian Bluethgen, Jip Prince, Curtis Langlotz, and Akshay S Chaudhari. MedVAE: Efficient
automated interpretation of medical images with large-scale generalizable autoencoders. In Medical Imaging
with Deep Learning, 2025.

[93] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural

information processing systems, 34:8780–8794, 2021.

[94] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based

generative models. Advances in neural information processing systems, 35:26565–26577, 2022.

[95] Muzaffer Özbey, Onat Dalmaz, Salman U. H. Dar, Hasan A. Bedel, ¸Saban Özturk, Alper Güngör, and Tolga
Çukur. Unsupervised medical image translation with adversarial diffusion models. IEEE Transactions on
Medical Imaging, 42(12):3524–3539, 2023.

[96] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit:
Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European
Conference on Computer Vision, pages 23–40. Springer, 2024.

[97] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo
Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for
training next generation image-text models. Advances in neural information processing systems, 35:25278–25294,
2022.

[98] Xiaofeng Yang, Chen Cheng, Xulei Yang, Fayao Liu, and Guosheng Lin. Text-to-image rectified flow as

plug-and-play priors. In The Thirteenth International Conference on Learning Representations, 2025.
[99] John Charles Butcher. Numerical methods for ordinary differential equations. John Wiley & Sons, 2016.
[100] Lawrence C Evans. Partial differential equations, volume 19. American mathematical society, 2022.
[101] Robert B Ash and Catherine A Doléans-Dade. Probability and measure theory. Academic press, 2000.
[102] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining
Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. In The
Thirteenth International Conference on Learning Representations, 2025.

[103] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej
Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet Large Scale Visual
Recognition Challenge. International Journal of Computer Vision (IJCV), 115(3):211–252, 2015.

[104] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance.

In NeurIPS 2021 Workshop on Deep

Generative Models and Downstream Applications, 2021.

[105] InstantX Team. Sd3-controlnet-canny. https://huggingface.co/InstantX/SD3-Controlnet-Canny,

2024. Hugging Face model card; accessed 2025-10-17.

[106] Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network

representations revisited. In International conference on machine learning, pages 3519–3529. PMlR, 2019.

[107] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole.
Score-based generative modeling through stochastic differential equations. In International Conference on
Learning Representations, 2021.

15

[108] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image
segmentation. In International Conference on Medical image computing and computer-assisted intervention,
pages 234–241. Springer, 2015.

A PREPRINT -

16

Supplementary Material for
Self-Supervised Semantic Bridge

A PREPRINT -

A Theoretical Details

Our theoretical analysis aims to quantify the difference between the ground-truth target x0 and the predicted translation
¯x0. We consider the ODE formulations of diffusion models. The following transparent assumptions are made throughout
our analysis.
Assumption A.1. Given a pair of source and target images (x(j), x(i)) that share the same latent code z, the learned
encoder Eϕ : Rn → Rm may not perfectly map both images to the identical latent code; that is, Eϕ(x(j)) ̸= Eϕ(x(i)).

This assumption accounts for the imperfection of the learned encoder. In our analysis, we address this by introducing an
error term associated with different initializations.
Assumption A.2. We assume the learned KL-VAE decoder Dφ : Rm → Rn satisfies the following conditions:

(a) Dφ has a bounded reconstruction error; that is, ∥Dφ(z) − x∥2 ≤ εDφ, where z denotes the latent code of x.
(b) Dφ is Lipschitz continuous with LDφ > 0 for any z1, z2 ∈ Rm

∥Dφ(z1) − Dφ(z2)∥2 ≤ LDφ ∥z1 − z2∥2.

Assumption A.2.(a) accounts for the imperfection of the learned decoder by assuming a bounded reconstruction error.
Assumption A.2.(b) imposes a standard regularity condition to ensure that the output of decoder does not change
drastically.
Assumption A.3. We assume the learned latent vector field vθ(t, z, zT ) satisfies the following conditions:

(a) For any t > 0, vθ(t, z, zT ) is Lipschitz continuous with Lv(t) > 0 for any z1, z2 ∈ Rn

∥vθ(t, z1, zT ) − vθ(t, z2, zT )∥2 ≤ Lv(t)∥z1 − z2∥2.

(b) For any t > 0, vθ(t, z, zT ) has a L2-accurate error

E (cid:2)∥vθ(t, z, zT ) − v(t, z, zT )∥2

2

(cid:3) ≤ εv(t),

where εv(t) < +∞.

Note that our analysis only requires L2-accurate vector field estimate under the measure pt, which is a relatively weaker
condition than the L∞-accurate estimate that assumes the error between the learned and ground-truth score is uniformly
bounded.

A.1 Derivation for Theorem 4.1

In this section, we present the detailed derivation for Theorem 4.1. The derivation is organized into four steps. The first
three steps progressively quantify the errors introduced by the encoder, the learned vector field, the Euler discretization,
and the decoder, while the final step derives the high-probability upper bound.

Step 1: Encoder Error & Continuous-Time Error Propagation. Consider the following ODEs characterized by
ideal and learned vector fields, respectively

dzt = v(t, zt, zT )dt, with zT = Eϕ(x(i))
d(cid:98)zt = vθ(t, (cid:98)zt, (cid:98)zT )dt, with (cid:98)zT = Eϕ(x(j))

where both ODEs evolve from T to 0. Define the time-varying difference as ∆(t) = zt − (cid:98)zt. We can write

d
dt

∆(t) = v(t, zt, zT ) − vθ(t, (cid:98)zt, (cid:98)zT )

(cid:16)

=

v(t, zt, zT ) − vθ(t, zt, zT )

(cid:17)

+

(cid:16)

(cid:17)
vθ(t, zt, zT ) − vθ(t, (cid:98)zt, (cid:98)zT )

(11)

(12)

(13)

17

A PREPRINT -

Taking the norm yields

∥

d
dt

∆(t)∥2 ≤ ∥vθ(t, zt, zT ) − vθ(t, (cid:98)zt, (cid:98)zT )∥2 + ∥v(t, zt, zT ) − vθ(t, zt, zT )∥2

= Lv(t)∥∆(t)∥2 + ∥e(t)∥2.

Here, we define e(t) = v(t, zt, zT )−vθ(t, zt, zT ). We can bound the growth of ∥∆(t)∥2 by using the Cauchy-Schwartz
inequality

d
dt

∥∆(t)∥2 =

∆(t)T d

dt ∆(t)

∥∆(t)∥2

≤

∥∆(t)∥2 · ∥ d
∥∆(t)∥2

dt ∆(t)∥2

≤ Lv(t)∥∆(t)∥2 + ∥e(t)∥2.

Due to the imperfection of the learned encoder, the two ODEs are run with different initialization; that is, ∥∆(T )∥2 =
∥Eϕ(x(j)) − Eϕ(x(i))∥2 ̸= 0. To quantify the difference at end point t = 0, we use the Grönwall Theorem (see
Theorem A.6) with time-varying coefficients

∥∆(0)∥2 ≤ exp

(cid:18) (cid:90) T

0

(cid:19)

Lv(s)ds

∥∆(T )∥2 +

(cid:90) T

0

exp

(cid:18) (cid:90) t

0

(cid:19)

Lv(s)ds

∥e(t)∥2dt

(14)

Equation (14) relates ∥∆(0)∥2 to the vector field approximation error in the continuous-time domain.

Step 2: Euler Discretization Error. We next incorporate the error caused by discretizing the ODE. Here, we consider
the Euler method, as it is the foundation of other popular first-order solvers. Our analysis builds on classical analysis on
the error of ODE discretization [99], and below we provide a brief derivation for the reader’s convenience. To begin
with, we introduce a standard assumption on the second-order time derivative of the solution path (cid:98)zt [99].
Assumption A.4. The second-order time derivative of the solution (cid:98)zt is bounded; that is, supt∈[0,T ] ∥ d2
where B > 0 is a finite constant.

dt2 (cid:98)zt∥2 ≤ B,

Consider the Euler method for solving the ODE in Equation 11 for t ∈ [0, T ].

¯zi+1 = ¯zi + τ vθ(ti, ¯zti, (cid:98)zT ),
where we use ¯zi to denote the discrete solution, τ = T /N the stepsize, N > 0 the number of steps, and ti the time
with ti = T − iτ . Applying the first-order Taylor expansion, we define the local truncation error for [ti, ti+1]
(cid:16)

(cid:17)

ηi = (cid:98)zti+1 −

(cid:98)zti + τ vθ(ti, (cid:98)zti, (cid:98)zT )

.

Define the discretization error at step i as εdisc

i = ∥(cid:98)zti − ¯zi∥2. We can write

(cid:16)

εdisc
i+1 = ∥(cid:98)zti+1 − ¯zi+1∥2
(cid:13)
(cid:13)
=
(cid:13)
(cid:13)
(cid:13)
(cid:13)(cid:98)zti − ¯zi + τ
≤ (1 + τ Lv(t))εdisc

(cid:98)zti + τ vθ(ti, (cid:98)zti, (cid:98)zT ) + ηi
(cid:16)

¯zi + τ vθ(ti, ¯zi, (cid:98)zT )
(cid:13)
(cid:17)
(cid:13)
vθ(ti, (cid:98)zti, (cid:98)zT ) − vθ(ti, ¯zi, (cid:98)zT )
(cid:13)2
i + ∥ηi∥2

+ ηi

−

=

(cid:16)

(cid:17)

(cid:17)(cid:13)
(cid:13)
(cid:13)2

(15)

The above inequality establishes the multiplicative recursion for εi with the truncation error ∥ηi∥2. From Assump-
tion A.4, we know that the truncation error is bounded

∥ηi∥2 ≤

τ 2
2

∥

d2
dt2 zξi∥2 ≤

τ 2
2

B,

where ξi ∈ (ti, ti+1). As we run the continuous-time and discretized ODEs with the same initialization (¯z0 = (cid:98)zt0 =
(cid:98)zT ), the initial discretization error is zero (εdisc
εdisc
N = ∥(cid:98)z0 − ¯zN ∥2

0 = 0). By unrolling (15), we obtain



N −1
(cid:89)

≤



(1 + τ Lv(tj))


 εdisc
0



+



B
2

N −1
(cid:88)

N −1
(cid:89)

(1 + τ Lv(tj))

i=0

j=i+1


 τ 2,

j=0

(cid:124)

(cid:123)(cid:122)
=0

(cid:125)

18

We can simplify the product in the nonzero term by using facts 1 + u ≤ eu

N −1
(cid:89)

(1 + τ Lv(tj)) ≤ exp

j=i+1





N −1
(cid:88)

j=i+1



τ Lv(tj)



Expressing the summation as an integral, we obtain

εdisc
N ≤





B
2

N −1
(cid:88)

N −1
(cid:89)

(1 + τ Lv(tj))

i=0

j=i+1


 τ 2 ≤





N −1
(cid:88)



exp



N −1
(cid:88)

i=0

j=i+1

B
2




 τ 2.

τ Lv(tj)



A PREPRINT -

(16)

Step 3: Decoder Error & Deterministic Upperbound. Now we derive the final bound by accounting the imperfection
of the learned decoder. According to Assumption A.2, we can write

∥x − ¯x∥2 = (cid:13)

(cid:13)Dφ(z0) − Dφ(¯zN ) + x − Dφ(z0)(cid:13)
(cid:13)2

≤ LDφ ∥z0 − ¯zN ∥2 + εDφ.

Applying triangle inequality yields to

∥z0 − ¯zN ∥2 ≤ ∥z0 − (cid:98)z0∥2 + ∥(cid:98)z0 − ¯zN ∥2
= ∥∆(0)∥2 + εN .

Combining equations (14) and (16), we can obtain the final bound

∥x − ¯x∥2 ≤ W (T )∥∆(T )∥2

+

(cid:124)

(cid:123)(cid:122)
Encoder Err.

(cid:125)

(cid:90) T

0

(cid:124)

W (t)∥e(t)∥2 dt

+ Cτ 2

+ εDφ

.

(cid:123)(cid:122)
Field Approx. Err.

(cid:125)

(cid:124)(cid:123)(cid:122)(cid:125)
Discret. Err.

(cid:124)(cid:123)(cid:122)(cid:125)
Decoder Err.

where W (t) and C are given by

W (t) = LDφ exp

(cid:18)(cid:90) t

(cid:19)

Lv(s) ds

,

N −1
(cid:88)

0


exp



N −1
(cid:88)



τ Lv(tj)

 .

i=0

j=i+1

C =

B
2

(17)

(18)

(19)

Step 4: Probabilistic Upperbound. In (18), randomness arises from the vector field approximation error according to
Assumption A.3(b). For notational convenience, we define the following quantities

E ≜

W ≜

Q ≜

(cid:90) T

0
(cid:90) T

0
(cid:90) T

0

εv(t) dt,

W (t)2 dt,

∥e(t)∥2

2 dt.

According to the Cauchy-Schwartz inequality, we can obtain

(cid:90) T

0

W (t) ∥e(t)∥2 dt ≤

(cid:115)

(cid:90) T

0

W (t)2 dt

(cid:115)

(cid:90) T

0

∥e(t)∥2

2 dt ≤

√

W Q.

(20)

Applying the Markov inequality (see Theorem A.7) to Q yields

(cid:18)

P

Q ≤

(cid:19)

E
δ

= 1 − P

(cid:18)

Q ≥

(cid:19)

E
δ

≥ 1 −

E[Q]
E/δ

.

19

A PREPRINT -

As ∥e(t)∥2

2 is nonnegative, we can swap the integral in E[Q] (Fubini-Tonelli Theorem) and write

E[Q] = E

∥v(t, zt, zT ) − vθ(t, zt, zT )∥2

(cid:35)
2dt

(cid:34)(cid:90) T

0

=

≤

(cid:90) T

0
(cid:90) T

0

E (cid:2)∥v(t, zt, zT ) − vθ(t, zt, zT )∥2

2

(cid:3) dt

εv(t)dt = E,

where the last equality uses Assumption A.3(b). Overall, we have

(cid:18)

P

Q ≤

(cid:19)

E
δ

≥ 1 −

E[Q]
E/δ

≥ 1 −

E
E/δ

≥ 1 − δ.

(21)

Note that, by applying (20), we can express (18) as

∥x − ¯x∥2 ≤ W (T )∥∆(T )∥2 +

√

WQ + Cτ 2 + εDφ.

Applying the result in (21), we derive the result in Theorem 4.1, that is, with probability at least 1 − δ,

∥x − ¯x∥2 ≤ W (T )∥∆(T )∥2 +

(cid:114)

WE
δ

+ Cτ 2 + εDφ .

A.2 Analysis of the DINO Encoders Training

Remark A.5 (Appearance invariance). Let augmentations factor as A = G ◦ C, with C ∈ C (appearance transforms, e.g.,
color, contrast, and style) and G ∈ G (geometric transforms, e.g., shape scale, and layout). At stationary (ϕ ≈ ϕ′), any
minimizer ϕ⋆ of L(ϕ; ϕ′) satisfies, for almost every x,

pϕ⋆ (G ◦ C(x)) ≈ pϕ⋆ (G(x)),

∀ C ∈ C, G ∈ G.

Proof. At stationarity (ϕ = ϕ′), the training loss reduces to

Lϕ = EA1,A2,x

(cid:2)− pϕ(A2(x))⊤ log pϕ(A1(x))(cid:3) .

(22)

For fixed (x, A1, A2) this is the cross-entropy CE(q, p) between q = pϕ(A2(x)) and p = pϕ(A1(x)). Since
CE(q, p) ≥ H(q) with equality iff p = q, the loss is minimized precisely when
∀ A1, A2.

pϕ(A1(x)) = pϕ(A2(x)),

(23)

Writing each augmentation as A = G ◦ C with C ∈ C (appearance) and G ∈ G (geometry), condition (23) implies

pϕ(G ◦ C(x)) = pϕ(G(x)),

∀ C ∈ C, G ∈ G,

(24)

which proves the remark.

A.3 Supporting Theorems

We here summarize all the theorems used in our proofs.
Theorem A.6 (Grönwall Theorem). Let f (t) be a nonnegative, absolutely continuous function on [0, T ], which satisfies
for almost everywhere t the differential inequality

df (t)
dt

≤ c(t)f (t) + h(t),

where f (t) and h(t) are nonnegative, summable functions on [0,T]. Then
(cid:90) t

(cid:18)(cid:90) t

(cid:19) (cid:20)

f (t) ≤ exp

c(s)ds

f (0) +

(cid:21)

h(s)ds

.

Proof. See Appendix B in [100].

0

0

Theorem A.7 (Markov Inequality). Let X be a nonnegative random variable and let a > 0. Then, we have

Proof. See pp 276 in [101].

P(X ≥ a) ≤

E[X]
a

.

20

A PREPRINT -

Figure 9: Visual Results. Qualitative results on natural image translation. We show unpaired horse→zebra and
apple→orange examples comparing SSB to prior GAN- and diffusion-based baselines. Our method achieves strong
appearance transfer (striping / color change) while better preserving object pose, background layout, and fine structural
details. The SSB results are obtained by fine-tuning SiT-XL/2.

B Datasets Usage

This section provides comprehensive information about the datasets used in our experiments, including data characteris-
tics, annotation details, and their roles in our experimental setup. We organize datasets into two categories: Medical
images and Natural images.

B.1 Medical Images

When constructing the CT training and test sets for all methods in this work, we apply TotalSegmentator [73] to each
CT volume to obtain a body mask and discard non-body regions. This removes the scanner table and other background
objects, making CT slices visually closer to MRI and reducing the cross-modality appearance gap. This preprocessing
step relies on an off-the-shelf tool and is applied only once per scan. It is therefore relatively cheap and unlikely to
become a computational bottleneck.

It is worth noting that, compared with large-scale natural image collections (e.g., LAION-5B [97]), publicly available
MRI/CT datasets are much smaller and often exhibit modality-specific artifacts (e.g., low-dose noise, motion artifacts,
and low spatial resolution). These factors limit the overall image quality and pose challenges for designing and training
high-capacity deep models. To better match, as far as possible, the training scale of DINOv2 on natural images, we
aggregate MRI/CT scans from multiple public datasets. On the other hand, medical images typically contain more
structured and recurring anatomical patterns (e.g., organs and bones) than natural images, which may reduce the amount
of data required to learn useful representations; quantifying this effect is an interesting direction for future work. Below,
we introduce each dataset individually.

SynthRAD2023. The SynthRAD2023 [70] radiotherapy dataset provides paired multimodal images (MRI–CT and
CBCT–CT) for brain and pelvic treatment sites from three Dutch university medical centers. Various scanner models

21

A PREPRINT -

and acquisition settings were used across patients at these centers. In total, we selected a subset of 350 MRI–CT patient
cases that are relevant to our study. The subset consists of approximately two-thirds T1-weighted spoiled gradient-echo
sequences and one-third T2-weighted fast spin-echo sequences, with no contrast used for either MRI or CT. We select
about 20 subjects for evaluation and treat the rest as training dataset.

SynthRAD2025. Similarly, SynthRAD2025 [71] benchmarks synthetic CT generation for MRI- and CBCT-based
radiotherapy workflows. The dataset contains MRI-to-CT pairs for MR-only and MR-guided photon/proton radiotherapy
(890 MRI–CT pairs) and CBCT-to-CT pairs for daily adaptive workflows (1,472 CBCT–CT pairs) from patients who
underwent radiotherapy in the head-and-neck, thorax, or abdomen. The population is predominantly adult, with no
gender restrictions applied during data collection, and the images are aggregated from five international centers. In this
work, we focus on the MRI-CT subset and use approximately 637 publicly available MRI–CT cases that match our
training experimental design and 30 for testing.

Multi-organ Abdominal Collection (AMOS). AMOS [72] is a multi-modal dataset from Longgang District People’s
Hospital, featuring 500 CT and 100 MRI scans from 600 patients with abdominal abnormalities. Acquired across eight
different scanner platforms, the dataset provides annotations for 15 anatomical structures in CT (spleen, right kidney,
left kidney, gallbladder, esophagus, liver, stomach, aorta, inferior vena cava, pancreas, right adrenal gland, left adrenal
gland, duodenum, bladder, and prostate/uterus) and 13 structures in MRI (excluding bladder and prostate in some cases).
We use both modalities for training dataset construction.

Whole-body PET/CT Collection (AutoPET). AutoPET [76] comprises 1,014 whole-body FDG-PET/CT studies,
balanced between 501 cases with confirmed malignancies (lymphoma, melanoma, NSCLC) and 513 negative control
cases. All scans include both PET and CT modalities with annotations for malignant lesions. We use the CT images for
our training dataset.

TotalSegmentator. TotalSegmentator [73] provides comprehensive whole-body segmentation datasets. The CT portion
comprises 1,229 scans with annotations for 104 anatomical structures spanning all major body regions, organs, vessels,
and skeletal structures. The MR portion includes 299 scans with annotations for 50 anatomical structures. Both datasets
aggregate scans from multiple sources and institutions, providing diverse imaging protocols and patient populations.
We adapt both modalities for unpaired training dataset construction.

IXI. The IXI dataset [74] contains nearly 600 brain MR scans from healthy subjects, acquired at three London hospitals
(Philips 3T, Philips 1.5T, GE 1.5T). For each subject, T1-, T2- and PD-weighted images, MRA, and diffusion-weighted
images are available. Following prior work, we chose a subset of IXI, containing about 300 subjects with T1/T2/PD
images. We exclude boundary slices that contain no brain tissue from each volume.

Pelvic MRI/CT Dataset. This Pelvic MRI/CT [75] dataset contains pelvic T1-, T2-weighted MRI and CT scans from
15 subjects. T1 images were acquired with TE ≈ 4.8–7.2 ms and TR ≈ 500–746 ms at either 0.88 × 0.88 × 3 mm3
or 1.10 × 1.10 × 2 mm3 resolution; T2 images with TE ≈ 91–97 ms, TR ≈ 6000–16000 ms and in-plane resolution
0.88–1.10 mm at 2.5 mm slice thickness. CT scans have 0.10 × 0.10 × 2–3 mm3 resolution with reconstruction kernels
B30f or FC17.

UKBB. UK Biobank’s imaging enhancement [78] is a large population study, each undergoing brain, cardiac, and
abdominal/whole-body MRI, plus DXA and carotid ultrasound. The abdominal and whole-body MRI are acquired
on a 1.5T system using multi-station Dixon sequences and additional liver/pancreas protocols. These body scans
provide volumetric measures of visceral and subcutaneous fat, muscle volume, and organ morphology, and are explicitly
designed as a gold standard for body composition and ectopic fat quantification. Since the UKBB whole-body MRI
dataset exhibits different appearance and resolution compared to the in-domain SynthRAD2023 and SynthRAD2025
data, we treat it as an out-of-domain MRI dataset. We select a small subset of UKBB whole-body MRI with both
fat-only and water-only Dixon reconstructions, containing about 100 subjects each. For each volume, we extract 5 axial
slices spanning from the mid-thigh to the chest.

In-domain vs. Out-of-domain (Empirical Justification). Although all models in our comparison are trained in an
unpaired fashion, the underlying MRI and CT datasets differ substantially in their acquisition compatibility. The
SynthRAD2023/2025 MRI and CT volumes were originally collected for supervised MRI→CT research and therefore
share similar spatial resolution, anatomical framing, and pre-registered geometry. As a result, the marginal distributions
of SynthRAD MRI and CT are naturally closer. Empirically, the Fréchet Inception Distance (FID) between SynthRAD
MRI and SynthRAD CT is 120.7. In contrast, UKBB fat- and water-suppressed MRI have different contrast mechanisms,
intensity statistics, and voxel sizes, and are not geometrically aligned to any CT dataset we use. Consequently, their
distributional distance to SynthRAD CT is substantially larger (FID of 177.2 and 191.6, respectively). Even UKBB fat-
and water-suppressed MRIs differ from each other (FID 69.6), highlighting their contrast heterogeneity.

22

A PREPRINT -

Table 3: FID distances between MRI and CT datasets. U-FAT = UKBB fat-suppressed MRI; U-WATER = UKBB
water-suppressed MRI; S-MRI = SynthRAD2023/2025 MRI; S-CT = SynthRAD2023/2025 CT. SynthRAD MRI
is substantially closer to SynthRAD CT (FID 120.7) than UKBB MRI (FID 177–192), supporting our definition of
SynthRAD MRI as “in-domain” and UKBB MRI as “OOD” with respect to CT acquisition.

FID Dist.↓

U-FAT U-WATER S-MRI

S-CT

U-FAT
U-WATER
S-MRI
S-CT

0
—
—
—

73.5
0
—
—

97.6
110.4
0
—

177.2
191.6
120.7
0

We therefore use “in-domain MRI” to denote MRI acquisitions whose marginal distribution is compatible with the
CT dataset used in evaluation (as in SynthRAD), and “out-of-domain MRI” to denote acquisitions that differ strongly
in resolution and contrast (as in UKBB). This distinction reflects data compatibility, not supervision. On the other
hand, this combination of in-domain and out-of-domain datasets can also be viewed as a form of multi-contrast MRI
translation, where diverse MRI contrasts (e.g., T1, T2, water–fat images) are translated into a unified CT domain.

B.2 Natural Images

In this section, we introduce the datasets used to finetune the SiT class-label ImageNet flow transformers and the
text–image SD3-M transformers employed in our main paper. For SiT, we follow [102] and first resize ImageNet-
1K [103] images to 256 × 256 and encode them into a VAE latent space of spatial size 32 × 32 with 4 channels.
ImageNet-1K spans 1000 object classes and contains 1,281,167 training images.

For SD3-M finetuning, we adopt LAION-POP and a high-resolution, non-overlapping subset of LAION-2B, both
derived from LAION-5B [97]. LAION-POP contains approximately 600,000 high-resolution images emphasizing
popular text-to-image concepts and is heavily filtered for high aesthetic scores and a minimum resolution of 768 px
on the shorter side, making it suitable for finetuning text–image models. We then sample an additional subset from
LAION-2B by requiring the shorter side to exceed 800 px and enforcing non-overlap with LAION-POP. Together, we
choose about 1.2M high-quality text–image pairs in total.

θ , v(j)

θ ; grid {tk}N

0 ; pw; ηt.

0 ; encoder Eϕ; decoder Dφ; inverter zinv

Algorithm 1 Simplified SSB for Conditional I2I Translation and Editing
T ; drifts v(i)
Require: Source x(j)
Ensure: Translated latent ¯z(i)
0 ; .
T (x(j)
1: y ← Eϕ(x(j)
0 )
2: zT ← y
3: z(i)
t0 , ˜z(i)
t0 , z(j)
4: for k = 0 to N − 1 do
5:
6:
7:

0 ) or zinv

t0 ← zT

else

, zT )

uncond)

cond − v(i)

uncond + s (v(i)

t ← tk, ∆t ← tk+1 − tk
cfg_on ← (s > 1) ∧ (t ∈ [tmin, tmax])
if cfg_on then
t ← v(i)
d(i)
t ← v(i)(t, z(i)
d(i)
end if
t ← v(j)(t, z(j)
d(j)
, zT )
t ← d(i)
d˜z(i)
t − d(i)
t + ηt(d(j)
t )
z(i)
← z(i)
tk
tk+1
{z(j), ˜z(i)}tk+1 ← {z(j), ˜z(i)}tk + ∆t · {d(j), d˜z(i)}t
tN ) (or Dφ(z(i)

0 ← Dφ(˜z(i)

+ ∆t · d(i)
t

t

t

tN ) if source model v(j)

θ

15:
16: end for
17: return ¯x(i)

8:
9:
10:
11:
12:

13:

14:

(Optional)
(Optional)

(Optional)

is not used)

23

A PREPRINT -

Figure 10: Data augmentation pipeline used for MRI-CT DINOv2 vision encoder. We adapt the widely used spatial-
temporal retinainspired filter for aligning MRI-CT appearance gap as illustrated in the bottom of this figure.

C Additional Implementation Details

Below we provide additional implementation details for our method and the baseline models that are obmitted from
the main paper due to space limitations. It is worth noting that in our work we do not explicitly tune the endpoint
Gaussian variance parameter b, which appears in zT = y ∼ N (Eϕ(x0), b2I). We include the notation for b primarily
to highlight the conceptual distinction between different bridge constructions discussed in the main paper—namely, the
medical image translation setting, where geometric alignment is strong, versus the natural image generation setting,
which involves richer semantic ambiguity. All baseline methods and our own models are trained on a cluster equipped
with AMD MI300 GPUs.

C.1 DINOv2 Pre-training for MRI-CT Translation

In this section, we briefly describe the MRI–CT augmentation pipeline used to train the DINOv2 ViT-B/8 encoder
(Fig. 10). As noted in the Remark in the main paper, our objective is to construct an augmentation strategy such that, at
convergence, the vision encoder becomes largely invariant to appearance variations while retaining stable geometric
structure. To achieve this, apart from commonly used augmentations, we incorporate a retinal-inspired contrast
filter implementing a spatial–temporal Gaussian center–surround mechanism (a weighted Difference-of-Gaussians).
This filter suppresses low-frequency appearance cues and enhances high-frequency structural information, providing
geometry-preserving augmentations that are suitable for contrast-agnostic MRI–CT representation learning. The
filtering effect on paired MRI-CT is introduced in Fig. 2 (bottom). The appearance gap between CT-MRI is largely
reduced by run few iterations of the filter. Following the original DINOv2 training settings, we set the batch size to 512.
The final model is obtained in approximately one day, requiring over 100,000 iterations.

Retina-Inspired Filter. Early retinal circuitry is well-approximated by a center–surround receptive field, in which a
narrow positive “center” is opposed by a broader negative “surround.” This structure functions as a spatial band-pass
operator that enhances local contrast and boundaries while suppressing slowly varying background intensity. Formally,
given an image f (x, y), we define a retinal-inspired response

R(x, y; t) = (cid:0)Gσc ∗ f (cid:1)(x, y) − ws

(cid:0)Gσs(t) ∗ f (cid:1)(x, y),

(25)

where Gσ is an isotropic Gaussian kernel of standard deviation σ, σc is the fixed center scale, and σs(t) > σc is a
surround scale that increases with a time-like parameter t. Larger t produces a broader surround and thus a stronger
center–surround effect.

24

A PREPRINT -

Practical Note.
In our preprocessing, we use cv::bioinspired::Retina. The parvocellular (detail) output of this
model behaves consistently with the formulation above and can be interpreted as snapshots of (φt ∗ f ) evaluated at
increasing effective times t. In this implementation, running the same input frame for m internal iterations corresponds
to evaluating the WDoG at a later time index, as each iteration expands the surround via a diffusion-like update. Typical
retinal settings follow σs ≈ 3σc with positive center and surround gains (wc, ws); we adopt the library’s default
configuration.

C.2 Medical I2I Translation Task—Bridge Model Training for CT Images

In this section, we provide additional details on training the diffusion bridge for MRI→CT translation. MRI and
CT exhibit strong geometric consistency and low semantic ambiguity, so the vision encoder Eϕ already produces a
shared latent space. This allows us to train a single bridge model with b = 0, yielding zT = y = Eϕ(x0). We use a
KL-VAE [69] with 64 × 64 spatial resolution and 8 latent channels, trained on the same combined MRI/CT dataset.
In practice, since all exiting medical SOTA baselines are implemented based on UNet diffusion model [21], so we
adapt the same type of UNet architecture. Our final bridge operates on 64 × 64 × 8 latents. Following standard bridge
constructions [35, 37, 40], we concatenate endpoints as UNet conditioning and apply a top-16 PCA channel sampling
(channel-wise shuffle) at each iteration to encourage appearance-invariant learning and improve robustness across MRI
contrasts. We set the training batch size to 128 and Adam optimizer with base lr=1e − 5, and the final model used in
this work is obtained in approximately one day.

C.3 Text-free I2I Translation Task—SiT Transformers Finetuning.

In this experiments, we further study SSB in a lower-capacity setting by fine-tuning it on class-conditional flow
transformers. Specifically, we use the pretrained 256×256 ImageNet-1K SiT-XL/2 model [96, 61], trained as a rectified
flow with a linear interpolant schedule and a 32 × 32 KL-VAE latent grid with 4 channels. To match the Gaussian
prior of the rectified-flow formulation, we set b = 1 during SSB adaptation, resulting in the endpoint distribution
zT = y ∼ N (Eϕ(x0), I). To align with SiT’s 4-channel latent space, we construct zT by average-pooling the top 16
PCA components of the DINOv3 embedding into a 4-channel tensor of size 32 × 32 × 4. These pooled features serve
as the terminal endpoint for the diffusion bridge.

Following the stand bridge design [35, 36, 40], we incorporate feature injection, allowing the drift network vθ to receive
auxiliary feature inputs from the source encoding E(x(j)
0 ). This conditioning is trained in a classifier-free guidance
manner [104], allowing controllable translation strength. In specific, we inject PCA-based structural conditioning
through a zero-initialized PatchEmbed module condembed, added to the SiT input tokens with a learnable scale c

,
x = x_embed(x) + pos + c cond_embed(w ⊙ cond)
(cid:125)

(cid:124)

(cid:123)(cid:122)
zero-init

(26)

with w ∼ Bernoulli(p). Zero-init ensures a smooth transition from the pretrained SiT, while stochastic conditioning
dropout (probability pw = 0.2) prevents over-reliance on the PCA signal. This yields more stable training and produces
a flexible, geometry-aware conditioning mechanism for structure-preserving edits. Similar to the original SiT-XL/2
training protocols, we set the batch size to 512 using the AdamW optimizer with a base learning rate of 5 × 10−7. The
fine-tuning process proves highly efficient, yielding the final model in approximately 4 hours.

C.4 Text-Guided Image Editing—SD3-M Transformer Finetuning

We fine-tune SSB on the SD3-M text–image flow transformer [52], which employs a 16-channel KL-VAE and preserves
the original 1024 × 1024 resolution (yielding a 128 × 128 × 16 latent). Following the rectified-flow setup, we set b = 1
so the endpoint distribution aligns with the Gaussian prior, giving zT = y ∼ N (Eϕ(x0), I). Unlike the SiT setting,
directly using the full top-16 DINOv3 PCA channels as endpoints severely degrades finetuning stability on SD3-M.
We attribute this to the model’s more complex flow trajectories and the larger mismatch between its 16-channel VAE
latent space and the DINOv3 embedding space. As a result, endpoints formed solely from DINO PCA features lie
outside the native SD3-M latent manifold and destabilize SSB training. To address this mismatch, we construct a hybrid
endpoint using channel-wise mixing between the two latent encoders. Specifically, we sample a binary channel mask
m ∈ {0, 1}C and define

ϕ(x0) = m ⊙ Eφ(x0) + (1 − m) ⊙ Eϕ(x0),
where Eφ is the SD3-M VAE encoder and Eϕ denotes the DINOv3 PCA features. This mechanism randomly replaces
a subset of DINO PCA channels with the corresponding VAE latent channels, thereby anchoring the endpoint closer
to the native SD3-M latent space while preserving the geometry-aware structure captured by DINO. We find this

E′

25

A PREPRINT -

Figure 11: Full trade-off curves complementing the quantitative results shown in the main paper Fig 6. As in the
main text, we plot text alignment (CLIP-T, x-axis) against multiple fidelity and perceptual metrics (y-axis), including
LPIPS, DINO, PSNR, SSIM, and CLIP-I. These expanded plots show the complete range of guidance settings used
for each method. Consistent with the main-paper results, our method provides a favorable balance between text
adherence and structural preservation, comparing competitively with—and in many cases outperforming—state-of-the-
art baselines such as FlowEdit. In addition, all methods exhibit smooth and monotonic behavior as guidance strength
varies, confirming the stability of our evaluation protocol. ControlNet (Canny) performs substantially worse due to its
edge-only conditioning, which discards most semantic and photometric information. Structural fidelity for scene-style
editing metrics (SSIM, PSNR, LPIPS) are computed on the luminance (YCbCr) channel to better reflect perceptual
structure consistency. Connected markers correspond to different hyperparameter values for each baseline.

hybrid construction essential for stable and high-quality SSB adaptation on SD3-M. See additional discussions and
experimental results in the next Section. Similar to SiT finetuning, in SD3-M, we also inject conditioning into several
transformer blocks using standard zero-initialized residual adapters. This allows the SSB signal to influence deeper
layers while preserving pretrained behavior at initialization. For SD3-M fine-tuning, we use a batch size of 256 and
the AdamW optimizer with a learning rate of 1.2 × 10−4. Similar to the SiT fine-tuning stage, convergence is rapid,
requiring approximately 8 hours to obtain the final model.

C.5 Parameter Settings of SSB at Inference

In Table 5, we summarize the hyperparameter settings used for MRI→CT, horse→zebra, text-guided scene editing,
and object-editing experiments. As noted earlier, the MRI→CT task does not require an inversion step, since our
pre-trained DINOv2 encoder already provides a strong geometry-preserving representation of the source MRI, making
direct conditioning sufficient. For scene-level image editing, we prioritize more dramatic global appearance changes. In
this setting, model inversion offers limited benefit because the structural prior encoded by Eϕ already retains rich spatial
information while largely disentangling appearance. This enables effective scene translation without relying heavily
on inversion, leading to diverse scene level translation, whereas object-level editing—where fine-grained control is
needed—may benefit more from it. Following the hyperparameter configurations detailed in Table 5, we present our
practical reverse ODE sampler based on Euler discretizations in Algorithm 1.

C.6 Baseline Methods Implementations

In this section, we briefly introduce each baseline methods used in the experiments and their implementation details.

GAN-based Methods. We include classical GAN-based I2I baselines—CycleGAN [15], UNIT [16], and CUT [1]—to
provide a complete comparison. CycleGAN and UNIT are trained on our MRI→CT data, while CUT is evaluated
on the official horse→zebra model and fine-tuned for apple→orange. Because these architectures have much smaller
capacity than modern diffusion or flow-based models, we construct a fair unpaired MRI/CT setting by sampling 5k MRI
and 5k CT images from SynthRAD2023/2025 and another 5k from the remaining data, ensuring comparable training
conditions without domain coupling.

Supervised Diffusion Bridge-based Methods. Since the MRI-CT dataset used in this paper is originally released for
supervised training design, with pre-registered MRI-CT pairs, we compare our methods with

26

A PREPRINT -

Table 4: Control strengths for inversion/editing baselines (unified across each method’s original notation). For SDEdit,
FlowEdit, and DDIB, a larger strength corresponds to weaker preservation of the source structure, whereas for
ControlNet the relationship is reversed.

Methods

T steps

SDEdit [26]
DDIB [27]
FlowEdit [55]
ControlNet [13]
iRFDS [98]

75
200
50
75
1000, 1400, 1800

tend (Control Strength*)
0.2, 0.3, 0.4, 0.5, 0.6, 0.75
0.7, 0.8, 0.9
0.5, 0.56, 0.6, 0.66, 0.7, 0.8
0.2, 0.5, 0.7, 0.8, 0.9
–

navg
1
1
1, 3, 5
1
1

CFG @ source

CFG @ target

–
1, 3.5
1, 3.5
–
–

7.5, 13.5, 16.5
13.5, 16.5, 19.5
13.5, 16.5, 19.5
7.5, 13.5, 16.5
official hyperparameters

Table 5: SSB hyperparameter configurations across different I2I translation tasks (medical and natural images).
We utilize standard diffusion hyperparameters (e.g., inversion steps, control strength) consistent with prior I2I
methods, adapting settings like stochasticity (pw) and guidance strength solely based on the task’s intrinsic ambiguity.
This demonstrates that SSB is a general framework achieving strong performance across domains using established
configurations, without relying on complex, task-specific heuristics.

Task

Setting

Inversion Step T steps

tend (Control Strength*) pw in Eq. (26) CFG @ target

Medical I2I

MRI→CT

Text-free I2I

Text-guided I2I

Horse→Zebra
Apple→Orange

Scene editing
Object editing

–

100
100

–
75

50

50
50

75
75

–

0.2, 0.4, 0.7
0.2, 0.4, 0.7

0.4, 0.7, 1
0, 0.2, 0.4, 0.7

–

0.5, 0.7
0.5, 0.7

–

3, 4.5
3, 4.5

0, 0.2, 0.5, 0.7
0, 0.2, 0.5, 0.7

4.5, 7.5, 13.5
4.5, 7.5, 13.5

Cycle-consistency Diffusion-based Methods. We compare against SynDiff [2], a recent cycle-consistent diffu-
sion–GAN method tailored for MRI→CT translation. SynDiff is designed for pre-registered MRI/CT pairs (e.g.,
SynthRAD2023/2025), where MRI and CT share similar resolution and alignment, and it trains a separate model for
each contrast (e.g., T1→CT, T2→CT). However, SynDiff is highly sensitive to domain coupling and does not train
reliably on large-scale, heterogeneous, or unregistered data such as UKBB MRI→CT. In practice, it collapses on
our full dataset, making a direct large-scale comparison infeasible and uninformative. To provide a fair comparison
under its intended operating regime, we therefore construct a smaller, higher-quality unpaired subset by combining
SynthRAD2023/2025 with an additional 10,000 unpaired MRI/CT images from the remaining training pool.
CycleNet [25] has demonstrated strong performance on classical unpaired I2I benchmarks such as horse→zebra
and apple→orange by fine-tuning pretrained text-to-image diffusion models (e.g., Stable Diffusion 2.1 [69]). For
comparison, we use the officially released CycleNet model pretrained on horse→zebra and further fine-tune it on the
apple→orange dataset [15] following its recommended protocol. This provides a comprehensive baseline representing
the line of diffusion-based cycle-consistency methods. We follow the official implementations of the parameter settings
for CycleNet.

Inversion-based Flow Methods. For natural image I2I, we evaluate inversion-based baselines including DDIB [27]
and SDEdit [26]. Both methods rely on a pretrained flow model and do not learn cross-domain mappings; accordingly,
we use the same official SiT model as their backbone, identical to ours. For text-guided I2I editing, we use the official
SD3-M model as the pretrained backbone for fair comparison. We additionally compare with FlowEdit [55], a recent
SOTA, inversion-free editing method specifically designed for rectified-flow models, providing a strong baseline for
text-guided structural transformations. In Table 4, we show the hyperparameter settings used in this work for each
above baselines.

ControlNet. For all exaperiments used related to ControlNet, we adopt the pre-trained released in SD3-Controlnet-
Canny [105] on huggingface.

C.7 Kernel Alignment Matrices

CKNNA. Centered Kernel Nearest-Neighbor Alignment (CKNNA) is a relaxed variant of the standard Centered
Kernel Alignment (CKA) [106], designed to soften CKA’s strict global alignment criterion by focusing only on locally
consistent neighborhoods. We adopt the notation and formulation introduced in Huh et al. [88]. We follow [61] for the
metrics implementation in Fig.3 in the main paper. In this paper, we randomly sample 1,000 paired MRI–CT slices
from the training set and report CKNNA@k = 10 computed on this subset and averaged across all layers of each image
encoder; following Huh et al. [88], smaller k provides a more stringent assessment of local neighborhood consistency
between representations.

27

Table 6: Efficiency Comparison on medical MRI→CT 256 × 256 images. All methods use a UNet backbone. We
evaluate SDEdit, DDIB, and DDBM using our own pre-trained diffusion models within their official frameworks. For
Syndiff, SelfDRB, and I2SB, we utilize their official implementations.

A PREPRINT -

Backbone: UNet-based [108, 93]

Methods

NFE↓ Params

Inference Time (s/image)↓

SDEdit [26]
DDIB [65]
Syndiff [2]
DDBM [35]
I2SB [37]
SelfDRB [41]
SSB (ours)

400
1000
4
300
1000
20
150

552.8M
552.8M
156.1M
489.7M
552.8M
46.9M
489.7M

8.03
20.15
0.13
5.21
24.97
0.29
2.57

Table 7: Efficiency Comparison on text-free I2I translation with Transformer-based diffusion denoisers. Methods
differ in base architectures (e.g., SiT/SD/SD3 backbones), reflecting the recent shift from U-Net to Transformer
denoisers. We use the official implementation of CycleNet and adapt publicly available weights for ControlNet-Canny.

Methods

Base-model

NFE↓

Inference Time (s/image) ↓ Parameters

SDEdit [26]
DDIB [27]
CycleNet [25]
ControlNet [13]
SSB (Ours)

SiT-XL/2 [61]
SiT-XL/2 [61]
SD-2.1 [69]
SD3-M [52]
SiT-XL/2 [61]

400
1000
200
150
250

9.22
22.06
5.21
5.75
6.56

675M
675M
865.91M
2.02B
675M

D Additional Details to Related Work

Connecting Conditional SI, ECSI, and DDBM. Beyond the vector field interpolation, one can also formulate
conditional sampling from the SDE perspective in practice. As is well known [21, 107, 26, 29], SDEs exhibit robustness
to initialization, with stability proportional to the variance of the additive noise. In contrast, ODE trajectories initialized
with corrupted or imperfect samples propagate these errors deterministically through the velocity field. By injecting
stochasticity at every step, SDEs behave as Markov processes that converge toward the intended invariant distribution,
thereby reducing sensitivity to initialization and improving the reliability of conditional generation.

We define the reverse SDE that shares the conditional marginals p(zt | zT ) with the PF-ODE as

dzt = (cid:2)v(t, zt, zT ) − gts(t, zt, zT )(cid:3)dt + (cid:112)2gtdwt,

(27)

where gt > 0 is the diffusion coefficient, s(t, zt, zT ) = ∇z log p(zt | zT ) is the conditional score, and wt is a standard
Wiener process. Setting gt ≡ 0 recovers the deterministic PF-ODE. Conditioned on the endpoint and using Tweedie’s
identity, the score is

s(t, z, zT ) := ∇z log p(z | zT ) = − 1
γt

E[ϵ | zt = z, zT ].

(28)

By combining this score identity with the stochastic interpolant parameterization, we obtain a closed-form correspon-
dence between score and velocity as

v(t, z, zT ) = E[ ∂tI(t, z0, zT ) | zt = z, zT ]
− ˙γtγt s(t, z, zT ).

(29)

We now show that the velocity–score reverse SDE for SI [39] coincides with the ESCI [36] formulation, which in turn is
equivalent to diffusion bridge models (DDBM) [35] under linear gaussian defined in the main paper and restated below.

For the linear–Gaussian SI kernel defined in the main paper, we have the form of

zt = αtz0 + βtzT + γtϵ,

ϵ ∼ N (0, I),

applying Eq. (29) the probability flow drift is

v(t, z, zT ) = ˙αtˆz0 + ˙βtzT − ˙γtγt sc(t, z, zT ),

(30)

(31)

28

A PREPRINT -

where we have ˆz0 = E[z0|zt, zT ] and s(t, z, zT ) = ∇z log ρt(z | zT ). By Tweedie’s identity [39] defined as

sc(t, z, zT ) = − 1
γt

ˆut,

ˆut = E[ϵ | zt, zT ].

we directly plug-in Eq. (32) to Eq.(27), this lead to

(cid:16)

dzt =

˙αtˆz0 + ˙βtzT + ( ˙γt + gt
γt

)ˆut

(cid:17)

dt + (cid:112)2gt d ¯wt,

(32)

(33)

which is exactly the ECSI reverse SDE drift form, see Sec. D in the supplement in [36]. Note that, the above conduct
establish that conditional SI, ECSI, and DDBM describe the same endpoint-conditioned marginals. The difference
lies only in training targets: ECSI/DDBM typically predict the posterior mean ˆz0, while FM-based models predict the
velocity v.

E Additional Experimental Results

In this section, we show additional experimental results that obmitted from the main paper due to limited spacing.

In Fig. 9, we present qualitative results for the horse→zebra and
Natural Image translation Visual Results.
apple→orange tasks. Overall, our method produces the most structurally consistent translations compared with existing
approaches on these two classical I2I benchmarks.

Additional Quantitative Results. In Fig. 11, we provide additional quantitative evaluations (e.g., SSIM, CLIP-I,
and CLIP-T) that were obmitted from the main paper. These figures show the full trade-off curves corresponding to
Fig. 6 of the main text. As before, we plot text alignment (CLIP-T, x-axis) against a variety of fidelity and perceptual
metrics (y-axis), including LPIPS, DINO, PSNR, SSIM, and CLIP-I. The expanded plots reveal the complete range
of guidance settings used for each method. Consistent with the main-paper results, our method achieves a highly
favorable balance between text adherence and structural preservation, performing competitively with—and often
surpassing—state-of-the-art approaches such as FlowEdit.

We additionally compare the efficiency of all baselines across different tasks in Table 6, Table 7, and Table 8. Rather
than relying solely on Number of Function Evaluations (NFE), we measure the actual inference time on a single
NVIDIA GeForce H200 GPU with a batch size of 1. The results demonstrate that our method achieves inference speeds
overall comparable to strong prior baselines. For text-guided I2I tasks (Table 8), where NFE varies significantly due to
hyperparameter settings (e.g., control strength), we omit a single NFE value and instead report the average running time
across different settings to reflect the actual computational cost more accurately.

Additional Qualitative Results. In Fig. 12, Fig. 13, and Fig. 14, we show additional visual results for in-domain and
OOD MRI→CT translation. Fig. 15 presents additional scene-level editing results using SSB, while Fig. 16 visualize
additional object level editing results.

F Limitations and Future Works

Medical Images. While SSB demonstrates strong 2D fully self-supervised MRI→CT translation, a current limitation
is that our medical pipeline operates on slice-wise inputs. This choice is primarily motivated by the substantial
computational cost of training full 3D vision encoders and 3D bridge models at clinical resolution. As with all 2D
diffusion and translation frameworks, this may introduce mild inter-slice inconsistencies due to the absence of explicit
through-plane anatomical modeling. Nevertheless, all baselines are evaluated under the same 2D setting, ensuring

Table 8: Efficiency Comparison on text-guided I2I translation and editing. We employ pre-trained public available
SD3-M for all methods. Since the NFE varies due to hyperparameter settings (e.g., control strength) for all baselines,
we omit a single NFE value and instead report the average running time across different settings to reflect the actual
computational cost more accurately.

Methods

Inference Time (s/image) ↓ Parameters

SDEdit [26]
DDIB [27]
FlowEdit [55]
iRFDS [98]
ControlNet [13]
SSB (Ours)

4.71
28.75
12.21
44.82
5.85
14.13

29

2.02B

A PREPRINT -

fair comparison. Extending SSB to 3D—either via slice-regularization techniques or fully 3D encoders and bridge
architectures—remains an important direction for future work.

General Domains Image Translation and Editing. Our method inherits two fundamental limitations stemming from
its use of a strong external structure prior and from the nature of cross-domain representation alignment. First, the
reliance on a geometry-preserving prior means that the model is highly effective for appearance-level edits—such as
color, style, or seasonal changes—where the underlying object shape and scene layout should remain stable. However,
this same inductive bias makes the method less suitable for large semantic transformations that require substantial
changes in global geometry or object morphology. As shown in Fig. 17 (Top), when the target prompt demands drastic
reinterpretation (e.g., transforming a small lizard into a large dragon), the model exhibits an inherent trade-off: stronger
edits relax the structure prior but may also distort the background, alter object pose, or disrupt the scene layout, while
preserving the structure suppresses the desired transformation.

Second, our approach struggles when the source and target domains have extremely large representation gaps. Domains
such as silhouettes, segmentation masks, sketches, or other abstract inputs lack the rich texture and depth characteristics
present in natural images. Since the structure prior was trained on natural imagery, it establishes meaningful correspon-
dences only when both domains share compatible geometric information. When applied to highly stylized or symbolic
inputs with no clear semantic grounding, the model may fail to generate realistic outputs or may collapse into painterly
or stylized renderings instead of photorealistic results as shown in Fig. 17 (bottom), silhouettes to several RGB image
style translations.

These limitations highlight directions for future work, including developing adaptive structure priors that modulate
geometric constraints based on target semantics, or designing cross-domain regularizers and multi-modal pretraining
strategies that bridge the gap between abstract representations and natural images, enabling more robust abstract-to-photo
translation.

30

A PREPRINT -

Figure 12: Additional Visual Results. Additional qualitative results for in-domain MRI → CT translation.

31

A PREPRINT -

Figure 13: Additional Visual Results. Additional qualitative results for out-of-distribution (OOD) MRI (UKBB
whole-body water-series) → CT translation. Segmentation masks are overlaid on the MRI source images in OOD
settings only to provide visual structural reference, since paired CT ground truth is unavailable. These masks are
not used during training or inference; they serve solely to illustrate anatomical fidelity without any segmentation
supervision.

32

A PREPRINT -

Figure 14: Additional Visual Results. Additional qualitative results for out-of-distribution (OOD) MRI (UKBB
whole-body fat-series) → CT translation. Segmentation masks are overlaid on the MRI source images in OOD settings
only to provide visual structural reference, since paired CT ground truth is unavailable. These masks are not used
during training or inference; they serve solely to illustrate anatomical fidelity without any segmentation supervision.

33

A PREPRINT -

Figure 15: Additional Visualization Results. We present additional qualitative results for text-guided image scene
editing using SD3-M [52]. Baseline methods are shown with numerical values in parentheses (∗), indicating the control
strengths defined in their original formulations. For ControlNet, we use Canny edges as the conditioning signal. For
example, FlowEdit (33) denotes a control parameter of nmax = 33, while ControlNet (0.7) refers to a conditioning
scale of α = 0.7, where larger values correspond to stronger Canny-based structural guidance.

34

A PREPRINT -

Figure 16: Additional Visualization Results. We present additional qualitative results for text-guided object editing
using SD3-M [52]. Baseline methods are shown with numerical values in parentheses (∗), indicating the control
strengths defined in their original formulations. For ControlNet, we use Canny edges as the conditioning signal. For
example, FlowEdit (33) denotes a control parameter of nmax = 33, while ControlNet (0.7) refers to a conditioning
scale of α = 0.7, where larger values correspond to stronger Canny-based structural guidance.

35

A PREPRINT -

Figure 17: Limitations in Natural Image Editing and Translation. Top: For large semantic changes (small lizard →
large dragon), SSB trades off between preserving the source structure and fully following the target prompt, and strong
edits can distort background geometry. Bottom: With highly abstract guidance such as silhouettes, the model tends to
produce stylized, painterly animals rather than fully photorealistic translations.

36

