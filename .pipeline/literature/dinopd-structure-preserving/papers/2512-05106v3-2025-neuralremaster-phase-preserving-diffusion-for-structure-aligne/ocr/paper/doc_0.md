6
2
0
2

r
a

M
5

]

V
C
.
s
c
[

3
v
6
0
1
5
0
.
2
1
5
2
:
v
i
X
r
a

NeuralRemaster: Phase-Preserving Diffusion for Structure-Aligned Generation

Yu Zeng1, Charles Ochoa1, Mingyuan Zhou2, Vishal M. Patel3,
Vitor Guizilini1, and Rowan McAllister1

1 Toyota Research Institute
2 University of Texas, Austin
3 Johns Hopkins University

Input e

FLUX-Kontext

QWen-Edit

Ours

Input

Cosmos-Transfer 2.5

Ours

Ours

Fig. 1: We present Phase-Preserving Diffusion (ϕ-PD), a model-agnostic reformula-
tion of the diffusion process that preserves an image’s phase while randomizing its
magnitude, enabling structure-aligned generation with no architectural changes or
additional parameters.

Abstract. Standard diffusion corrupts data using Gaussian noise whose
Fourier coefficients have random magnitudes and random phases. While
effective for unconditional or text-to-image generation, corrupting phase
components destroys spatial structure, making it ill-suited for tasks re-
quiring geometric consistency, such as re-rendering, simulation enhance-
ment, and image-to-image translation. We introduce Phase-Preserving
Diffusion (ϕ-PD), a model-agnostic reformulation of the diffusion pro-
cess that preserves input phase while randomizing magnitude, enabling
structure-aligned generation without architectural changes or additional

 
 
 
 
 
 
2

Y. Zeng et al.

parameters. We further propose Frequency-Selective Structured (FSS)
noise, which provides continuous control over structural rigidity via a sin-
gle frequency-cutoff parameter. ϕ-PD adds no inference-time cost and is
compatible with any diffusion model for images or videos. Across photo-
realistic and stylized re-rendering, as well as sim-to-real enhancement for
driving planners, ϕ-PD produces controllable, spatially aligned results.
When applied to the CARLA simulator, ϕ-PD significantly improves
sim-to-real planner transfer performance. The method is complementary
to existing conditioning approaches and broadly applicable to image-to-
image and video-to-video generation. Videos, additional examples, and
code are available on our project page.

1

Introduction

Recent advances in diffusion models have revolutionized image generation, achiev-
ing high-fidelity results for unconditional or text-conditioned synthesis. Yet many
practical applications do not require generating a scene from scratch. Instead,
they operate within an image-to-image setting where the spatial layout, such
as object boundaries, geometry and scene structures, should remain fixed while
the appearance is modified. Examples include neural rendering, stylization, and
sim-to-real transfer for autonomous driving or robotics simulation. We refer to
this broad class of problems as structure-aligned generation.

Although these tasks are conceptually easier than generating from scratch,
existing solutions are unnecessarily complex. Methods such as ControlNet [43],
T2I-Adapter [21], and related variants attach auxiliary branches to inject struc-
tural input into the model. While effective, this introduces additional parame-
ters and computational cost, paradoxically making structure-aligned generation
harder than it should be.

We argue that this inefficiency stems not from the network architecture, but
from the diffusion process itself. The forward diffusion process injects Gaus-
sian noise, which destroys both the magnitude and phase components in the
frequency domain. Classical signal processing [23, 31, 38], however, tells us that
phase encodes structure while magnitude encodes texture. Destroying the phase
means destroying the very spatial coherence that structure-aligned generation
depends on, forcing the model to reconstruct structure from scratch.

Motivated by this insight, we propose Phase-Preserving Diffusion (ϕ-PD). In-
stead of corrupting data with Gaussian noise, ϕ-PD constructs structured noise
whose magnitude matches that of Gaussian noise while preserving the input
phase. This naturally maintains spatial alignment throughout sampling (Fig-
ure 1) with no architectural modification, no extra parameters (Figure 2),
and is compatible with any DDPM or flow-matching model for images or videos.
To provide controllable levels of structural rigidity, we further introduce
Frequency-Selective Structured (FSS) noise, which interpolates between
input phase and pure Gaussian noise via a single cutoff parameter (Figure 4).
This allows us to control the trade-off between strict alignment and creative
flexibility.

Phase-Preserving Diffusion

3

Prior methods encode structural
input with additional modules (yel-
low), that depend on the model
(green) and incur additional com-
putation.

Our method incurs no ad-
ditional overhead and works
with any model (green).

Our method extends to
video.

Fig. 2: Unlike prior approaches that modify architectures and add overhead, ϕ-PD
preserves structure via phase consistency, remaining lightweight and model-agnostic,
reflecting that image-conditioned generation should be simpler, not harder.

We evaluate ϕ-PD across photorealistic re-rendering, stylized re-rendering
and simulation enhancement for embodied-AI agents. ϕ-PD consistently main-
tains geometry alignment while producing high-quality visual outputs, outper-
forming prior methods across both quantitative and qualitative metrics. When
used to enhance CARLA simulations, ϕ-PD improves planner transfer to the
Waymo Open Dataset by 50%, substantially narrowing the sim-to-real gap. In
summary, our contributions include:

– Phase-preserving diffusion process: A diffusion process that preserves
phase while randomizing magnitude in frequency domain, maintaining spa-
tial structure without architectural changes.

– Frequency-selective structured noise: A single-parameter mechanism

that enables continuous control over structural alignment rigidity.

– Unified and efficient framework applicable to both images and videos,
compatible with DDPMs and flow-matching, and requires no inference-time
overhead.

2 Related Work

Diffusion Models. Diffusion models have become a dominant paradigm for
generative modeling, capable of representing complex data distributions with
remarkable fidelity [11, 16]. They progressively corrupt data into Gaussian noise
through a forward diffusion process, then learn to invert this process via iterative
denoising. This framework has demonstrated state-of-the-art performance across
diverse domains, including image, video, and audio generation [3,4,12,17,25,29,
32], as well as reinforcement learning [14, 26, 37] and robotics [1, 5, 36].

Phase Preserving (PP) NoiseGaussian NoiseFrame-wise PP noise4

Y. Zeng et al.

Frequency-Domain Manipulation for Diffusion. Recent work has explored
frequency domain operations for diffusion models. [7] argues that diffusion mod-
els of images perform approximate autoregression in the frequency domain. [9]
shows that the standard diffusion forward process corrupts high-frequency com-
ponents faster than low-frequency ones, and introduces an alternate process that
corrupts all frequencies at the same rate. [41] shows that modifying the UNet
frequency domain features significantly improves the generating quality for im-
age or video generation. FreeDiff [39] introduces a fine-tuning free approach for
image editing that employs progressive frequency truncation to refine the guid-
ance of diffusion models. [45] proposes a training-free style transfer method that
modulates the intermediate samples with the image phase. [27] proposed to use
a frequency-dependent moving average during sampling. [2] proposes a training-
free approach for image inpainting that optimizes the initial seed noise in the
spectral domain.

Structure-Aligned Generation with Diffusion. Most existing methods achieve
structure-aligned generation by modifying the network architecture and intro-
ducing additional adaptation components. ControlNet [43] copies the entire U-
Net encoder into a trainable encoder branch, which adds significant computation
overhead. T2I-Adapter [21] reduces computation overhead using a lightweight
adapter module but sacrifices control precision. Uni-ControlNet [46] enables si-
multaneous utilization of multiple local controls by training two adapters. Omni-
Control [33] integrates image conditions into Diffusion Transformer (DiT) archi-
tectures with only 0.1% additional parameters by re-using the VAE and trans-
former blocks of the base model. ControlNeXt [24] uses a lightweight convolu-
tional module to inject control signals, and directly finetune selective parame-
ters of the base model to reduce training costs and latency increase. SCEdit [15]
proposes an efficient finetuning framework that edits skip connections using a
lightweight module. NanoControl [13] aims to achieve efficient control with a
LoRA-style control module. The above methods all rely on an additional mod-
ule to incorporate the control signal, though some are more lightweight than
others. CosmosTransfer [22] achieves multi-modal control by combining mul-
tiple ControlNet branches, demonstrating promising applications on physical
tasks; however, multiple branches introduce significant computation overhead.
In contrast, ϕ-PD does not introduce any computation overhead or additional
parameters while enabling universal spatial control.

Training-Free Guidance Methods. Recently, several training-free meth-
ods have been developed. [42] introduced FreeDoM, which leverages off-the-shelf
pre-trained networks to construct time-independent energy functions that guide
generation. [6] proposed ZestGuide for zero-shot spatial layout conditioning, uti-
lizing implicit segmentation maps extracted from cross-attention layers to align
generation with input masks. [20] presented FreeControl, a training-free ap-
proach that enforces structure guidance with the base model feature extracted
from the control signal. Although these methods avoid training cost, they intro-
duce additional overhead at test time, either an external model, DDIM inversion,

or multiple inferences of the base model. In contrast, ϕ-PD achieves spatial con-
trol without any additional inference time overhead.

Phase-Preserving Diffusion

5

3 Method

3.1 Frequency Domain Fundamentals

In the frequency domain, any image I(x, y) can be represented through the 2D
Fourier transform:

F (u, v) = F{I(x, y)} =

W −1
(cid:88)

H−1
(cid:88)

x=0

y=0

I(x, y)e−2πj(ux/W +vy/H),

(1)

where F (u, v) is a complex-valued function that can be decomposed into mag-
nitude and phase components:

F (u, v) = |F (u, v)| · ejϕ(u,v) = A(u, v) · ejϕ(u,v).

(2)

Here, A(u, v) = |F (u, v)| is the magnitude spectrum and ϕ(u, v) the phase spec-
trum. The inverse Fourier transform uses magnitude and phase to reconstruct
the original image:

I(x, y) = F −1{F (u, v)} =

W −1
(cid:88)

H−1
(cid:88)

u=0

v=0

F (u, v)e2πj(ux/W +vy/H).

(3)

Phase-Magnitude Separation in
Signal Processing. Foundational
work by Oppenheim [23] shows that
phase primarily determines spatial
structure, while magnitude largely
controls
texture statistics. Mixing
phase and magnitude from differ-
ent images produces reconstructions
the
whose
source of the phase, not magnitude
(see Figure 3). This observation mo-
tivates our approach: if diffusion de-
stroys phase, it destroys spatial geom-
etry; if we preserve phase, we preserve
structure.

follows

spatial

layout

Fig. 3: Mixing phase and magnitude from
two images. The mixture keeps the struc-
ture of the image where the phase is taken.

3.2 Phase-Preserving Diffusion

Standard diffusion corrupts data using Gaussian noise whose Fourier coefficients
have random magnitudes and random phases. As a result, even early diffusion

ImagesPhase Magnitude Car PhaseDog Magnitude6

Y. Zeng et al.

steps erase spatial alignment. We propose a simple alternative: preserve the in-
put image’s phase and randomize the magnitude, by using structured noise that
shares the input phase.
Structured Noise Construction. Given an input image I, we compute its
Fourier transform:

FI = AI · ejϕI .

(4)

We construct phase-preserving noise by pairing the input image phase with
a random magnitude:

and invert it:

Fˆϵ = Aϵ · ejϕI ,

ˆϵ = F −1{Fˆϵ},

(5)

(6)

where the random magnitude Aϵ can be obtained from the Fourier transform of
Gaussian noise:

Aϵ = |F{ϵ}|,

ϵ ∼ N (0, I).

Alternatively, Aϵ can be sampled from a scaled Rayleigh distribution [10]:

√

Aϵ =

−N ln U , U ∼ Uniform(0, 1),

(7)

(8)

where N = H × W , matching the magnitude statistics of the DFT of unit-
variance Gaussian noise.

This structured noise is used in place of Gaussian noise in forward diffusion
for training. It injects randomness while maintaining the phase of the input.
At test time, we achieve structure-aligned generation by starting sampling from
structured noise constructed with input image phase.
Frequency Selective Structured (FSS) Noise. In practice, we often want to
control to what extent we keep the structure from the input image. Some tasks
require strict structure preservation, while others benefit from partial freedom to
reinterpret the scene. To provide this control, we introduce Frequency Selective
Structured (FSS) noise, which only keep the image within a radius r and use
the phase from the noise for the remainder. We define a smooth frequency mask
M (u, v) based on the cutoff radius r:

M (u, v) =

(cid:40)1

exp

(cid:16)

√

− (

u2+v2−r)2

(cid:17)

2σ2

√

√

if

if

u2 + v2 ≤ r

u2 + v2 > r

(9)

where (u, v) are centered frequency coordinates with the DC component at the
origin, and r is specified in pixels relative to the frequency grid of size W × H.
In practice, we use fftshift to center the spectrum before applying the mask.
The FSS noise ˆϵ is the combination of image phase and noise phase using the

mask:

Fˆϵ = Aϵ · ejϕI ⊙M +jϕϵ⊙(1−M ),

(10)

Phase-Preserving Diffusion

7

Fig. 4: Frequency Selective Structured (FSS) Noise with increasing cutoff radius r.

Input

r = 1

r = 6

r = 10

r = 20

r = 30

Fig. 5: Image generated with the same noise and different cutoff radius r. Results are
based on SD1.5.

where ⊙ represents element-wise multiplication. We can sample phase ϕϵ as the
Fourier transform of Gaussian noise:

ϵ ∼ N (0, 1),

ϕϵ = arg(F{ϵ}) ∼ Uniform(−π, π).

(11)

When the mask is all zero, we take a random phase for all frequencies, then
this FFS noise becomes Gaussian noise and ϕ-PD becomes standard diffusion.
Figure 4 visualizes FSS noise with different cutoff radius r. We can see that the
noise becomes increasingly more structured with increasing r. Figure 5 shows
images generated from the same input with different cutoff radius where the
generated image aligns more tightly to the input with larger r.

3.3 Training Objective

ϕ-PD does not depend on model architecture or diffusion formulation. In the
experiment section, we demonstrate integration both DDPM and Flow Matching,
without modifying their architectures of loss functions.

The flow matching objective learns a vector field that transports the struc-
tured noise distribution to the target image distribution. During training, given
a target image I and a structured noise ˆϵ, and a timestep t ∈ [0, 1], an interme-
diate image xt is obtained using a linear combination between I and ˆϵ following
Rectified Flows [19]:

The ground-truth velocity is

xt = t ˆϵ + (1 − t) I.

vt =

dxt
dt

= ˆϵ − I.

(12)

(13)

8

Y. Zeng et al.

With this ground-truth, we can then train the model by minimizing the mean
squared error between the model output and the ground-truth:

L = EI,ˆϵ,t∥u(xt, t; θ) − vt∥2
2.

(14)

In the Fourier domain, when using full phase-preservation, the velocity becomes

Fvt = (Aˆϵ − AI )ejϕI

(15)

which has phase ϕI or ϕI + π, preserving structural alignment. With FSS noise,
phase is preserved within the cutoff radius r, providing controllable alignment
strength.

In DDPMs [11], data x0 is gradually corrupted by Gaussian noise:

q(xt | xt−1) = N ((cid:112)1 − βt xt−1, βtI).

(16)

The model learns the reverse process by predicting the added noise ϵ at each
step using the loss

LDDPM = Ex0,ϵ,t

(cid:2)∥ϵ − ϵθ(xt, t)∥2

2

(cid:3) ,

(17)

where xt =

√

¯αtx0 +

√

1 − ¯αtϵ and ¯αt = (cid:81)t

s=1(1 − βs).

In our formulation, we replace the Gaussian noise ϵ with structured noise ˆϵ

that preserves the input phase:

√

xt =

¯αt x0 +

√

1 − ¯αt ˆϵ.

The training objective follows the same form,

Lϕ-PD = Ex0,ˆϵ,t

(cid:2)∥ˆϵ − ϵθ(xt, t)∥2

2

(cid:3) ,

(18)

(19)

but the expectation is now over structured noise ˆϵ rather than Gaussian noise ϵ.
Although ˆϵ is non-Gaussian and formally violates standard DDPM assumptions,
we observe high-quality, structure-aligned outputs using standard samplers after
finetuning with this objective.

3.4 Extension to Videos

ϕ-PD extends to video by constructing phase-preserving noise frame-by-frame.
For a video {I1, I2, ..., IT }, we construct structured noise for each frame and
concatenate along the time dimension. Since current video diffusion models pro-
duce lower-fidelity individual frames than image models, we adopt a two-stage
pipeline: generating the initial frame with image-based ϕ-PD, then extending
temporally with first-frame-conditioned video ϕ-PD. No architectural changes
are required.

Phase-Preserving Diffusion

9

Input

QWen-Edit

FLUX.1 Kontext

Ours

Fig. 6: Results on UnrealCV compared to FLUX-Kontext and QWenEdit.

4 Experiments

We evaluate ϕ-PD across three settings: photorealistic re-rendering, stylized
re-rendering, and simulation enhancement for autonomous driving, comparing
against state-of-the-art methods. To demonstrate its broad applicability, we im-
plement ϕ-PD on three representative diffusion models: SD 1.5, FLUX-dev, and
WAN 2.2 14B, which vary in size, formulation, and modality, covering both image
and video generation. Please refer to the supplementary materials for additional
results and code.

During training, we use real images: construct structured noise from the im-
age’s phase, corrupt the image, and train the model to denoise. At inference, ϕ-
PD supports two inference modes. (1) Structured SDEdit: add structured noise
to the source image at some noise level t, then denoise. (2) From noise: de-
noise from pure structured noise without any pixel contribution from the source;
structure is encoded entirely in the phase. Mode (1) generally yields stronger
structural alignment; mode (2) offers more flexibility with larger appearance
changes.

4.1

Implementation Details

Datasets UnrealCV4 is a open-source tool that includes multiple assets. We
created a diverse test set consisting of around 5,000 images across all available
assets, for a total of around 200 scenes. Figure 6 shows examples from this test
set. This dataset covers a diverse range of scenes, including outdoor and indoor,
city and natural etc, with geometry diversity while lacking photorealism. This
dataset evaluates photorealistic enhancement and structure preservation.

4 https://github.com/unrealcv/unrealcv

10

Y. Zeng et al.

“Pencil Sketch of a Castle. "

“Picture of a Husky"

Input

PNP

SDEdit

ControlNet

Ours

Fig. 7: Stylized re-rendering results.

ImageNetR is a test set proposed by [35], including 29 images of various objects
and styles. While the original dataset provides prompts, these are generic image
editing prompts. Since our work primarily focuses on re-rendering, we keep the
editing prompts with style hints in the original dataset and added additional
style prompts, resulting in a total of 8 prompts for each image. This benchmark
assesses stylized re-rendering and structure preservation.
CARLA is an open-source driving simulator [8]. We collect 5.5 hours of driving
videos from CARLA Town 4 using the simulator’s default autopilot. We then
split the videos into 25 second clips and annotate a caption for each clip. For
simulation enhancement, we use these captions combined with the style hint
“A photorealistic video of driving". We evaluate the effectiveness of sim-to-real
transfer by testing the CARLA-trained planner on Waymo’s WOD-E2E [40]
validation set.

Model architecture We integrate ϕ-PD into: SD 1.5 (image DDPM), FLUX-
dev (image flow matching), and Wan2.2-14B (video flow matching). We either
fully finetune or LoRA-finetune each model using phase-preserving noise; no ar-
chitectural changes are introduced. Notably, this finetuning is highly efficient:
adapting the Wan2.2-14B video model with LoRA required only a single GPU
while still yielding high-quality results, further demonstrating the lightweight
nature of ϕ-PD. To evaluate structure alignment, we compute LPIPS [44] be-
tween the input and output pairs, as well as the error between their depth map
using metrics SSIM (Structural Similarity Index Measure) and ABSREL (Abso-
lute Relative error). To evaluate text prompt alignment, we compute CLIP [28]
similarity between the generated images and the input text prompt.

Training and Inference Details We start from the officially released check-
points of SD 1.5 [30], FLUX-dev [18], and Wan2.2-14B [34], and finetune each

Table 1: Quantitative evaluation results for photorealistic re-rendering on UnrealCV.

Phase-Preserving Diffusion

11

Prompt Align.

Structure Alignment

Method

CLIP ↑

LPIPS ↓ SSIM ↑ ABSREL ↓

ControlNet-Tile
PNP
FBSDiff
SDEdit
Ours

0.3018
0.2978
0.3255
0.2989
0.3212

0.5831
0.4076
0.3558
0.4540
0.2397

0.6804
0.6993
0.6792
0.6754
0.7481

0.9484
0.8870
0.9362
0.8932
0.5589

model with phase-preserving noise. For SD 1.5, we experiment with both full fine-
tuning and LoRA finetuning, while for FLUX-dev and Wan2.2-14B we use LoRA
finetuning due to computational constraints. At inference time, for Wan2.2-14B
we adopt the 4-step LoRA from LightX2V5 and apply it directly on top of our
finetuned LoRA weights to accelerate sampling. FLUX and SD 1.5-based mod-
els are trained on PhotoConceptBucket dataset6; WAN2.2-14B based models are
trained on OpenSoraPlan dataset7.

We LoRA finetune Wan2.2-14B for 1,200 iterations and FLUX-dev for 10,000
iterations, while SD 1.5 is fully finetuned for 140,000 iterations. Each training
run takes approximately 48 hours on an NVIDIA A100 GPU.

For each training iteration, we sample a cutoff radius r from an exponential
distribution and add a constant offset r0 to ensure a minimum amount of phase
information is always preserved:

r = r0 + r′,

r′ ∼ Exp(λ),

(20)

where λ > 0 is the rate parameter of the exponential distribution and r0 controls
the minimum cutoff. In our experiments, we set λ = 0.1 empirically. We set the
transition bandwidth parameter σ = 2, which controls the smoothness of the
frequency mask M (u, v) around the cutoff radius r.

4.2 Results

Photorealistic Re-Rendering. Quantitative results on UnrealCV re-rendering
are summarized in Table 1, where all methods are implemented using the SD 1.5-
based models for fair quantitative comparison. We evaluate SDEdit with Gaus-
sian noise (SDEdit) and structured noise (Ours) at identical noise levels (t=0.5).
This comparison shows that, compared to Gaussian noise, structured noise con-
sistently improves structure preservation (near 90% improvements on LPIPS)
while maintaining text prompt alignment. Qualitative examples are shown in
Figure 6, which compares our Flux-based model against stronger recent models

5 https://huggingface.co/lightx2v/Wan2.2-Lightning
6 https://huggingface.co/datasets/bghira/photo-concept-bucket
7 https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main

12

Y. Zeng et al.

Table 2: Quantitative evaluation for stylized re-rendering.

Prompt Align.

Structure Alignment

Method

CLIP ↑

LPIPS ↓ SSIM ↑ ABSREL ↓

ControlNet-Tile
PNP
FBSDiff
SDEdit
Ours

0.3022
0.2983
0.3091
0.3011
0.3017

0.5849
0.3029
0.3150
0.3456
0.2759

0.7508
0.7796
0.7562
0.7588
0.7842

0.9085
0.8816
0.9736
0.9148
0.8087

Table 3: Efficiency comparison in extra parameters and FLOPs relative to the base
model as well as wall-clock time.

ControlNet SDEdit PNP FBSDiff Ours

Extra Params
Extra FLOPs
Time (s/image)

+50%
+50%
28.02

+0% +0% +0% +0%
+0% +100% +1100% +0%
133.8 20.22
20.31

20.40

such as FLUX-Kontext, and Qwen-Edit. Across both settings, all methods im-
prove photorealism-as reflected by higher AS scores than the input images while
ϕ-PD achieves the highest photorealism and superior structure alignment. We
observe that QWen-Edit produces visually high-quality results but often fails to
maintain structural alignment with the input image; for example, in the first
four cases, it enlarges the main subjects significantly. FLUX-Kontext aligns bet-
ter with the input structure but provides only limited improvement in visual
quality. Our method achieves both high visual fidelity and consistent structural
alignment across frames.
Stylized Re-rendering. Results of stylized re-rendering are presented in Ta-
ble 2, with representative examples shown in Figure 7. All models are based on
SD 1.5. This task evaluates the model’s ability to alter appearance while pre-
serving scene structure. As shown, ϕ-PD produces visually coherent stylizations
that maintain object boundaries and spatial consistency, while prior methods
often distort geometry or introduce texture misalignment. Quantitatively, ϕ-PD
achieves similar prompt alignment and significantly higher structure alignment.
Simulation Enhancement. For this
experiment, we generate 5.5 hours
of demonstration driving videos from
CARLA using its autopilot. Then we
train an end2end planner on the re-
rendered CARLA videos from each
method using a ResNet backbone
with a GRU to take temporal
in-
put and an MLP head to output a
trajectory ∈ R16×2 (4s predictions
at 4Hz in XY space). As baseline,
we also present the results from a
purely CARLA-trained model. Open-

Fig. 8: Planner error on Waymo validation
set. Lower is better. 1zero-shot, 2finetuned
on Waymo training set videos.

ADEFDE051015202530Error8.217.111.228.84.19.14.210.0CARLACosmos Transfer 2.51Ours1Ours2Phase-Preserving Diffusion

13

t
u
p
n
I

-
s
o
m
s
o
C

5
.
2

r
e
f
s
n
a
r
T

s
r
u
O

Fig. 9: Video re-rendering results on CARLA.

Fig. 10: LPIPS and CLIP scores with different r0 and r.

loop imitation driving results are
given in Figure 8. ϕ-PD boosts plan-
ner generalization by 50% in zero-shot setting, demonstrating that structure-
preserving appearance enhancement significantly reduces the sim-to-real gap.
Video examples in Figure 9 show that ϕ-PD maintains road boundaries, vehi-
cle shapes, and spatial layout consistently across frames, whereas the compared
method produces distorted trees and multi-object artifacts.

Efficiency Comparison. Table 3 compares the extra parameters and FLOPs
relative to the base model, as well as the wall-clock time each method takes
to process an image. ControlNet Tile adds 50% parameters and computation
overhead (UNet + ControlNet model per step. PNP is 2 × more expensive
than the base model due to inversion and sampling, however achieves similar
wall-clock time thanks to optimized implementation from diffuser. FBSDiff is
extremely expensive (12 × base model FLOPs) due to 1000 inversion steps and
100 sampling steps. Our method does not introduce extra parameters or FLOPs.

2481632Inference Cutoff Radius0.300.320.340.360.380.400.420.44LPIPSTraining r0r0 = 0r0 = 2r0 = 4r0 = 82481632Inference Cutoff Radius0.30250.30500.30750.31000.31250.31500.31750.3200CLIP SimilarityTraining r0r0 = 0r0 = 2r0 = 4r0 = 814

Y. Zeng et al.

5 Ablation Studies

5.1 Cutoff Radius

We ablate the choices of r0, the minimal cutoff radius at training time, and
the inference-time cutoff radius r in this section. All ablation experiments are
conducted using SD 1.5 with LoRA finetuning and evaluated on 1, 000 randomly
selected samples from the UnrealCV test set. Figure 10 shows how r0 and r
affects these metrics. Increasing the inference-time cutoff radius r significantly
improves structural alignment: LPIPS drops from 0.44 − 0.45 to 0.30 − 0.38,
while the change in text alignment is small: CLIP scores stay around 0.30 −
0.32. The minimal cutoff threshold r0 during training influences the structural
alignment with a large inference time cutoff radius r. It also affects performance
across different inference-time radii r. A higher r0 during training leads to better
performance with higher r during inference, while a lower r0 favors scenarios with
smaller inference-time r.

5.2 Phase Preservation in Latent Space.

Classical results on phase-structure correspondence [23] were established in pixel
space, while modern diffusion models operate in VAE latent space. We ap-
ply phase preservation directly to latents. While a complete theoretical anal-
ysis of how phase-structure relationships transfer through nonlinear encoders is
beyond the scope of this work, we provide empirical validation in Figure 11.
We encode two images I1 and I2 into
VAE latent representations z1 and
z2, then swap their Fourier magni-
tude and phase components. Combin-
ing the phase from z1 with the magni-
tude from z2 produces a decoded im-
age that preserves the structure of I1
(the dog’s silhouette), as shown in the
top row. The bottom row shows the
reverse combination which preserves
the structure of I2 (road, horizon, ve-
hicles). This suggests that VAE la-
tents, which are trained to preserve
perceptually meaningful spatial struc-
ture, maintain sufficient phase-structure correspondence for our method to be
effective.

Fig. 11: Phase-structure correspondence in
VAE latent space.

6 Conclusion

We introduced Phase-Preserving Diffusion (ϕ-PD), a simple yet effective re-
formulation of the diffusion process that replaces Gaussian noise with struc-
tured noise, preserving image phase while randomizing the magnitude in the

I1I2Phase(z1) + Mag(z2)I2I1Phase(z2) + Mag(z1)Structurefrom I1Structurefrom I2Phase-Preserving Diffusion

15

frequency domain. This simple change retains spatial alignment throughout sam-
pling without modifying the architecture, altering training objectives, or intro-
ducing inference-time overhead. We also introduced Frequency-Selective Struc-
tured (FSS) noise, which provides continuous control over structural alignment
rigidity through a single frequency cutoff parameter, making it broadly applica-
ble to different applications.
Limitation. ϕ-PD assumes image-like inputs; modalities such as depth or nor-
mals may require a lightweight prior to produce an initial image representation.
Future work. ϕ-PD is orthogonal to existing conditioning or adapter methods
and can be integrated with them for enhanced control. Future work includes
extending ϕ-PD to tasks such as deblurring, relighting, super-resolution, and
general image restoration.

References

1. Anurag Ajay, Yilun Du, Abhi Gupta, Joshua Tenenbaum, Tommi Jaakkola, and
Is conditional generative modeling all you need for decision-

Pulkit Agrawal.
making? arXiv preprint arXiv:2211.15657, 2022. 3

2. Seungyeon Baek, Erqun Dong, Shadan Namazifard, Mark J Matthews, and
Kwang Moo Yi. Sonic: Spectral optimization of noise for inpainting with con-
sistency. arXiv preprint arXiv:2511.19985, 2025. 4

3. Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng
Zhang, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, et al. eDiff-I: Text-
to-image diffusion models with an ensemble of expert denoisers. arXiv preprint
arXiv:2211.01324, 2022. 3

4. Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu
Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. VideoCrafter1:
Open diffusion models for high-quality video generation.
arXiv preprint
arXiv:2310.19512, 2023. 3

5. Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burch-
fiel, and Shuran Song. Diffusion policy: Visuomotor policy learning via action
diffusion. arXiv preprint arXiv:2303.04137, 2023. 3

6. Guillaume Couairon, Marlène Careil, Matthieu Cord, Stéphane Lathuilière, and
Jakob Verbeek. Zero-shot spatial layout conditioning for text-to-image diffusion
models. In Proceedings of the IEEE/CVF International Conference on Computer
Vision (ICCV), pages 2174–2183, October 2023. 4
7. Sander Dieleman. Diffusion is spectral autoregression.

urlhttps://sander.ai/2024/09/02/spectral-autoregression.html, September 2024.
Accessed: 7 Dec 2025. 4

8. Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen

Koltun. CARLA: An open urban driving simulator, 2017. 10

9. Fabian Falck, Teodora Pandeva, Kiarash Zahirnia, Rachel Lawrence, Richard
Turner, Edward Meeds, Javier Zazo, and Sushrut Karmalkar. A fourier space
perspective on diffusion models. arXiv preprint arXiv:2505.11278, 2025. 4
10. Joseph W. Goodman. Statistical Optics. Wiley, 2 edition, 2015. Section 2.9.3. 6
11. Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic mod-
els. Advances in Neural Information Processing Systems, 33:6840–6851, 2020. 3,
8

16

Y. Zeng et al.

12. Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad
Norouzi, and David J Fleet. Video diffusion models. Advances in Neural In-
formation Processing Systems, 35:8633–8646, 2022. 3

13. Huang et al. NanoControl: A lightweight framework for precise and efficient control

in diffusion transformer. arXiv preprint arXiv:2508.10424, 2024. 4

14. Michael Janner, Yilun Du, Joshua B Tenenbaum, and Sergey Levine. Planning
with diffusion for flexible behavior synthesis. arXiv preprint arXiv:2205.09991,
2022. 3

15. Zeyinzi Jiang, Chaojie Mao, Yulin Pan, Zhen Han, and Jingfeng Zhang. SCEdit:
Efficient and controllable image diffusion generation via skip connection editing.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 8995–9004, June 2024. 4

16. Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design
space of diffusion-based generative models. In Alice H. Oh, Alekh Agarwal, Danielle
Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing
Systems, 2022. 3

17. Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. DiffWave:
A versatile diffusion model for audio synthesis. arXiv preprint arXiv:2009.09761,
2020. 3

18. Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.

10

19. Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning
to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003,
2022. 7

20. Sicheng Mo, Fangzhou Mu, Kuan Heng Lin, Yanli Liu, Bochen Guan, Yin Li,
and Bolei Zhou. FreeControl: Training-free spatial control of any text-to-image
diffusion model with any condition. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), pages 7465–7475, June
2024. 4

21. Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and
Ying Shan. T2I-Adapter: Learning adapters to dig out more controllable ability
for text-to-image diffusion models. In AAAI, volume 38, pages 4296–4304, 2024.
2, 4

22. NVIDIA. Cosmos-Transfer1: Conditional world generation with adaptive multi-

modal control. ArXiv, abs/2503.14492, 2025. 4

23. Alan V. Oppenheim and Jae S. Lim. The importance of phase in signals. Proceed-

ings of the IEEE, 69(5):529–541, 1981. 2, 5, 14

24. Bohao Peng, Jian Wang, Yuechen Zhang, Wenbo Li, Ming-Chang Yang, and Jiaya
Jia. ControlNeXt: Powerful and efficient control for image and video generation.
arXiv preprint arXiv:2408.06070, 2024. 4

25. Vadim Popov, Ivan Vovk, Vladimir Gogoryan, Tasnima Sadekova, and Mikhail
Kudinov. Grad-TTS: A diffusion probabilistic model for text-to-speech. In Inter-
national Conference on Machine Learning, pages 8599–8608. PMLR, 2021. 3
26. Michael Psenka, Alejandro Escontrela, Pieter Abbeel, and Yi Ma. Learning
arXiv preprint

a diffusion model policy from rewards via q-score matching.
arXiv:2312.11752, 2023. 3

27. Yurui Qian, Qi Cai, Yingwei Pan, Yehao Li, Ting Yao, Qibin Sun, and Tao Mei.
Boosting diffusion models with moving average sampling in frequency domain. In
Proceedings of the IEEE/CVF conference on computer vision and pattern recogni-
tion, pages 8911–8920, 2024. 4

28. Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sand-
hini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al.

Phase-Preserving Diffusion

17

Learning transferable visual models from natural language supervision. pages 8748–
8763. PmLR, 2021. 10

29. Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen.
Hierarchical text-conditional image generation with clip latents. arXiv preprint
arXiv:2204.06125, 2022. 3

30. Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn
Ommer. High-resolution image synthesis with latent diffusion models. In CVPR,
pages 10684–10695, June 2022. 10

31. Daniel L. Ruderman and William Bialek. The statistics of natural images. Network:

Computation in Neural Systems, 5(4):517–548, 1994. 2

32. Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily
Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S Sara Mahdavi,
Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models with deep
language understanding. arXiv preprint arXiv:2205.11487, 2022. 3

33. Zhenxiong Tan et al. OminiControl2: Efficient conditioning for diffusion trans-

formers. arXiv preprint arXiv:2503.08280, 2025. 4

34. Alibaba Team Wan. Wan: Open and advanced large-scale video generative models.

arXiv preprint arXiv:2503.20314, 2025. 10

35. Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play dif-
fusion features for text-driven image-to-image translation. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
pages 1921–1930, June 2023. 10

36. Julen Urain, Niklas Funk, Jan Peters, and Georgia Chalvatzaki.

SE(3)-
DiffusionFields: Learning smooth cost functions for joint grasp and motion op-
timization through diffusion. In 2023 IEEE International Conference on Robotics
and Automation (ICRA), pages 5923–5930. IEEE, 2023. 3

37. Zhendong Wang, Jonathan J Hunt, and Mingyuan Zhou. Diffusion policies
as an expressive policy class for offline reinforcement learning. arXiv preprint
arXiv:2208.06193, 2022. 3

38. Zhou Wang and Eero P. Simoncelli. Translation insensitive image similarity in
complex wavelet domain. IEEE Transactions on Image Processing, 14(4):466–479,
2005. 2

39. Wei Wu, Qingnan Fan, Shuai Qin, Hong Gu, Ruoyu Zhao, and Antoni B Chan.
Freediff: Progressive frequency truncation for image editing with diffusion models.
In European Conference on Computer Vision, pages 194–209. Springer, 2024. 4
40. Runsheng Xu, Hubert Lin, Wonseok Jeon, Hao Feng, Yuliang Zou, Liting Sun, John
Gorman, Ekaterina Tolstaya, Sarah Tang, Brandyn White, Ben Sapp, Mingxing
Tan, Jyh-Jing Hwang, and Dragomir Anguelov. WOD-E2E: Waymo open dataset
for end-to-end driving in challenging long-tail scenarios, 2025. 10

41. Cuihong Yu, Cheng Han, Chao Zhang, Yuewei Wang, Qihang Hu, Yin Yan, Moran
Zhan, Meng Li, and Guangjin Bi. DMFFT: improving the generation quality of
diffusion models using fast fourier transform. Scientific Reports, 15, March 2025.
4

42. Jiwen Yu, Yinhuai Wang, Chen Zhao, Bernard Ghanem, and Jian Zhang. Free-
In Proceedings
DoM: Training-free energy-guided conditional diffusion model.
of the IEEE/CVF International Conference on Computer Vision (ICCV), pages
23174–23184, October 2023. 4

43. Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to

text-to-image diffusion models. In ICCV, pages 3836–3847, 2023. 2, 4

44. Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang.
The unreasonable effectiveness of deep features as a perceptual metric. In CVPR,
pages 586–595, 2018. 10

18

Y. Zeng et al.

45. Siyuan Zhang, Wei Ma, Libin Liu, Zheng Li, and Hongbin Zha. Training-free fourier
phase diffusion for style transfer. In Proceedings of the Thirty-Fourth International
Joint Conference on Artificial Intelligence, pages 2386–2394, 2025. 4

46. Shihao Zhao, Dongdong Chen, Yen-Chun Chen, Jianmin Bao, Shaozhe Hao, Lu
Yuan, and Kwan-Yee K. Wong. Uni-ControlNet: All-in-one control to text-to-
image diffusion models. In Advances in Neural Information Processing Systems 36
(NeurIPS), 2023. 4

