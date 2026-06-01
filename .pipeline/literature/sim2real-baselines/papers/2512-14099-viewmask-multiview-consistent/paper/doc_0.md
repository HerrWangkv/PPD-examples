6
2
0
2

r
a

M
3
1

]

V
C
.
s
c
[

2
v
9
9
0
4
1
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

ViewMask-1-to-3: Multi-View Consistent Image Generation via Multimodal
Diffusion Models

Ruishu Zhu1,2*
Ping Luo4

Zhihao Huang1,2*

Jiacheng Sun3
Hongyuan Zhang2,4† Xuelong Li2†

1School of Artificial Intelligence, OPtics and ElectroNics (iOPEN), Northwestern Polytechnical University
2Institute of Artificial Intelligence of China Telecom (TeleAI)
3Huawei Technologies Co., Ltd.
4The University of Hong Kong
{zhuruishu0848, xysunjiacheng, hyzhang98}@gmail.com
huangzhihao@mail.nwpu.edu.cn, pluo@cs.hku.hk, xuelong li@ieee.org

Abstract

Motivated by discrete diffusion’s success in language-vision
modeling, we explore its potential for multi-view genera-
tion, a task dominated by continuous approaches. We in-
troduce ViewMask-1-to-3, formulating multi-view synthe-
sis as a discrete sequence modeling problem where each
viewpoint is represented as visual tokens from MAGVIT-v2.
Through masked token prediction, our approach enables
progressive multi-view generation via iterative token un-
masking, unifying language and vision in a shared token
space. Importantly, simple random masking combined with
self-attention naturally encourages cross-view consistency
without specialized architectures or 3D geometric priors.
Our method outperforms the baseline on the GSO and 3D-
FUTURE benchmarks, ranking first on average across stan-
dard image metrics and improving IoU by 10.6% on 3D-
FUTURE. This validates discrete diffusion as a promising
candidate for multi-view generation.

1. Introduction

Multi-view image generation has emerged as a fundamen-
tal challenge [14, 15, 29], with applications spanning virtual
reality [58] and 3D reconstruction [1]. The task aims to gen-
erate multiple consistent viewpoints of an object or scene
from limited input—typically a single image and accompa-
nying text description. Computational approaches face sig-
nificant challenges in maintaining geometric consistency,
preserving fine-grained details, and ensuring semantic co-
herence across viewpoints.

*Equal Contribution.
†Corresponding Author.

Recent advances in this domain have achieved impres-
sive results through diverse technical approaches. 3D rep-
resentation methods [30, 42, 46] leverage explicit 3D rep-
resentations, such as neural radiance fields, 3D Gaussian
primitives, or structured latent spaces, to ensure geomet-
ric consistency through native 3D modeling and rendering.
Camera-conditioned diffusion models [19, 22, 28, 39]
have demonstrated remarkable flexibility by directly gen-
erating multi-view images in 2D space, either controlling
the views with precise camera parameters or directly pro-
ducing a fixed views. Image Editing model [18, 27, 31]
explore integrating multi-view generation with image edit-
ing manner, enabling seamless transitions between different
vision tasks. These methods, predominantly built on contin-
uous diffusion in latent space, have collectively established
strong baselines.

Meanwhile, a distinct paradigm has shown remarkable
success in unified language-vision modeling: discrete diffu-
sion models based on masked token prediction [35, 52, 54].
By operating in a unified vocabulary space, they offer in-
triguing properties: (1) native multi-modal fusion of text
and visual tokens, (2) bidirectional, parallel context model-
ing during generation, and (3) architectural alignment with
large language models. This raises a compelling yet un-
explored question: Can this discrete paradigm serve as a
viable alternative for the geometrically demanding task of
multi-view generation?

In this work, we present ViewMask-1-to-3, an explo-
ration of discrete diffusion models for multi-view consistent
image generation. Our key insight is to formulate multi-
view synthesis as a discrete sequence modeling problem,
where each viewpoint is represented as a sequence of vi-
sual tokens obtained through learned tokenization [55]. By

 
 
 
 
 
 
Figure 1. Comparison of multi-view image generation approaches. (Left-Top) 3D representation methods with 3D reconstruction and
rendering. (Left-Bottom) Camera-conditioned diffusion methods with precise camera parameters. (Left-middle) Image editing models
generate views with sequential manner. (Right) Our discrete diffusion approach enables parallel multi-view generation through unified
sequence modeling with flexible text-based control.

extending masked token prediction, a technique proven ef-
fective in language modeling [35], to the visual domain
with multiple viewpoints, we enable progressive genera-
Importantly, we
tion through iterative token unmasking.
find that a simple random masking strategy, when com-
bined with bidirectional self-attention, naturally encourages
cross-view consistency without requiring specialized archi-
tectural components or explicit 3D geometric priors. Our
contributions are as follows:

• We present a systematic exploration of discrete diffusion
models for multi-view image generation, demonstrating
that masked token prediction can achieve competitive per-
formance in this domain.

• The proposed simple masking strategy, when combined
with bidirectional attention mechanisms, naturally en-
courages cross-view consistency without requiring 3D
geometric priors or specialized architectures.

• We evaluate ViewMask-1-to-3 on standard multi-view
generation benchmarks, where it tops the average rank-
ings on image-level metrics (PSNR, SSIM, LPIPS, CD,
IOU) for GSO and 3D-FUTURE, particularly achieving
a 10.6% higher IoU on 3D-FUTURE.

2. Related Work

2.1. Multi-View Image Generation

3D Representation-Based Methods. Early approaches en-
sure geometric consistency through explicit 3D modeling.
3D-aware GANs such as π-GAN [4] and EG3D [5] leverage
neural radiance fields or tri-plane representations for view-
consistent synthesis. More recently, DreamGaussian [42]
employs 3D Gaussian Splatting for efficient content cre-
ation; TRELLIS [46] adopts flow matching to obtain struc-
tured 3D latents and then extracts multi-view images; MV-
Dream [40] and SyncDreamer [30] integrate NeRF [34] into
diffusion frameworks with denoising across viewpoints.

Camera-Conditioned Diffusion Methods. An alternative
approach generates multi-view images directly in 2D space
through camera-conditioned diffusion. Zero-1-to-3 [28] pi-
oneered fine-tuning pretrained diffusion models with cam-
era pose conditioning for zero-shot novel view synthesis,
generating each view independently. Subsequent works
address cross-view consistency through various strategies:
Zero123++ [39] models the joint distribution of multiple
views by concatenating them into a unified representation;
SyncDreamer [30] and One-2-3-45 [27] incorporate syn-
chronized denoising and post-processing refinement. Re-
cent extensions include ViVid-1-to-3 [22], which lever-
ages video diffusion for temporal coherence; MV-AR [17],

TokenizerDetokenizerText PromptDiscrete DiffusionInput ImageMulti-View ImageBGBG Discrete Diffusion Method (Ours, 2025)ObjectMulti-View ObjectFlexible: Design degree easily, unified sequence modelingDiffusion StepRotate 90° 180° 270°of the imageMask TokenText TokenImage Token3D Representation Method (TRELLIS, 2024)3D ConstructorView ExtractionInput Image3D modelMulti-View ImageNative 3D Modeling: Geometry construction with rendering pipelineImage Editing Model (Lumina-DiMOO, 2025)TokenizerDetokenizerInput ImageText PromptMLLMLeft view of the imageSingle ViewMultimodal Editing: Instruction-based view editingDiffusionVAEEncoderVAEDecoderPose Conditioning: Camera parameter guided 2D diffusionCamera ParamInput ImageCamera-condition Method (Zero1-to-3, 2023)Multi-View Imagewhich introduces autoregressive viewpoint generation; and
EpiDiff [19], which enforces epipolar constraints through
specialized attention mechanisms. However, these methods
typically focus on image-to-multi-view synthesis, with text-
to-multi-view requiring additional text-to-image models to
generate reference images first.

Image Editing Methods. Recent methods attempt to gen-
erate multi-view images via image editing strategies. These
approaches leverage various techniques: MV-Adapter [18]
transforms text-to-image models into multi-view genera-
tors without altering their structure; ImageBrush [53] em-
ploys visual instructions rather than text for precise ma-
nipulation; SceneScape [10] ensures geometric consistency
through depth-aware generation; and Lumina-DiMOO [50]
introduces discrete diffusion in image editing with impres-
sive results. While these methods rely on specialized edit-
ing strategies, ViewMask-1-to-3 naturally achieves cross-
view consistency through token-level modeling using dis-
crete diffusion, offering a more unified and elegant solution
to multi-view synthesis.

2.2. Discrete Diffusion Models

D3PM [2] established the foundational framework for dis-
crete diffusion, extending traditional diffusion to discrete
state spaces through structured denoising processes. Diffu-
sionBERT [16] demonstrated that pre-trained BERT mod-
els could be adapted for text generation through discrete
diffusion training. A significant breakthrough came with
LLaDA [35], which showed that masked diffusion mod-
els can match autoregressive models like LLaMA3-8B in
language generation while offering parallel decoding and
bidirectional context modeling. LLaDA-V [54] extended
this framework to multimodal understanding, demonstrat-
ing the potential of discrete diffusion in vision-language
tasks. These advances have established discrete diffusion as
a viable alternative to autoregressive approaches, for tasks
requiring controllable and parallel generation.

2.3. Visual Tokenization

The effectiveness of discrete diffusion for visual tasks crit-
ically depends on visual tokenization quality. VQVAE [43]
introduced discrete visual representations through vector
quantization,
laying the groundwork for discrete visual
modeling. MaskGIT [6] applied masked token prediction
to image generation using vector-quantized representations.
MAGVIT-v2 [55] enabling high-quality discrete tokeniza-
tion for both images and videos through lookup-free quanti-
zation and architectural improvements. The success of these
tokenizers make discrete approaches increasingly competi-
tive with continuous methods in visual generation tasks.

3. ViewMask-1-to-3

3.1. Overview
Given a single input image I0 ∈ RH×W ×3 and an optional
text description T , our goal is to generate N = 3 con-
sistent target views {I1, I2, I3} representing the same ob-
ject from different viewpoints. Beyond image-conditioned
view synthesis, our model also achieve text-only genera-
tion: provided solely with a description T , it can synthe-
size N = 4 new views {I1, I2, I3, I4} that correspond to
distinct viewpoints, even in the absence of an input image.
This unified formulation enables both image-to-multi-view
and text-to-multi-view generation within a single frame-
work. Unlike Zero-1-to-3 that operates in continuous latent
spaces, we formulate multi-view generation as a discrete
sequence modeling problem. By representing each view as
a sequence of discrete visual tokens, we can leverage the
parallel processing and bidirectional context modeling
capabilities of masked diffusion models.

3.2. Model Design

Our approach consists of three main components: (1) Visual
Tokenization, where we convert images to discrete token
sequences using MAGVIT-v2; (2) Cross-View Sequence
Construction, where we build unified sequences encoding
multiple viewpoints with special tokens; and (3) Masked
Diffusion Training, where we learn to predict masked vi-
sual tokens through iterative refinement.

Visual Tokenization. We adopt MAGVIT-v2 [55] as our
visual tokenizer due to its superior performance in discrete
visual representation. For an input image Ii, the tokenizer
produces a sequence of discrete visual tokens,

V (i) = Tokenize(Ii) = {v(i)

1 , v(i)

2 , . . . , v(i)

L },

(1)

where V (i) ∈ V L represents the token sequence, V is the
visual vocabulary with |V| = 218 tokens, and L is the se-
quence length. Each token v(i)
corresponds to a spatial re-
j
gion of the image and encodes local visual features. The
discrete representation enables us to apply language model-
ing techniques directly to visual content while maintaining
reconstruction quality through the learned tokenizer.

The tokenized sequences are then embedded into the lan-

guage model’s embedding space:

e(i) = Embed(V (i)) ∈ RL×d,

(2)

where d is the embedding dimension of the transformer.

Cross-View Sequence Construction.
The unified se-
quence construction is a core step of ViewMask-1-to-3,
which enables coordinated generation across multiple view-
points under both image-to-multi-view (I2MV) and text-to-
multi-view (T2MV) settings. By using the same masked-
sequence format, we minimize the structural gap between

Figure 2. Overview of our three-stage training framework. (I) Stage 1: Pretraining aligns text and image modalities using masked
image-caption sequences. (II) Stage 2: Image-to-Multi-View (I2MV) masks the target view and learns to reconstruct it conditioned on a
reference view, thereby acquiring multi-view generation capability. (III) Stage 3: Text-to-Multi-View (T2MV) masks all views and learns
to generate multiple views of an object conditioned on a text description.

I2MV and T2MV inputs, enabling the model to treat both
tasks similarly (see Section 3.3 for details).

imagenet-1k-vl-enriched [24], to facilitate alignment be-
tween modalities.

Cross-View Masking Strategy. During training, we em-
ploy a simple yet effective random masking strategy. For
each training sample, target tokens are first selected based
on the current training stage (see Section 3.3), and a mask-
ing ratio r ∼ Uniform(0, 1) is sampled to randomly replace
a subset of them with a special [MASK] token.

3.3. Multi-Stage Training Strategy

As shown in Fig. 2, our training framework comprises three
stages.

Stage1: Pretraining for Multimodal Alignment. To en-
able the LLADA model to learn representations of image
tokens, we leverage image-caption pairs to align the two
modalities. The input sequence is formatted as:

[T2I] [SOT] caption tokens [EOT] [SOI] image tokens [EOI],

where [T2I] indicates the image-caption alignment, [SOT]
and [EOT] denote the start and end of the text, and [SOI]
and [EOI] denote the start and end of the image. Both cap-
tion tokens and image tokens are randomly masked with a
certain probability during training. This design encourages
the model to fully capture the co-occurring semantic infor-
mation in image–caption pairs, thereby aligning the repre-
sentations of the two modalities.

We construct a multimodal pretraining corpus of 1.2
million image–caption pairs using the images from Im-
ageNet [9] images combined with captions provided by

Stage2: Image to Multi-View. To enable the LLADA
model to learn representations of image tokens, we lever-
age image–caption pairs to align the two modalities. The
input sequence is formatted as:

[I2MV] [SOT] P. [EOT] [SOI] R. [EOI] [SOI] G1 G2 G3 [EOI],

where [I2MV] indicates the image-to-multi-view task, P.
represents the prompt tokens, R. represents the reference
image tokens, and Gi represents the generated image to-
kens. For brevity, the repeated [EOI] [SOI] between con-
secutive Gi are omitted. Gi are randomly masked with a
certain probability during training.

At this stage, the model is trained on a curated sub-
set of Objaverse [8] and Habitat Synthetic Scene Dataset
(HSSD) [21]. Following the data split provided by Obja-
verse++ [25], we removed samples labeled as Low Quality
(No semantic meaning. Objects that annotators cannot iden-
tify or are corrupted), to ensure more stable training.

Stage3: Text to Multi-View.
To extend the I2MV
paradigm to T2MV, we incorporate an additional textual de-
scription specifying the target views immediately after the
prompt. Accordingly, the input sequence is formatted as:

[T2MV] [SOT] P. D. [EOT] [SOI] G1 G2 G3 G4 [EOI],

where [T2MV] indicates the text-to-multi-view task, D. rep-
resents the description tokens. For brevity, the repeated
[EOI] [SOI] between consecutive Gi are omitted. Gi are
randomly masked with a certain probability during training.

Figure 3. Overview of inference framework. (a) I2MV: The tokenized prompt and reference view provide a template for generating three
consistent output images from different viewpoints. (b) T2MV: Tokenized prompt and description, providing a template for generating
four output images. At each timestep t, the tokens are further predicted, and under the guidance of the mask schedule, remasking is
performed based on confidence.

We enhance Objaverse with Cap3D [32, 33], which of-

fers detailed textual descriptions for 3D assets.

Loss. ViewMask-1-to-3 is trained using a cross-entropy
loss that predicts original tokens at masked positions.

LCE = −

3
(cid:88)

(cid:88)

i=1

j∈Mi

log P (v(i)
j

|s\M),

(3)

where Mi denotes the set of masked positions in view i, and
s\M represents the sequence with masked tokens. This ob-
jective encourages the model to reconstruct original visual
tokens while leveraging unmasked context from the input
image, text description, and partially observed target views.
The cross-attention mechanism naturally enables informa-
tion flow between different modalities and viewpoints.

3.4. Inference Process

Construction of Input Sequences. During inference, we
perform iterative denoising starting from fully masked tar-
get views, as illustrated in Fig. 3. For I2MV, the model is
provided with a tokenized prompt and the reference view.
The template is designed to generate three output images.
For T2MV, the input consists of a tokenized prompt and a
textual description of the desired view. This template sup-
ports the generation of four output images.

Predict and Remask. We employ an iterative denoising-
based sampling strategy. The model progressively pre-
dicts token distributions over T iterations. At each step,
the model performs a forward pass to obtain logits for all

masked positions and estimates the confidence of each to-
ken prediction. A scheduling function (cosine, linear, or
quadratic) determines the number of tokens to be re-masked
in the next step, where low-confidence tokens are re-masked
while high-confidence ones are retained. This predict–re-
mask loop continues until all masked tokens are generated,
progressively refining the sequence toward the final output.
We further discuss the effects of different scheduling strate-
gies in the ablation study (see Section 4.3).

4. Experiments

Datasets. Our model was trained on a subset of Obja-
verse [8] and HSSD [21]. Following the data split provided
by Objaverse++ [25], we removed samples labeled as Low
Quality, to ensure more stable training. To enrich the textual
supervision, we adopted the Cap3D [32, 33] annotations,
which provide descriptive captions for 3D objects. In to-
tal, we used 180K 3D objects, each rendered as an 8-frame
RGBA orbit sequence at a resolution of 256×256, with the
elevation fixed at 30°. During training, one frame was ran-
domly selected as the reference image for each object, re-
sulting in a multi-view generation dataset.

Metrics. We evaluate the multi-view generation results
from two perspectives. First, for the directly generated
novel views, we adopt common image-level metrics, includ-
ing Peak Signal-to-Noise Ratio (PSNR), Structural Similar-
ity Index Measure (SSIM) [44], and Learned Perceptual Im-
age Patch Similarity (LPIPS) [56], to assess visual fidelity
and perceptual quality. Then, we employ InstantMesh [51]

Figure 4. Qualitative comparison on the 3D-Future dataset. Visual results on the 3D-Future dataset, where the bottom-right shows the
reconstructed mesh, demonstrating that our ViewMask-1-to-3 produces more consistent and realistic novel views compared with recent
methods.

to reconstruct 3D geometry from the generated views, and
evaluate the resulting point clouds using Chamfer Distance
(CD) and Intersection over Union (IoU) to quantify geomet-
ric consistency and accuracy.

4.1. Image to Multi-view Images

Baseline. We compare our method against a comprehensive
set of state-of-the-art diffusion-based novel view generation
approaches, including Zero-1-to-3 [28], Zero-1-to-3 XL [7,
28], ViVid-1-to-3 [22], Zero123++ [39], MV-Adapter [18],
EpiDiff [19], AR-1-to-3 [57], and TRELLIS [46].

Qualitative Results.

Fig. 4 presents a visual comparison between our
method and several state-of-the-art multi-view generation
approaches. In the top three rows, ViewMask-1-to-3 pre-
serves accurate chair legs and seat curvature, avoiding the
distortions and artifacts seen in other methods. In the bot-

tom three rows, it achieves the best overall consistency, cap-
turing both the recessed details on the desktop and the metal
frame. Compared with these baselines, our method pro-
duces sharper visual quality, better preserves fine-grained
textures, and maintains higher cross-view consistency.

Quantitative Results. To quantitatively evaluate the zero-
shot generalization capability of our model, we conduct ex-
periments on two aspects: (1) image-level quality, measured
on novel views generated from a single input image, and
(2) 3D-level geometric accuracy, evaluated on 3D recon-
structions. These experiments are evaluated on the Google
Scanned Objects (GSO) dataset [50] and the 3D-FUTURE
dataset [11].

The input views are selected with controlled offsets in
azimuth and elevation relative to the canonical frontal view
of each 3D object. For models that support arbitrary-angle
rendering, such as Zero-1-to-3 and Zero-1-to-3 XL, the

Table 1. Quantitative comparison on multi-view generation over the GSO and 3D-FUTURE datasets. We report PSNR, SSIM, and
LPIPS.

Method

Architecture

GSO

3D-FUTURE

PSNR↑

SSIM↑

LPIPS↓

PSNR↑

SSIM↑

LPIPS↓

Zero-1-to-3 [28]
Zero-1-to-3 XL [28]
ViVid-1-to-3 [22]
Zero123++ [39]
Epidiff [19]
MV-Adapter [18]
AR-1-to-3 [57]
TRELLIS [46]
ViewMask-1-to-3

2D Cont. Diff.
2D Cont. Diff.
2D Cont. Diff.
2D Cont. Diff.
2D Cont. Diff.
2D Cont. Diff.
Video Cont. Diff.
3D Rectified Flow
2D Disc. Diff.

18.8219
19.6839
19.7978
19.6373
18.9917
19.4673
13.5084
14.4493
20.3868

0.8294
0.8381
0.8566
0.8045
0.8244
0.7518
0.7376
0.8058
0.8549

0.1659
0.1518
0.1764
0.3550
0.1667
0.1503
0.3514
0.3223
0.1537

17.0526
18.4702
18.3241
23.5001
17.4592
19.3375
13.7724
14.4604
19.8186

0.8163
0.8337
0.8437
0.8527
0.8108
0.7217
0.7479
0.7896
0.8610

0.1760
0.1485
0.1682
0.1529
0.1785
0.4036
0.3034
0.2990
0.1315

Rank

5.3
3.0
3.5
4.3
5.7
5.8
8.5
7.2
1.7

Table 2. Quantitative comparison on 3D reconstruction over
the GSO and 3D-FUTURE datasets. We report chamfer distance
(CD) value and IOU. ↑ indicates higher is better, ↓ indicates lower
is better.

Method

Zero-1-to-3
Zero-1-to-3 XL
Zero123++
ViVid-1-to-3
AR-1-to-3
EpiDiff
TRELLIS
ViewMask-1-to-3

GSO

3D-FUTURE

CD↓

IOU↑

CD↓

0.0163
0.0159
0.0163
0.0163
0.0372
0.0166
0.0496
0.0149

0.5665
0.5799
0.5832
0.5841
0.4380
0.5457
0.4026
0.5847

0.0113
0.0106
0.0314
0.0105
0.0148
0.0121
0.0178
0.0106

IOU↑

0.5005
0.5271
0.4250
0.5246
0.4286
0.4784
0.4396
0.5315

views are rendered at the same angles as those used by
ViewMask-1-to-3 to ensure a fair comparison. We then
evaluate PSNR, SSIM, and LPIPS following each model’s
respective settings, and report the results in Tab. 1. Over-
all, our method, ViewMask-1-to-3, achieves the best com-
bined performance across both datasets and attains the high-
est rank (1.7), demonstrating its superiority in multi-view
generation.

To evaluate geometric consistency, we reconstruct 3D
shapes from the generated multi-view images for each
model. We then compute Chamfer Distance (CD) and
IoU between the reconstructed shapes and the ground-truth
meshes. The quantitative results are reported in Tab. 2,
demonstrating that our method achieves lower CD and
higher IoU, indicating more accurate 3D reconstruction and
better cross-view consistency.

4.2. Text to Multi-view Images

In the T2MV task, we demonstrate the ability to gener-
ate multi-view images of an object directly from text in-
puts. Models like MV-Adapter [18] and TRELLIS [46]
support T2MV and I2MV using different checkpoints or

Figure 5. Qualitative results on T2MV. Examples of multi-view
object generation from text prompts using our method, achieved
directly without any intermediate steps.

architectures. Zero123++ [39] and ViVid-1-to-3 [22] gen-
erate reference images with a T2I model before extracting
the subject for novel views.
In contrast, our method di-
rectly produces novel views that preserve both the specified
viewpoint and the prompt’s semantic content. As shown
in Fig. 5, we present representative test samples from the
Objaverse dataset and compare qualitative results with MV-
Adapter [18], TRELLIS [46], and SPAD [20], showing that
our method achieves comparable visual quality and consis-
tency.

4.3. Ablation study

Mask Schedule. We investigate the impact of different
mask scheduling strategies on our model’s performance, in-
cluding linear, quadratic, and cosine schedules.

Figure 6. Qualitative results of different mask scheduling
strategies on a subset of the GSO dataset. Compared with lin-
ear, quadratic, and fixed schedules, the cosine schedule produces
sharper details and more coherent structures in the generated novel
views.

In Tab. 3, our experiments on a subset of the GSO
dataset indicate that the cosine schedule yields smoother op-
timization and superior final results compared with the other
strategies, highlighting the importance of gradually adjust-
ing the masking ratio over timesteps. In addition to quan-
titative evaluation, we provide qualitative visualizations of
the generated images under different schedules, as shown
in Fig. 6. For the boot, both the linear and quadratic sched-
ules hallucinate an additional zipper on the opposite side,
which is clearly inconsistent with reality. For the chair,
the quadratic schedule yields incorrect thickness estimation.
For the toy, both the linear and quadratic schedules distort
the teddy bear’s facial region.

Table 3. Ablation study on different mask scheduling strate-
gies. The metrics (e.g., PSNR, SSIM, LPIPS) are reported for
each schedule.

Schedule

PSNR ↑

SSIM ↑ LPIPS ↓

Linear
Quadratic
Cosine

17.9882
17.9785
18.0974

0.8435
0.8431
0.8437

0.1882
0.1865
0.1834

4.4. Task Extension

Unified Multimodal Understanding and Generation. In
addition to the aforementioned I2MV and T2MV capabili-
ties, the Stage-1 pretraining endows our model with funda-
mental multimodal understanding and text-to-image gener-
ation abilities. As shown in Fig. 7, the model can perform
basic understanding and generation tasks without any ad-
ditional fine-tuning. Building on these abilities, our model

Figure 7. Task Extension. Left: basic multimodal understanding
(MMU), e.g., captioning. Middle: text-to-image (T2I) generation
without fine-tuning. Right: training-free 2D turn-style and com-
pletion preserving the background.

naturally extends toward unified multimodal understanding
and generation, evaluated on GenEval [13] for composi-
tional text-to-image generation. ViewMask-1-to-3 achieves
the best overall score (0.72), outperforming unified mod-
els Show-o (0.68) [49] and MMaDA (0.63) [52], and sur-
passing generation-only models like DALL·E 3 (0.67) [3],
comparable to SANA-1.5 (0.72) [48], as shown in Ta-
ble 4. Compared with the architecturally similar MMaDA,
improvement
ViewMask-1-to-3 achieves a 14% overall
(0.72 vs. 0.63). This significant gain highlights the prac-
tical potential of discrete diffusion architectures.

Training-Free 2D Turn-Style. Inspired by Adobe’s Project
Turn Style, which enables re-angling and rotating 2D illus-
trations as if they were 3D objects, we extend this capa-
bility in a training-free manner based on our ViewMask-
1-to-3 framework. Benefiting from the inherent properties
of discrete diffusion, our model can rotate a specified ob-
ject by a given angle and perform image completion while
preserving the original background, all without additional
fine-tuning, as these abilities are implicitly embedded dur-
ing diffusion training. These cases show that our framework
extends to a unified diffusion architecture while maintaining
strong flexibility and generalization across diverse tasks.

5. Conclusion

In this work, we introduce ViewMask-1-to-3, a discrete
diffusion framework for multi-view image generation. Ex-
periments on GSO and 3D-FUTURE demonstrate its strong
performance, showcasing discrete diffusion as an effective
and flexible approach for multi-view synthesis. Our re-
sults also highlight its potential for MMU, T2I, T2MV, and

a green door with a blackmetal windowa red crossbill is perchedon a tree stumpA mountain with a snowcapped peak.Training-Free Turn-StyleT2IMMUTask3: Visualize theobject  from its 90°,180°, and 270°Task2: Generate the imageby the input textTask1: Render a clear andconcise summary of thephoto.A butterfly is sittingon a leaf.90°180°270°training-free 2D turn-style, paving the way toward more
unified and scalable multi-view generation.

Despite strong performance, there is room to expand
viewpoint diversity and improve lighting robustness. Fu-
ture work will focus on enhancing scalability, resolution,
and generalization for more unified multi-view generation.

Acknowledgements

This paper is partially supported by the National Key R&D
Program of China No.2022ZD0161000.

References

[1] Hadi Alzayer, Kevin Zhang, Brandon Feng, Christopher A
Metzler, and Jia-Bin Huang. Seeing the world through your
eyes. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 4864–4873,
2024. 1

[2] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tar-
low, and Rianne Van Den Berg. Structured denoising dif-
fusion models in discrete state-spaces. Advances in neural
information processing systems, 34:17981–17993, 2021. 3

[3] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng
Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce
Lee, Yufei Guo, et al.
Improving image generation with
better captions. Computer Science. https://cdn. openai.
com/papers/dall-e-3. pdf, 2(3):8, 2023. 8, 1

[4] Eric R Chan, Marco Monteiro, Petr Kellnhofer, Jiajun Wu,
and Gordon Wetzstein. pi-gan: Periodic implicit generative
adversarial networks for 3d-aware image synthesis. In Pro-
ceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 5799–5809, 2021. 2

[5] Eric R Chan, Connor Z Lin, Matthew A Chan, Koki Nagano,
Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas J
Guibas, Jonathan Tremblay, Sameh Khamis, et al. Efficient
geometry-aware 3d generative adversarial networks. In Pro-
ceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 16123–16133, 2022. 2
[6] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and
William T. Freeman. Maskgit: Masked generative image
In Proceedings of the IEEE/CVF Conference
transformer.
on Computer Vision and Pattern Recognition (CVPR), pages
11315–11325, 2022. 3

[7] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong
Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Chris-
tian Laforte, Vikram Voleti, Samir Yitzhak Gadre, Eli
VanderBilt, Aniruddha Kembhavi, Carl Vondrick, Georgia
Gkioxari, Kiana Ehsani, Ludwig Schmidt, and Ali Farhadi.
Objaverse-xl: A universe of 10m+ 3d objects. arXiv preprint
arXiv:2307.05663, 2023. 6, 1

[8] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs,
Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana
Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse:
In Proceedings of
A universe of annotated 3d objects.
the IEEE/CVF conference on computer vision and pattern
recognition, pages 13142–13153, 2023. 4, 5

[9] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li,
and Li Fei-Fei. Imagenet: A large-scale hierarchical image
database. In 2009 IEEE conference on computer vision and
pattern recognition, pages 248–255. Ieee, 2009. 4

[10] Rafail Fridman, Amit Abecasis, Yoni Kasten, and Tali
Dekel. Scenescape: Text-driven consistent scene genera-
tion. Advances in Neural Information Processing Systems,
36:39897–39914, 2023. 3

[11] Huan Fu, Rongfei Jia, Lin Gao, Mingming Gong, Binqiang
Zhao, Steve Maybank, and Dacheng Tao. 3d-future: 3d fur-
niture shape with texture, 2020. 6

[12] Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin
Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Mul-
timodal models with unified multi-granularity comprehen-
sion and generation. arXiv preprint arXiv:2404.14396, 2024.
1

[13] Dhruba Ghosh, Hanna Hajishirzi, and Ludwig Schmidt.
Geneval: An object-focused framework for evaluating text-
to-image alignment, 2023. 8

[14] Qianyun He, Xinya Ji, Yicheng Gong, Yuanxun Lu, Zhengyu
Diao, Linjia Huang, Yao Yao, Siyu Zhu, Zhan Ma, Song-
cen Xu, et al. Emotalk3d: High-fidelity free-view synthesis
of emotional 3d talking head. In European Conference on
Computer Vision, pages 55–72. Springer, 2024. 1

[15] Yuxiao He, Yiyu Zhuang, Yanwen Wang, Yao Yao, Siyu Zhu,
Xiaoyu Li, Qi Zhang, Xun Cao, and Hao Zhu. Head360:
Learning a parametric 3d full-head for free-view synthesis in
360 degree. In European Conference on Computer Vision,
pages 254–272. Springer, 2024. 1

[16] Zhengfu He, Tianxiang Sun, Qiong Tang, Kuanning Wang,
Xuan-Jing Huang, and Xipeng Qiu. Diffusionbert: Improv-
ing generative masked language models with diffusion mod-
els. In Proceedings of the 61st Annual Meeting of the Asso-
ciation for Computational Linguistics (Volume 1: Long Pa-
pers), pages 4521–4534, 2023. 3

[17] JiaKui Hu, Yuxiao Yang, Jialun Liu, Jinbo Wu, Chen Zhao,
and Yanye Lu. Auto-regressively generating multi-view con-
sistent images. arXiv preprint arXiv:2506.18527, 2025. 2

[18] Zehuan Huang, Yuanchen Guo, Haoran Wang, Ran Yi,
Lizhuang Ma, Yan-Pei Cao, and Lu Sheng. Mv-adapter:
Multi-view consistent image generation made easy. arXiv
preprint arXiv:2412.03632, 2024. 1, 3, 6, 7

[19] Zehuan Huang, Hao Wen, Junting Dong, Yaohui Wang,
Yangguang Li, Xinyuan Chen, Yan-Pei Cao, Ding Liang, Yu
Qiao, Bo Dai, and Lu Sheng. Epidiff: Enhancing multi-view
synthesis via localized epipolar-constrained diffusion, 2024.
1, 3, 6, 7

[20] Yash Kant, Ziyi Wu, Michael Vasilkovsky, Guocheng
Qian, Jian Ren, Riza Alp Guler, Bernard Ghanem, Sergey
Tulyakov, Igor Gilitschenski, and Aliaksandr Siarohin. Spad
: Spatially aware multiview diffusers, 2024. 7

[21] Mukul Khanna*, Yongsen Mao*, Hanxiao Jiang, Sanjay
Haresh, Brennan Shacklett, Dhruv Batra, Alexander Clegg,
Eric Undersander, Angel X. Chang, and Manolis Savva.
Habitat Synthetic Scenes Dataset (HSSD-200): An Analy-
sis of 3D Scene Scale and Realism Tradeoffs for ObjectGoal
Navigation. arXiv preprint, 2023. 4, 5

[22] Jeong-gi Kwak, Erqun Dong, Yuhe Jin, Hanseok Ko, Shweta
Mahajan, and Kwang Moo Yi. Vivid-1-to-3: Novel view
arXiv preprint
synthesis with video diffusion models.
arXiv:2312.01305, 2023. 1, 2, 6, 7

[23] Black Forest Labs. Flux. https://github.com/

[24] Visual Layer.

black-forest-labs/flux, 2024. 1
https : / /
imagenet-1k-vl-enriched.
huggingface . co / datasets / visual - layer /
imagenet-1k-vl-enriched, 2024. 4

[25] Chendi Lin, Heshan Liu, Qunshu Lin, Zachary Bright, Shi-
tao Tang, Yihui He, Minghao Liu, Ling Zhu, and Cindy Le.
Objaverse++: Curated 3d object dataset with quality annota-
tions, 2025. 4, 5

[26] Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel.
World model on million-length video and language with
ringattention. arXiv preprint, 2024. 1

[27] Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Mukund
Varma T, Zexiang Xu, and Hao Su. One-2-3-45: Any single
image to 3d mesh in 45 seconds without per-shape optimiza-
tion. Advances in Neural Information Processing Systems,
36:22226–22246, 2023. 1, 2

[28] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tok-
makov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-
In Proceedings of
3: Zero-shot one image to 3d object.
the IEEE/CVF international conference on computer vision,
pages 9298–9309, 2023. 1, 2, 6, 7

[29] Tianqi Liu, Guangcong Wang, Shoukang Hu, Liao Shen,
Xinyi Ye, Yuhang Zang, Zhiguo Cao, Wei Li, and Ziwei Liu.
Mvsgaussian: Fast generalizable gaussian splatting recon-
struction from multi-view stereo. In European Conference
on Computer Vision, pages 37–53. Springer, 2024. 1
[30] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie
Liu, Taku Komura, and Wenping Wang. Syncdreamer: Gen-
erating multiview-consistent images from a single-view im-
age. In The Twelfth International Conference on Learning
Representations, 2024. 1, 2

[31] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu,
Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang,
Marc Habermann, Christian Theobalt, et al. Wonder3d: Sin-
gle image to 3d using cross-domain diffusion. In Proceed-
ings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 9970–9980, 2024. 1

[32] Tiange Luo, Chris Rockwell, Honglak Lee, and Justin
Johnson. Scalable 3d captioning with pretrained models.
Advances in Neural Information Processing Systems, 36:
75307–75337, 2023. 5

[33] Tiange Luo, Justin Johnson, and Honglak Lee. View selec-
In European
tion for 3d captioning via diffusion ranking.
Conference on Computer Vision, pages 180–197. Springer,
2024. 5

[34] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM, 65(1):99–106, 2021.
2

[35] Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang
Ou, Jun Hu, JUN ZHOU, Yankai Lin, Ji-Rong Wen, and

Chongxuan Li. Large language diffusion models. In ICLR
2025 Workshop on Deep Generative Model in Machine
Learning: Theory, Principle and Efficacy, 2025. 1, 2, 3
[36] Dustin Podell, Zion English, Kyle Lacey, Andreas
Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and
Improving latent diffusion mod-
Robin Rombach. Sdxl:
arXiv preprint
els for high-resolution image synthesis.
arXiv:2307.01952, 2023. 1

[37] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu,
and Mark Chen. Hierarchical text-conditional image gener-
ation with CLIP latents. CoRR, abs/2204.06125, 2022. 1
[38] Robin Rombach, Andreas Blattmann, Dominik Lorenz,
Patrick Esser, and Bj¨orn Ommer. High-resolution image syn-
thesis with latent diffusion models. In CVPR, pages 10684–
10695, 2022. 1

[39] Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu,
Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao
Su. Zero123++: a single image to consistent multi-view dif-
fusion base model. arXiv preprint arXiv:2310.15110, 2023.
1, 2, 6, 7

[40] Yichun Shi, Peng Wang, Jianglong Ye, Long Mai, Kejie Li,
and Xiao Yang. MVDream: Multi-view diffusion for 3d gen-
eration. In The Twelfth International Conference on Learn-
ing Representations, 2024. 2

[41] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue
Peng, Ping Luo, and Zehuan Yuan. Autoregressive model
beats diffusion: Llama for scalable image generation. arXiv
preprint arXiv:2406.06525, 2024. 1

[42] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang
Zeng. Dreamgaussian: Generative gaussian splatting for ef-
ficient 3d content creation. In The Twelfth International Con-
ference on Learning Representations, 2024. 1, 2

[43] Aaron van den Oord, Oriol Vinyals, and koray kavukcuoglu.
Neural discrete representation learning. In Advances in Neu-
ral Information Processing Systems. Curran Associates, Inc.,
2017. 3

[44] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli.
Image quality assessment: from error visibility to structural
similarity. IEEE Transactions on Image Processing, 13(4):
600–612, 2004. 5

[45] Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma,
Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai
Yu, Chong Ruan, et al. Janus: Decoupling visual encoding
for unified multimodal understanding and generation. arXiv
preprint arXiv:2410.13848, 2024. 1

[46] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng
Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong
Yang. Structured 3d latents for scalable and versatile 3d gen-
eration. arXiv preprint arXiv:2412.01506, 2024. 1, 2, 6, 7

[47] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xin-
grun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun
Huang, and Zheng Liu. Omnigen: Unified image genera-
In Proceedings of the Computer Vision and Pattern
tion.
Recognition Conference, pages 13294–13304, 2025. 1
[48] Enze Xie, Junsong Chen, Yuyang Zhao, Jincheng Yu, Ligeng
Zhu, Chengyue Wu, Yujun Lin, Zhekai Zhang, Muyang Li,
Junyu Chen, et al. Sana 1.5: Efficient scaling of training-time

fied understanding and generation in a visual autoregres-
arXiv preprint
sive multimodal
large language model.
arXiv:2501.12327, 2025. 1

and inference-time compute in linear diffusion transformer.
arXiv preprint arXiv:2501.18427, 2025. 8, 1

[49] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang,
Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie
Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o:
One single transformer to unify multimodal understanding
and generation, 2025. 8, 1

[50] Yi Xin, Qi Qin, Siqi Luo, Kaiwen Zhu, Juncheng Yan, Yan
Tai, Jiayi Lei, Yuewen Cao, Keqi Wang, Yibin Wang, Jin-
bin Bai, Qian Yu, Dengyang Jiang, Yuandong Pu, Haoxing
Chen, Le Zhuo, Junjun He, Gen Luo, Tianbin Li, Ming Hu,
Jin Ye, Shenglong Ye, Bo Zhang, Chang Xu, Wenhai Wang,
Hongsheng Li, Guangtao Zhai, Tianfan Xue, Bin Fu, Xiao-
hong Liu, Yu Qiao, and Yihao Liu. Lumina-dimoo: An omni
diffusion large language model for multi-modal generation
and understanding, 2025. 3, 6

[51] Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang,
Shenghua Gao, and Ying Shan.
Instantmesh: Efficient 3d
mesh generation from a single image with sparse-view large
reconstruction models. arXiv preprint arXiv:2404.07191,
2024. 5

[52] Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke
Shen, Yunhai Tong, and Mengdi Wang. Mmada: Mul-
arXiv preprint
timodal large diffusion language models.
arXiv:2505.15809, 2025. 1, 8

[53] Yifan Yang, Houwen Peng, Yifei Shen, Yuqing Yang, Han
Hu, Lili Qiu, Hideki Koike, et al.
Imagebrush: Learning
visual in-context instructions for exemplar-based image ma-
nipulation. Advances in Neural Information Processing Sys-
tems, 36:48723–48743, 2023. 3

[54] Zebin You, Shen Nie, Xiaolu Zhang, Jun Hu, Jun Zhou,
Zhiwu Lu, Ji-Rong Wen, and Chongxuan Li. Llada-v: Large
language diffusion models with visual instruction tuning,
2025. 1, 3

[55] Lijun Yu, Jose Lezama, Nitesh Bharadwaj Gundavarapu,
Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng,
Agrim Gupta, Xiuye Gu, Alexander G Hauptmann, Boqing
Gong, Ming-Hsuan Yang, Irfan Essa, David A Ross, and Lu
Jiang. Language model beats diffusion - tokenizer is key to
visual generation. In The Twelfth International Conference
on Learning Representations, 2024. 1, 3

[56] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shecht-
man, and Oliver Wang. The unreasonable effectiveness of
deep features as a perceptual metric. In Proceedings of the
IEEE conference on computer vision and pattern recogni-
tion, pages 586–595, 2018. 5

[57] Xuying Zhang, Yupeng Zhou, Kai Wang, Yikai Wang, Zhen
Li, Shaohui Jiao, Daquan Zhou, Qibin Hou, and Ming-
Ming Cheng. Ar-1-to-3: Single image to consistent 3d
object generation via next-view prediction. arXiv preprint
arXiv:2503.12929, 2025. 6, 7, 1

[58] Ziyang Zhou, Pinghui Wang, Zi Liang, Haitao Bai, and
Ruofei Zhang. Cross-modal 3d representation with multi-
view images and point clouds. In Proceedings of the Com-
puter Vision and Pattern Recognition Conference, pages
3728–3739, 2025. 1

[59] Xianwei Zhuang, Yuxin Xie, Yufan Deng, Liming Liang,
Jinghan Ru, Yuguo Yin, and Yuexian Zou. Vargpt: Uni-

A. Additional Experimental Details

A.1. Baseline Setting

Most of the baselines we select—such as Zero-1-to-3 [28], Zero-1-to-3 XL [7, 28], ViVid-1-to-3 [22], Zero123++ [39],
EpiDiff [19], AR-1-to-3 [57], TRELLIS [46], and our own method ViewMask-1-to-3 are trained on data rendered with
perspective projection. Among all these methods, only MV-Adapter [18] is trained using data rendered with orthographic
projection.

A.2. Experimental Settings
We train ViewMask-1-to-3 using 24 H200 GPUs with a total batch size of 384 and a learning rate of 3×10−5. A cosine
learning rate scheduler is adopted, and training is performed under the Zero-2 optimization strategy. During inference, we
set the sampling steps to 20.

A.3. Ground-Truth Rendering

For Zero123++ and AR-1-to-3, the ground-truth images are rendered from 3D assets using their default six viewpoint con-
figuration. MV-Adapter, follows its original training protocol and renders four orthographic views consistent with its ortho-
graphic camera model. For the remaining baselines that support arbitrary multi-view rendering, we use the same viewpoint
and camera parameters as ours to ensure fair comparison.

For models such as Zero123++ that by default generate images with a gray background, we render the corresponding
ground-truth views using the same background color when reporting PSNR, SSIM, and LPIPS. Although some implemen-
tations offer background removal via rembg, this often removes thin or delicate parts of the object as well, which in turn
compromises the reliability of the baseline’s quantitative results.

Table 4. Evaluation on Image Generation Benchmarks (GenEval). Our model achieves strong performance in unified models and is
also comparable to methods among generation-only models.

Model

Architecture

Single Obj. Two Obj. Counting Colors Position Color Attr. Overall

Generation-Only

SDv1.5 [38]
SDv2.1 [38]
DALL-E 2 [37]
DALL-E 3 [3]
LlamaGen [41]
FLUX.1 [Dev] [23]
SDXL [36]
OmniGen [47]
SANA-1.5 [48]

Diffusion
Diffusion
-
-
AR
Diffusion
Diffusion
Diffusion
Diffusion

Unified Understanding & Generation

SEED-X [12]
LWM [26]
Janus [45]
Show-o [49]
VAR-GPT [59]
MMaDA [52]
ViewMask-1-to-3

AR
AR
AR
AR+Diff.
AR
Discrete Diff.
Discrete Diff.

0.97
0.98
0.94
0.96
0.71
0.98
0.98
0.98
0.99

0.97
0.93
0.97
0.98
0.96
0.99
0.99

B. Additional Results

B.1. Image to Multi-view Images

0.38
0.51
0.66
0.87
0.34
0.81
0.74
0.84
0.85

0.58
0.41
0.68
0.80
0.53
0.76
0.88

0.35
0.44
0.49
0.47
0.21
0.74
0.39
0.66
0.77

0.26
0.46
0.30
0.66
0.48
0.61
0.50

0.76
0.85
0.77
0.83
0.58
0.79
0.85
0.74
0.87

0.80
0.79
0.84
0.84
0.83
0.84
0.86

0.04
0.07
0.10
0.43
0.07
0.22
0.15
0.40
0.34

0.19
0.09
0.46
0.31
0.13
0.20
0.60

0.06
0.17
0.19
0.45
0.04
0.45
0.23
0.43
0.54

0.14
0.15
0.42
0.50
0.21
0.37
0.53

0.43
0.50
0.52
0.67
0.32
0.66
0.55
0.68
0.72

0.49
0.47
0.61
0.68
0.53
0.63
0.72

We provide additional I2MV comparison results, as shown in Fig. 8. Our model demonstrates more accurate control over
fine-grained details.

Figure 8. Qualitative comparison on the 3D-Future dataset. Visual results on the 3D-Future dataset, showing that our ViewMask-1-to-3
produces more consistent and realistic novel views compared with recent methods.

B.2. Task Extension

Unified Multimodal Understanding and Generation. During inference, ViewMask-1-to-3 employs task-specific input
templates that leverage discrete tokens for both vision and language modalities. For multimodal understanding (MMU)

tasks, the input sequence is structured as:

[MMU] [SOT] prompt [EOT] [SOI] image tokens [EOI]

[SOT] answer [EOT],

where the text prompt and visual content are presented sequentially to guide the model’s comprehension and response gen-
eration. The red-highlighted tokens correspond to the fully masked answer region, representing the content the model must
produce.

For text-to-image (T2I) task, the sequence follows:

[T2I] [SOT] caption [EOT] [SOI] image tokens [EOI],

where the model autoregressively generates image tokens after the caption prompt.

We quantify the text-to-image generation capability of our model on GenEval, which evaluates object-centric generation
under compositional prompts with diverse object attributes. Table 4 shows that ViewMask-1-to-3 achieves a strong overall
score of 0.72, outperforming the architecturally similar MMaDA by 14% (0.72 vs. 0.63), highlighting the practical potential
of discrete diffusion architectures.

