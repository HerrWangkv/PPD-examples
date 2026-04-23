6
2
0
2

r
p
A
2

]

V
C
.
s
c
[

1
v
1
6
7
1
0
.
4
0
6
2
:
v
i
X
r
a

Control-DINO: Feature Space Conditioning for
Controllable Image-to-Video Diffusion

Edoardo A. Dominici1⋆, Thomas Deixelberger2,3⋆, Konstantinos Vardis1, and
Markus Steinberger2,3

1 Huawei Technologies, Switzerland
2 Huawei Technologies, Austria
3 Graz University of Technology, Austria
https://dedoardo.github.io/projects/Control-DINO

Fig. 1: We introduce a DINO based ControlNet for Image-to-Video diffusion models
that allows using DINOv3 features for structural and semantic guidance, in the image
we show: (1) DL3DV style transfer; (2) VKITTI synthetic scene to real weather transfer;
(3) Video-from-3DGS. We apply a transfer function on the first frames with FLUX.1
KREA [22].

Abstract. Video models have recently been applied with success to
problems in content generation, novel view synthesis, and, more broadly,
world simulation. Many applications in generation and transfer rely on
conditioning these models, typically through perceptual, geometric, or
simple semantic signals, fundamentally using them as generative render-
ers. At the same time, high-dimensional features obtained from large-
scale self-supervised learning on images or point clouds are increasingly
used as a general-purpose interface for vision models. The connection
between the two has been explored for subject specific editing, aligning
and training video diffusion models, but not in the role of a more gen-
eral conditioning signal for pretrained video diffusion models. Features
obtained through self-supervised learning like DINO, contain a lot of
entangled information about style, lighting and semantics of the scene.
This makes them great at reconstruction tasks but limits their genera-
tive capabilities. We show in this paper how we can use the features for

⋆ The authors contributed equally to this work.

 
 
 
 
 
 
2

Dominici, Deixelberger, Vardis and Steinberger

tasks such as video domain transfer and video-from-3D generation. We
introduce a lightweight architecture and training strategy that decou-
ples appearance from other features that we wish to preserve, enabling
robust control for appearance changes such as stylization and relighting.
Furthermore, we show that low spatial resolution can be compensated
by higher feature dimensionality, improving controllability in generative
rendering from explicit spatial representations.

Keywords: video diffusion · controllable generation · self-supervised
features · feature conditioning

1

Introduction

Generative video models have advanced rapidly and now serve as strong back-
bones for editing, stylization, simulation, and world modeling [28, 30, 40, 52]. In
many practical scenarios, generation does not proceed de novo; rather, partial
or complete information about the scene is already available, such as a reference
video, sparse views, or underlying 3D geometry, and the objective is to preserve
this structure while enabling controlled variation.

Most current control pipelines rely on low-level geometric or perceptual in-
puts, including edges, depth maps, colored points, pose skeletons and semantic
masks [27, 28]. These controls are often effective, but each captures only a nar-
row aspect of scene structure, and different tasks tend to require different control
types. This fragmentation makes it difficult to build a unified conditioning inter-
face that transfers cleanly across tasks like video restyling, video-from-3D, and
long-horizon view synthesis.

At the same time, universal self-supervised features such as DINO [4] are
increasingly used as a common interface across vision models. Trained on large-
scale 2D and 3D data, these high-dimensional features encode semantics, geom-
etry, and appearance. They achieve great performance in downstream tasks and
have also shown promise as latent spaces for prediction [1,52] or completion [15].
This makes them an attractive candidate for conditioning video diffusion models
with richer and more reusable control signals than traditional low-level inputs.
However, feature conditioning is not straightforward. The same expressivity that
makes foundation features useful also makes them difficult to tame: appearance,
geometry and other properties are often entangled, making the process challeng-
ing to generalize, especially when training data is limited (Figure 2). Recent work
has shown promising subject-specific video transfer with DINO features [13], but
robust general-purpose conditioning remains underexplored.

We study DINO-based conditioning for video diffusion in a unified frame-
work that operates across both 2D and 3D feature sources. Our key idea is
to disentangle structural information from appearance during training. Specif-
ically, we condition the diffusion model on features extracted from an input
video, while training it to reconstruct appearance-augmented versions of the
same scene. Because identical features correspond to multiple appearance varia-
tions, the model is encouraged to rely on spatial and semantic cues rather than

Control-DINO

3

appearance-specific signals. This induces invariance to appearance changes and
enables controlled manipulation of style, lighting, and texture at inference time.
Furthermore, we observe that low spatial resolution features, when paired
with high feature dimensionality, provide sufficiently rich structural guidance.
This allows us to bridge diverse input modalities, including 3D renderings and
voxel-based representations, to photorealistic video generation.

Our contributions are as follows:

– We propose a general-purpose feature conditioning framework for video dif-
fusion based on self-supervised representations, enabling structural and se-
mantic guidance while decoupling appearance from the conditioning signal.
– We apply the ControlNet to video transfer and video from 3D tasks showing

its versatility and robustness.

– We demonstrate new applications such as rendering low-resolution 3D voxel

structures and connecting foundational 3D models to video diffusion.

2 Related Work

Controllable diffusion via spatial conditioning. Modern generative video mod-
els [2, 12] perform denoising in the latent space of a variational autoencoder,
either with UNets or more recently diffusion transformers [33, 41, 44]. Within
this paradigm, controllable generation is commonly achieved by injecting aux-
iliary signals into the denoising network interleaved between the transformer
blocks. ControlNet [49] introduced a widely adopted design in which a train-
able copy of the encoder injects residuals into the frozen generator via zero-
initialized convolutions. Lighter-weight variants such as T2I-Adapter [29] and
IP-Adapter [45] offer more parameter-efficient or reference-based alternatives.
Several works extend spatial conditioning to video by adapting image-level con-
trol architectures to temporal backbones [5, 11, 16, 23, 24], while alternative de-
signs such as FullDiT [17] tokenize all conditions for joint self-attention at the
cost of full retraining. To date, these methods have predominantly been applied
with geometric signals (edges, depth, pose), yet foundation model features re-
main underexplored as conditioning inputs. Recent methods that generate videos
from 3D representations [19] also rely on edge conditioning for structure transfer.
For a broader overview of controllable video generation, see [27].

Self-supervised features for generation. Self-supervised vision models such as DI-
NOv2 [31] learn patch-level representations that capture object semantics and
support tasks from segmentation to depth estimation, all from a single frozen
backbone. DINOv3 [35] scales this family to denser features that preserve fine-
grained local structure across model sizes. Multiple studies observe that self-
supervised and diffusion features capture complementary information: the for-
mer provide high-level semantic structure, while the latter encode dense spatial
coherence [26, 37, 39, 48]. This motivates conditioning diffusion models on foun-
dation features to supply semantic guidance beyond what the generation process
captures on its own. REPA [47] further shows that frozen DINOv2 features align

4

Dominici, Deixelberger, Vardis and Steinberger

Fig. 2: DINO features are prone to overfitting when simply training a ControlNet. We
study the conditioning effect by dropping the first frame conditioning and only relying
on the text embedding for guidance. We set the prompt to "blue" and show that in our
method we significantly reduce the bias of the ControlNet towards the training data
domain.

with internal DiT representations, though as a training regularizer rather than
an explicit conditioning signal. Directly conditioning on such features was first
explored by Bordes et al. [3] using global DINO embeddings via FiLM modu-
lation. Graikos et al. [10] scale this to local, domain-specific SSL embeddings
with cross-attention for large-image generation in medical and remote-sensing
domains. AnyDoor [6] injects DINOv2 patch tokens into a pretrained backbone
for object identity, yet like the preceding works targets images and not scene-
level spatial control. Self-supervised representations have also been extended to
3D, with methods such as Concerto [51] and Sonata [43] learning dense, multi-
view consistent descriptors from point clouds that remain close to DINO-style
embeddings. We leverage this property by rendering such 3D features into 2D
conditioning maps for our video diffusion adapter.

Foundation features for video. Recent work has begun to explore foundation
features as conditioning signals for video generation and editing. CAGE [9] in-
troduces a video model trained jointly with sparse DINOv2 patches to enable
conditioning for unconditional generation for unsupervised scene composition
and object animation on synthetic scenes. DIVE [13] leverages DINOv2 features
as implicit motion correspondences for subject-driven video editing through per-
video MLP optimization, targeting appearance transfer rather than scene-level
spatial control. Concurrent to our work, Driving with DINO (DwD) [7] injects
DINOv3 features through a ControlNet-style branch into a frozen video back-
bone for sim-to-real driving video translation, introducing PCA-based spectral
pruning tailored to its synthetic-to-real setting. However, none of these works tar-
gets dense foundation-feature conditioning as a general-purpose mechanism on
a pretrained video backbone. We explore this direction with a ControlNet-style
adapter that operates on spatially dense features without per-video optimization,
and evaluate it across video transfer and 3D-guided video generation.

3 Method

The goal of our model is to condition video diffusion on dense features while
making their control practical (Figure 3). We achieve this by injecting founda-

Input image (t = 0)Condition (t = T/2)Real training data only (t = T/2)Mixed training data (t = T/2)Training

Inference

Control-DINO

5

Fig. 3: (Left) During training, DINOv3 features from the original video condition
the trainable Control-DINO branch, while the frozen backbone denoises appearance-
augmented latents of the same scene. (Right) At inference, conditioning features (2D
or 3D-rendered) guide generation through Control-DINO. A transferred first frame sets
the target appearance.

tion features into a frozen video transformer backbone through a ControlNet-
style residual adapter (Sec. 3.1). We train on paired augmented data where the
conditioning features come from the original video but the denoising target is a
structurally similar version of the same scene, enabling the adapter to rely on ge-
ometry and semantics alone (Sec. 3.2). A property of feature-level conditioning is
that it operates at low spatial resolution but high channel dimensionality, which
means the same interface naturally extends to coarse 3D representations where
pixel-level detail is unavailable. Since conditioned video models are increasingly
used for video-from-3D generation [19], we describe how features can be obtained
from 3D data and used as conditioning inputs to our adapter (Sec. 3.3).

3.1 Diffusion and Conditioning

Let x1:T denote an RGB video and z1:T its latent representation under the video
VAE encoder. For a sampled diffusion timestep t, we follow the standard diffusion
formulation:

zt = αtz0 + σtϵ,

ϵ ∼ N (0, I).

(cid:2)∥vt − vθ(zt, t, c)∥2

The denoiser vθ, a video transformer model, is trained to predict the velocity to
(cid:3), where vt = αtϵ −
move the next sample towards L = Ez0,ϵ,t
σtz0 and c denotes the conditioning signal. We consider per-frame conditioning
signals c1:T (edges, depth, semantic maps, or feature maps) pixel aligned with
the output frames. Conditioning signals that are encoded with the video VAE
(e.g., depth or edges) match the latent-space temporal resolution directly, while
signals that bypass the VAE are rescaled spatially and temporally. With the
spatial resolution adapted, we use a temporal adapter with the same causal
architecture as the video’s encoder to map the conditioning temporal length
from 49 to 13 frames to match latent temporal resolution.

2

We condition the video diffusion model with residual connections to the first
16 layers from the output of the transformer blocks that processes the noisy la-
tent concatenated with the features along the feature dimension. The projections

6

Dominici, Deixelberger, Vardis and Steinberger

Zℓ are zero initialized and can be additionally masked. High-resolution masks,
obtained for example by rendering geometry, can also be masked:

hl = hl + M ⊙ Zl(hc

l ),

l = 1, . . . , L.

Here hl is the current hidden state being denoised and hc
state from the transformer blocks.

l is the output hidden

3.2 Feature Conditioning

A side effect of DINO features encoding rich appearance information (as illus-
trated in Figure 2) is that structural signals can be entangled with color, texture,
and style. We therefore aim for the ControlNet to faithfully follow the spatial
structure depicted in the conditioning features while remaining agnostic to the
visual style of the source, enabling applications such as relighting, stylization,
and scene transfer.

We extract dense patch-level features from a frozen DINOv3 vision encoder
D applied to each frame of the source video. For x1:T ∈ RT ×3×H×W we obtain
per-frame feature maps

Fi = D(xi) ∈ R384×60×90,

i = 1, . . . , T,

for patch size 16. To align conditioning with the video diffusion latent grid, we
bicubic-upsample each input frame before DINO encoding so that Fi operates
at the transformer latent spatial scale. This is also the latent space resolution
of the video model’s VAE. These feature maps serve as the conditioning signal
c1:T = F1:T for the ControlNet adapter described in Sec. 3.1.

2

(cid:2)∥vt − vθ(zt, t, F1:T )∥2

Training with augmentation. To force style-invariant conditioning, we construct
training pairs (F1:T , ˜z1:T ), where F1:T are DINO features from the original video
and ˜z1:T are VAE latents of an appearance-augmented version of the same
(cid:3) , where zt is obtained
scene. We train with L = Eϵ,t
from ˜z1:T using the forward diffusion process in Sec. 3.1. During training we
employ a hierarchy of augmentation strategies at varying levels of severity to
mimic appearance changes expected at test time: (i) photometric augmentations
(hue, grayscale, saturation, brightness, contrast, gamma, etc.) applied uniformly
across frames; (ii) 5 common GAN based neural style transfer (cartoon, anime,
etc.) [8, 38]; and (iii) simple blur to measure quality variations. During training,
we sample from a mixture uniformly distributed between real and these 3 groups,
all paired with features extracted from the original unstylized video. At inference,
we apply a more generic transfer function only to the first frame while condi-
tioning on features from the original source domain (real or rendered) [42,50]: as
long as the first-frame transfer preserves structure and semantics, the model can
follow the target appearance consistently over time. Because the same features
correspond to multiple appearances, the model learns to rely on spatial and se-
mantic information rather than appearance cues in the features. We compare
this strategy against PCA-based feature projection in Sec. 4.6.

Control-DINO

7

3.3

3D Feature Conditioning

Although DINO features are typically extracted on relatively coarse patch grids
(low spatial resolution), they remain highly informative because each token is
high-dimensional and semantically rich. In practice, this means that a h × w
grid with C=384 channels can preserve object identity, part-level layout, and
scene semantics even when fine pixel detail is absent. This representation is
complementary to geometry controls: geometry anchors where content should
appear, while DINO channels encode what the content is and how regions relate
semantically. To exploit this complementarity, we align both signals at the same
low spatial resolution and use them jointly as conditioning inputs. In our method,
the primary geometric representation is a voxel grid, which provides a stable
3D structural prior across views and time, but also other 3D structures like
point clouds or 3DGS [18] can be used. This avoids expensive full-resolution
conditioning: voxel geometry enforces camera-consistent structure and DINO
features stabilize subject/background semantics. Because both inputs are low-
resolution, compute remains modest; because both are high-level, the model
still receives strong guidance for temporally coherent generation. Notably, high-
resolution geometry is not required in our setting: low-resolution voxel cues are
sufficient and can be obtained from either LiDAR reconstructions or occupancy
prediction models.

DINO-like features from uncolored geometry. Importantly, useful feature priors
do not require RGB textures at inference time. Recent 2D-3D self-supervised
methods, such as Concerto [51], can produce dense geometric descriptors from
uncolored geometry like point clouds that remain close to DINO-style semantic
embeddings after projection. We therefore treat these geometry-derived descrip-
tors as a drop-in conditioning signal: they can be rendered to the target view,
projected to the ControlNet token grid, and fused with standard controls ex-
actly as in Figure 5. This enables controllable generation in settings where only
geometry is available, while retaining semantic consistency typically associated
with appearance-based foundation features.

4 Experiments

4.1 Datasets

Our work combines video diffusion conditioning from both 2D and 3D represen-
tations, with a focus on generative rendering. To emphasize generation quality
and cross-dataset generalization, we train on DL3DV [25] and evaluate primarily
on Tanks & Temples [20], which provides longer and smoother camera trajec-
tories. From these trajectories we sample two evaluation settings, stride 8 and
stride 16. For Table 3, we also report results on DL3DV140 [25]. In total, we train
on about 4.5M unique frames and evaluate on around 10k unique frames. The
larger camera motion in Tanks & Temples stresses both generative quality and
conditioning robustness. For 3D rendering experiments, we fine-tune the same

8

Dominici, Deixelberger, Vardis and Steinberger

Fig. 4: Results for DINO stylization and dramatic light changes (GT Frame is in the
bottom left of the image). Our method is able to strongly retain the original geometry
and semantics while allowing for appearance and color to change.

Scene 1

Scene 2

Scene 3

Scene 4

Fig. 5: Visualisation of 3D structure used on ScanNet++ first frames. Top row: voxel
RGB renderings. Bottom row: mesh video renderings.

base model on ScanNet++ [46] and evaluate on its official validation split. On
ScanNet++, we sample training clips at stride 1, yielding nearly 8k scene sam-
ples with 49 frames each, and evaluate video-from-3D pipelines that construct
DINO and DINO-like conditioning features from explicit 3D geometry (Table 4).

4.2 Model Details

We implement our approach on CogVideoX-5B-I2V [44] with residual connec-
tions [49] on the first L=16 transformer blocks, using dense DINOv3-ViT-S/16
patch features as input. The features are bicubically upscaled 2x and then en-
coded. To adapt the latent space dimension to the one of the hidden state, two
causal Conv3D layers temporally compress the sequence from 49 to 13 frames,
matching the VAE latent temporal resolution (and architecture), and the result
is concatenated channel-wise with the VAE latents before being fed into the con-

Control-DINODINO (PCA)Wan 2.2 CannyCannyWan 2.2 DepthDepthAnyV2VGT Framet=0t=T/2t=Tt=0t=T/2t=TCond.Painting (InstantStyle)RelightingControl-DINO

9

Table 1: Quantitative comparison on T&T Appearance Augmented. All models are
trained on DL3DV and evaluated on the T&T (for two frame sampling strides) with a
combination of 15 different lighting or stylistic variations per scene different from the
training data. Overall our method produces the most 3D consistent videos, especially
for longer camera trajectory, while maintaining a competitive CLIP similarity with the
first frame in the sequence. Our same model but trained only on real data performs
much worse in style following.

Tanks and Temples - Out of domain - Transfer

Consistency

Video Quality (VBench)

CLIP Similarity COLMAP

Method

Subject ↑ Background ↑ Motion Smoothness ↑ Aesthetic ↑ Imaging ↑ CLIP Sim. ↑ Registration ↑

Wan2.2 Fun Depth [41]
Wan2.2 Fun Canny [41]
AnyV2V [21]

0.8853
0.8778
0.8124

Control-DINO Real Only 0.8767
0.8913
Control-DINO

Wan2.2 Fun Depth [41]
Wan2.2 Fun Canny [41]
AnyV2V [21]

0.8222
0.8226
0.7621

Control-DINO Real Only 0.8544
0.8461
Control-DINO

0.8916
0.8848
0.8818

0.8465
0.9060

0.8527
0.8512
0.8579

0.8906
0.8862

Stride = 8

0.9421
0.9599
0.9258

0.9680
0.9640

Stride = 16

0.9410
0.9357
0.9162

0.9396
0.9361

0.5388
0.5325
0.5390

0.5533
0.5610

0.5044
0.5104
0.5072

0.5485
0.5508

0.6880
0.7568
0.6590

0.7372
0.7146

0.6497
0.7235
0.6299

0.6908
0.7109

0.4626
0.5020
0.4875

0.4408
0.4717

0.4804
0.4630
0.4864

0.4406
0.4707

0.8430
0.9770
0.5470

0.8840
0.9530

0.4956
0.5918
0.1875

0.9911
0.9932

Table 2: We also compare against the only publicly available segmentation ControlNet,
we show that our method produces competitive results, especially with larger camera
motions, even against models larger in terms of scale, training data and architecture.

Tanks and Temples - Out of domain - Transfer
Video Quality (VBench)
Consistency

CLIP Similarity Coverage

Method

Subject ↑ Background ↑ Motion Smoothness ↑ Aesthetic ↑ Imaging ↑ CLIP Sim. ↑ RegRate ↑

Cosmos-Segmentation (S=8)
Control-DINO (S=8)

0.9128
0.8913

Cosmos-Segmentation (S=16) 0.8315
0.8461
Control-DINO (S=16)

0.9191
0.9060

0.8958
0.8862

0.9678
0.9640

0.9475
0.9361

0.4987
0.5610

0.4547
0.5508

0.6666
0.7146

0.6190
0.7109

0.7306
0.4717

0.7147
0.4707

0.9374
0.9530

0.7755
0.9932

ditioning branch. We set the hidden dimension of the conditioning branch to 256
to further reduce the risk of overfitting. The last linear layer is zero initialized
for more stable training. At inference time, we scale the conditioning residuals
by 0.8.

We train the temporal adapter and lightweight transformer blocks from
scratch with AdamW (β1=0.9, β2=0.95) at a peak learning rate of 2×10−4 with
cosine annealing and 500 warmup steps, a batch size of 8, and gradient clipping
at 1.0. We apply conditioning dropout during training, setting the conditioning
input to zero with probability 0.1.

4.3 Metrics

For the main benchmark on out-of-domain evaluation (Table 1) we assess video
quality using VBench [14], reporting consistency, motion and imaging scores.
Additionally we measure 3D consistency through COLMAP-based reconstruc-

10

Dominici, Deixelberger, Vardis and Steinberger

Table 3: We illustrate the tradeoff between structural control signals and generation
determinism. DINOv3 features jointly encode geometry and appearance, constraining
the diffusion output more than other signals which allow more interpretation, including
worse generations which results in lower video quality.

DL3DV140 - In domain - Real
Reconstruction

Video Quality (VBench)

Method

FID ↓ Latent Var. ↓ CLIP Var. mIOU ↑ Flickering ↑ Aesthetic ↑ Imaging ↑

Wan2.2 Depth [41] 47.98
Wan2.2 Canny [41] 37.65

0.5115
0.4787

0.0151
0.0150

0.0362
0.0322

0.9192
0.9154

0.4343
0.4496

0.5469
0.6169

Control-DINO

20.23

0.1944

0.0051

0.2992

0.9208

0.5133

0.6283

Table 4: Video-from-3D comparison on ScanNet++. We evaluate geometry-driven
3D-derived conditioning methods. We compare quantitative results for image-space
baselines (Wan and 2D Control-DINO) and report results for 3D-derived methods.

ScanNet++ - In domain - Video from 3D

Reconstruction

Video Quality (VBench)

Method

PSNR ↑ SSIM ↑ LPIPS ↓ FID ↓ Subject ↑ Background ↑ Flickering ↑ Motion Smooth ↑ Aesthetic ↑ Imaging ↑

Image-space baselines

W2.2 Depth
W2.2 Canny
Control-DINO video (reference)

18.23
19.82
27.93

0.7237
0.7968
0.9084

0.2782 92.1072 0.9027
0.1987 72.1929 0.9001
0.0963 38.5853 0.9183

0.9234
0.9251
0.9256

21.41
Control-DINO from Mesh
22.80
Control-DINO Voxel
21.19
Control-DINO Concerto
Control-DINO Concerto (inpainted) 19.66
27.53
Control-DINO 3DGS

0.8010
0.8153
0.7805
0.7542
0.9031

0.1807 67.1730 0.9145
0.1572 65.3810 0.9216
0.1752 67.9680 0.9195
0.2060 78.8411 0.9122
0.0948 36.7027 0.9178

0.9242
0.9293
0.9263
0.9267
0.9226

3D-derived methods (ours)

0.9680
0.9590
0.9662

0.9647
0.9658
0.9638
0.9635
0.9659

0.9881
0.9894
0.9931

0.9919
0.9928
0.9926
0.9917
0.9932

0.4111
0.4100
0.4187

0.4198
0.4188
0.4200
0.4184
0.4209

0.5714
0.6625
0.5821

0.5852
0.5979
0.6120
0.6326
0.5949

tion errors and registration success rate. We quantify style transfer fidelity via
CLIP style similarity with the first frame image. On the validation split of the
training set (Table 3) we additionally evaluate model generalizability by looking
at the distribution of the output latent space and features. For the video from
3D tasks, we focus on standard reconstruction metrics when evaluating video
from 3D.

4.4 Baselines

Our main baselines use alternative conditioning signals on image-to-video dif-
fusion models with similar architecture and scale. We use the official Wan2.2-
Fun-5B-Control (I2V) checkpoints with Canny and depth conditioning [41]. We
also compare against the publicly available semantic ControlNet from Cosmos-
Transfer-2.5 [30], while noting that Cosmos base model and ControlNet are sub-
stantially larger models. For domain transfer, our setup is closest to AnyV2V [21],
which uses DDIM-inverted latents [36] for appearance transfer during diffusion.
For first-frame appearance transfer baselines, we use IC-Light for relighting [50],
InstantStyle for style transfer [42] picking 10 styles from the default ones, and
five additional cartoon neural styles that were excluded from training.

Control-DINO

11

Table 6: We ablate architectural choices
for the model. In particular we find fea-
tures after PCA projection to be less ef-
fective than raw features.

Architectural ablation on
DL3DV140.

Method

FID ↓ Subj. Cons. ↑ BG Cons. ↑ Aesthetic ↑ Imaging ↑
0.955
71.67
Control-DINO
0.933
w/o Spatial Mixing
177.80
0.940
w/ style invariant PCA 89.14
0.920
108.87
w/ Tail Drop (k=64)
0.908
156.67
w/ Tail Drop (k=8)

0.959
0.919
0.940
0.918
0.845

0.542
0.495
0.524
0.494
0.490

0.504
0.474
0.479
0.459
0.468

Table 5: Effect of projection basis on
DINOv3 feature disentanglement. Cosine
similarity measures agreement between
real and stylized scene features after pro-
jection, and explained variance reports re-
tained feature variance.

Projection basis disentanglement

Method

Cosine Similarity ↑ Explained Variance (%) ↑

No projection (384-d)

Sim2Real reference (DwD [7])

Standard PCA
Style-Invariant PCA
Bottom Eigenvectors
Random Orthogonal

0.9194

0.9598

0.9110
0.9481
0.9929
0.9226

100.0

–

72.5
30.4
2.9
16.7

4.5 Experiments and Results

Representation and transfer. We evaluate representation-level conditioning
for long-horizon transfer on the out-of-domain Tanks & Temples benchmark.
Quantitative results in Table 1 and Table 2 show that DINO-based conditioning
provides strong and stable guidance under large camera motion, with improved
temporal and structural consistency across stride-8 and stride-16 settings. As
also illustrated in Figure 9, our method has lower generative flexibility than
weaker controls, but this is compensated by substantially higher output quality
and consistency. Camera control is stronger because the model can offload part of
the generative burden to the base diffusion prior, allowing the diffusion pathway
to focus on finer structural details (Figure 4).

Our appearance-decoupled training makes this representation practical for
transfer: the same conditioning interface supports RGB-to-stylized transfer and
geometry-driven transfer (e.g., mesh-to-real), as discussed in Sec. 3.3. Qualitative
examples are shown in Figure 7 and Figure 9, with additional results in Figure 8.
We observe particularly strong relighting behavior, consistent with the built-in
invariances of DINO features and the pixel-aligned geometric constraints of the
task.

Video from 3D. We evaluate three 3D-to-video conditioning pipelines that
all use a ControlNet branch driven by DINO-like feature maps. For the mesh
setting, we render untextured meshes from target camera trajectories, upscale
the rendered views, and extract DINOv3-S features as conditioning maps with
shape 384 × 60 × 90. Because mesh renderings contain holes and missing regions,
we additionally encode a binary visibility mask with the CogVideoX VAE and
provide it as an auxiliary conditioning signal, so the diffusion model can explicitly
inpaint invalid regions instead of overfitting to artifacts.

For the voxel setting, we investigate whether low-resolution geometry with
color cues is sufficient. Starting from an input video sequence, we reconstruct
geometry from a dense point cloud and associate DINO-encoded image features
to 3D points, then voxelize this feature-augmented point cloud. The resulting

12

Dominici, Deixelberger, Vardis and Steinberger

GT

2D

Voxel

Mesh

Concerto

Concerto
inpainted

3DGS

Scene 1

5
3
=
t

9
4
=
t

0
=
t

5
3
=
t

9
4
=
t

0
=
t

5
3
=
t

9
4
=
t

0
=
t

GT

2D

Voxel

Mesh

Concerto

Concerto
inpainted

3DGS

Scene 2

GT

2D

Voxel

Mesh

Concerto

Concerto
inpainted

3DGS

Scene 3

Fig. 6: Qualitative comparison at matched time steps for four scenes of the ScanNet++
dataset with conditioning signal example. We show generated outputs at t=35 and t=49
for 2D, voxel, Concerto, mesh and 3DGS conditioning. The last row at t=0 shows the
conditioning dino feature maps signal for each method, while the GT frame at t=0 is
the common starting frame for all methods.

Control-DINO

13

Fig. 7: We compare to related work on 2 scenes from DL3DV140 for reconstruction.
Our method can faithfully regenerate the output with minor lighting and color vari-
ations while the other methods have significantly weaker semantic following and only
preserve the approximate scene layout.

Fig. 8: Additional 2D transfer results on different scenes and different validation styles.

voxel grid is rendered back to the target views to produce DINO-like conditioning
maps; as in the mesh case, we propagate hole masks so the model can ignore
unobserved areas during denoising, visualized in Figure 5.

Finally, for Concerto-based conditioning we target the realistic case where
only sparse geometry is available and RGB coverage is limited to one input
view. We extract Concerto point-cloud descriptors, voxelize them, and render
Concerto feature maps as the control signal. This setup allows generation from
sparse, mostly uncolored 3D observations while preserving semantic consistency.
Quantitative comparisons are reported in Table 4, and qualitative conditioning
examples are shown in Figure 6.

The quantitative results show that our approach outperforms state-of-the-art
depth- and edge-conditioned baselines (notably Wan2.2) in reconstruction qual-
ity for image-based conditioning. We also report quantitative results for our 3D-
derived pipelines, but we avoid direct one-to-one comparison with image-based
methods because the application setting is different. Among our 3D variants,

GTt=2t=T/2t=T1t=2t=T/2t=T1Control-DINOWan 2.2 CannyWan 2.2 DepthFrancis  Drawingt=0t=T/2t=T1Horse  Soft StudioCourthouse  MondrianTruck  Mosaict=0t=T/2t=T1Temple  Bright NeonTrain  Arcane14

Dominici, Deixelberger, Vardis and Steinberger

Control-DINO performs best when full color information is available, consis-
tent with the 2D or 3DGS rendering setting where appearance cues are strong.
Methods based on untextured 3D geometry show lower reconstruction quality, as
expected, because missing appearance information reduces feature fidelity and
weakens structural disambiguation during generation. Furthermore, the sparsity
of the 3D information tends to introduce holes that provide no conditioning
signal.

Finally, Table 4 reports results for Concerto-based conditioning in two regimes:
with and without color information (the uncolored point-cloud setting). Despite
the stronger input sparsity, the color-free Concerto setup still provides usable ap-
pearance priors and enables plausible scene generation, highlighting its practical
value for generation-focused use-cases rather than strict reconstruction.

4.6 Ablations

We first ablate the effect of training data composition (real-only vs. mixed
real+augmented). As shown in Table 1, mixing data can slightly reduce genera-
tion quality, while improving transfer fidelity. We ablate the feature conditioning
against two PCA decomposition strategies (Table 6): (1) Given paired real and
stylized DINOv3 features freal and fstyle, we compute difference vectors d =
freal − fstyle, perform eigendecomposition on their covariance S = (1/M )DT D
to obtain the top-K style directions Vk, project them out via P = I −VkV T
k , then
run standard PCA on the cleaned features P freal to extract the top-D structure
directions as our final projection basis; (2) A tail dropping strategy described
in [13] where we first compute a 64-component PCA basis during training, then
randomly retain only the first k components with k ∈ {8, 16, 32, 64}. At inference,
we ablate against k = 8 and k = 64. We have found (and show in Table 5) that
PCA decomposition, in part because of the already small DINO model we are
using, is quite disruptive of the information signal. Even when computing a style
invariant decomposition with paired data the amount of information explained
remains < 40%, which indicates substantial information loss. We consider the
necessity of spatially mixing the signal with the hidden state in a transformer
block by replacing it with a simple linear projection directly into the residual
(like [13]). Interestingly, we have found that the quality of the images remains
quite high but the model has difficulty following the conditioning under large
camera motion.

Control-DINO

15

Fig. 9: We show two more applications of Control-DINO: (a) without lighting specific
changes we are able to produce relighting results which are comparable to methods
designed for the task [53]; (b) We can also perform domain transfer from simulated
videos to photorealistic ones. The model has never seen sim/real pairs of driving videos
but still succeeds in preserving the real appearance while being controlled by sim
features.

5 Discussion

We introduced a DINO-based conditioning mechanism for video diffusion, show-
ing that dense self-supervised features can provide robust structural control while
allowing flexible appearance variation and new applications of generative ren-
dering. The same interface supports 2D transfer tasks such as stylization and
relighting, as well as 3D-guided generation from meshes, voxels, and point-cloud
descriptors. Several limitations remain: appearance and semantics are not always
perfectly disentangled: the model occasionally leaks source appearance, particu-
larly for styles far from the training distribution. Lower guidance helps, but we
report all results at a fixed guidance scale for fairness.

More broadly, using high-dimensional features as conditioning implies a trade-
off with generative flexibility (Table 3): stronger control restricts the diversity
of plausible outputs. For 3D-derived conditioning, generation quality depends
on the fidelity of the underlying reconstruction and on how well the projected
features approximate the 2D DINO space the model was trained on.

We believe that learning spatial weighting of the conditioning signal, as well
as extending to other foundation feature spaces, could further improve the bal-

16

Dominici, Deixelberger, Vardis and Steinberger

ance between control and generation quality. We release source code and weights
for the community to build upon.

Control-DINO

17

References

1. Baldassarre, F., Szafraniec, M., Terver, B., Khalidov, V., Massa, F., LeCun, Y.,
Labatut, P., Seitzer, M., Bojanowski, P.: Back to the features: Dino as a foundation
for video world models. arXiv preprint arXiv:2507.19468 (2025) 2

2. Blattmann, A., Dockhorn, T., Kulal, S., Mendelevitch, D., Kilian, M., Lorenz, D.,
Levi, Y., English, Z., Voleti, V., Letts, A., Jampani, V., Rombach, R.: Stable video
diffusion: Scaling latent video diffusion models to large datasets (2023) 3

3. Bordes, F., Balestriero, R., Vincent, P.: High fidelity visualization of what your
self-supervised representation knows about. Transactions on Machine Learning Re-
search (2022) 4

4. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A.:
Emerging properties in self-supervised vision transformers. In: Proceedings of the
IEEE/CVF International Conference on Computer Vision (ICCV). pp. 9650–9660
(October 2021) 2

5. Chen, W., Wu, J., Xie, P., Wu, H., Li, J., Xia, X., Xiao, X., Lin, L.: Control-a-video:

Controllable text-to-video generation with diffusion models (2023) 3

6. Chen, X., Huang, L., Liu, Y., Shen, Y., Zhao, D., Zhao, H.: Anydoor: Zero-shot
object-level image customization. In: Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR). pp. 6593–6602 (June 2024) 4
7. Chen, X., Zhang, C., Fu, C., Yang, Z., Zhou, K., Zhang, Y., He, J., Zhang, Y., Sun,
M., Wang, Z., Dong, Z., Long, X., Meng, L.: Driving with dino: Vision foundation
features as a unified bridge for sim-to-real generation in autonomous driving. arXiv
preprint arXiv:2602.06159 (2026) 4, 11

8. Chen, Y., Lai, Y.K., Liu, Y.J.: Cartoongan: Generative adversarial networks for
photo cartoonization. In: IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) (2018) 6

9. Davtyan, A., Sameni, S., Ommer, B., Favaro, P.: Cage: Unsupervised visual com-
position and animation for controllable video generation. Proceedings of the AAAI
Conference on Artificial Intelligence 39(15), 16163–16171 (Apr 2025) 4

10. Graikos, A., Yellapragada, S., Le, M.Q., Kapse, S., Prasanna, P., Saltz, J., Sama-
ras, D.: Learned representation-guided diffusion models for large-image generation.
In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition. pp. 8532–8542 (2024) 4

11. Guo, Y., Yang, C., Rao, A., Agrawala, M., Lin, D., Dai, B.: Sparsectrl: Adding
sparse controls to text-to-video diffusion models. In: Leonardis, A., Ricci, E., Roth,
S., Russakovsky, O., Sattler, T., Varol, G. (eds.) Computer Vision – ECCV 2024.
pp. 330–348. Springer Nature Switzerland, Cham (2025) 3

12. Ho, J., Salimans, T., Gritsenko, A., Chan, W., Norouzi, M., Fleet, D.J.: Video
diffusion models. In: Koyejo, S., Mohamed, S., Agarwal, A., Belgrave, D., Cho,
K., Oh, A. (eds.) Advances in Neural Information Processing Systems. vol. 35, pp.
8633–8646. Curran Associates, Inc. (2022) 3

13. Huang, Y., Xiong, W., Zhang, H., Chen, C., Liu, J., Yan, M., Chen, S.: Dive:
Taming dino for subject-driven video editing. In: Proceedings of the IEEE/CVF
International Conference on Computer Vision (ICCV). pp. 16004–16014 (October
2025) 2, 4, 14

14. Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin,
Q., Chanpaisit, N., Wang, Y., Chen, X., Wang, L., Lin, D., Qiao, Y., Liu, Z.:
VBench: Comprehensive Benchmark Suite for Video Generative Models . In: 2024
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
pp. 21807–21818. IEEE Computer Society, Los Alamitos, CA, USA (Jun 2024) 9

18

Dominici, Deixelberger, Vardis and Steinberger

15. Jevtić, A., Reich, C., Wimbauer, F., Hahn, O., Rupprecht, C., Roth, S., Cre-
mers, D.: Feed-forward SceneDINO for unsupervised semantic scene completion.
In: IEEE/CVF International Conference on Computer Vision (ICCV) (2025) 2
16. Jiang, Z., Han, Z., Mao, C., Zhang, J., Pan, Y., Liu, Y.: Vace: All-in-one video
creation and editing. In: Proceedings of the IEEE/CVF International Conference
on Computer Vision (ICCV). pp. 17191–17202 (October 2025) 3

17. Ju, X., Ye, W., Liu, Q., Wang, Q., Wang, X., Wan, P., Zhang, D., Gai, K., Xu, Q.:
Fulldit: Video generative foundation models with multimodal control via full at-
tention. In: Proceedings of the IEEE/CVF International Conference on Computer
Vision (ICCV). pp. 15737–15747 (October 2025) 3

18. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for
real-time radiance field rendering. ACM Transactions on Graphics 42(4) (July
2023), https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ 7
19. Kim, G., Han, J., Cho, S.: Videofrom3d: 3d scene video generation via comple-
mentary image and video diffusion models. In: SIGGRAPH Asia 2025 Conference
Papers (SA Conference Papers ’25). pp. 1–11. ACM, Hong Kong, Hong Kong (2025)
3, 5

20. Knapitsch, A., Park, J., Zhou, Q.Y., Koltun, V.: Tanks and temples: Benchmark-
ing large-scale scene reconstruction. In: Proceedings of the IEEE International
Conference on Computer Vision (ICCV) (2017) 7

21. Ku, M., Wei, C., Ren, W., Yang, H., Chen, W.: Anyv2v: A tuning-free framework
for any video-to-video editing tasks. Transactions on Machine Learning Research
(2024), reproducibility Certification 9, 10

22. Lee, S., Ebbecke, T., Millon, E., Beddow, W., Zhuo, L., García-Ferrero, I., Es-
parraguera, L., Petrescu, M., Saß, G., Menezes, G., Perez, V.: Flux.1 krea [dev].
https://github.com/krea-ai/flux-krea (2025) 1

23. Liang, F., Wu, B., Wang, J., Yu, L., Li, K., Zhao, Y., Misra, I., Huang, J.B.,
Zhang, P., Vajda, P., Marculescu, D.: Flowvid: Taming imperfect optical flows for
consistent video-to-video synthesis. In: Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR). pp. 8207–8216 (June 2024)
3

24. Lin, H., Cho, J., Zala, A., Bansal, M.: Ctrl-adapter: An efficient and versatile
framework for adapting diverse controls to any diffusion model. In: The Thirteenth
International Conference on Learning Representations (2025) 3

25. Ling, L., Sheng, Y., Tu, Z., Zhao, W., Xin, C., Wan, K., Yu, L., Guo, Q., Yu,
Z., Lu, Y., Li, X., Sun, X., Ashok, R., Mukherjee, A., Kang, H., Kong, X., Hua,
G., Zhang, T., Benes, B., Bera, A.: Dl3dv-10k: A large-scale scene dataset for
deep learning-based 3d vision. In: Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR). pp. 22160–22169 (June 2024)
7

26. Luo, G., Dunlap, L., Park, D.H., Holynski, A., Darrell, T.: Diffusion hyperfeatures:
Searching through time and space for semantic correspondence. In: Advances in
Neural Information Processing Systems (2023) 3

27. Ma, Y., Feng, K., Hu, Z., Wang, X., Wang, Y., Zheng, M., Wang, B., Wang, Q.,
He, X., Wang, H., Zhu, C., Liu, H., He, Y., Wang, Z., Li, Z., Li, X., Han, S., Guo,
Y., Liu, W., Xu, D., Zhang, L., Chen, Q.: Controllable video generation: A survey
(2026) 2, 3

28. Melnik, A., Ljubljanac, M., Lu, C., Yan, Q., Ren, W., Ritter, H.: Video diffu-
sion models: A survey. Transactions on Machine Learning Research (2024), survey
Certification 2

Control-DINO

19

29. Mou, C., Wang, X., Xie, L., Wu, Y., Zhang, J., Qi, Z., Shan, Y., Qie, X.: T2i-
adapter: Learning adapters to dig out more controllable ability for text-to-image
diffusion models (2023) 3

30. NVIDIA: Cosmos world foundation model platform for physical ai (2025) 2, 10
31. Oquab, M., Darcet, T., Moutakanni, T., Vo, H.V., Szafraniec, M., Khalidov, V.,
Fernandez, P., HAZIZA, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N.,
Galuba, W., Howes, R., Huang, P.Y., Li, S.W., Misra, I., Rabbat, M., Sharma, V.,
Synnaeve, G., Xu, H., Jegou, H., Mairal, J., Labatut, P., Joulin, A., Bojanowski,
P.: DINOv2: Learning robust visual features without supervision. Transactions on
Machine Learning Research (2024), featured Certification 3

32. Parmar, G., Zhang, R., Zhu, J.Y.: On aliased resizing and surprising subtleties in

gan evaluation. In: CVPR (2022) 21

33. Peebles, W., Xie, S.: Scalable diffusion models with transformers. In: Proceedings of
the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 4195–
4205 (October 2023) 3

34. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G.,
Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from
natural language supervision. In: International conference on machine learning. pp.
8748–8763. PmLR (2021) 21

35. Siméoni, O., Vo, H.V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov,
V., Szafraniec, M., Yi, S., Ramamonjisoa, M., Massa, F., Haziza, D., Wehrstedt,
L., Wang, J., Darcet, T., Moutakanni, T., Sentana, L., Roberts, C., Vedaldi, A.,
Tolan, J., Brandt, J., Couprie, C., Mairal, J., Jégou, H., Labatut, P., Bojanowski,
P.: Dinov3 (2025) 3

36. Song, J., Meng, C., Ermon, S.: Denoising diffusion implicit models. In: International

Conference on Learning Representations (2021) 10

37. Stracke, N., Baumann, S.A., Bauer, K., Fundel, F., Ommer, B.: Cleandift: Diffusion
features without noise. In: Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (2025) 3

38. Tachibana, Y., Tomita, S., Tanno, T., Kaji, H., Nagai, T.: Animeganv2: Rethinking
the high-quality anime image generation. In: IEEE/CVF Conference on Computer
Vision and Pattern Recognition Workshops (CVPRW) (2021) 6

39. Tang, L., Jia, M., Wang, Q., Phoo, C.P., Hariharan, B.: Emergent correspondence
from image diffusion. In: Thirty-seventh Conference on Neural Information Pro-
cessing Systems (2023) 3

40. Team, R., Gao, Z., Wang, Q., Zeng, Y., Zhu, J., Cheng, K.L., Li, Y., Wang, H.,
Xu, Y., Ma, S., Chen, Y., Liu, J., Cheng, Y., Yao, Y., Zhu, J., Meng, Y., Zheng,
K., Bai, Q., Chen, J., Shen, Z., Yu, Y., Zhu, X., Shen, Y., Ouyang, H.: Advancing
open-source world models (2026) 2

41. Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.W., Chen, D., Yu, F., Zhao,
H., Yang, J., Zeng, J., Wang, J., Zhang, J., Zhou, J., Wang, J., Chen, J., Zhu, K.,
Zhao, K., Yan, K., Huang, L., Feng, M., Zhang, N., Li, P., Wu, P., Chu, R., Feng,
R., Zhang, S., Sun, S., Fang, T., Wang, T., Gui, T., Weng, T., Shen, T., Lin, W.,
Wang, W., Wang, W., Zhou, W., Wang, W., Shen, W., Yu, W., Shi, X., Huang,
X., Xu, X., Kou, Y., Lv, Y., Li, Y., Liu, Y., Wang, Y., Zhang, Y., Huang, Y., Li,
Y., Wu, Y., Liu, Y., Pan, Y., Zheng, Y., Hong, Y., Shi, Y., Feng, Y., Jiang, Z.,
Han, Z., Wu, Z.F., Liu, Z.: Wan: Open and advanced large-scale video generative
models (2025) 3, 9, 10

42. Wang, H., Spinelli, M., Wang, Q., Bai, X., Qin, Z., Chen, A.: Instantstyle: Free

lunch towards style-preserving in text-to-image generation (2024) 6, 10

20

Dominici, Deixelberger, Vardis and Steinberger

43. Wu, X., DeTone, D., Frost, D., Shen, T., Xie, C., Yang, N., Engel, J., Newcombe,
R., Zhao, H., Straub, J.: Sonata: Self-supervised learning of reliable point rep-
resentations. In: Proceedings of the Computer Vision and Pattern Recognition
Conference (CVPR). pp. 22193–22204 (June 2025) 4

44. Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W.,
Zhang, X., Feng, G., Yin, D., Yuxuan.Zhang, Wang, W., Cheng, Y., Xu, B., Gu,
X., Dong, Y., Tang, J.: Cogvideox: Text-to-video diffusion models with an expert
transformer. In: The Thirteenth International Conference on Learning Represen-
tations (2025) 3, 8, 21

45. Ye, H., Zhang, J., Liu, S., Han, X., Yang, W.: Ip-adapter: Text compatible image

prompt adapter for text-to-image diffusion models (2023) 3

46. Yeshwanth, C., Liu, Y.C., Nießner, M., Dai, A.: Scannet++: A high-fidelity dataset
of 3d indoor scenes. In: Proceedings of the IEEE/CVF International Conference
on Computer Vision. pp. 12–22 (2023) 8

47. Yu, S., Kwak, S., Jang, H., Jeong, J., Huang, J., Shin, J., Xie, S.: Representation
alignment for generation: Training diffusion transformers is easier than you think.
In: The Thirteenth International Conference on Learning Representations (2025)
3

48. Zhang, J., Herrmann, C., Hur, J., Cabrera, L.P., Jampani, V., Sun, D., Yang, M.H.:
A tale of two features: Stable diffusion complements DINO for zero-shot semantic
correspondence. In: Thirty-seventh Conference on Neural Information Processing
Systems (2023) 3

49. Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to text-to-image dif-
fusion models. In: 2023 IEEE/CVF International Conference on Computer Vision
(ICCV). pp. 3813–3824 (2023) 3, 8

50. Zhang, L., Rao, A., Agrawala, M.: Scaling in-the-wild training for diffusion-based
illumination harmonization and editing by imposing consistent light transport. In:
The Thirteenth International Conference on Learning Representations (2025) 6,
10

51. Zhang, Y., Wu, X., Lao, Y., Wang, C., Tian, Z., Wang, N., Zhao, H.: Con-
certo: Joint 2d-3d self-supervised learning emerges spatial representations. In: The
Thirty-ninth Annual Conference on Neural Information Processing Systems (2025)
4, 7

52. Zhou, G., Pan, H., Lecun, Y., Pinto, L.: DINO-WM: World models on pre-trained
visual features enable zero-shot planning. In: Singh, A., Fazel, M., Hsu, D., Lacoste-
Julien, S., Berkenkamp, F., Maharaj, T., Wagstaff, K., Zhu, J. (eds.) Proceedings
of the 42nd International Conference on Machine Learning. Proceedings of Machine
Learning Research, vol. 267, pp. 79115–79135. PMLR (13–19 Jul 2025) 2

53. Zhou, Y., Bu, J., Ling, P., Zhang, P., Wu, T., Huang, Q., Li, J., Dong, X., Zang, Y.,
Cao, Y., Rao, A., Wang, J., Niu, L.: Light-a-video: Training-free video relighting
via progressive light fusion. arXiv preprint arXiv:2502.08590 (2025) 15

6 Supplemental

6.1 Additional Evaluation Details

For Tanks and Temples we sample 3 videos from 10 scenes for each one of the 3
times 5 transfer modes for a total of approximately 20 thousand unique frames,
including some crashes and failures. For DL3DV140, we sample one video per

Control-DINO

21

scene with stride 1 and a random starting point, for a single style and a total of
6.8 thousand unique frames.

The styles used for training and validation are illustrated in Figure 10. The
model appears to generalize from simple photometric augmentations to more
dramatic lighting variations and has learned the neural style transformations ef-
fectively. However, under the most extreme InstantStyle transfers, it occasionally
struggles to remain within the training data distribution. We hypothesize that
incorporating stronger stylistic augmentations during training could improve ro-
bustness in such cases.

Additional qualitative results on Tanks and Temples are shown in Figures 18,
19, 20, 21, 22, 23, and 24. Additional results on DL3DV140 are shown in Fig-
ure 17.

We evaluate geometric consistency by running COLMAP on generated video
frames using pycolmap, reporting the fraction of frames successfully registered.
Fréchet Inception Distance (FID) is measured with clean-fid [32] using the
default Inception-V3 features, with frames sampled every 4 steps from both
generated and reference videos. Style alignment is assessed via cosine similarity
between CLIP ViT-L/14 [34] embeddings of the style reference image and the
mean embedding across all generated frames.

Furthermore, we present supplementary visualizations on the ScanNet++
dataset. Figure 11 shows the conditioning signals used by our approach at their
native resolution of 60x90. Figure 15 shows qualitative examples corresponding
to the results reported in Table 4 in the main paper. We further demonstrate
results on longer video sequences generated autoregressively with our model.
Figure 14 shows random ScanNet++ snippets, while Figure 13 illustrates the
temporal stability of DINO features by revisiting the same frames at different
points in time. We additionally show the effect of different augmentations during
conditioning signal generation (Figure 12).

6.2 Dynamic Video

As shown in Figure 16, although the model was trained only with camera motion,
it is able to generalize to scenes containing object motion.

6.3 Causal Temporal Adapter.

To compress the per-frame DINO features from T =49 frames to match the VAE
latent temporal dimension of T ′=13, we employ a causal temporal encoder that
mirrors the design of the CogVideoX 3D Causal VAE [44]. Like the VAE encoder,
we use temporally causal 3D convolutions that pad only from past frames, ensur-
ing each output depends solely on current and preceding inputs without access
to future information. The encoder consists of two stages, each performing tem-
poral downsampling with stride 2:

h(1) = SiLU (GN(CausalConv3D(F))),
h(2) = SiLU (GN(CausalConv3D(h(1)))),

T : 49 → 25

T : 25 → 13

(1)

(2)

22

Dominici, Deixelberger, Vardis and Steinberger

Then h(2) is concatenated with the noisy state and passed to the 16 trans-

former blocks to obtain the residual for each of the blocks hc
l .

6.4 Prompt Conditioning

For the transfer experiments, during both training and inference, we append
style-specific keywords to the text prompt before passing it to the text encoder
and diffusion model. For example we add "paprika", "drawing contour" or "soft
light", this has a much weaker conditioning effect compared to the first frame
image, but it helps to align the distributions.

6.5 Long Video Sequences

Our model currently supports up to 49 frames per inference. Nevertheless, ar-
bitrarily long video sequences can be generated by running the model autore-
gressively. In particular, inference is performed on blocks of 49 frames, where
the conditioning frame for each subsequent block is set to the last output frame
of the previously generated block. We demonstrate this procedure in Figures 13
and 14.

Empirically, we observe that long video sequences can be generated without
any noticeable degradation in quality. Camera poses remain highly consistent
across time, with only minor deviations relative to the initial conditioning frame.
This indicates that our model maintains temporal coherence and stability even
over extended autoregressive rollouts, highlighting the robustness of both the
generated content and the DINO feature representations.

6.6 Conditioning Signal Augmentation

In Figure 12, we provide further insight into the robustness of our control network
under changes to the conditioning signal. To do this, we modify the 3D structure
in various ways, such as changing the original voxel size, performing point cloud
rendering instead of using voxels, and downscaling the feature maps by reducing
their resolution by a factor of 2. As can be seen, our model is not entirely
unaffected by these changes, but it handles them reasonably well depending on
the magnitude of the modifications.

Control-DINO

23

Fig. 10: Overview of appearance augmentations used during training and validation.
Training: We apply neural style transfer (AnimeGAN, CartoonGAN, FastStyle, Draw-
ing) and photometric augmentations (color temperature, contrast, saturation shifts) to
the training videos. Validation: We evaluate generalization using held-out styles in-
cluding InstantStyle with famous paintings, unseen neural style transfer models, and
IC-Light relighting with varying lighting conditions.

24

Dominici, Deixelberger, Vardis and Steinberger

2D

Voxel

Mesh

Concerto

Scene 1

p
0
2
7

0
9
x
0
6

Fig. 11: PCA visualization of the high-resolution and true resolution conditioning
feature maps (60×90).

Voxel size 0.05

Voxel size 0.03

Point cloud

Features 30x45

2

e
n
e
c
S

3

e
n
e
c
S

Fig. 12: Visual examples of augmentations to the conditioning signal: different voxel
sizes(default 0.02), point clouds in place of voxels, and downscaled feature maps.

t=1

t=25

t=49

t=74

t=98

t=123

t=147

t=172

t=196

1

e
n
e
c
S

2

e
n
e
c
S

3

e
n
e
c
S

4

e
n
e
c
S

Fig. 13: Temporal stability of the generated content when looping the conditioning
signal while running our method autoregressively. Frames with identical camera poses
occur at t=1/98/196, t=25/74/123/172, and t=49/147, allowing comparison of scene
consistency over long time horizons.

Control-DINO

25

0
=
t

6
1
=
t

2
3
=
t

8
4
=
t

4
6
=
t

0
8
=
t

6
9
=
t

2
1
1
=
t

8
2
1
=
t

4
4
1
=
t

0
6
1
=
t

6
7
1
=
t

2
9
1
=
t

8
0
2
=
t

0
4
2
=
t

Fig. 14: Visual examples of our model running in autoregressive mode to generate long
video sequences on the ScanNet++ dataset.

26

Dominici, Deixelberger, Vardis and Steinberger

t=1

Scene 1
t=35

t=49

t=1

Scene 2
t=35

t=49

T
G

s
r
u
O

e
g
d
E

h
t
p
e
D

T
G

s
r
u
O

e
g
d
E

h
t
p
e
D

t=1

Scene 3
t=35

t=49

T
G

s
r
u
O

e
g
d
E

h
t
p
e
D

T
G

s
r
u
O

e
g
d
E

h
t
p
e
D

t=1

Scene 4
t=35

t=49

Fig. 15: Qualitative comparison on the ScanNet++ dataset between DINO-Control
2D and Wan2.2 under Canny edge and depth conditioning.

Control-DINO

27

Fig. 16: Control-DINO applied to a dynamic video with moving pedestrians, demon-
strating compatibility with neural style transfer (CartoonGAN) and photometric aug-
mentation (Noir).

Control-DINOScene 1+ CartoonGAN+ IC-LightControl-DINOScene 2+ CartoonGANt=0+ IC-Lightt=T/2t=TStyle Transfer on Dynamic Video28

Dominici, Deixelberger, Vardis and Steinberger

Fig. 17: We show here additional results on DL3DV140 to generate "real" sequences.

Fig. 18: Control-DINO (first 3 columns) and edges (first 3 columns) conditioning re-
sults for Tanks and Temples transfer on "Bright Daylight".

Control-DINO

29

Fig. 19: Control-DINO (first 3 columns) and edges (first 3 columns) conditioning re-
sults for Tanks and Temples transfer on "CartoonGAN Paprika".

Fig. 20: Control-DINO (first 3 columns) and edges (first 3 columns) conditioning re-
sults for Tanks and Temples transfer on "Cool Moonlight".

30

Dominici, Deixelberger, Vardis and Steinberger

Fig. 21: Control-DINO (first 3 columns) and edges (last 3 columns) conditioning results
for Tanks and Temples transfer on "Drawing Contour Style".

Fig. 22: Control-DINO (first 3 columns) and edges (last 3 columns) conditioning results
for Tanks and Temples transfer on "FastStyle Udnie".

Control-DINO

31

Fig. 23: Control-DINO (first 3 columns) and edges (last 3 columns) conditioning results
for Tanks and Temples transfer on "Sunset Left".

Fig. 24: Control-DINO (first 3 columns) and edges (last 3 columns) conditioning results
for Tanks and Temples transfer on "Van Gogh Starry Night".

