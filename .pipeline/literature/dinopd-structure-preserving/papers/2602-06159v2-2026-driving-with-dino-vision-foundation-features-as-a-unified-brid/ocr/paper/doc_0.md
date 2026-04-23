6
2
0
2

b
e
F
9

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
5
1
6
0
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

Driving with DINO: Vision Foundation Features as a Unified Bridge for
Sim-to-Real Generation in Autonomous Driving

Xuyang Chen1,2∗ Conglang Zhang3,4∗ Chuanheng Fu3,4∗

Zihao Yang5 Kaixuan Zhou3†‡

Yizhi Zhang3

Jianan He3 Yanfeng Zhang2 Mingwei Sun3,4
Liqiu Meng1
Zengmao Wang4‡

Zhen Dong4

Xiaoxiao Long6
1Technical University of Munich
4Wuhan University

2Huawei Hilbert Research Center

5University of Science and Technology of China

3Huawei Riemann Lab
6Nanjing University

∗Equal contribution

†Project lead

‡Corresponding author

Figure 1. Driving with DINO (DwD) achieves photorealistic simulation-to-real video translation with superior structural consistency.
As shown, our method generalizes remarkably well across diverse scenarios, ranging from town and urban environments to challenging
weather conditions such as rain and frost.

Abstract

Driven by the emergence of Controllable Video Diffusion,
existing Sim2Real methods for autonomous driving video
generation typically rely on explicit intermediate represen-
tations to bridge the domain gap. However, these modalities
face a fundamental Consistency-Realism Dilemma. Low-
level signals (e.g., edges, blurred images) ensure precise
control but compromise realism by ”baking in” synthetic
artifacts, whereas high-level priors (e.g., depth, seman-
tics, HDMaps) facilitate photorealism but lack the struc-

tural detail required for consistent guidance. In this work,
we present Driving with DINO (DwD), a novel frame-
work that leverages Vision Foundation Module(VFM) fea-
tures as a unified bridge between the simulation and real-
world domains. We first identify that these features en-
code a spectrum of information, from high-level seman-
tics to fine-grained structure. To effectively utilize this, we
employ Principal Subspace Projection to discard the high-
frequency elements responsible for ’texture baking,’ while
concurrently introducing Random Channel Tail Drop to
mitigate the structural loss inherent in rigid dimensional-

1

Simulated Driving VideosPhotorealistic Videos 
 
 
 
 
 
ity reduction, thereby reconciling realism with control con-
sistency. Furthermore, to fully leverage DINOv3’s high-
resolution capabilities for enhancing control precision, we
introduce a learnable Spatial Alignment Module that adapts
these high-resolution features to the diffusion backbone. Fi-
nally, we propose a Causal Temporal Aggregator employing
causal convolutions to explicitly preserve historical motion
context when integrating frame-wise DINO features, which
effectively mitigates motion blur and guarantees temporal
stability. Extensive experiments show that our approach
achieves State-of-the-Art performance, significantly outper-
forming existing baselines in generating photorealistic driv-
ing videos that remain faithfully aligned with the simula-
tion.

1. Introduction

Closed-loop validation is critical for autonomous driving
safety. However, applying it in the real world is risky and
inefficient due to the massive data requirements and the
scarcity of rare corner cases. Simulation offers a scalable
alternative, yet creating photorealistic assets remains costly.
Consequently, simulators like CARLA [10] often compro-
mise on visual fidelity, resulting in a significant domain
gap between simulated and real-world data. This discrep-
ancy hinders the generalization of simulation-trained mod-
els to physical environments, necessitating robust Sim-to-
Real transfer techniques to effectively utilize simulation as-
sets.

Figure 2. Consistency-Realism-Dilemma

To address this challenge, Diffusion Models have
emerged as a promising paradigm, offering superior stabil-
ity and high-fidelity generation capabilities. While initial
attempts [28, 31] adapted these image diffusion models for
Sim-to-Real in a frame-wise manner; however, treating con-
tinuous driving video as independent images ignores tem-
poral correlations, inevitably leading to severe flickering ar-
tifacts and texture incoherence. To remedy this, leverag-
ing video diffusion models augmented with structural con-
trol mechanisms (e.g., ControlNet [53]) emerges as a supe-
rior paradigm [16, 17, 23, 27, 43, 47, 55]. Video diffusion
inherently model temporal dynamics to ensure coherence,
while the integration of ControlNet adapters strictly anchors
the generative process to the simulation’s geometric layout.
This integration effectively reconciles the tension between

2

temporal and spatial consistency, making it a highly promis-
ing direction for robust Sim-to-Real transfer.

In this context, intermediate representations such as edge
maps, blurred images, semantic masks, and depth maps
serve as the common bridge between simulation and real-
ity, and have been widely integrated into ControlNet-based
frameworks [1, 2, 13, 14]. Additionally, High-Definition
Maps (HDMaps) is a widely adopted structural representa-
tion in the autonomous driving domain. However, achiev-
ing both high photorealism and spatial consistency simul-
taneously remains a significant challenge, which we call
as Consistency-Realism-Dilemma. As depicted in Fig. 2,
current widely adopted control signals could not balance
them well:
• Low-level signals (e.g., edges, blurred images) provide
strong structural guidance but often impose rigid con-
straints that ”bake in” synthetic textures. This forces the
model to retain the simulation’s artificial appearance, hin-
dering the generation of true photorealism.

However,

• High-level signals (e.g., depth, semantics, HDMaps)
heavily discard structural information to allow for pho-
this lack of fine-
torealistic generation.
grained structural information creates ambiguity, which
frequently leads to generative hallucinations and control
errors, such as generating spurious or incorrect lane lines.
To leverage the advantages of both feature types, recent
multi-control frameworks attempt to mitigate these trade-
offs, yet they face significant challenges. Jointly training
multiple ControlNets incurs prohibitive GPU memory costs
and suffers from gradient conflicts, where heterogeneous
modalities compete and fail to converge [7, 18]. Con-
versely, combining separately trained adapters during infer-
ence is equally problematic. Aggregating distinct control
signals often leads to feature interference, necessitating la-
borious weight tuning and frequently resulting in incoherent
fusion or visual artifacts due to conflicting guidance [57].

Departing from complex multi-branch designs, we pro-
pose a novel conditioning strategy leveraging Vision Foun-
dation Models (VFMs), specifically DINOv3 [37]. Unlike
explicit multi-control approaches, DINO features inherently
encode a rich spectrum of scene properties ranging from
low-level to high-level cues within a unified latent space,
a capability validated by its robust performance across a
wide spectrum of task granularities [24, 41, 45, 49, 50, 58].
By utilizing DINO as a holistic condition for Sim-to-Real
transfer, we achieve a natural fusion of structural and se-
mantic information. Crucially, this approach eliminates the
feature competition and gradient conflicts typical of multi-
branch architectures, allowing for efficient training and in-
ference without additional computational overhead or com-
plex manual tuning.

Nevertheless, our analysis reveals that naively incorpo-
rating DINO features into ControlNet results in textural in-

ConsistencyRealismBlurEdgeDepthSemanticHD MapOriginal CGformation leakage. Since DINO representations are rich
enough to serve as powerful encoders for auto-encoding
tasks—a property explored in prior work [21, 36, 58]—they
inadvertently carry over fine-grained pixel information. In
our Sim-to-Real context, the model essentially memorizes
and reproduces the synthetic textures of the simulation in-
stead of generating realistic details, thereby defeating the
purpose of domain adaptation. Furthermore, the spatial res-
olution of DINO features is typically down-sampled by a
factor of 16. This significant compression results in the
loss of critical structural details, particularly along object
boundaries, which inevitably degrades control consistency.
Conversely, in the temporal domain, DINO features exhibit
significant redundancy due to their independent frame-wise
processing nature. While existing methods often resort to
naive keyframe sampling to mitigate this [4, 35], our analy-
sis suggests that such aggressive subsampling disrupts mo-
tion continuity, inevitably leading to temporal aliasing and
incoherence in the generated video.

To address these challenges, we propose Driving with
DINO (DwD), a controllable video diffusion framework
leveraging DINO features to effectively balance visual re-
alism and control consistency through the following contri-
butions:

Minor Components Pruning. We reveals that the mi-
nor components of VFM features primarily encode high-
frequency textural details, which lead to texture baking in
Sim-to-Real transfer. Then, we leverage Principal Subspace
Projection to attenuate high-frequency details, and intro-
duce Random Channel Tail Drop to prevent the significant
structural information loss caused by rigid dimensionality
reduction. This training strategy stochastically prunes mi-
nor components after PCA projection, enabling the model
to effectively balance the utilization of semantic and struc-
tural information during training, thereby achieving a trade-
off between realism and controllability.

Spatial Resolution Enhancer. To compensate for the sig-
nificant spatial information loss of DINO backbones, we
leverage DINOv3 to process input videos at higher reso-
lutions. We then introduce a learnable Spatial Alignment
Module to bridge the consequent dimensionality mismatch
with the diffusion backbone, thereby significantly enhanc-
ing control consistency.

Causal Temporal Aggregator. We introduce the Causal
Temporal Aggregator, a downsampling module driven by
causal convolutions. It addresses temporal redundancy by
preserving historical context, effectively mitigating motion
blur and enhancing stability.

2. Related Works

2.1. Generative Adversarial Networks

Early Sim-to-Real approaches utilized GANs (e.g., Cycle-
GAN [59], CUT [30]) to align simulated images with real-
world distributions. However, these adversarial translations
frequently introduce geometric distortions and semantic in-
consistencies. While utilizing simulator-derived G-buffers
(e.g., depth, normals) can constrain the generation process
to preserve geometry [34], such methods often retain adver-
sarial artifacts and lack fine-grained texture fidelity.

2.2. 2D Image and Video Diffusion

Diffusion models have mitigated GAN-based instability
through superior fine-grained fidelity. SDEdit [28] facil-
itates transfer via SDE inversion [38], injecting photore-
alistic priors while maintaining the simulator’s semantic
structure. Extensions to video (e.g., Pix2Video [6], To-
kenFlow [15]) enforce temporal coherence within zero-shot
frameworks by propagating features or injecting optical
flow priors (FLATTEN [9]).

However, methods relying on inversion or diffusion-
based relighting [25, 54] struggle to effectively disentan-
gle semantic structure from synthetic style, resulting in se-
vere “texture baking.” To address this, ControlNet [53] in-
troduced explicit spatial conditioning to decouple geometry
from appearance. With the rise of Video Diffusion Trans-
formers (DiTs) [32], controllable generation has shifted to-
wards inherently modeling motion dynamics, offering tem-
poral consistency superior to zero-shot adaptations.

2.3. Controllable Video Diffusion

In autonomous driving simulation, methods like Magic-
Drive [13, 14] and DriveDreamer [44, 56] utilize sparse
high-level priors (e.g., HDMaps, 3D boxes). While effec-
tive for layout, the spatial sparsity of these signals leads to
hallucinations in unconstrained regions. Conversely, volu-
metric approaches like WoVoGen [26] utilize 3D occupancy
for consistency but remain too coarse-grained for fine geo-
metric details.

Recent efforts to unify these paradigms, such as Cos-
mos Transfer [1, 2], incorporate both high-level semantic
and low-level geometric cues. However, fusing such het-
erogeneous signals introduces significant architectural com-
plexity and instability, often hitting a Consistency-Realism
Dilemma. Existing multi-control fusion strategies are either
limited to similar modalities [12] or suffer from prohibitive
training costs [48]. To address these limitations, we pro-
pose utilizing DINOv3 as a unified representation that en-
capsulates both semantics and structures, allowing it to be
seamlessly integrated into a standard single-control archi-
tecture. While DINO has been applied to object-level con-
trol [8, 22], we extend this to scene-level Sim-to-Real trans-

3

Figure 3. The framework of DwD. (a) Training: The model is trained on real-world driving videos using a controllable diffusion architec-
ture. The core module, VFM-Prism, processes DINOv3 features through Spatial Resolution Enhancement, Minor Components Pruning,
and Causal Temporal Aggregation. These refined features are injected via a Control Branch to guide the reconstruction of the original
video. (b) Inference: The model performs Sim-to-Real translation using synthetic inputs. By leveraging the domain-invariant structural
features extracted by VFM-Prism (specifically via PCA-based pruning to mitigate texture leakage), DwD generates high-fidelity photore-
alistic videos that strictly preserve the simulation’s geometric layout.

fer, utilizing a unified representation to reconcile photoreal-
ism with strict structural consistency and eliminate texture
baking.

3. Method

3.1. Overview

DwD is a diffusion-based framework designed to translate
synthetic driving simulations into photorealistic video se-
quences. At its core, the framework leverages Vision Foun-
dation Model (VFM) latents as an intermediate semantic
bridge. The methodology is organized as follows: Sec. 3.2
details the backbone video diffusion architecture. Sec. 3.3
introduces VFM-Prism, a feature processing module tai-
lored to ensure robust Sim2Real alignment during training.
Finally, Sec. 3.4 delineates the inference pipeline used to
transform synthetic inputs into realistic videos.

3.2. Controllable Video Diffusion Model

The DwD framework leverages the architecture of the
pretrained Cosmos-Predict2.5 [2], which integrates the
WAN2.1 3D VAE [40] with fine-tuned Diffusion Trans-
former (DiT) blocks. To enable structurally controllable
generation, we augment the video diffusion framework with
a dedicated control branch, as illustrated in Fig. 3(a). This

branch comprises conditioning DiT blocks initialized with
weights from the denoising DiT backbone. Following the
injection strategy proposed in VACE [23], control features
are modulated into the base model at uniform intervals of
N blocks. The control branch encodes processed DINOv3
latents as conditioning inputs. During the training phase,
the model is optimized to reconstruct ground-truth realistic
videos. At inference( Fig. 3(b), the model processes syn-
thetic simulation inputs to generate photorealistic counter-
parts, effectively bridging the domain gap between simula-
tion and reality.

3.3. Robust Training with VFM-Prism

To integrate VFM features into Sim2Real translation, we
propose a module named VFM-Prism to mitigate the afore-
mentioned challenges—specifically the high spatial com-
pression of DINOv3, textural leakage, and the latent di-
mensionality mismatch with DiT blocks. As illustrated in
Fig. 3(a), the pipeline comprises three core modules: Spa-
tial Resolution Enhancer, Minor Components Pruning, and
Causal Temporal Aggregator, each detailed in the following
sections.

Spatial Resolution Enhancer.

Unlike previous
VFMs [29, 33, 39] that exhibit bias against high-resolution
inputs, DINOv3 demonstrates robustness to variable reso-

4

Driving Video 𝑽𝑽∈ℝ𝑇𝑇×𝐻𝐻×𝑊𝑊×3Z ∈ℝ𝑻𝑻𝟒𝟒×𝑯𝑯𝟖𝟖×𝑾𝑾𝟖𝟖×𝟏𝟏𝟏𝟏Noisy Latent(a) Training PipelinePatchify𝑻𝑻×𝑯𝑯𝑯𝑯𝟏𝟏𝟏𝟏×𝑾𝑾𝑯𝑯𝟏𝟏𝟏𝟏×𝑪𝑪′DiTBlockDiTBlockZeroLinearCondition DiT(Trainable Copy)Denoising DiT𝑻𝑻𝟒𝟒×𝑯𝑯𝟏𝟏𝟏𝟏×𝑾𝑾𝟏𝟏𝟏𝟏×𝑪𝑪𝒑𝒑VFM-Prism𝑻𝑻×𝑯𝑯𝟏𝟏𝟏𝟏×𝑾𝑾𝟏𝟏𝟏𝟏×𝑪𝑪′Spat. Align×1/𝑆𝑆𝑻𝑻×𝑯𝑯𝑯𝑯𝟏𝟏𝟏𝟏×𝑾𝑾𝑯𝑯𝟏𝟏𝟏𝟏×𝑪𝑪VAEDec Generated Video 𝑽𝑽𝑽∈ℝ𝑇𝑇×𝐻𝐻×𝑊𝑊×30ZeroPadding123012Causal Strided ConvTrainable ModulesFrozen ModulesSimulated Driving VideoControlNet + Video DiTGenerated Video(b) Inference PipelineVFM-PrismPCA Projection RanChannel Tail DropChannel DimSpat. Up×SSpatial Res. EnhancerMinor Comp. PruningTemporal AggregatorVFM Figure 4. Lower-dimensional PCA components encode coarse semantic layouts, whereas increasing the dimensions leads to the refinement
of high-frequency details. Similarity maps are shown for two anchor points (top: × on building, bottom: × on one car).

16 × W ×S

lutions due to its specialized positional encoding strategy
(RoPE-box jittering). To compensate for the high spa-
tial compression rate of DINOv3, we upsample the in-
put video V ∈ RT ×H×W ×3 by a scale factor S prior to
encoding, yielding a higher-resolution feature map Z c ∈
RT × H×S
16 ×C. While post-hoc feature map upsam-
pling methods like Featup [11] and Anyup [46] offer com-
putationally cheaper alternatives, our empirical findings
suggest they compromise feature map quality, negatively
impacting the consistency between generated videos and in-
put simulations. Thus, input-space upscaling ensures su-
perior structural fidelity. However, this results in a latent
tensor size misaligned with the input requirements of the
Condition DiT blocks, which should spatially be H
16 × W
16
rather than H×S
16 . Therefore, we introduce a Spatial
Alignment Module to bridge this gap, which is composed
of strided convolutions with residual blocks, as detailed in
the Appendix.

16 × W ×S

Minor Components Pruning. To mitigate texture leak-
age and extract domain-invariant structural conditions, we
employ Principal Component Analysis (PCA) to construct
a semantic bottleneck. By projecting the high-dimensional
feature space onto a lower-dimensional subspace Rk, we
effectively filter out low-variance components associated
with appearance while retaining dominant structural infor-
mation. Qualitative analysis of feature similarity (Fig. 4)
indicates that the dimensionality k acts as a critical hyper-
parameter for this disentanglement. Extremely low dimen-
sions (e.g., k = 3) compromise spatial granularity, causing
semantic ambiguity. Conversely, a high-dimensional spec-
trum (e.g., k = 64) retains excessive high-frequency in-
formation, resulting in similarity hotspots driven by local
texture rather than geometry—indicative of domain noise
leakage. We observe that a moderate subspace (e.g., k ∈
{8, 16, 32}) strikes an optimal balance, preserving struc-
tural layout while suppressing domain-specific texture.

Although the leading PCA components primarily en-
code semantics, a distinct boundary between structure and
style is ill-defined. To prevent the loss of structural infor-
mation associated with rigid dimensionality reduction, we
propose Random Channel Tail Drop, a stochastic regular-
ization strategy. During training, rather than using a fixed

5

truncation point, we sample the number of active channels
k from a predefined discrete set of candidate dimensions
K = {k1, k2, . . . , km}. We then apply a channel-wise bi-
nary mask M ∈ {0, 1}km with entries Mi = 1 if i ≤ k and
0 otherwise.

Causal Temporal Aggregator. The absence of temporal
compression in DINOv3, results in a temporal dimension
misalignment with the input requirements of the Condition
DiT blocks. While prior methods rely on heuristic down-
sampling strategies such as keyframe sampling [4, 20] or bi-
linear interpolation [35], these approaches often fail to cap-
ture fine-grained spatial details and temporal dynamics. To
bridge this gap without incurring the computational over-
head of a full CausalVAE, we insert extra causal convolu-
tion blocks into aforementioned Spatial Alignment Module
(see Appendix).

3.4. Video Sim2Real Translation

Once trained, DwD is employed to perform Video
Sim2Real Translation. As illustrated in Fig. 3(b), the infer-
ence process takes a synthetic video rendered from a driv-
ing simulator as input. While the pipeline largely mirrors
the training phase, we introduce a key flexibility during
inference: we manually modulate the number k of active
PCA components (i.e., the spectral truncation threshold).
This adjustment allows us to explicitly control the guidance
strength, effectively balancing structural fidelity with real-
istic texture generation.

4. Experiments

4.1. Experiment Setup

We train DwD on the nuPlan dataset [5] and evaluate perfor-
mance using videos rendered from the CARLA simulator
(see Appendix for more details). To ensure a comprehensive
assessment, we compare our method against representative
baselines across different categories. For training-free im-
age diffusion models like TC-Light [25], we adhere to the
official protocols using appropriate relighting prompts. For
FRESCO [51], we additionally incorporate nuPlan images
as reference style inputs. In the video generation domain,
we compare against Cosmos-Transfer2.5 [2] with various

dim = 16×dim = 8dim = 3dim = 32dim = 64dim = 1024×BuildingCarinput modalities. Crucially, to ensure a fair comparison, we
fine-tune the ControlNet modules of these video generation
baselines on the same training set used for our method.

4.2. Evaluation Metrics.

Sim-to-Real video translation must balance consistency and
realism. To provide a comprehensive evaluation, we assess
the results along the following three dimensions:
• Visual Fidelity: We employ sFID and sKID to mea-
sure distribution discrepancy, adopting a semantics-aware
sampling strategy [34] to mitigate layout mismatches. By
matching VGG features between CG and real patches, we
construct a benchmark of 400,000 pairs for robust evalu-
ation.

• Perceptual Realism: We quantify authenticity via CLIP-
Real [52], formulated as CLIP-Real = (x⊤tp)/(x⊤tn).
Here, x, tp, tn represent embeddings of the output frame,
positive (e.g., “Photo”), and negative (e.g., “Game”)
prompts, respectively. Higher scores indicate better align-
ment with real photography.

• Temporal Consistency: We assess temporal coher-
ence using Motion-S [19] for motion plausibility and
WarpSSIM. The latter computes SSIM between a frame
and its neighbors warped via optical flow derived from
source CG videos.

• Sim2Real Consistency: We assess semantic preserva-
tion using the mean Intersection over Union (mIoU).
Specifically, we employ the state-of-the-art perception
model [42] (pretrained on Cityscapes) to perform seman-
tic segmentation on the translated frames. We then calcu-
late the mIoU by comparing the predicted segmentation
masks against the ground truth semantic labels derived
from the source CG engine.

4.3. Quantitative Evaluation.

Tab. 1 highlights DwD’s substantial advantage in fidelity
metrics (sKID and sFID). As previously analyzed, low-level
conditions (e.g., edge, blur) exhibit poor fidelity primar-
ily due to their CG-like textures, which significantly devi-
ate from real-world data distributions. Counter-intuitively,
high-level modalities like segmentation and depth also fail
to achieve competitive sFID/sKID scores. This stems from
their insufficient control capability, which leads to pro-
nounced layout discrepancies between the generated con-
tent and the ground truth. Since sFID and sKID are com-
puted based on aligned spatial regions, such structural mis-
alignments inevitably penalize the fidelity scores. In con-
trast, DwD effectively bridges this gap by balancing precise
layout guidance with high-quality synthesis, resulting in su-
perior performance.

In terms of photorealism (measured by CLIP-Real),
DwD achieves performance comparable to methods uti-
lizing high-level structural conditions, such as depth and

In contrast, low-level conditions tend to
semantic maps.
produce textures that closely mirror the original synthetic
videos (CG), resulting in a CLIP-Real score (100.63 for
blur and 112.38 for edge) similar to that of the CG base-
line (98.79). TC-Light exhibits a similar trend (102.46) as
it primarily modifies environment lighting without altering
underlying textures.

Regarding temporal consistency, DwD demonstrates su-
perior performance on the Motion-S metric.
In terms of
WarpSSIM, DwD is only slightly below Edge-based meth-
ods while outperforming Seg- and Depth-based approaches.
Notably, TC-Light achieves an outlier WarpSSIM score by
nearly identical preservation of original CG textures; how-
ever, this comes at a severe expense of realism.

Table 3 reports the Sim2Real consistency reflected by
mIoU. Our method exhibits high mIoU scores across mul-
tiple categories, demonstrating consistency comparable to
robust low-level control signals such as Edge and Blur.
Furthermore, a comparison with the low-resolution alter-
native verifies that spatial resolution enhancement signifi-
cantly contributes to structural consistency.

Table 1. Quantitative evaluations. Top results are colored in
Gold , Silver , and Bronze (also applicable to Tabs. 2 and 3.). †
denotes modules fine-tuned on the same dataset as DwD. C-T2.5
stands for Cosmos-Transfer 2.5.

Method

Mot-S↑ W-SSIM↑ CLIP-R↑

sKID↓

sFID↓

CG

FRESCO

TC-Light

C-T2.5 Edge
C-T2.5 Edge†
C-T2.5 Blur
C-T2.5 Depth
C-T2.5 Edge+Depth
C-T2.5 Edge+Seg
C-T2.5 Seg

Ours

(%)

-

98.37

98.79

98.61
98.73
98.02
98.47
93.97
98.62
98.85

98.94

-

93.32

96.59

93.60
94.37
88.14
92.48
93.97
93.97
90.92

93.07

98.79

29.59

43.65

109.92

32.55

56.28

102.46

22.12

44.47

112.38
112.58
100.63
116.50
112.15
112.13
119.73

119.11

22.82
21.22
29.09
26.98
23.71
23.32
23.81

9.30

37.44
32.59
39.57
38.44
36.69
36.62
37.20

21.62

4.4. Qualitative Evaluation.

We conduct a qualitative comparison between the photore-
alistic videos generated by DwD and baseline methods. As
illustrated in Fig. 5, DwD not only ensures semantic con-
sistency but also maintains precise control over structural
details, particularly for road lines and markings. This effec-
tively addresses the challenge of balancing low-level and
high-level control signals discussed in the Introduction.

Specifically, high-level signals (e.g., depth or semantic
maps) often cause hallucinations in background regions.
While semantic maps can produce accurate lane markings
when guided by explicit instance segmentation, they tend
to hallucinate non-existent lines when the input lacks such

6

Figure 5. Qualitative comparison with state-of-the-art methods. The red boxes highlight inconsistencies with respect to the CG input,
while the green boxes point out low-fidelity textures.

7

CGC-T2.5 Deblur[Ali et al . 2025]C-T2.5 Depth[Ali et al . 2025]C-T2.5 Edge[Ali et al . 2025]C-T2.5 Seg[Ali et al . 2025]C-T2.5 Depth+Edge[Ali et al . 2025]FRESCO[Yang et al. 2024b]TC-Light[Liu et al. 2025]DwD(ours)markings (see Fig. 12 and Fig. 13). Conversely, low-level
control signals tend to yield oversimplified outputs that ad-
here too closely to the synthetic domain. For instance,
Cosmos-T2.5 Blur merely reproduces the synthetic input.
Similarly, Cosmos-T2.5 Edge lacks realism because the
edges extracted from CG are excessively clean and regu-
lar, thereby inherently carrying texture information from the
CG source that biases the generation toward a synthetic ap-
pearance. Although the multi-control variant, Cosmos-T2.5
Edge+Depth, attempts to strike a balance, its performance
remains inferior to DwD.

Table 2. Ablation studies. ×S means the upscaling spatial resolu-
tion S times.

Method

pca3 (×1)
pca8 (×1)
pca16 (×1)
pca32 (×1)

pca8 (×2)
pca8 (×4)

pca8 (×4)+Temp (final)

W-SSIM↑ CLIP-R↑

sKID↓

sFID↓

90.94
91.34
91.49
90.29

90.73
92.69

93.07

119.63
118.78
116.00
114.38

118.38
118.89

119.11

10.70
9.08
9.92
9.91

9.79
9.68

9.30

23.32
21.37
22.60
23.01

23.06
21.92

21.62

4.5. Ablation Studies

We systematically evaluate the impact of key hyperparam-
eters and components in Tab. 2. First, we investigate the
influence of the active PCA channels k during inference.
We observe that fidelity metrics gradually improve as k
decreases; we attribute this to high-level texture leakage
caused by insufficient pruning of PCA-projected features.
However, excessive pruning (e.g., k = 3) causes fidelity
to degenerate, as the model struggles to maintain seman-
tic consistency—a limitation shared with Cosmos T2.5 Seg.
Given these findings, we select k = 8 as our optimal dimen-
sionality for inference, as it strikes the best balance between
preserving structural guidance and ensuring generative real-
ism.

Subsequently, we analyze the influence of the feature up-
scaling factor S. While a factor of ×4 leads to a slight de-
crease in sFID and sKID, it yields a substantial improve-
ment in temporal consistency (W-SSIM). As illustrated in
the Appendix, higher S indeed leads to better structural con-
sistency, which should be beneficial to sFID and sKID. We
attribute the subsequent decrease in fidelity metrics to the
domain gap in sharpness: the generated videos preserve the
pristine edges of the source CG, which statistically diverges
from the inherent motion blur and noise found in real video.
Finally, we verify the efficacy of the Causal Temporal Ag-
gregator. Compared to the ×4 baseline, this component si-
multaneously enhances Warp-SSIM and improves fidelity
metrics (sFID, sKID). This performance gain is primarily
attributed to the aggregator’s ability to mitigate blur and cor-
rupted frames in the generated videos, which significantly
elevates overall generative quality. Furthermore, the mIoU
scores in Tab. 3 provide additional evidence that both the
Spatial Resolution Enhancer and the Causal Temporal Ag-
gregator substantially boost controllability. To offer a more
intuitive perspective that quantitative metrics may obscure,
we provide additional visual comparisons in the appendix,
clearly illustrating the distinct effects of varying hyperpa-
rameters.

8

Table 3. Comparison of mIoU across different methods and cate-
gories.

Category

pca8 (×1)

Edge

Blur

Depth

Seg

Ours

Road
Sidewalk
Building
Fence
Pole
Traffic Light
Traffic Sign
Vegetation
Terrain
Sky
Person
Car
Bus
Bicycle

0.9574
0.5464
0.7802
0.0978
0.2757
0.4354
0.2014
0.6907
0.3171
0.8945
0.5380
0.7522
0.0989
0.0599

0.9591
0.5914
0.8193
0.0767
0.3665
0.6655
0.3838
0.6833
0.3561
0.8912
0.6795
0.7973
0.0223
0.0242

0.9636
0.6332
0.8127
0.0790
0.3214
0.5501
0.2862
0.7210
0.4476
0.9028
0.6729
0.7819
0.0620
0.0132

0.9475
0.5253
0.7725
0.0789
0.3548
0.5392
0.2601
0.6936
0.3284
0.9115
0.6438
0.7828
0.0259
0.0294

0.9345
0.5010
0.5781
0.0848
0.2555
0.3174
0.1505
0.5773
0.3210
0.7691
0.5828
0.7044
0.0049
0.0612

0.9605
0.5610
0.8087
0.1187
0.3435
0.4503
0.2681
0.7150
0.3245
0.9111
0.6256
0.7756
0.0296
0.0666

mIoU (Avg)

0.4675

0.5015

0.5105

0.4843

0.4132

0.4967

5. Conclusion

In this paper, we propose Driving with DINO (DwD),
which leverages Vision Foundation Models to resolve
the “Consistency-Realism-Dilemma” in Sim-to-Real video
translation. By comparing DwD with state-of-the-art base-
lines on diverse simulated datasets and our self-collected
data, we verify the significant advantages of DwD in
achieving superior visual realism and structural consistency.
While our results are promising, we acknowledge that com-
putational constraints currently limit the exploration of the
model’s full capacity. Accordingly, future work will focus
on validating the scalability of our approach on large-scale,
high-resolution driving datasets and investigating its de-
ployment within closed-loop autonomous driving systems.

References

[1] Hassan Abu Alhaija, Jose Alvarez, Maciej Bala, Tiffany Cai,
Tianshi Cao, Liz Cha, Joshua Chen, Mike Chen, Francesco
Ferroni, Sanja Fidler, et al. Cosmos-transfer1: Conditional
world generation with adaptive multimodal control. arXiv
preprint arXiv:2503.14492, 2025. 2, 3

[2] Arslan Ali, Junjie Bai, Maciej Bala, Yogesh Balaji, Aaron
Blakeman, Tiffany Cai, Jiaxin Cao, Tianshi Cao, Eliza-

beth Cha, Yu-Wei Chao, et al. World simulation with
arXiv preprint
video foundation models for physical ai.
arXiv:2511.00062, 2025. 2, 3, 4, 5

[3] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui
Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao,
Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang,
Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhao-
hai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Jun-
yang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu,
Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu
Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang
Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jian-
qiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yux-
uan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu,
Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen
Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen
Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu,
and Ke Zhu. Qwen3-vl technical report. arXiv preprint
arXiv:2511.21631, 2025. 12

[4] Ryan Burgert, Yuancheng Xu, Wenqi Xian, Oliver Pilarski,
Pascal Clausen, Mingming He, Li Ma, Yitong Deng, Lingx-
iao Li, Mohsen Mousavi, et al. Go-with-the-flow: Motion-
controllable video diffusion models using real-time warped
noise. In Proceedings of the Computer Vision and Pattern
Recognition Conference, pages 13–23, 2025. 3, 5

[5] Holger Caesar, Juraj Kabzan, Kok Seang Tan, Whye Kit
Fong, Eric Wolff, Alex Lang, Luke Fletcher, Oscar Beijbom,
and Sammy Omari. nuplan: A closed-loop ml-based plan-
ning benchmark for autonomous vehicles. arXiv preprint
arXiv:2106.11810, 2021. 5, 12

[6] Duygu Ceylan, Chun-Hao P Huang, and Niloy J Mitra.
Pix2video: Video editing using image diffusion. In Proceed-
ings of the IEEE/CVF International Conference on Com-
puter Vision, pages 23206–23217, 2023. 3

[7] Weifeng Chen, Jie Wu, Pan Pan, Wanyi Xing, Yi Lu, Zhou
Li, and Gang Zhao. Control-a-video: Controllable text-
to-video generation with diffusion models. arXiv preprint
arXiv:2305.13840, 2023. 2

[8] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao,
and Heng-Guan Heng. Anydoor: Zero-shot object-level im-
age customization. In CVPR, 2024. 3

[9] Yuren Cong, Mengmeng Xu, Christian Simon, Shoufa Chen,
Jiawei Ren, Yanping Xie, Juan-Manuel Perez-Rua, Bodo
Rosenhahn, Tao Xiang, and Sen He. Flatten: optical flow-
guided attention for consistent text-to-video editing. arXiv
preprint arXiv:2310.05922, 2023. 3

[10] Alexey Dosovitskiy, German Ros, Felipe Codevilla, Anto-
nio Lopez, and Vladlen Koltun. Carla: An open urban driv-
ing simulator. In Conference on robot learning, pages 1–16.
PMLR, 2017. 2

[11] Stephanie Fu, Mark Hamilton, Laura Brandt, Axel Feldman,
Zhoutong Zhang, and William T Freeman. Featup: A model-
agnostic framework for features at any resolution. arXiv
preprint arXiv:2403.10516, 2024. 5

[12] J. Gao, Z. Chen, X. Liu, J. Feng, C. Si, Y. Fu, et al. Longvie:
Multimodal-guided controllable ultra-long video generation.
arXiv preprint arXiv:2508.03694, 2025. 3

[13] Ruiyuan Gao, Kai Chen, Enze Xie, Lanqing Hong, Zhenguo
Li, Dit-Yan Yeung, and Qiang Xu. Magicdrive: Street view
generation with diverse 3d geometry control. arXiv preprint
arXiv:2310.02601, 2023. 2, 3

[14] Ruiyuan Gao, Kai Chen, Bo Xiao, Lanqing Hong, Zhen-
guo Li, and Qiang Xu. Magicdrive-v2: High-resolution long
video generation for autonomous driving with adaptive con-
trol. In Proceedings of the IEEE/CVF International Confer-
ence on Computer Vision, pages 28135–28144, 2025. 2, 3

[15] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel.
Tokenflow: Consistent diffusion features for consistent video
editing. arXiv preprint arXiv:2307.10373, 2023. 3

[16] Yuwei Guo, Ceyuan Yang, Anyi Rao, Maneesh Agrawala,
Dahua Lin, and Bo Dai. Sparsectrl: Adding sparse controls
to text-to-video diffusion models. In European Conference
on Computer Vision, pages 330–348. Springer, 2024. 2
[17] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo
Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling
camera control for text-to-video generation. arXiv preprint
arXiv:2404.02101, 2024. 2

[18] Zhihao Hu and Dong Xu. Videocontrolnet: A motion-
guided video-to-video translation framework by using diffu-
sion model with controlnet. In Proceedings of the 31st ACM
International Conference on Multimedia, pages 6467–6476,
2023. 2

[19] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si,
Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin,
Nattapol Chanpaisit, et al. Vbench: Comprehensive bench-
mark suite for video generative models. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 21807–21818, 2024. 6

[20] Sungwon Hwang, Hyojin Jang, Kinam Kim, Minho Park,
and Jaegul Choo. Cross-frame representation alignment
arXiv preprint
for fine-tuning video diffusion models.
arXiv:2506.09229, 2025. 5

[21] Mingkai Jia, Mingxiao Li, Liaoyuan Fan, Tianxing Shi, Ji-
axin Guo, Zeming Li, Xiaoyang Guo, Xiao-Xiao Long, Qian
Zhang, Ping Tan, et al. Dino-tok: Adapting dino for visual
tokenizers. arXiv preprint arXiv:2511.20565, 2025. 3
[22] J. Jiang, G. Hong, M. Zhang, H. Hu, K. Zhan, R. Shao, and
L. Nie. Dive: Efficient multi-view driving scenes genera-
tion based on video diffusion transformer. arXiv preprint
arXiv:2504.19614, 2025. 3

[23] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang,
Yulin Pan, and Yu Liu. Vace: All-in-one video creation and
editing. arXiv preprint arXiv:2503.07598, 2025. 2, 4
[24] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao,
Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer White-
head, Alexander C Berg, Wan-Yen Lo, et al. Segment any-
thing. In Proceedings of the IEEE/CVF international confer-
ence on computer vision, pages 4015–4026, 2023. 2
[25] Yang Liu, Chuanchen Luo, Zimo Tang, Yingyan Li, Yuany-
ong Ning, Lue Fan, Junran Peng, Zhaoxiang Zhang, et al.
Tc-light: Temporally coherent generative rendering for real-
istic world transfer. In The Thirty-ninth Annual Conference
on Neural Information Processing Systems, 2025. 3, 5

9

[26] Jiachen Lu, Ze Huang, Zeyu Yang, Jiahui Zhang, and Li
Zhang. Wovogen: World volume-aware diffusion for con-
In Eu-
trollable multi-camera driving scene generation.
ropean Conference on Computer Vision, pages 329–345.
Springer, 2024. 3

[27] Wan-Duo Kurt Ma, John P Lewis, and W Bastiaan Kleijn.
Trailblazer: Trajectory control for diffusion-based video
In SIGGRAPH Asia 2024 Conference Papers,
generation.
pages 1–11, 2024. 2

[28] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jia-
jun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided
image synthesis and editing with stochastic differential equa-
tions. arXiv preprint arXiv:2108.01073, 2021. 2, 3

[29] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy
Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,
Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al.
Dinov2: Learning robust visual features without supervision.
arXiv preprint arXiv:2304.07193, 2023. 4

[30] Taesung Park, Alexei A Efros, Richard Zhang, and Jun-
Yan Zhu. Contrastive learning for unpaired image-to-image
In Computer Vision–ECCV 2020: 16th Euro-
translation.
pean Conference, Glasgow, UK, August 23–28, 2020, Pro-
ceedings, Part IX 16, pages 319–345. Springer, 2020. 3
[31] Gaurav Parmar, Taesung Park, Srinivasa Narasimhan, and
Jun-Yan Zhu. One-step image translation with text-to-image
models. arXiv preprint arXiv:2403.12036, 2024. 2

[32] William Peebles and Saining Xie. Scalable diffusion mod-
els with transformers. In Proceedings of the IEEE/CVF In-
ternational Conference on Computer Vision (ICCV), pages
4195–4205, 2023. 3

[33] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning
transferable visual models from natural language supervi-
sion. In International conference on machine learning, pages
8748–8763. PmLR, 2021. 4

[34] Stephan R Richter, Hassan Abu AlHaija, and Vladlen
Koltun. Enhancing photorealism enhancement. IEEE Trans-
actions on Pattern Analysis and Machine Intelligence, 45(2):
1700–1715, 2022. 3, 6

[35] Lloyd Russell, Anthony Hu, Lorenzo Bertoni, George Fe-
doseev, Jamie Shotton, Elahe Arani, and Gianluca Corrado.
Gaia-2: A controllable multi-view generative world model
for autonomous driving. arXiv preprint arXiv:2503.20523,
2025. 3, 5

[36] Minglei Shi, Haolin Wang, Wenzhao Zheng, Ziyang Yuan,
Xiaoshi Wu, Xintao Wang, Pengfei Wan, Jie Zhou, and Ji-
wen Lu. Latent diffusion model without variational autoen-
coder. arXiv preprint arXiv:2510.15301, 2025. 3

[37] Oriane Sim´eoni, Huy V Vo, Maximilian Seitzer, Federico
Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov,
Marc Szafraniec, Seungeun Yi, Micha¨el Ramamonjisoa,
et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025. 2

[38] Jiaming Song, Chenlin Meng,

Denoising diffusion implicit models.
arXiv:2010.02502, 2020. 3

and Stefano Ermon.
arXiv preprint

[39] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muham-
Ibrahim Alabdulmohsin, Nikhil

mad Ferjad Naeem,

10

Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil
Mustafa, et al. Siglip 2: Multilingual vision-language en-
coders with improved semantic understanding, localization,
and dense features. arXiv preprint arXiv:2502.14786, 2025.
4

[40] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao,
Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianx-
iao Yang, et al. Wan: Open and advanced large-scale video
generative models. arXiv preprint arXiv:2503.20314, 2025.
4

[41] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea
Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Vi-
sual geometry grounded transformer. In Proceedings of the
Computer Vision and Pattern Recognition Conference, pages
5294–5306, 2025. 2

[42] Wenhai Wang, Jifeng Dai, Zhe Chen, Zhenhang Huang,
Zhiqi Li, Xizhou Zhu, Xiaowei Hu, Tong Lu, Lewei Lu,
Hongsheng Li, et al. Internimage: Exploring large-scale vi-
In
sion foundation models with deformable convolutions.
Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition, pages 14408–14419, 2023. 6

[43] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Ji-
uniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jin-
gren Zhou. Videocomposer: Compositional video synthesis
with motion controllability. Advances in Neural Information
Processing Systems, 36:7594–7611, 2023. 2

[44] Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Jia-
gang Zhu, and Jiwen Lu. Drivedreamer: Towards real-world-
In European
drive world models for autonomous driving.
conference on computer vision, pages 55–72. Springer, 2024.
3

[45] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang,
Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chun-
hua Shen, and Tong He. π3: Permutation-equivariant visual
geometry learning. arXiv preprint arXiv:2507.13347, 2025.
2

[46] Thomas Wimmer, Prune Truong, Marie-Julie Rakotosaona,
Michael Oechsle, Federico Tombari, Bernt Schiele, and
Jan Eric Lenssen. Anyup: Universal feature upsampling.
arXiv preprint arXiv:2510.12764, 2025. 5, 13

[47] Dianbing Xi, Jiepeng Wang, Yuanzhi Liang, Xi Qiu, Yuchi
Huo, Rui Wang, Chi Zhang, and Xuelong Li. Omnivdiff:
Omni controllable video diffusion for generation and under-
standing. arXiv preprint arXiv:2504.10825, 2025. 2

[48] D. Xi, J. Wang, Y. Liang, X. Qiu, J. Liu, H. Pan, et al. Ctr-
lvdiff: Controllable video generation via unified multimodal
video diffusion. arXiv preprint arXiv:2511.21129, 2025. 3

[49] Gangwei Xu, Haotong Lin, Hongcheng Luo, Xianqi Wang,
Jingfeng Yao, Lianghui Zhu, Yuechuan Pu, Cheng Chi,
Haiyang Sun, Bing Wang, et al. Pixel-perfect depth with
semantics-prompted diffusion transformers. arXiv preprint
arXiv:2510.07316, 2025. 2

[50] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi
Feng, and Hengshuang Zhao. Depth anything: Unleashing
the power of large-scale unlabeled data. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 10371–10381, 2024. 2

[51] Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy.
Fresco: Spatial-temporal correspondence for zero-shot video
In Proceedings of the IEEE/CVF Conference
translation.
on Computer Vision and Pattern Recognition, pages 8703–
8712, 2024. 5

[52] Yu Zeng, Charles Ochoa, Mingyuan Zhou, Vishal M Pa-
tel, Vitor Guizilini, and Rowan McAllister. Neuralremaster:
Phase-preserving diffusion for structure-aligned generation.
arXiv preprint arXiv:2512.05106, 2025. 6

[53] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding
conditional control to text-to-image diffusion models.
In
Proceedings of the IEEE/CVF international conference on
computer vision, pages 3836–3847, 2023. 2, 3

[54] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Scal-
ing in-the-wild training for diffusion-based illumination har-
monization and editing by imposing consistent light trans-
port. In The Thirteenth International Conference on Learn-
ing Representations, 2025. 3

[55] Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng
Zhang, Wangmeng Zuo, and Qi Tian.
Controlvideo:
Training-free controllable text-to-video generation 2023.
URL https://arxiv. org/abs/2305.13077, 2023. 2

[56] Guosheng Zhao, Xiaofeng Wang, Zheng Zhu, Xinze
Chen, Guan Huang, Xiaoyi Bao, and Xingang Wang.
Drivedreamer-2: Llm-enhanced world models for diverse
driving video generation. In Proceedings of the AAAI Con-
ference on Artificial Intelligence, pages 10412–10420, 2025.
3

[57] Shihao Zhao, Dongdong Chen, Yen-Chun Chen, Jianmin
Bao, Shaozhe Hao, Lu Yuan, and Kwan-Yee K. Wong.
Uni-controlnet: All-in-one control to text-to-image diffusion
models. In Advances in Neural Information Processing Sys-
tems, pages 1354–1367, 2023. 2

[58] Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining
Xie. Diffusion transformers with representation autoen-
coders. arXiv preprint arXiv:2510.11690, 2025. 2, 3
[59] Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A
Efros. Unpaired image-to-image translation using cycle-
consistent adversarial networks. In Proceedings of the IEEE
international conference on computer vision, pages 2223–
2232, 2017. 3

11

A. Implementation details

A.1. Training Details

We leverage the Nuplan dataset [5] to train DwD. To ad-
dress the inconsistent quality of raw sequences, we imple-
ment an automated data curation pipeline utilizing Qwen3-
VL [3]. This pipeline filters out samples exhibiting visual
degradation—such as over-saturation, monochromatic arti-
facts, or other anomalies—and generates captions for the re-
maining data. Consequently, we compile a curated dataset
of 6,000 videos, each containing 200 frames.To ensure a
stable input space for DwD, we address the inherent sign
ambiguity of PCA eigenvectors by establishing a consis-
tent global feature space via Incremental PCA. Specifically,
we adopt a sparse sampling strategy that extracts a single
random frame from each video to compute a fixed global
basis (comprising the mean and eigenvectors). This ap-
proach maximizes environmental diversity (e.g., weather
conditions, time of day) while mitigating sequence-specific
biases. All feature embeddings are projected onto this basis
prior to training.During training, video sequences are re-
sized to a resolution of 1280 × 704 and randomly cropped
into temporal chunks of 93 frames. We adopt a focused
fine-tuning strategy: the parameters of the original denois-
ing DiT are frozen, while only the copied conditional DiT
is updated. Due to memory constraints (48GB VRAM), we
employ Context Parallelism on a single node with 8 GPUs.
The model is trained for 20k iterations with a total batch
size of 1 and a learning rate of 5 × 10−5.

A.2. Data rendered from Simulator

We synthesized video sequences using the CARLA simula-
tor across various pre-defined maps (Towns) with random-
ized spawn and destination points. Dynamic agents, includ-
ing vehicles and pedestrians, were governed by CARLA’s
built-in Traffic Manager to simulate realistic traffic flow. In
total, we generated 149 videos, each averaging 200 frames.
To further evaluate the generalizability of our DwD, we ex-
tended our experiments to several custom-built scenes. One
visual result is illustrated in Fig. 9.

A.3. Spatial Alignment Module and Causal Tempo-

ral Aggregator

As illustrated in Figure 6, the Spatial Alignment Module
consists of spatial convolutions and residual blocks. When
integrating the Causal Temporal Aggregator, we prefix
the input DINO latent sequence with zero-padding to en-
sure the temporal convolutions maintain causality.

B. Qualitative Comparisons across Ablated

Hyperparameters

In this section, we provide extensive visual comparisons to
substantiate the quantitative observations discussed in the

Figure 6. The Spatial Alignment Module and Causal Temporal
Aggregator.

main paper. specifically, we examine the impact of key hy-
perparameters and architectural choices on the visual qual-
ity, structural fidelity, and temporal consistency of the gen-
erated videos.

Figure 7. Qualitative Comparison across Minor Components
Pruning Degree (k)

B.1. Analysis of Naive DINO Latent Injection

As hypothesized in the main methodology, directly utiliz-
ing non-pruned DINO latents as the control signal intro-
duces significant information redundancy. We visualize this
phenomenon in the first row of Fig. 7. Compared to DwD,
the naive implementation exhibits severe texture leakage,
where the high-frequency surface details of the source CG

12

1x3x3ConvCasual ZeroPad3x1x1ConvResBlock3D×max 𝑆2,𝑇23x3x3Conv3x3x3ConvGroupNormSiLU×2⭐⭐The stride is configurable to 1 or 2.Temporal Ops⭐CGDwD DINOv3-LDwD pca32DwD pca8(final)DwD pca3input (e.g., wireframes or flat shading artifacts) are inad-
vertently preserved in the generated output. This leakage
prevents the generative model from effectively synthesiz-
ing new textures, resulting in a video that appears merely
as a filtered version of the input rather than a semantically
consistent regeneration. In contrast, DwD successfully dis-
entangles structure from texture, preserving the geometric
layout while allowing for realistic rendering.

B.2. Impact of Minor Components Pruning Degree

(k′)

The pruning of minor components, modulated by the PCA
clamping factor k′, functions as a critical regularizer within
the information bottleneck framework. As illustrated in
Fig. 7, this parameter regulates the trade-off between struc-
tural guidance and generative flexibility:
• Insufficient Pruning (k′ = 32): A larger k preserves ex-
cessive trivial feature variance. Consistent with the quan-
titative results in the main paper, this results in a degra-
dation of realism, as the model becomes over-constrained
by noise and irrelevant artifacts inherited from the source
domain.

• Excessive Pruning (k′ = 3): Conversely, overly aggres-
sive pruning compromises the semantic integrity of the
scene. At k = 3, the control signal becomes too sparse
to accurately encapsulate complex geometries, leading to
structural hallucinations where the generated content de-
viates from the layout of the input CG scene. For in-
stance, as shown in Fig. 7 (Row 5), a bridge is erroneously
synthesized in the distant background. Furthermore, this
configuration undermines a key advantage of DwD: its ro-
bustness against degradation during autoregressive long-

Figure 8. Qualitative Comparison across different Upscaling Fac-
tors ×S in Spatial Resolution Enhancement. The red boxes high-
light inconsistencies with respect to the CG input.

13

video generation, which will be discussed subsequently.

B.3. Effect of Spatial Upscaling Factor (S)

We further investigate the upscaling factor S of Spatial Res-
olution Enhancement. Visual results in Fig. 8 indicate a
positive correlation between the spatial upscaling factor S
and structural consistency. S = 4 provide fine-grained geo-
metric cues, enabling the model to align generated instances
more precisely with the input. Lower resolutions (S = 1)
tend to produce incorrect stuffs, like translating street lamps
into traffic light.

B.4. Efficacy of the Causal Temporal Aggregator

Finally, we evaluate the contribution of the Causal Tem-
poral Aggregator by comparing it with a naive and popu-
lar key-frame sampling strategy. As illustrated in Fig. 10,
under fast camera jittering, The naive approach leads to
In contrast,
observable motion blurs and discontinuities.
our Temporal Aggregator effectively smooths the transi-
tions across frames. The visual comparison highlights that
our module significantly mitigates motion blur and flicker-
ing artifacts, yielding video sequences with superior tempo-
ral coherence and stability.

C. Upsampling using Anyup

While feature upsampling methods such as AnyUp [46] ap-
pear to offer a computationally efficient alternative for ob-
taining high-resolution DINO feature maps, we empirically
find that they consistently lead to unstable controllability
in video generation, resulting in the suboptimal quantita-
tive metrics and poor visual quality. To investigate this, we
visualize the PCA components of feature maps upsampled
via AnyUp versus naive input upscaling in Fig. 11.We ob-
serve that since AnyUp incorporates high-resolution RGB
images as guidance, it inadvertently introduces unnecessary
high-frequency pixel information, leading to feature quality
degradation. As highlighted by the red boxes, the pixel-
level guidance causes AnyUp to misinterpret texture differ-
ences (e.g., between the curb and the neighboring ground)
as distinct semantic categories, whereas naive upscaling
maintains a more coherent semantic representation. Fur-
thermore, as shown in the green boxes, AnyUp fails to re-
cover fine-grained structural details despite an 8× upsam-
pling factor; for instance, the boundaries between distant
street lamps and the sky remain indistinct. These visual ob-
servations corroborate our experimental findings, confirm-
ing that the feature degradation inherent in AnyUp is the
primary cause of the generation instability.

D. Robust Long Video Generation using DwD

A distinctive advantage of DwD is its superior robustness
in autoregressive long-horizon driving video generation.

CGDwD pca8+x1DwD pca8+x4Figure 9. Qualitative comparison of autoregressive long video generation.

14

Frame 622CGC-T2.5 Deblur[Ali et al . 2025]C-T2.5 Depth[Ali et al . 2025]C-T2.5 Edge[Ali et al . 2025]C-T2.5 Seg[Ali et al . 2025]C-T2.5 Depth+Edge[Ali et al . 2025]FRESCO[Yang et al. 2024b]TC-Light[Liu et al. 2025]DwD(ours)Frame 0Figure 10. Qualitative Comparison w. and w./o. Causal Temporal Aggregator

Figure 11. PCA visualization comparing feature maps upsampled via AnyUp (×4 and ×8) versus naive input upscaling(×4). The red
boxes highlight semantic inconsistencies introduced by AnyUp, where high-frequency pixel information leads to false semantic boundaries
(e.g., misclassifying the curb). The green boxes demonstrate the failure of AnyUp to recover fine-grained structural details, such as the
boundaries of distant street lamps, despite the high upsampling factor.

Leveraging the Cosmos Predict 2.5 backbone with a 93-
frame chunk size, we observe that most Cosmos Transfer
2.5 control branches suffer from severe error accumulation
and degradation after approximately three generation cy-
cles. In contrast, our method effectively maintains visual
fidelity, avoiding common artifacts such as color oversatu-
ration. Qualitative comparisons are illustrated in Fig. 9; we
explicitly recommend reviewing the supplementary videos
for a dynamic evaluation of temporal consistency.

E. Extra Qualitative Comparisons

15

w/ Tempw/o TempAnyup x4Input Anyup x8Figure 12. Extra Qualitative comparison with state-of-the-art methods. The red boxes highlight inconsistencies with respect to the CG
input, while the green boxes point out low-fidelity textures.

16

CGC-T2.5 Deblur[Ali et al . 2025]C-T2.5 Depth[Ali et al . 2025]C-T2.5 Edge[Ali et al . 2025]C-T2.5 Seg[Ali et al . 2025]C-T2.5 Depth+Edge[Ali et al . 2025]FRESCO[Yang et al. 2024b]TC-Light[Liu et al. 2025]DwD(ours)Figure 13. Extra Qualitative comparison with state-of-the-art methods. The red boxes highlight inconsistencies with respect to the CG
input, while the green boxes point out low-fidelity textures.

17

CGC-T2.5 Deblur[Ali et al . 2025]C-T2.5 Depth[Ali et al . 2025]C-T2.5 Edge[Ali et al . 2025]C-T2.5 Seg[Ali et al . 2025]C-T2.5 Depth+Edge[Ali et al . 2025]FRESCO[Yang et al. 2024b]TC-Light[Liu et al. 2025]DwD(ours)