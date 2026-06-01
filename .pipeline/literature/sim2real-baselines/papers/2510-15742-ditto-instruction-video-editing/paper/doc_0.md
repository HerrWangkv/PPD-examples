Scaling Instruction-Based Video Editing with a High-Quality Synthetic Dataset

Qingyan Bai1,2, Qiuyu Wang2, Hao Ouyang2, Yue Yu1,2, Hanlin Wang1,2,
Wen Wang2,3, Ka Leong Cheng2, Shuailei Ma2,4, Yanhong Zeng2,
Zichen Liu1,2, Yinghao Xu2, Yujun Shen2, Qifeng Chen1
3Zhejiang University

2Ant Group

4Northeastern University

1HKUST

5
2
0
2

c
e
D
7
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
2
4
7
5
1
.
0
1
5
2
:
v
i
X
r
a

Figure 1. Our proposed synthetic data generation pipeline can automatically produce high-quality and highly diverse video editing data,
encompassing both global and local editing tasks. We highly recommend the readers see the supplementary video samples.

Abstract

Instruction-based video editing promises to democratize
content creation, yet its progress is severely hampered by
the scarcity of large-scale, high-quality training data. We
introduce Ditto, a holistic framework designed to tackle
this fundamental challenge. At its heart, Ditto features a
novel data generation pipeline that fuses the creative di-
versity of a leading image editor with an in-context video
generator, overcoming the limited scope of existing mod-
els. To make this process viable, our framework resolves the
prohibitive cost-quality trade-off by employing an efficient,
distilled model architecture augmented by a temporal en-

hancer, which simultaneously reduces computational over-
head and improves temporal coherence. Finally, to achieve
full scalability, this entire pipeline is driven by an intelli-
gent agent that crafts diverse instructions and rigorously
filters the output, ensuring quality control at scale. Using
this framework, we invested over 12,000 GPU-days to build
Ditto-1M, a new dataset of one million high-fidelity video
editing examples. We trained our model, Editto, on Ditto-
1M with a curriculum learning strategy. The results demon-
strate superior instruction-following ability and establish a
new state-of-the-art in instruction-based video editing. The
data, model, and code can be found at the project page.

1

Make it in the style of Japanese anime.Transform the scene into a whimsical pastry dream world where the cake is a floating island.Replace the black dog with a white fox sitting calmly beside them.Imitate the look of the 3D Chibi style.Add a glowing vintage streetlamp casting a warm yellow hue on the pavement near the couple.Reimagine the industrial setting in a cybernetic brain lab where the tablet is a neural interface. 
 
 
 
 
 
1. Introduction

Recently, the field of visual generative models has wit-
nessed a remarkable divergence: while instruction-based
image editing has achieved unprecedented levels of pre-
cision and user-friendliness with models like Instruct-
Pix2Pix [6], FLUX.1 Kontext [3], Qwen-Image [47], and
its video
Gemini 2.5 Flash Image (Nano-Banana) [16],
counterpart has lagged significantly behind. This grow-
ing capabilities gap stems from the inherent complexities
of the temporal dimension. Editing a video requires not
only modifying content but also ensuring these changes
are propagated coherently across frames—a challenge that
has proven formidable. The primary obstacle impeding
progress is a well-understood but unsolved problem:
the
profound scarcity of large-scale, high-quality, and diverse
paired data for training end-to-end video editing mod-
els [53, 54].

Existing works have attempted to address this data
scarcity challenge through various synthetic data genera-
tion strategies. Earlier approaches either relied on compu-
tationally prohibitive per-video optimization methods [36]
or adopted training-free image-to-video propagation tech-
niques [49, 53]. However, these pipelines suffer from a per-
sistent trade-off: they sacrifice editing diversity, temporal
consistency, and visual quality for scalability, or vice-versa.
A scalable, cost-efficient data pipeline that generate high-
fidelity results remains an open challenge.

To address these shortcomings, we introduce Ditto1, a
scalable and cost-efficient data synthesis pipeline archi-
tected to systematically dismantle these trade-offs. Our
approach first
tackles the challenge of editing fidelity
and diversity. Capitalizing on the advanced maturity of
instruction-based image editors, the pipeline generates a
high-quality edited reference frame that acts as a strong
visual prior. This anchor frame then guides an in-context
video generator [20] to synthesize a temporally coherent
video that faithfully matches the edit, overcoming the qual-
ity limitations of previous methods. Second, to resolve the
critical efficiency-coherence trade-off where high-fidelity
generation is prohibitively expensive, our pipeline inte-
grates a distilled video model with a temporal enhancer.
This innovative combination reduces computational costs
to just 20% of the original while preserving temporal sta-
bility and avoiding visual artifacts. Finally, to achieve true
scalability and eliminate the bottleneck of manual curation,
we deploy an autonomous Vision-Language Model (VLM)
agent. This agent carries dual responsibilities: program-
matically generating diverse instructions for both local and
global edits, and serving as a flaw-detection mechanism to
automatically filter out low-quality or failed video pairs, en-

1The name “Ditto” is chosen to reflect the model’s core function: mak-
ing the output video a faithful reflection, or “ditto,” of the user’s textual
instruction.

suring the final dataset’s integrity.

We invested over 12,000 GPU-days using this pipeline to
construct Ditto-1M, a new large-scale dataset comprising
over one million source-instruction-edited video triplets, as
demonstrated in Fig. 1. The dataset is meticulously struc-
tured to cover a wide spectrum of editing tasks and is cu-
rated via our VLM agent to ensure instruction consistency
and high aesthetic quality.

With the proposed dataset, we train our final editing
model, Editto. To bridge the gap between our visually-
guided data synthesis and the goal of purely instruction-
driven inference, we propose a modality curriculum learn-
ing strategy [4]. Our curriculum begins by providing the
model with both the text instruction and the edited reference
image as a “scaffold.” As training progresses, we gradually
anneal the visual guidance, compelling the model to learn
the more difficult, abstract mapping from text instruction
alone.

Our contributions are as follows:

• A novel, scalable synthesis pipeline, Ditto,

that effi-
ciently generates high-fidelity and temporally coherent
video editing data.

• The Ditto-1M Dataset, a million-scale, open-source col-
lection of instruction-video pairs to facilitate community
research.

• A state-of-the-art editing model, trained on Ditto-1M, that
demonstrates superior performance on established bench-
marks.

• A modality curriculum learning strategy that effec-
tively enables a visually-conditioned model to perform
language-driven editing.

2. Related Work

2.1. Instruction-based Image Editing

Visual generative models have advanced rapidly [1, 5, 7,
12, 13, 15, 17–19, 23–25, 27, 30, 37, 39–45, 50, 56].
Instruction-based image editing has also rapidly evolved,
moving beyond simple text-to-image generation to enable
nuanced, user-guided modifications. Early and influential
methods like InstructPix2Pix [6] demonstrated the feasibil-
ity of fine-tuning diffusion models on generated datasets of
image triplets (source image, instruction, edited image) to
perform edits. This was achieved by ingeniously combin-
ing a large language model (GPT-3) to generate textual edit
instructions and a text-to-image model (Stable Diffusion) to
synthesize the corresponding image pairs, creating a large-
scale training corpus without manual annotation. More re-
cent advancements, particularly with the advent of pow-
erful models like FLUX.1 Kontext [3], Qwen-Image [47],
and Gemini 2.5 Flash Image (Nano-Banana) [16], have un-
locked even more sophisticated capabilities. These mod-
els can process both text and reference images as inputs,

2

enabling targeted local edits, robust character consistency
across multiple turns, and complex scene transformations
within a unified architecture, often without requiring fine-
tuning. Our work builds upon this progress by integrating
a state-of-the-art instruction-based image editor as a critical
component in our data synthesis pipeline, using it to manip-
ulate keyframes that guide the subsequent video-level edit.

2.2. Instruction-based Video Editing

Video editing has gained remarkable progress [8, 9, 28, 46,
51] with the development of the base generative models.
Extending instruction-based editing to video requires main-
taining temporal consistency and preserving background
content. Current approaches fall into two main categories:
Inversion-based Methods. These methods avoid paired
video-text-edit data but are computationally intensive.
Tune-A-Video [48] fine-tunes a text-to-image model on a
single video, enabling personalized edits but lacking scal-
ability. Zero-shot techniques like TokenFlow [14] and
FateZero [35] use DDIM inversion and feature propagation
to enforce the consistency of the edited video. However,
their quality relies on inversion fidelity and often struggles
with complex motion or occlusions.
Feed-forward Methods. These end-to-end models aim
to overcome inversion-based limitations but face the fun-
damental challenge of data scarcity. The development of
feed-forward approaches is tightly coupled with the creation
of synthetic datasets, as large-scale human-annotated video
edit data is notoriously scarce. Early works [36, 55] at-
tempted to synthesize data using computationally expensive
one-shot tuning methods [32, 48], which limited the scale
and quality of the resulting datasets. More recent paradigms
have sought greater scalability with the approach of “lift
and propagate”, employed by methods like VEGGIE [53]
and InsViE [49]. These methods edit a single keyframe and
then use an image-to-video model to propagate the change,
but often suffer from temporal inconsistencies as the qual-
ity is capped by the propagation model. Se˜norita [57]
achieves notable progress with a sophisticated “expert sys-
tem” paradigm. It systematically categorizes editing tasks
into 18 sub-classes and employs a large suite of special-
ized expert models - some newly trained - to generate high-
quality data for each specific task. While ensuring quality
for predefined tasks, this approach is less scalable and re-
quires significant effort to develop and maintain numerous
task-specific models. In stark contrast to these specialized
or propagation-based pipelines, our work introduces a uni-
fied and scalable ”All-in-One” data synthesis framework.
Our pipeline is centered around a single in-context video
generator that conditions on a reference edited frame from a
single image editor and a depth-derived motion representa-
tion, enabling more direct and high-quality video synthesis
without relying on a multitude of disparate expert models.

Crucially, our contribution extends beyond the dataset itself.
We propose a novel Modality Curriculum Learning (MCL)
strategy during training. This allows our final model, Editto,
to perform edits based purely on text instructions at infer-
ence time, bridging the gap between multi-modal data syn-
thesis and single-modality deployment. Finally, the concur-
rent work EditVerse [21] also explores in-context learning
to unify editing tasks. Instead, we leverage in-context gen-
eration primarily for high-quality data synthesis.

3. Ditto-1M

Our methodology begins with the construction of a large-
scale, high-quality dataset. We designed a novel, scal-
able data generation pipeline to synthesize over one mil-
lion instruction-video triplets, as in Fig. 2. The architec-
ture of this pipeline was specifically engineered to address
four critical challenges inherent to existing data synthesis
approaches:
1. Overcoming Limited Editing Diversity and Fidelity.
Current data pipelines of instruction-based video editing
often rely on training-free inversion techniques [36, 54],
which tend to yield synthetic data of limited quality. To ad-
dress this, we propose to leverage an in-context video gen-
erator to produce high-quality editing samples with visual
contexts. Capitalizing on the more advanced development
of image-based editing models, we incorporate strong pri-
ors from these image editors to serve context and guide the
video generation for better editing quality. This is combined
with depth-guided video context to ensure spatiotemporal
coherence, significantly improving the diversity and fidelity
of generated edits.
2. Resolving the Efficiency-Quality Trade-off. A major
technical hurdle is the trade-off between generation cost and
data quality. Current high-fidelity methods are prohibitively
expensive (e.g., 50 GPU-minutes per sample on a single
GPU), while faster, distilled models often introduce arti-
facts like temporal flickering. Our pipeline is designed with
a cost-aware workflow that significantly reduces computa-
tional overhead without compromising the temporal coher-
ence of the videos.
3. Automating Instruction Generation and Quality Con-
trol. To achieve true scalability, manual creation of instruc-
tions and verification of outputs is infeasible. Our pipeline
integrates an automated agent with two primary responsi-
bilities: (a) programmatically generating diverse and mean-
ingful instructions for both local and global edits, and (b)
serving as a flaw-detection mechanism to automatically fil-
ter out generated pairs that are of low quality or fail to fol-
low the instructions.
4. Ensuring High Aesthetic and Motion Quality. Unlike
general-purpose video datasets (e.g., Panda-70M), which
are not optimized for editing tasks, our pipeline priori-
tizes the generation of content with high aesthetic value

3

Figure 2. Our scalable data synthesis pipeline. (1) Pre-processing: A diverse video pool is curated via automated deduplication and motion
filtering. (2) The core engine synthesizes video triplets, conditioning an in-context generator on automated instructions, appearance context
from edited key-frames, and structural context from depth maps. (3) Post-processing: Final visual quality is guaranteed by a VLM-based
filter and a denoising enhancer. For the sake of reproducibility, every component in our pipeline employs an open-source model.

and natural motion dynamics. This focus ensures the re-
sulting dataset, and the models trained upon it, are well-
aligned with real-world usage scenarios where visual appeal
is paramount.

The following sections will detail the architecture of our
data generation pipeline, explaining how each component
systematically addresses these challenges.

3.1. Source Video Filtering

is built exclusively from high-
The Ditto-1M dataset
resolution videos sourced from Pexels [34], a platform for
professional-grade footage under the Pexels License. Un-
like datasets derived from uncurated web scrapes, this strat-
egy provides a foundation of superior aesthetic and techni-
cal quality, suitable for video editing tasks. We also first
apply a rigorous filtering and pre-processing protocol. This
protocol examines videos in the following aspects:
Near-Duplicate Removal: To prevent dataset redundancy
and ensure broad content diversity, we implement a rigor-
ous deduplication process. We employ a powerful visual
encoder [31] to extract compact feature representations for
each video. Pairwise similarity between these feature vec-
tors is then computed. Videos exceeding a pre-defined simi-
larity threshold are systematically filtered out, guaranteeing
the uniqueness of each source video in our collection.

Motion Scale: Videos that contain little or no motion over
time—such as fixed-camera surveillance footage, still na-
ture scenes, or unmoving interior shots—are considered
less valuable for video editing tasks because they lack dy-
namic visual changes. To automatically identify such low-
dynamic content, we employ a tracking-based method that
analyzes frame-to-frame motion across the video sequence.
Specifically, for each video, we first sample points on a grid
layout and then use CoTracker3 [22] to track these points,
obtaining their trajectories. We then compute the average of
the cumulative displacements of all tracked points over the
entire video as the motion score of the video. By setting a
threshold, we filter out videos with low motion scores, ef-
fectively removing those with negligible temporal variation.
Videos that pass this filtering stage are then standardized.
Each video is resized to a uniform resolution and its frame
rate is converted to 20 FPS. This standardization simplifies
the training process and ensures consistency across the en-
tire dataset.

3.2. Instruction Generation

For each filtered source video Vs, we generate a set of cor-
responding editing instructions p. We employ a powerful
VLM Qwen2.5 VL [2] for this task with a two-step prompt-
ing strategy. First, we prompt the VLM to generate a dense

4

VLMFilteringLow noiseDenoisingEnhancerRule-based Filtering of Edited VideosQuality Enhancing of Edited VideosMotion  Filtering via TrackingPointTracking끫룊2끫룊3끫룊6(cid:2201)(cid:2191)(cid:2195)>(cid:2202)(cid:2190)(cid:2200)Delete끫룊0끫룊4끫룊1끫룊5끫룊7끫룊(cid:3041)Deduplication with RepresentationsVisionEncoderSource Videos…TagsSceneSubject …VLMExtraction of Tags and Representations RepresentationsKey-Frames1. Video Caption2. Edit Instructions Source videoImage EditorDepthPredictorVLMIn-Context                            Video GeneratorEdited Key-FramesDepth VideosEdit Instructions Edited Videos1. Pre-process2. Generation3. Post-process(~60GPU-Days)(~6000 GPU-Days)(~6000 GPU-Days)Edited Key-Framescaption c that describes the video’s content, subjects, and
scenery:

c = VLM(Vs, pcaption).

(1)

This caption serves as a semantic anchor. Next, we feed
both the video Vs and its caption c back into the VLM,
prompting it to devise a creative and plausible editing in-
struction p:

p = VLM(Vs, c, pinstruct).

(2)

This conditioned approach ensures that instructions are con-
textually grounded in the video’s content, yielding a diverse
set of commands ranging from global style transformations
to specific, localized object modifications.

3.3. Visual Context Preparation

Our generation process is heavily guided by a rich visual
context, which consists of two key components: an edited
reference frame that specifies the target appearance, and a
depth video that enforces spatiotemporal consistency.
Key-Frame Editing for Appearance Guidance. We first
select a key-frame fk from the source video Vs as the anchor
for the editing. This frame is then edited by the instruction-
guided image editor Qwen-Image [47] Eimg, using the in-
struction p generated in the previous step:

f ′
k = Eimg(fk, p).

(3)

This resulting frame f ′
k, serves as the visual prototype for
the edit, defining the final appearance including style and
textures.
Depth Video Prediction for Spatiotemporal Structure.
To preserve the geometric structure and motion dynamics
of the original scene, we extract a dense depth video Vd
from Vs with a video depth predictor D [10]. The predicted
depth video acts as a dynamic structural scaffold, provid-
ing an explicit, frame-by-frame guide for the structure and
geometry of the scene during the video generation.

3.4. In-Context Video Generation

With the editing instruction p and the multi-modal visual
context f ′
k and Vd prepared, we synthesize the edited video
Ve with the in-context video generator [20], which is de-
noted as G. VACE is a feed-forward video generative model
designed to condition its generation on rich visual prompts
such as images, masks, and videos by learning a context
branch beyond the base generative model [43]. In our de-
sign, we adopt G to synthesize the edited video by taking the
textual prompt p as a high-level semantic guide, the edited
key-frame f ′
k as the primary appearance condition, and the
depth video Vd as a strict spatiotemporal constraint. This
generation process is formulated as:

Ve = G(Vd, f ′

k, p).

(4)

5

By integrating these three modalities with the attention
mechanism, VACE can faithfully propagate the edit defined
in f ′
k across the entire sequence, adhering to the motion and
structure laid out by Vd, while ensuring the result is seman-
tically aligned with the instruction p. Our pipeline achieves
high-quality and coherent video edits without costly per-
video optimization. Please refer to the supplementary ma-
terials for additional analysis of the data generator. To fa-
cilitate scalable synthetic data generation and further re-
duce the computational burden, we employ model quantiza-
tion and knowledge distillation techniques [52]. We apply
post-training quantization to reduce the model’s memory
footprint and inference cost with minimal impact on out-
put quality. Furthermore, we adopt the generative video
model [52] distilled from the teacher model, preserving
editing fidelity while significantly accelerating the gen-
eration process with few-step inference. This optimized
pipeline is crucial for producing large-scale video editing
data efficiently.

3.5. Edited Video Curation and Enhancing

To guarantee the highest quality, the generated triplets (Vs,
p, Ve) undergo a final two-stage curation and refinement in-
cluding VLM filtering and denoiser enhancing.
VLM-Based Curation. We first use a VLM [2] as an au-
tomated judge to perform rejection sampling. Each triplet
is evaluated against two criteria: (1) Instruction Fidelity:
whether the edit in Ve accurately reflects the prompt p. (2)
Fidelity: whether Ve preserves the semantic and motion
from Vs.
(3) Visual quality: whether the videos are vi-
sual appealing without significant distortion or artifacts. (4)
Safety & Appropriateness: whether the content has unsafe
or inappropriate material, such as pornography, violence, or
horror, ensuring the dataset is ethically compliant and suit-
able. Triplets that fail to meet our quality thresholds on
these criteria are discarded.
Quality Enhancement via Denoising. The curated edited
videos are then enhanced using the state-of-the-art open-
source Text-to-Video (T2V) model, Wan2.2 [43]. Unlike
post-processing in prior work that performs simple upscal-
ing, our objective is to achieve perceptual refinement with-
out introducing semantic deviations from edited content of
Ve. This requirement aligns perfectly with the specialized
design of Wan2.2’s Mixture-of-Experts (MoE) architecture,
which employs a coarse denoiser for structural and semantic
formation under high noise, and a fine denoiser specialized
in later-stage refinement under low noise. We specifically
leverage the fine denoiser for a short, 4-step reverse process.
For each video Ve, we first add a small amount of Gaussian
noise. The fine denoiser then inverts this process utilizing
its expert prior to remove subtle artifacts and enhance textu-
ral details precisely because it is optimized for making min-
imal, semantic-preserving adjustments to nearly-complete

Figure 3. Model training pipeline. We train the context blocks based on the in-context video generator with curriculum learning by
gradually annealing and eventually dropping the reference frame.

Table 1. Quantitative comparisons with prior arts. The best results are bolded.

Automatic Metric

Human Evaluation

Method

CLIP-T ↑

CLIP-F ↑

VLM ↑

Edit-Acc ↑

Temp-Con ↑

Overall ↑

TokenFlow [14]
InsV2V [11]
InsViE [49]
Ours

23.63
22.49
23.56
25.54

98.43
97.99
98.78
99.03

7.10
6.55
7.35
8.10

1.70
2.17
2.28
3.85

1.97
1.96
2.30
3.76

1.70
2.07
2.36
3.86

videos. This yields a high-quality output with improved res-
olution and visual fidelity that remains strictly semantically
consistent with our initial edit.

3.6. Details of Ditto-1M

We collected a total of over 200k source videos, approx-
imately half of which feature human activities. After un-
dergoing a filtering process, these videos were edited using
editing instructions generated by a VLM, followed by an ad-
ditional round of filtering. This pipeline ultimately yielded
approximately 1M edited videos. Among these, about 700k
video triplets involve global editing (including changes to
style, environment, etc.), while roughly 300k pertain to lo-
cal editing (encompassing object replacing, adding, and re-
moval). The final enhanced videos have a resolution of
1280x720, each comprising 101 frames at 20 FPS. The vi-
sual quality of the final samples significantly surpasses that
of previous datasets - we strongly recommend reviewing the
video samples provided on the project page.

4. Modality Curriculum Model Learning

We select the in-context video generator VACE [20] as our
backbone, inspired by its strong prior for generating videos
that are spatially and structurally aligned with a source
video. VACE’s original capability is to condition its gener-
ation on two visual contexts (and prompts): a source video
and a reference image. Our goal is to repurpose this power-
ful visual generator into a proficient editor that operates on
abstract textual instructions. However, directly fine-tuning

the model to bridge the vast semantic gap from visual to
textual conditioning is prone to instability. We therefore
adapt its architecture, as shown in Fig. 3. It consists of a
Context Branch for extracting spatiotemporal features from
the source video and reference frame, and a DiT-based [33]
Main Branch that synthesizes the edited video under the
joint guidance of the visual context and the new textual em-
beddings from the instruction.

To ease the training difficulty and stably bridge this
modality gap, we introduce a modality curriculum learning
(MCL) strategy. The core idea is to leverage the model’s
inherent ability to process the reference image context as a
temporary aid. In the initial training phase, we provide the
edited reference frame as a strong visual ”scaffold” along-
side the new text instruction. As training progresses, we
gradually anneal the probability of providing this visual
scaffold, eventually dropping it entirely. This process com-
pels the model to shift its dependency from the concrete
visual target it already understands to the more abstract tex-
tual instruction, transforming it into a purely instruction-
based video editing model. We train the model using the
flow matching [26] objective:

L = Et,z0,c∥vt(zt, t, c) − (z0 − zt)∥2,

(5)

where z0 is the clean latent encoded from the target edited
video, zt is its noised version at timestep t, c represents
the conditioning from text and visual contexts, and vt is the
model’s predicted vector field pointing from zt to z0.

6

“Make it an Ukiyo-e style video.”+Input Video……DiTBlockText EncoderEdited VideoVAE DecoderVAE EncoderEdited VideoDiTBlockDiTBlockContext BlockNoiseVAE EncoderContext BlockEdit InstructionContext BranchMain BranchReference FrameFigure 4. Qualitative comparisons with prior arts TokenFlow [14], InsV2V [11], InsViE [49] and Gen4-Aleph [38].

Figure 5. Our data and learned model enable the translation from synthetic videos to the real domain.

5. Experiments

5.1. Experimental Settings

Our model is built upon the pre-trained in-context video
generator [20, 43] and is fine-tuned on our newly proposed
large-scale dataset, which comprises over one million high-
quality video triplets. To maintain the strong generative
prior of the base model and ensure training efficiency, we
freeze the majority of the pre-trained model’s parameters,
and only fine-tune the linear projection layers of context
blocks. The model is trained for approximately 16,000 steps
using the AdamW optimizer [29] with a constant learn-

ing rate of 1e-4 on a cluster of 64 GPUs. We employ
our modality curriculum learning strategy, where the initial
5,000 steps serve as a curriculum warm-up phase.

5.2. Experimental Results

Quantitative Comparison. We perform quantitative com-
parisons using automatic metrics and a user study, summa-
rized in Tab. 1. To ensure a fair comparison, our test set
consists of 50 videos collected from various online sources,
deliberately excluding Pexels videos to ensure the data is
out-of-distribution relative to our training set. For each
video, we provide 5 distinct editing instructions. For au-

7

SourceTokenFlowInsV2VInsViEGen4-AlephOursRender it in the style of pixel art.Replace the man's cloth to black suit.Make it the LEGO toy style.Figure 6. Unlike the original data generator, which fails to handle newly emerging information beyond key frames, our model - trained with
filtering and scaling techniques - outperforms it.

Figure 7. Ablation studies on training data scale and modality curriculum learning (MCL).

tomatic evaluation, we employ three metrics: CLIP-T mea-
sures the CLIP text-video similarity to assess how well the
edit follows the instruction; CLIP-F calculates the aver-
age inter-frame CLIP similarity to gauge temporal consis-
tency; and a VLM score provides a holistic assessment of
edit effectiveness, semantic preservation, and overall aes-
thetic quality. A user study based on 1,000 votes from post-
graduates and researchers also rates instruction-following
(Edit-Acc), temporal consistency (Temp-Con), and overall
quality (Overall). As shown, our method significantly out-
performs all baselines across metrics, achieving the highest
automatic scores and a strong preference in human evalu-
ations, which confirms its superior instruction adherence,
temporal smoothness, and visual quality. Please refer to the
supplementary materials for details of the user study.

Qualitative Comparison. As shown in Fig. 4, our method
consistently produces visually superior results that better
adhere to edit instructions compared to prior arts. For com-
plex stylizations, our model generates temporally coherent
videos that accurately match the target style, while com-
petitors often yield blurry or inconsistent results. For local
attribute changes (e.g., “black suit”), our method precisely
edits the target object while preserving identity and back-
ground details, a task where Gen4-Aleph slightly changes
the man’s identity and other methods largely fail. We
recommend reviewing the video samples provided on the
project page for a better understanding.

Additional Results. We showcase the synthetic-to-real
(syn2real) capability in Fig. 5 benefited from our data
by training the model to map the stylized videos in our
dataset back to their original, real-world source videos.
This successful transfer highlights the rich and photoreal-
istic information contained within our dataset, demonstrat-
ing its utility beyond standard editing tasks. Also, our final
trained model substantially outperforms the raw data gener-

ator from our pipeline, demonstrating superior handling of
newly emerged content as in Fig. 6. This superiority stems
from our scaled training regimen, including the curriculum
learning and exposure to the filtered, high-quality data.

5.3. Ablation Studies

We conduct ablation studies to validate the key components
of our framework, with results presented in Fig. 7. We
find that our model’s performance scales effectively with
the training data - as the number of samples increases, both
the quality of the stylistic edits and the fidelity to the orig-
inal video’s content and motion improve significantly, con-
firming the value of our large-scale dataset. Furthermore,
we ablate our modality curriculum learning (MCL) strategy
and find that, without MCL, the model often struggles to in-
terpret the instruction’s full semantic intent. Therefore it is
crucial for bridging the modality gap and learning to follow
instructions.

6. Conclusion

We have presented Ditto, a scalable framework that sig-
nificantly advances instruction-based video editing by sys-
tematically addressing the core challenge of data scarcity
through a new paradigm for large-scale data synthesis. Our
synthetic data generation pipeline overcomes the fidelity-
diversity and efficiency-coherence trade-offs plaguing prior
methods by leveraging strong image-editing priors, a dis-
tilled in-context video generator with a temporal enhancer,
and autonomous VLM-based quality control. This en-
ables the creation of the large-scale, high-quality Ditto-1M
dataset. The proposed modality curriculum learning strat-
egy further ensures our model Editto achieves state-of-the-
art performance by effectively transitioning from visual-
textual conditioning to purely instruction-driven inference.

8

SourceOursData GeneratorRobot armsPencil sketchSourcew/ MCLw/o MCLBearChinese clothesOrigamiPixel~60K Samples~120K Samples~250K Samples~500K SamplesSourceReferences

[1] Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lian-
rui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai,
Pengfei Wan, et al.
Recammaster: Camera-controlled
generative rendering from a single video. arXiv preprint
arXiv:2503.11647, 2025. 2

[2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin
Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun
Tang, et al. Qwen2.5-VL technical report. arXiv preprint
arXiv:2502.13923, 2025. 4, 5

[3] Stephen Batifol, Andreas Blattmann, Frederic Boesel, Sak-
sham Consul, Cyril Diagne, Tim Dockhorn, Jack English,
Zion English, Patrick Esser, Sumith Kulal, et al. FLUX. 1
Kontext: Flow matching for in-context image generation and
editing in latent space. arXiv e-prints, 2025. 2

[4] Yoshua Bengio, J´erˆome Louradour, Ronan Collobert, and Ja-
son Weston. Curriculum learning. In Int. Conf. Mach. Learn.,
2009. 2

[5] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel
Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi,
Zion English, Vikram Voleti, Adam Letts, et al. Stable video
diffusion: Scaling latent video diffusion models to large
datasets. arXiv preprint arXiv:2311.15127, 2023. 2

[6] Tim Brooks, Aleksander Holynski, and Alexei A Efros. In-
structPix2Pix: Learning to follow image editing instructions.
In IEEE Conf. Comput. Vis. Pattern Recog., 2023. 2

[7] Huanqia Cai, Sihan Cao, Ruoyi Du, Peng Gao, Steven
Hoi, Shijie Huang, Zhaohui Hou, Dengyang Jiang, Xin Jin,
Liangchen Li, et al. Z-image: An efficient image generation
foundation model with single-stream diffusion transformer.
arXiv preprint arXiv:2511.22699, 2025. 2

[8] Duygu Ceylan, Chun-Hao P Huang, and Niloy J Mitra.
In Int.

Pix2Video: Video editing using image diffusion.
Conf. Comput. Vis., 2023. 3

[9] Wenhao Chai, Xun Guo, Gaoang Wang, and Yan Lu. Stable-
Video: Text-driven consistency-aware diffusion video edit-
ing. In Int. Conf. Comput. Vis., 2023. 3

[10] Sili Chen, Hengkai Guo, Shengnan Zhu, Feihu Zhang, Zi-
long Huang, Jiashi Feng, and Bingyi Kang. Video depth
anything: Consistent depth estimation for super-long videos.
In IEEE Conf. Comput. Vis. Pattern Recog., 2025. 5

[11] Jiaxin Cheng, Tianjun Xiao, and Tong He. Consistent video-
to-video transfer using synthetic dataset. In Int. Conf. Learn.
Represent., 2024. 6, 7

[12] Ruihang Chu, Yefei He, Zhekai Chen, Shiwei Zhang, Xi-
aogang Xu, Bin Xia, Dingdong Wang, Hongwei Yi, Xi-
hui Liu, Hengshuang Zhao, et al. Wan-move: Motion-
controllable video generation via latent trajectory guidance.
arXiv preprint arXiv:2512.08765, 2025. 2

[13] Junyu Gao, Kunlin Yang, Xuan Yao, and Yufan Hu. Unity
in diversity: Video editing via gradient-latent purification. In
IEEE Conf. Comput. Vis. Pattern Recog., 2025. 2

[14] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel.
TokenFlow: Consistent diffusion features for consistent
video editing. In Int. Conf. Learn. Represent., 2024. 3, 6,
7

[15] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat
Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan
Misra. Imagebind: One embedding space to bind them all.
In IEEE Conf. Comput. Vis. Pattern Recog., 2023. 2

[16] Google.

Gemini 2.5 Flash Image.

https : / /
aistudio.google.com/models/gemini- 2- 5-
flash-image, 2025. 2

[17] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang,
Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and
Bo Dai. AnimateDiff: Animate your personalized text-to-
image diffusion models without specific tuning. In Int. Conf.
Learn. Represent., 2024. 2

[18] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo
Dai, Hongsheng Li, and Ceyuan Yang. CameraCtrl: En-
abling camera control for text-to-video generation. arXiv
preprint arXiv:2404.02101, 2024.

[19] Yi Huang, Wei Xiong, He Zhang, Chaoqi Chen, Jianzhuang
Liu, Mingfu Yan, and Shifeng Chen. Dive: Taming dino
for subject-driven video editing. In Int. Conf. Comput. Vis.,
2025. 2

[20] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang,
Yulin Pan, and Yu Liu. VACE: All-in-one video creation
and editing. arXiv preprint arXiv:2503.07598, 2025. 2, 5, 6,
7

[21] Xuan Ju, Tianyu Wang, Yuqian Zhou, He Zhang, Qing
Liu, Nanxuan Zhao, Zhifei Zhang, Yijun Li, Yuanhao Cai,
Shaoteng Liu, et al. Editverse: Unifying image and video
arXiv
editing and generation with in-context
preprint arXiv:2509.20360, 2025. 3

learning.

[22] Nikita Karaev,

Iurii Makarov, Jianyuan Wang, Natalia
Neverova, Andrea Vedaldi, and Christian Rupprecht. Co-
tracker3: Simpler and better point tracking by pseudo-
arXiv preprint arXiv:2410.11831,
labelling real videos.
2024. 4

[23] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai,
Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang,
et al. HunyuanVideo: A systematic framework for large
video generative models. arXiv preprint arXiv:2412.03603,
2024. 2

[24] Shanchuan Lin, Xin Xia, Yuxi Ren, Ceyuan Yang, Xuefeng
Xiao, and Lu Jiang. Diffusion adversarial post-training for
one-step video generation. arXiv preprint arXiv:2501.08316,
2025.

[25] Shanchuan Lin, Ceyuan Yang, Hao He, Jianwen Jiang, Yuxi
Ren, Xin Xia, Yang Zhao, Xuefeng Xiao, and Lu Jiang.
Autoregressive adversarial post-training for real-time inter-
active video generation. arXiv preprint arXiv:2506.09350,
2025. 2

[26] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximil-
ian Nickel, and Matt Le. Flow matching for generative mod-
eling. arXiv preprint arXiv:2210.02747, 2022. 6

[27] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng
Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli
Ouyang. Flow-grpo: Training flow matching models via on-
line rl. arXiv preprint arXiv:2505.05470, 2025. 2

[28] Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, and Jiaya
Jia. Video-P2P: Video editing with cross-attention control.
In IEEE Conf. Comput. Vis. Pattern Recog., 2024. 3

9

[29] Ilya Loshchilov and Frank Hutter. Decoupled weight decay

regularization. arXiv preprint arXiv:1711.05101, 2017. 7

[30] Guoqing Ma, Haoyang Huang, Kun Yan, Liangyu Chen, Nan
Duan, Shengming Yin, Changyi Wan, Ranchen Ming, Xi-
aoniu Song, Xing Chen, et al. Step-video-t2v technical re-
port: The practice, challenges, and future of video founda-
tion model. arXiv preprint arXiv:2502.10248, 2025. 2
[31] Maxime Oquab, Timoth´ee Darcet, Theo Moutakanni, Huy V.
Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,
Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Rus-
sell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-
Wen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nico-
las Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou,
Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bo-
janowski. DINOv2: Learning robust visual features without
supervision, 2023. 4

[32] Hao Ouyang, Qiuyu Wang, Yuxi Xiao, Qingyan Bai, Juntao
Zhang, Kecheng Zheng, Xiaowei Zhou, Qifeng Chen, and
Yujun Shen. CoDeF: Content deformation fields for tempo-
In IEEE Conf. Comput.
rally consistent video processing.
Vis. Pattern Recog., 2024. 3

[33] William Peebles and Saining Xie. Scalable diffusion models
with transformers. In Int. Conf. Comput. Vis., 2023. 6

[34] Pexels. Pexels. https://www.pexels.com/, 2025. 4
[35] Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei,
Xintao Wang, Ying Shan, and Qifeng Chen. FateZero: Fus-
ing attentions for zero-shot text-based video editing. In Int.
Conf. Comput. Vis., 2023. 3

[36] Bosheng Qin, Juncheng Li, Siliang Tang, Tat-Seng Chua,
and Yueting Zhuang.
Instructvid2vid: Controllable video
editing with natural language instructions. In Int. Conf. Mul-
timedia and Expo, 2024. 2, 3

[37] Robin Rombach, Andreas Blattmann, Dominik Lorenz,
Patrick Esser, and Bj¨orn Ommer. High-resolution image syn-
thesis with latent diffusion models. In IEEE Conf. Comput.
Vis. Pattern Recog., 2022. 2

[38] Runway.

https :
Introducing Runway Gen-4.
/ / runwayml . com / research / introducing -
runway-gen-4, 2025. 7

[39] Chitwan Saharia, William Chan, Saurabh Saxena, Lala
Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour,
Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans,
et al. Photorealistic text-to-image diffusion models with deep
In Adv. Neural Inform. Process.
language understanding.
Syst., 2022. 2

[40] Tiancheng Shen, Zilong Huang, Xiangtai Li, Zhijie Lin,
Jiyang Liu, Yitong Wang, Jiashi Feng, Ming-Hsuan Yang,
and Jun Hao Liew. Qk-edit: Revisiting attention-based in-
jection in mm-dit for image and video editing. In Int. Conf.
Comput. Vis., 2025.

[41] Yang Song and Prafulla Dhariwal.

niques for training consistency models.
arXiv:2310.14189, 2023.

Improved tech-
arXiv preprint

[42] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue,
and Xinchao Wang. Ominicontrol: Minimal and universal
control for diffusion transformer. In Int. Conf. Comput. Vis.,
2025.

[43] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao,
Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianx-
iao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jin-
gren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao,
Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang,
Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei
Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui,
Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang,
Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu,
Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu
Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yi-
tong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun
Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi
Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open
and advanced large-scale video generative models. arXiv
preprint arXiv:2503.20314, 2025. 5, 7

[44] Shengzhi Wang, Yingkang Zhong, Jiangchuan Mu, Kai Wu,
Mingliang Xiong, Wen Fang, Mingqing Liu, Hao Deng, Bin
He, Gang Li, et al. Align-a-video: Deterministic reward tun-
ing of image diffusion models for consistent video editing.
In IEEE Conf. Comput. Vis. Pattern Recog., 2025.

[45] Yukun Wang, Longguang Wang, Zhiyuan Ma, Qibin Hu, Kai
Xu, and Yulan Guo. Videodirector: Precise video editing via
In IEEE Conf. Comput. Vis. Pattern
text-to-video models.
Recog., 2025. 2

[46] Cong Wei, Quande Liu, Zixuan Ye, Qiulin Wang, Xintao
Wang, Pengfei Wan, Kun Gai, and Wenhu Chen. Univideo:
Unified understanding, generation, and editing for videos.
arXiv preprint arXiv:2510.08377, 2025. 3

[47] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan
Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei
Chen, et al. Qwen-image technical report. arXiv preprint
arXiv:2508.02324, 2025. 2, 5

[48] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian
Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu
Qie, and Mike Zheng Shou. Tune-A-Video: One-shot tuning
of image diffusion models for text-to-video generation.
In
Int. Conf. Comput. Vis., 2023. 3

[49] Yuhui Wu, Liyi Chen, Ruibin Li, Shihao Wang, Chenxi
Xie, and Lei Zhang. InsViE-1M: Effective instruction-based
In Int.
video editing with elaborate dataset construction.
Conf. Comput. Vis., 2025. 2, 3, 6, 7

[50] Weihan Xu, Yimeng Ma, Jingyue Huang, Yang Li, Wenye
Ma, Taylor Berg-Kirkpatrick, Julian McAuley, Paul Pu
Liang, and Hao-Wen Dong. Regen: Multimodal retrieval-
embedded generation for long-to-short video editing. arXiv
preprint arXiv:2505.18880, 2025. 2

[51] Xiangpeng Yang, Linchao Zhu, Hehe Fan, and Yi Yang.
VideoGrain: Modulating space-time attention for multi-
grained video editing. In Int. Conf. Learn. Represent., 2025.
3

[52] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Free-
man, Fredo Durand, Eli Shechtman, and Xun Huang. From
slow bidirectional to fast autoregressive video diffusion mod-
els. In IEEE Conf. Comput. Vis. Pattern Recog., 2025. 5
[53] Shoubin Yu, Difan Liu, Ziqiao Ma, Yicong Hong, Yang
Zhou, Hao Tan, Joyce Chai, and Mohit Bansal. VEG-
GIE: Instructional editing and reasoning video concepts with

10

grounded generation.
2025. 2, 3

arXiv preprint arXiv:2503.14350,

[54] Chi Zhang, Chengjian Feng, Feng Yan, Qiming Zhang,
Mingjin Zhang, Yujie Zhong, Jing Zhang, and Lin Ma. In-
structVEdit: A holistic approach for instructional video edit-
ing. arXiv preprint arXiv:2503.17641, 2025. 2, 3

[55] Zhenghao Zhang, Zuozhuo Dai, Long Qin, and Weizhi
Wang. EffiVED: Efficient video editing via text-instruction
diffusion models. arXiv preprint arXiv:2403.11568, 2024. 3
[56] Yixuan Zhu, Haolin Wang, Shilin Ma, Wenliang Zhao, Yan-
song Tang, Lei Chen, and Jie Zhou. Fade: Frequency-aware
In IEEE
diffusion model factorization for video editing.
Conf. Comput. Vis. Pattern Recog., 2025. 2

[57] Bojia Zi, Penghui Ruan, Marco Chen, Xianbiao Qi, Shaozhe
Hao, Shihao Zhao, Youze Huang, Bin Liang, Rong Xiao, and
Kam-Fai Wong. Se˜norita-2M: A high-quality instruction-
based dataset for general video editing by video specialists.
arXiv preprint arXiv:2502.06734, 2025. 3

11

