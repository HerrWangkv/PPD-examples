Accelerating Diffusion Model Training under Minimal Budgets:
A Condensation-Based Perspective

Rui Huang1,2⋆

Shitong Shao1⋆ Zikai Zhou1

Pukun Zhao1 Hangyu Guo3

Tian Ye1 Lichen Bai1

Shuo Yang3 Zeke Xie1†

1 xLeaF Lab, HKUST (GZ), 2 UESTC, 3 HIT (SZ)

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

G
L
.
s
c
[

3
v
4
1
9
5
0
.
7
0
5
2
:
v
i
X
r
a

Abstract

Diffusion models have achieved remarkable performance
on a wide range of generative tasks, yet training them from
scratch is notoriously resource-intensive, typically requir-
ing millions of training images and many GPU days. Moti-
vated by a data-centric view of this bottleneck, we adopt a
condensation-based perspective: given a large training set,
the goal is to construct a much smaller condensed dataset
that still supports training strong diffusion models under
minimal data and compute budgets. To operationalize this
perspective, we introduce Diffusion Dataset Condensation
(D2C), a two-phase framework comprising Select and At-
tach. In the Select phase, a diffusion difficulty score com-
bined with interval sampling is used to identify a compact,
informative training subset from the original data. Build-
ing on this subset, the Attach phase further strengthens the
conditional signals by augmenting each selected image with
rich semantic and visual representations. To our knowl-
edge, D2C is the first framework that systematically inves-
tigates dataset condensation for diffusion models, whereas
prior condensation methods have mainly targeted discrim-
inative architectures. Extensive experiments across data
budgets (0.8%–8% of ImageNet), model architectures, and
image resolutions demonstrate that D2C dramatically ac-
celerates diffusion model training while preserving high
generative quality. On ImageNet 2562 with SiT-XL/2, D2C
attains a FID of 4.3 in just 40k steps using only 0.8% of the
training images, corresponding to about 233× and 100×
faster training than vanilla SiT-XL/2 and SiT-XL/2 + REPA,
respectively.

1. Introduction

Generative models, such as score-based [1–3] and flow-
based [4] approaches, have achieved remarkable success
in various generative tasks [5], producing high-quality

⋆ Equal contribution.

† Correspondence to zekexie@hkust-gz.edu.cn.

and diverse data across domains [6–8]. However, these
approaches are notoriously data and compute intensive
to train, often requiring millions of samples and hun-
dreds of thousands of iterations to capture complex high-
The resulting cost
dimensional distributions [9–11].
presents a significant barrier to broader application and it-
eration within the AIGC community, making efficient train-
ing increasingly important across both academic and indus-
trial settings [12–14]. Recent efforts have improved dif-
fusion training efficiency through various strategies, such
as architectural redesigns [9, 11, 15], attention optimiza-
tion [16, 17], reweighting strategies [18], and representation
learning [10, 19, 20]. In parallel, data-centric approaches
such as patch-based methods [21, 22], Infobatch [23] and
Reweighting[24] aim to better exploit the potential of ex-
isting data. Despite these advances, directly constructing a
much smaller yet informative training subset as a primary
condensation-based route to accelerate diffusion training
remains largely unexplored, even though it provides a par-
ticularly direct and effective way to reduce data budgets.

Dataset condensation [25–28] aims to construct a much
smaller condensed sub-dataset with significantly fewer
samples than the original dataset, such that a model trained
from scratch on this subset achieves performance compara-
ble to one trained on the full dataset while converging much
faster. In practice, existing DC methods typically instantiate
this objective in two ways [25]: (i) pixel-level dataset distil-
lation, which directly optimizes synthetic images [27, 29],
and (ii) image-level condensation, which operates on real
images through selection and transformation [30–32]. Un-
like classical data pruning or selection [33], which passively
select a fixed subset of existing samples, both branches of
dataset condensation actively construct condensed training
sets, either by optimizing synthetic images or by selecting
and enriching real ones, thereby enabling more aggressive
data reduction and higher training efficiency [30]. However,
these methods have been developed almost exclusively for
discriminative tasks. Compared to discriminative learning,
generative diffusion training is substantially more complex
and demands higher dataset quality [34]; directly applying

 
 
 
 
 
 
Figure 1. D2C framework significantly accelerates diffusion model training with limited data. (a) Overview of our D2C pipeline,
which consists of a Select phase that filters a compact and diverse subset via diffusion difficulty score and interval sampling, and an Attach
phase that enriches samples with semantic and visual information. (b) D2C achieves over 100× faster convergence compared to REPA and
over 233× faster than vanilla SiT-XL/2, reaching a FID of 4.3 at just 40k steps. (c) Under a strict 4% data budget (0.05M), our method
achieves a FID of 2.7 at 180k iterations, demonstrating its strong training efficiency and rapid convergence.

popular DC algorithms (e.g., SRe2L [27], RDED [30]) to
diffusion models often leads to synthetic images that lack
structural and semantic fidelity, resulting in degraded sam-
ple quality and unstable convergence (see Sec. 4).

We raise a key question: “Can we train diffusion models
dramatically faster with significantly less data, while retain-
ing high generation quality?” The answer is affirmative. In
this paper, we make three main contributions.

First, to the best of our knowledge, we are the first to
formally study the dataset condensation task for diffusion
models, a new challenging problem setting that aims at con-
structing a “condensed” sub-dataset with significantly fewer
samples than the original dataset for training high-quality
diffusion models significantly faster. We address a funda-
mental academic gap concerning the application of dataset
condensation in diffusion models. More specifically, our
explorations with the diffusion model provide the first in-
sights into the challenges and potential solutions for ap-
plying dataset condensation to vision generation tasks. We
note that while conventional dataset condensation has made
great progress and sometimes uses diffusion models to con-
struct a subset, this line of research only focused on training
discriminative models instead of generative models.

Second, we propose D2C, a novel two-stage dataset con-
densation framework tailored for training diffusion models.
Our framework addresses the challenges of dataset conden-
sation for diffusion models by decomposing the problem
into two key aspects: the Select stage identifies an informa-
tive, compact, and learnable subset by ranking samples us-
ing the diffusion difficulty score derived from a pre-trained
diffusion model; the Attach stage enriches each selected
sample by adding semantic and visual representations, fur-
ther enhancing the training efficiency while preserving per-
formance.

Third, extensive experiments demonstrate great empiri-

cal success that the proposed D2C can train diffusion mod-
els significantly faster with dramatically fewer data while
retaining high visual quality, substantiating the effective-
ness and scalability. Specifically, D2C significantly outper-
forms random sampling and several popular dataset conden-
sation algorithms across data compression ratios of 0.8%,
4%, and 8%, at resolutions of 256×256 and 512×512, and
with both SiT [11] and DiT [9] architectures. In particular,
D2C achieves a FID of 4.3 in merely 40k training steps us-
ing SiT-XL/2 [11], demonstrating a 100× acceleration over
REPA [10] and a 233× speed-up compared to vanilla SiT.
Furthermore, it improves to a FID of 2.7 using only 50k
condensed images with CFG (refer to Fig. 1 (c)).

2. Preliminaries and Related Work
Diffusion Models. We briefly introduce the standard latent-
space noise injection formulation [9], which defines a for-
ward process that gradually perturbs input data x0 ∼ q0(x)
with Gaussian noise:

qt(xt | x0) = N (xt; αtx0, σ2

t I),

(1)

where αt, σt ∈ R+ are differentiable functions of t with
bounded derivatives. The choice for αt and σt is referred to
as the noise schedule of a diffusion model. After that, we
need to train a neural network ϵθ(·, ·, ·) to approximate the
reverse denoising process (i.e., predict the added noise ϵ) for
sampling (see Appendix B for more details). The training
objective is to minimize the mean squared error between the
predicted and the ground true noise:

Ldiff = Ex0∼q0(x),ϵ∼N (0,I),t∼U [0,1]

(cid:3) ,
(2)
Here, c is a conditional input, such as class labels or text
embeddings. In some cases, the prediction target is replaced
with the v-prediction, which corresponds to flow matching.

(cid:2)∥ϵ − ϵθ(xt, t, c)∥2

2

b) Acceleration Comparisona) 𝐷2𝐶 Pipelinec) Final Performance ComparisonTotal DatasetSubsetSelectAttachFinalDatasetSemanticInformationVisualInformationFigure 2. Overview of Diffusion Dataset Condensation (D2C). D2C employs a two-stage process: Select and Attach. The Select stage
identifies a compact and diverse subset by interval sampling using the diffusion difficulty score derived from a pre-trained diffusion model.
The Attach stage further enriches each selected sample by adding semantic information and visual information.

Data-centric Efficient Training.
Beyond model-side
improvements, a complementary line of work takes a
Patch-based
data-centric view on diffusion efficiency.
schemes [21, 22] and Infobatch [23] focus on reallocating
training effort over existing samples by reweighting or re-
sampling informative regions and instances. However, com-
paratively few methods directly tackle diffusion training ef-
ficiency by explicitly reducing and restructuring the over-
In this setting, given an original dataset
all training set.
D = {(ˆxi, ˆyi)}|D|
i=1, where each ˆyi is the label correspond-
ing to sample ˆxi, dataset compression aims to reduce the
size of training data while preserving model performance.
Two primary strategies have been extensively studied in this
context: dataset pruning and dataset condensation.
1) Dataset Pruning. Dataset pruning selects an information-
enriched subset from the original dataset, i.e., Dcore ⊂ D
with |Dcore| ≪ |D|, and directly minimizes the training loss
over the subset:

min
θ

E(x,y)∼Dcore [ℓ(ϕθDcore (x), y)] ,

(3)

where ℓ(·, ·) denotes the empirical training loss, and ϕθDcore
is the model parameterized by θDcore. Classical data pruning
methods like random sampling, K-Center [35], and Herd-
ing [36] can be used with diffusion models, but they offer
minimal performance improvements. Very recently, Li et
al. [24] investigate data-efficient diffusion training from the
perspective of dataset pruning by selecting a coreset with
surrogate features and then performing class-wise reweight-
ing. While this approach substantially reduces training cost
and improves over naive pruning, it does not attach any ad-
ditional information to the selected samples and is mainly
validated on relatively small-scale or latent diffusion set-
tings, which limits its ability to fully exploit the potential

of condensed training data for large-scale, high-resolution
diffusion models.
2) Dataset Condensation. Following recent work [25],
dataset condensation aims to synthesize a small, com-
pact, and diverse synthetic dataset DS = (X, Y) =
{(xj, yj)}|DS |
j=1 to replace the original dataset D. The syn-
thetic dataset DS is generated by a condensation algorithm
C such that DS ∈ C(D), with |DS | ≪ |D|. Each yj corre-
sponds to the synthetic label for the sample xj.

The key motivation for dataset condensation is to cre-
ate DS such that models trained on it can achieve perfor-
mance within an acceptable deviation η compared to mod-
els trained on D. This can be formally expressed as:

sup

(cid:110)(cid:12)
(cid:12)
(cid:12)ℓ(ϕθD (ˆx), ˆy) − ℓ(ϕθS

D

(cid:12)
(cid:12)
(ˆx), ˆy)
(cid:12)

(cid:111)

(ˆx,ˆy)∼D

≤ η,

(4)

where θD is the parameter set of the neural network ϕ
optimized on D: θD = arg minθ E(ˆx,ˆy)∼D [ℓ(ϕθ(ˆx), ˆy)] .
A similar definition applies to θS
D, which is optimized on
the synthetic dataset DS . Existing DC methods can be
broadly divided into two families. Pixel-level approaches
perform dataset distillation by directly optimizing synthetic
training images in pixel space (e.g., using gradient- or
matching-based objectives) [27–29, 37]. In contrast, image-
level condensation operates on real images via selection
and transformation, as in patch-based or quantization-style
schemes [25, 30, 32]. These methods have been developed
mainly for discriminative models; when naively applied to
diffusion training, they tend to produce images that devi-
ate from the target data distribution, which harms genera-
tive quality (see Appendix K for visualizations). Our D2C
framework follows the image-level condensation route, but
goes beyond passive pruning by not only selecting informa-

Original datasetClass 1Class 3Class N…−𝒑𝒑𝜽𝜽(𝒙𝒙|𝒄𝒄)Stage I: SelectDensityInterval Sampling Distribution with 50K DataDiffusionDifficulty ScoreminmaxSubsetSubsetSubsetStage II: AttachAttach Visual InformationEmbeddingGenerate semantic information for each classSavevisual informationVisual InformationText EncoderVision Encoder…SubsetSubsetSubsetAttachSaveAttachDC-Embedding…AttachClass 1：Class 2：Class N：…Sort images within each classFrom low to high0.320.350.370.470.630.510.270.210.25SubsetSubsetSubsetSelectDiffusion Difficulty ScoreClass 2“a photo of magpie”“a photo of king penguin”“a photo of lesser panda”Interval Samplingtive real samples, but also attaching rich semantic and visual
representations tailored to diffusion training.

3. Diffusion Dataset Condensation

As illustrated in Fig. 2, D2C consists of two stages: Se-
lect (Sec. 3.1), which identifies a compact set of diverse and
learnable real images using diffusion difficulty score and in-
terval sampling techniques; and Attach (Sec. 3.2), which
augments each selected image with semantic and visual in-
formation to improve generation performance. Finally, we
describe how to train diffusion models on the condensed
dataset produced by D2C in Sec. 3.3. (Sec. 3.3).

3.1. Select: Difficulty-Aware Selection

In this work, we focus on class-to-image (C2I) synthesis,
aligned with the setting in (author?) [10], and show that
our framework also applies to the text-to-image (T2I) set-
ting with only minor changes; see Appendix G for details.
Given a class-conditioned dataset D = (cid:83)C
y=1 Dy, where C
denotes the class number and Dy = {xi}|Dy|
i=1 denotes all
samples of class y, our aim is to select a compact subset for
efficient diffusion training. To achieve this, we propose the
diffusion difficulty score to quantify the denoising difficulty
of each sample, followed by our designed interval sampling
to ensure diversity within the selected subset.
Diffusion Difficulty Score. The arrangement of samples
from easy to hard is crucial for revealing underlying data
patterns and facilitating difficulty-aware selection. Recent
work [38, 39] demonstrates that diffusion models inher-
ently encode semantic-related class-conditional probability
pθ(c | x) through the variational lower bound (i.e., diffu-
sion loss Eq. 2) of log pθ(x | c) [1, 3]. This conditional
probability admits the standard Bayesian form

pθ(c | x) =

pθ(x | c) p(c)
ˆc pθ(x | ˆc) p(ˆc)

(cid:80)

.

(5)

Intuitively, a larger pθ(c | x) indicates that sample x can
be more confidently identified as belonging to class c, thus
suggesting lower learning difficulty. Computing the full de-
nominator in Eq. (5) for every sample is expensive, while
we only need a score that orders samples by difficulty.
Since the class label y ∼ U {1, . . . , C} is obtained by
uniform sampling and the average likelihood over classes
does not vary too much across samples, i.e., we assume
(cid:12)
(cid:12)Eˆc[pθ(x1 | ˆc)] − Eˆc[pθ(x2 | ˆc)](cid:12)
supx1,x2∼D
(cid:12) ≤ η,where
η > 0 is a small tolerance, the denominator in Eq. (5) can be
treated as approximately constant with respect to x. Conse-
quently, the posterior is proportional to the class-conditional
likelihood,

pθ(c | x) ∝ pθ(x | c).

(6)

We define the diffusion difficulty score based on this pos-

terior:

sdiff(x) = −pθ(c | x) ∝ −pθ(x | c)
= −Eϵ∼N (0,I), t∼U [0,1]

(cid:104)

∥ϵ − ϵθ (xt, t, c)∥2
2

(cid:105) (7)

The higher the score sdiff(x), the more difficult it is, and
the lower the score sdiff(x), the easier it is. To simplify our
presentation, we define the diffusion loss −pθ(x|c) as the
diffusion difficulty score.

By computing sdiff(x) for all training samples, we con-
struct a ranked dataset. As shown in Fig. 3, these scores
exhibit a skewed unimodal distribution. Selecting the eas-
iest samples (Min) yields a subset dominated by clean,
background-simple images with high learnability but lim-
ited diversity. In contrast, selecting only the highest-score
samples (Max) results in cluttered, noisy, and ambiguous
images that are difficult to optimize. Meanwhile, many
samples lie in the middle range, offering moderate learn-
ability but richer contextual information. Selecting an ap-
propriate value within this range is therefore critical; we
provide a more detailed discussion in Appendix H.1.
Interval Sampling. To balance diversity and learnabil-
ity, we propose an interval sampling strategy. Specifically,
we sort its images Dy within each class y in ascending
order of sdiff(x) and select samples at a fixed interval k:
(cid:12)
(cid:12) i ∈ {0, k, 2k, . . . }(cid:9), where DIS
DIS = (cid:83)C
denotes the selected subset constructed by interval sam-
pling, k is the fixed sampling interval, and x(i) is the i-
th sample in the sorted list (e.g., x(0) corresponds to the
sample with the lowest diffusion difficulty score). Interval
sampling with a larger interval k promotes diversity in the
sampled data while potentially hindering learnability. As
shown in Fig. 3 (Left), this trade-off arises from a shift in
the sample distribution: a larger k leads to a reduction in the
number of easy samples and a corresponding increase in the
representation of standard and difficult samples.
Extended Discussion. Training exclusively on the easiest
(Min) or the hardest (Max) samples is suboptimal. Instead,
a balanced curriculum comprising easy, medium, and diffi-
cult examples yields a training subset that is both learnable
and diverse, ultimately leading to stronger generative per-
formance. We further offer more discussions and insights
on interval sampling in Appendix H.2.

(cid:8)x(i) ∈ Dy

y=1

3.2. Attach: Semantic and Visual Information En-

hancement

To complement the Select phase, which yields a compact
subset of informative real images, the Attach phase enriches
each selected instance with additional semantic and visual
information. In particular, we attach semantic information
via a Dual Conditional Embedding (DC-Embedding) mod-
ule and inject visual information through visual representa-
tion, resulting in a more expressive condensed dataset and
improved generalization of the trained diffusion models.

Figure 3. Left: Distribution of diffusion difficulty scores under different interval values k. Smaller intervals (e.g., 1, 2) favor low-loss
samples, while larger intervals (e.g., 64, 128) result in a distribution closer to random sampling, thus approximating the original data
distribution. Moderate intervals (e.g., 16) provide balanced coverage across difficulty levels. Right: Representative samples selected
by three strategies: Min (lowest score), Max (highest score), and Interval (our proposed strategy). Interval sampling achieves a balance
between structural clarity and contextual richness.

in Fig. 4, the text embedding tc and the text mask tmask un-
dergo a 1D convolution and are fused with a learnable class
embedding ec using a residual MLP:

˜tc = Conv1d(tc × tmask),

ytext = MLP(˜tc) + ˜tc + ec.

(9)
This resulting vector ytext then serves as a semantic con-
ditioning token for the conditional diffusion model. Com-
pared to using simple class embeddings alone, this formu-
lation offers richer semantic information while retaining the
learnability of class embeddings.
Visual Information Injection. While semantic informa-
it often
tion aids in distinguishing inter-class structure,
fails to capture the intra-class variability essential for high-
fidelity generation. To address this, we integrate instance-
specific visual representations into the attached informa-
tion. For each image x ∈ R3×H×W , a pre-trained vision en-
coder fvis (e.g., DINOv2 [41]) extracts patch-level semantic
representations:

yvis = fvis(x) ∈ RN ×dtext

(10)

where N is the number of image patches and dtext is the fea-
ture dimension. We retain the first h (i.e., number of tokens
in the diffusion transformer) tokens of yvis to form a com-
pact representation of the dominant structure: yvis = yvis[:
h, :] ∈ Rh×dtext. As outlined in REPA [10], this visual in-
formation provides a semantic prior for the diffusion model
and thus significantly benefits data-centric efficient training.
Similar to the text information ytext, the visual information
yvis is also stored on disk as attached metadata alongside the
selected subset DIS.

Figure 4. Overview of DC-Embedding.

Dual Conditional Embedding (DC-Embedding). Exist-
ing C2I synthesis methods [9, 11] commonly rely on class
embeddings trained from scratch, which often fail to ef-
fectively capture inherent semantic information (see Ap-
pendix I.1). We enrich the class embedding by incor-
porating text representations derived from a pre-trained
text encoder (e.g., T5-encoder [40]). For each class c ∈
{1, . . . , C}, a descriptive prompt P (c) (e.g., “a photo of a
cat”) is encoded by a pre-trained text encoder ftext, yielding
its corresponding text embedding tc and text mask tmask:

tc, tmask = ftext(P (c)),

(8)

The resulting text embedding and text mask are stored on
disk as attached text information alongside the subset DIS
generated in the preceding phase, ready for import during
formal training. During the formal training, as illustrated

MinMaxInterval (Ours)Diffusion Difficulty ScoreDensitySiT/DiT-BlockLinearPatchingInputLinearUnpatchingOutputTextEmbeddingProjectorNTextMaskConv1dClass EmbeddingClassFigure 5. D2C improves visual quality under tight data budgets. We compare Random sampling and D2C on DiT-L/2 at 10k and 50k
data budgets, and neither setting uses classifier-free guidance.

3.3. D2C Training Process

Here, we detail the training process of the diffusion model
using our condensed dataset, which comprises a compact
subset selected during the Select phase and subsequently
enriched with semantic and visual information during the
Attach phase. Our goal is to fully leverage the informa-
tion contained in our condensed dataset to accelerate train-
ing without compromising performance.

(cid:2)∥ϵ − ϵθ(xt, t, y, ytext)∥2

We employ a conditional diffusion model Dθ and,
as an example, utilize the optimization objective of
predicting the added
score-based diffusion models:
noise ϵ from the perturbed latent input xt at time step
t, conditioned on the text information ytext and the class
label y. The new denoising loss is defined as Ldiff =
Ex0∼q0(x),ϵ∼N (0,I),t∼U [0,1]
where the specific injected forms of y and ytext can be
found in Sec. 3.2. Then, to maximize the utilization of
visual
the same formulation as
REPA [10], which involves aligning the encoder’s output
(i.e., the decoder’s input) within the diffusion model with
the visual representation yvis = {vi}h
i=1. Concretely, from
a designated intermediate layer of the diffusion backbone,
we obtain token features {hi ∈ Rd}h
i=1. A projection head
ϕ maps these tokens from Rd to Rdtext , and we compute a
semantic alignment loss:

information, we adopt

(cid:3),

2

Lproj = −

1
h

h
(cid:88)

i=1

(cid:28) ϕ(hi)
∥ϕ(hi)∥

,

vi
∥vi∥

(cid:29)

.

(11)

This loss encourages the model to align its encoder’s out-
put with visual representations, promoting localized realism
and spatial consistency [41] in generation.
Overall Training Objective. The final training loss com-
bines the denoising objective and the semantic alignment
term (with the balance weight λ is set to 0.5 by default):

Ltotal = Ldiff + λEx,ϵ∼N (0,I),t∼U [0,1],y,ytext,yvis [Lproj] . (12)

This training strategy enables D2C to effectively learn from

limited yet enhanced data, offering a practical solution for
efficient diffusion training under minimal budgets.

4. Experiments

In this section, we validate the performance of D2C and an-
alyze the contributions of its components through extensive
experiments. In particular, we aim to answer the following
questions: 1) Can D2C improve training speed and reduce
data usage of diffusion models? 2) Does D2C generalize
well across backbones, data scales, and resolutions? 3) How
do D2C’s components and hyperparameter choices affect its
overall effectiveness?

4.1. Setup

Experiment settings. We conduct experiments on the
ImageNet-1K dataset [42], using subsets of 10K, 50K, and
100K images, corresponding to 0.8%, 4%, and 8% of the
full dataset, respectively. To further demonstrate the gener-
alization and effectiveness of our method, Appendix J re-
ports additional results of D2C on CIFAR datasets. All
images are center-cropped and resized to 256×256 and
512×512 resolutions using the ADM [43] preprocessing
pipeline. Furthermore, we use [·]-L/2 and [·]-XL/2 archi-
tectures in both DiT [9] and SiT [11] backbones, following
the standard settings outlined in (author?) [11].
Evaluation and baselines. We train models from scratch
on the collected subset and evaluate them using gFID [44],
sFID, Inception Score [45] and Precision, adhering to stan-
dard evaluation protocols [9, 11, 43]. We compare our
method against REPA [10], REPA-E [46], REG [19] and
various data condensation and selection baselines, including
SRe2L [27], RDED [30], Herding, K-Center, and random
sampling, using SiT and DiT architectures [9, 11]. Further
details regarding evaluation metrics and baseline methods
can be found in Appendix D and E.
4.2. Main Result
Training Performance and Speed. We evaluate D2C us-
ing 10K and 50K data budgets, comparing its performance

Training Iteration60k180k120k60k180k120k40k80kRandom 50kD²C(Ours)50kRandom 10kD²C(Ours)10kTable 1. Comparison of gFID-50K across various dataset condensation methods and data budgets using DiT-L/2 and SiT-L/2 on ImageNet
256×256. We use CFG=1.5 for evaluation. D2C surpasses other methods at all settings.

Data Budget

Iter.

0.8% (10K)
0.8% (10K)

4.0% (50K)
4.0% (50K)

8.0% (100K)
8.0% (100K)

100k
300k

100k
300k

100k
300k

DiT-L/2

Random K-Center Herding

D2C Random K-Center Herding

SiT-L/2

35.86
4.19

36.78
11.55

41.02
11.49

50.77
13.5

69.86
38.54

71.31
37.35

40.75
22.35

32.38
22.44

36.37
15.23

4.20
4.13

14.81
5.99

22.55
6.49

4.35
4.33

31.13
14.18

36.64
12.56

14.77
13.58

61.66
39.69

66.96
39.08

22.96
22.55

29.11
22.44

32.3
16.17

D2C

3.98
3.98

11.21
5.66

15.01
5.65

Figure 6. Left: Interval-sampling ablation. Small k speeds early training. The best final gFID-10K appears at k=96 for the 10K
budget and k=16 for the 50K budget, roughly scaling with data size. Right: DC-Embedding ablation at 10K. Combining text and class
embeddings outperforms either alone; “Only Class” denotes the baseline that injects class embeddings only.

Table 2. Comparison with a strict data budget 0.8% (10K) on Ima-
geNet 512×512. We use CFG=1.5 for evaluation. D2C surpasses
random sampling at all settings.

Model Method

Iter.

gFID↓

sFID↓

Inception Score↑ Precision↑

DiT-L/2 Random
DiT-L/2 D2C (Ours)

DiT-L/2 Random
DiT-L/2 D2C (Ours)

SiT-L/2 Random
SiT-L/2 D2C (Ours)

SiT-L/2 Random
SiT-L/2 D2C (Ours)

100k
100k

300k
300k

100k
100k

300k
300k

24.8
14.8

17.1
5.8

13.3
9.1

5.0
4.22

11.9
6.9

12.8
15.1

22.8
14.3

13.6
11.6

74.3
109.2

130.6
318.9

197.1
261.7

316.9
289.7

0.65
0.63

0.64
0.77

0.69
0.72

0.76
0.79

against REPA and a vanilla SiT model trained on the full
ImageNet dataset (a 1.28M data budget), as well as ran-
dom selection with 10K and 50K data budgets. As shown
in Table 3 and Fig. 1 (b), our method achieves a gFID-
50K of 4.23 at only 40K iterations with 10K training data.
In contrast, REPA requires 4 million steps and the vanilla
SiT model needs 7 million steps to reach comparable per-
formance, representing an acceleration of over 100× and
233×, respectively. Under a 4% data budget (50K) with
CFG set to 1.5, our method achieves an FID of 2.78 at 180K
steps, further demonstrating significant data and compute
efficiency (Fig. 1 (c)). Moreover, Fig. 5 presents a visual
comparison between random selection and our D2C at 10K
and 50K data sizes. Our method demonstrates superior vi-
sual quality compared to the baseline and generates higher-
quality images, even during the early iterations of training.

Table 3. Comparison of acceleration algorithms on ImageNet-1K.

Model

Training Set

Iter.

gFID↓

DiT L-2
+ REPA
+ D2C
+ D2C

SiT L-2
+ REPA
+ D2C

SiT XL-2
+ REPA
+ REPA-E
+ REG
+ D2C
+ D2C

1.28M 400k
1.28M 400k
0.05M 10k
0.01M 10k

1.28M 400k
1.28M 700k
0.01M 80k

1.28M
7M
4M
1.28M
1.28M 235k
1.28M 200k
0.01M 40k
0.05M 180k

23.3
15.6
14.81
4.2

18.8
8.4
7.07

8.3
5.9
5.9
5.0
4.3
2.78

Comparison on ImageNet 256×256. We compare D2C
with random sampling, Herding [36], K-Center [35],
SRe2L [27], RDED [30] under various data budgets and
backbones. As shown in Table 1, D2C consistently achieves
the lowest FID across all settings. For instance, using only
0.8% of the data and 100K iterations with early stopping,
our method achieves a gFID-50K of 4.20 on DiT-L/2 and
3.98 on SiT. These results demonstrate the superiority of
our approach over existing methods. Notably, SRe2L and

40K60K80K100K120K140K160K180K200KTraining Steps8163264gFID-10K (log scale)Interval Sampling Ablation Study (10K)DC-Embedding+Interval=1DC-Embedding+Interval=4DC-Embedding+Interval=16DC-Embedding+Interval=64DC-Embedding+Interval=96DC-Embedding+Interval=112Random40K60K80K100K120K140K160K180K200KTraining Steps8163264128Interval Sampling Ablation Study (50K)DC-Embedding+Interval=1DC-Embedding+Interval=4DC-Embedding+Interval=8DC-Embedding+Interval=16DC-Embedding+Interval=32MediumRandom60K100K140K01020304050gFID-10K36.599.017.7946.1917.398.5747.6937.0713.19DC Embedding Ablation Study (10K)Class+TextOnly TextOnly Class050100Interval1015gFID-10K at 200K iterations020Interval2025gFID-10K at 200K iterationsTable 4. D2C vs. SRe2L [27] and RDED [30] on ImageNet
256×256 with a data budget 0.8% (10K).

Table 5. Ablation studies on the Select and Attach phases. Sel.:
Select. Vis.: Vision.

Model Method

gFID↓

sFID↓

Inception Score↑ Precision↑

DiT-L/2 RDED
DiT-L/2 SRe2L
DiT-L/2 D2C (Ours)

SiT-L/2 RDED
SRe2L
SiT-L/2
SiT-L/2 D2C (Ours)

166.2
104.2
4.2

97.5
82.3
3.9

60.1
20.2
11.0

66.8
19.8
10.7

10.8
14.1
283.6

65.63
18.1
289.7

0.09
0.20
0.72

0.22
0.27
0.73

RDED, which perform well in classification task, fail on this
generative task (see Table 4) due to their focus on category-
discriminative features. Similarly, geometry-based meth-
ods like Herding and K-Center, along with random sam-
pling, prove inadequate for achieving efficient and high-
performing training.
Comparison on ImageNet 512×512. As shown in Table 2,
D2C achieves a gFID of 5.8 on DiT-L/2, a significant im-
provement over the 17.1 achieved by random sampling at
300k iterations under the ImageNet 512×512 settings. On
SiT-L/2, similar improvements are observed. These demon-
strate that D2C generalizes well to higher resolutions.
4.3. Ablation Study

Ablation on Select Phase. We investigate the impact of
the interval value k in the Select phase, as shown in Fig. 6
(Left). Using a small value accelerates early training by
prioritizing min-loss samples, which are simpler and eas-
ier to learn. However, the limited diversity of such samples
leads to degraded performance in later stages, eventually
being overtaken by settings with moderate interval values.
In contrast, large intervals or random selection introduce ex-
cessive max-loss or uncurated samples, destabilizing train-
ing (Fig. 3). As k increases, we observe that gFID-10K first
decreases and then worsens, revealing an optimal trade-off
between diversity and learnability. Empirically, the best re-
sults are achieved with an interval of 96 for the 10K budget
and 16 for 50K, approximately following the ratio of data
budgets (50K/10K). Table 5 further shows that using the
Select stage alone reduces gFID from 37.07 to 14.96, un-
derscoring its effectiveness and usefulness.
Ablation on Attach Phase. We evaluate Attach from two
angles. First, as shown in Fig. 6 (Right), DC embed-
ding consistently outperforms using either alone under a
10K budget, with text-only better than class-only, indicating
richer semantics from textual descriptions. Second, Table 5
shows steady gains from the injection modules: baseline
gFID-10K is 14.96, adding only visual information reaches
10.37, adding only DC embedding reaches 9.01, and com-
bining both achieves the best 7.62. Appendix I.2 further
ablates the visual encoder and demonstrates the robustness
of our approach.
Effect of Pretrained Diffusion Models and Wall-Clock
Cost. Our D2C pipeline does not inherently require a pow-
erful pretrained model. As shown in Table 6, when the

Model Sel. DC Emb. Vis. Emb. gFID↓

DiT-L/2 (cid:37) (cid:37)
DiT-L/2 (cid:37) (cid:33)

DiT-L/2 (cid:33) (cid:37)
DiT-L/2 (cid:33) (cid:37)
DiT-L/2 (cid:33) (cid:33)
DiT-L/2 (cid:33) (cid:33)

(cid:37)
(cid:33)

(cid:37)
(cid:33)
(cid:37)
(cid:33)

37.07
8.79

14.96
10.37
9.01
7.62

Table 6. A breakdown of the computational overhead for sub-
processes in D2C. Compared to the REPA baseline, the additional
scoring time is negligible, demonstrating D2C’s efficiency.

Method

Score Model Score Time Iter. Train Time gFID↓

REPA
D2C
(w/o select)
D2C
(w/ select)
D2C
(w/ select)

N/A

N/A

N/A

N/A

4M

750h

0.04M 7.4h

5.9

5.6

From Scratch

1.9h

0.04M (7.4+26.2)h 4.9

Pretrained

2.1h

0.04M 7.4h

4.3

scoring network is a strong DiT-XL/2 with base gFID 2.27
from [9], D2C reaches an FID of 4.3; with a weaker DiT-
L/2 that we trained from scratch achieving a base gFID of
11.5, it reaches 4.9. Using only the Attach stage, without
Select, still reaches 5.6 and surpasses REPA at 5.9. In wall-
clock terms, the Attach-only variant finishes in 7.4h, which
is 0.99% of REPA’s 750h and about 101× faster. With a pre-
trained scorer, the end-to-end pipeline totals 9.5h, with 2.1h
for scoring and 7.4h for training; this is 1.27% of REPA
and about 79× faster. With a scorer trained from scratch,
the pipeline totals 35.5h, with 1.9h for scoring, 26.2h for
training the scorer, and 7.4h for diffusion training; this is
4.7% of REPA and about 21× faster. These results show
that whether the scorer is strong, weak, or omitted, D2C
consistently accelerates diffusion training while maintain-
ing competitive quality.

5. Conclusion

In this paper, we introduce D2C, the first dataset conden-
sation framework that significantly accelerates diffusion
model training for generative tasks. D2C follows a two-
phase pipeline, Select and Attach, which selects a compact
yet diverse subset via a diffusion difficulty score with in-
terval sampling and enriches it with semantic and visual
signals. On ImageNet-1K, D2C achieves 100–233× faster
training than strong baselines while maintaining competi-
tive generative quality, and we hope it will motivate further
research on data-centric efficiency for diffusion models.

Acknowledgement. This work was supported by the
National Natural Science Foundation of China under Grant
No. 62506317.

References

[1] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar,
S. Ermon, and B. Poole, “Score-based generative modeling
through stochastic differential equations,” in International
Conference on Learning Representations, 2021. 1, 4

[2] J. Song, C. Meng, and S. Ermon, “Denoising diffusion im-
plicit models,” in International Conference on Learning Rep-
resentations, 2021.

[3] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion proba-
bilistic models,” in Neural Information Processing Systems,
(Virtual Event), pp. 6840–6851, NeurIPS, Dec. 2020. 1, 4
[4] X. Liu, C. Gong, and Q. Liu, “Flow straight and fast: Learn-
ing to generate and transfer data with rectified flow,” arXiv
preprint arXiv:2209.03003, 2022. 1

[5] C. Chen, S. Hu, J. Zhu, M. Wu, J. Chen, Y. Li, N. Huang,
C. Fang, J. Wu, X. Chu, et al., “Taming preference mode
collapse via directional decoupling alignment in diffusion
reinforcement learning,” arXiv preprint arXiv:2512.24146,
2025. 1

[6] T. Karras, M. Aittala, T. Aila, and S. Laine, “Elucidating
the design space of diffusion-based generative models,” Ad-
vances in neural information processing systems, vol. 35,
pp. 26565–26577, 2022. 1

[7] Y. Guo, C. Yang, A. Rao, Z. Liang, Y. Wang, Y. Qiao,
M. Agrawala, D. Lin, and B. Dai, “Animatediff: Animate
your personalized text-to-image diffusion models without
specific tuning,” arXiv preprint arXiv:2307.04725, 2023.
[8] C. Chen, J. Zhu, X. Feng, et al., “S2-guidance: Stochas-
tic self guidance for training-free enhancement of diffusion
models,” arXiv preprint arXiv:2508.12880, 2025. 1

[9] W. Peebles and S. Xie, “Scalable diffusion models with
transformers,” in Proceedings of the IEEE/CVF Interna-
tional Conference on Computer Vision, pp. 4195–4205,
2023. 1, 2, 5, 6, 8

[10] S. Yu, S. Kwak, H. Jang, J. Jeong, J. Huang, J. Shin, and
S. Xie, “Representation alignment for generation: Training
diffusion transformers is easier than you think,” in Interna-
tional Conference on Learning Representations, 2025. 1, 2,
4, 5, 6

[11] N. Ma, M. Goldstein, M. S. Albergo, N. M. Boffi, E. Vanden-
Eijnden, and S. Xie, “Sit: Exploring flow and diffusion-
based generative models with scalable interpolant transform-
ers,” in European Conference on Computer Vision, pp. 23–
40, Springer, 2024. 1, 2, 5, 6, 3

[12] S. Shitong, G. Yufei, and X. Zeke, “Fastlightgen: Fast and
light video generation with fewer steps and parameters,” Pro-
ceedings of the IEEE Conference on Computer Vision and
Pattern Recognition (CVPR), 2026. 1

[13] J. Liu, P. Cai, Q. Zhou, Y. Lin, D. Kong, B. Huang, Y. Pan,
H. Xu, C. Zou, J. Tang, S. Zheng, and L. Zhang, “Fre-
qca: Accelerating diffusion models via frequency-aware
caching,” 2025.

[14] Y. Gu and Z. Xie, “Mano: Restriking manifold optimization
for llm training,” arXiv preprint arXiv:2601.23000, 2026. 1
[15] H. Zheng, W. Nie, A. Vahdat, and A. Anandkumar, “Fast
training of diffusion models with masked transformers,”
arXiv preprint arXiv:2306.09305, 2023. 1

[16] D. Bolya and J. Hoffman, “Token merging for fast stable dif-
fusion,” in Proceedings of the IEEE/CVF conference on com-
puter vision and pattern recognition, pp. 4599–4603, 2023.
1

[17] H. Li, S. Shao, W. Zhong, Z. Zhou, L. Bai, H. Xiong, and
Z. Xie, “Pisa: Piecewise sparse attention is wiser for efficient
diffusion transformers,” arXiv preprint arXiv:2602.01077,
2026. 1

[18] T. Hang, S. Gu, C. Li, J. Bao, D. Chen, H. Hu, X. Geng,
and B. Guo, “Efficient diffusion training via min-snr weight-
ing strategy,” in Proceedings of the IEEE/CVF international
conference on computer vision, pp. 7441–7451, 2023. 1
[19] G. Wu, S. Zhang, R. Shi, S. Gao, Z. Chen, L. Wang, Z. Chen,
H. Gao, Y. Tang, jian Yang, M.-M. Cheng, and X. Li, “Rep-
resentation entanglement for generation: Training diffusion
transformers is much easier than you think,” in The Thirty-
ninth Annual Conference on Neural Information Processing
Systems, 2025. 1, 6, 4

[20] S. Zening, X. Zhengpeng, B. Lichen, S. Shitong, S. Yang,
and X. Zeke, “Craft: Aligning diffusion models with fine-
tuning is easier than you think,” Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition
(CVPR), 2026. 1

[21] Z. Ding, M. Zhang, J. Wu, and Z. Tu, “Patched denoising
diffusion models for high-resolution image synthesis,” in The
twelfth international conference on learning representations,
2023. 1, 3

[22] Z. Wang, Y. Jiang, H. Zheng, P. Wang, P. He, Z. Wang,
W. Chen, M. Zhou, et al., “Patch diffusion: Faster and more
data-efficient training of diffusion models,” Advances in neu-
ral information processing systems, vol. 36, pp. 72137–
72154, 2023. 1, 3

[23] Z. Qin, K. Wang, Z. Zheng, J. Gu, X. Peng, Z. Xu, D. Zhou,
L. Shang, B. Sun, X. Xie, et al., “Infobatch: Lossless training
speed up by unbiased dynamic data pruning,” arXiv preprint
arXiv:2303.04947, 2023. 1, 3

[24] Y. Li, Y. Zhang, S. Liu, and X. Lin, “Pruning then reweight-
ing: Towards data-efficient training of diffusion models,”
in ICASSP 2025-2025 IEEE International Conference on
Acoustics, Speech and Signal Processing (ICASSP), pp. 1–
5, IEEE, 2025. 1, 3

[25] H. Wu, D. Su, J. Hou, and G. Li, “Dataset condensation with
color compensation,” Transactions on Machine Learning Re-
search, 2025. 1, 3

[26] K. Wang, B. Zhao, X. Peng, Z. Zhu, S. Yang, S. Wang,
G. Huang, H. Bilen, X. Wang, and Y. You, “Cafe: Learn-
ing to condense dataset by aligning features,” in Computer
Vision and Pattern Recognition, (New Orleans, LA, USA),
pp. 12196–12205, IEEE, Jun. 2022.

[27] Z. Yin, E. P. Xing, and Z. Shen, “Squeeze, recover and
relabel: Dataset condensation at imagenet scale from A
new perspective,” in Neural Information Processing Systems,
NeurIPS, 2023. 1, 2, 3, 6, 7, 8

[28] S. Shao, Z. Yin, X. Zhang, and Z. Shen, “Generalized large-
scale data condensation via various backbone and statistical
matching,” arXiv preprint arXiv:2311.17950, 2023. 1
[29] G. Cazenavette, T. Wang, A. Torralba, A. A. Efros, and
J. Zhu, “Dataset distillation by matching training trajecto-
ries,” in Computer Vision and Pattern Recognition, (New Or-
leans, LA, USA), IEEE, Jun. 2022. 1, 3

[30] P. Sun, B. Shi, D. Yu, and T. Lin, “On the diversity and
realism of distilled dataset: An efficient dataset distillation
paradigm,” in Computer Vision and Pattern Recognition,
IEEE, 2024. 1, 2, 3, 6, 7, 8

[31] C. Gao, H. Li, L. Liu, Z. Xie, P. Zhao, and zhiqiang xu, “Prin-
cipled data selection for alignment: The hidden risks of diffi-
cult examples,” in Forty-second International Conference on
Machine Learning, 2025.

[32] S. K. A. Khatib, A. ElHagry, S. Shao, and Z. Shen, “Od3:
Optimization-free dataset distillation for object detection,”
arXiv preprint arXiv:2506.01942, 2025. 1, 3

[33] S. Yang, Z. Xie, H. Peng, M. Xu, M. Sun, and P. Li, “Dataset
pruning: Reducing training data by examining generaliza-
tion influence,” in The Eleventh International Conference on
Learning Representations, 2023. 1

[34] L. Manduchi, K. Pandey, C. Meister, R. Bamler, R. Cotterell,
S. D¨aubener, S. Fellenz, A. Fischer, T. G¨artner, M. Kirchler,
et al., “On the challenges and opportunities in generative ai,”
arXiv preprint arXiv:2403.00025, 2024. 1

[35] M. Jones, H. Nguyen, and T. Nguyen, “Fair k-centers via
maximum matching,” in International conference on ma-
chine learning, pp. 4940–4949, PMLR, 2020. 3, 7, 2
[36] Y. Chen and M. Welling, “Parametric herding,” in Proceed-
ings of the Thirteenth International Conference on Artificial
Intelligence and Statistics, pp. 97–104, JMLR Workshop and
Conference Proceedings, 2010. 3, 7, 2

[37] S. Shao, Z. Zhou, H. Chen, and Z. Shen, “Elucidating
the design space of dataset condensation,” arXiv preprint
arXiv:2404.13733, 2024. 3

[38] A. C. Li, M. Prabhudesai, S. Duggal, E. Brown, and
D. Pathak, “Your diffusion model is secretly a zero-shot clas-
sifier,” in Proceedings of the IEEE/CVF International Con-
ference on Computer Vision, pp. 2206–2217, 2023. 4
[39] Q. Zipeng, L. Buhua, Z. Shiyan, L. Bao, X. Zhiqiang,
X. Haoyi, and X. Zeke, “A simple and efficient base-
line for zero-shot generative classification,” arXiv preprint
arXiv:2412.12594, 2024. 4

[40] J. Ni, G. H. Abrego, N. Constant, J. Ma, K. B. Hall,
D. Cer, and Y. Yang, “Sentence-t5: Scalable sentence en-
coders from pre-trained text-to-text models,” arXiv preprint
arXiv:2108.08877, 2021. 5, 2

[41] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec,
V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-
Nouby, et al., “Dinov2: Learning robust visual features with-
out supervision,” arXiv preprint arXiv:2304.07193, 2023. 5,
6, 1, 2, 4

[42] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh,
S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein,
et al., “Imagenet large scale visual recognition challenge,”
International Journal of Computer Vision, vol. 115, no. 3,
pp. 211–252, 2015. 6

[43] P. Dhariwal and A. Nichol, “Diffusion models beat gans on
image synthesis,” in Neural Information Processing Systems,
vol. 34, (Virtual Event), pp. 8780–8794, NeurIPS, Dec. 2021.
6

[44] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and
S. Hochreiter, “Gans trained by a two time-scale update rule
converge to a local nash equilibrium,” in Neural Information
Processing Systems, vol. 30, (Long Beach Convention Cen-
ter, Long Beach), NeurIPS, Dec. 2017. 6, 2

[45] T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Rad-
ford, and X. Chen, “Improved techniques for training gans,”
in Neural Information Processing Systems, vol. 29, (Centre
Convencions Internacional Barcelona, Barcelona SPAIN),
NeurIPS, Dec. 2016. 6, 2

[46] X. Leng, J. Singh, Y. Hou, Z. Xing, S. Xie, and L. Zheng,
“Repa-e: Unlocking vae for end-to-end tuning with latent
diffusion transformers,” arXiv preprint arXiv:2504.10483,
2025. 6

[47] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn,
X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer,
G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An im-
age is worth 16x16 words: Transformers for image recogni-
tion at scale,” in International Conference on Learning Rep-
resentations, 2021. 1

[48] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh,
S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark,
G. Krueger, and I. Sutskever, “Learning transferable visual
models from natural language supervision,” 2021. 1

[49] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna,
“Rethinking the inception architecture for computer vision,”
in Proceedings of the IEEE conference on computer vision
and pattern recognition, pp. 2818–2826, 2016. 2

[50] C. Nash, J. Menick, S. Dieleman, and P. W. Battaglia, “Gen-
erating images with sparse representations,” arXiv preprint
arXiv:2103.03841, 2021. 2

[51] T. Kynk¨a¨anniemi, T. Karras, S. Laine, J. Lehtinen, and
T. Aila, “Improved precision and recall metric for assessing
generative models,” Advances in neural information process-
ing systems, vol. 32, 2019. 2

[52] K. He, X. Chen, S. Xie, Y. Li, P. Doll´ar, and R. Girshick,
“Masked autoencoders are scalable vision learners,” in Pro-
ceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pp. 16000–16009, 2022. 2

[53] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, “Momentum
contrast for unsupervised visual representation learning,” in
Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition, pp. 9729–9738, 2020. 2
[54] D. Xie, S. Shao, L. Bai, zikai zhou, B. Cheng, S. Yang,
W. JUN, and Z. Xie, “Guidance matters: Rethinking the
evaluation pitfall for text-to-image generation,” in The Four-
teenth International Conference on Learning Representa-
tions, 2026. 3

[55] C. Shi, S. Li, S. Guo, S. Xie, W. Wu, J. Dou, C. Wu, C. Xiao,
C. Wang, Z. Cheng, et al., “Where culture fades: Revealing
the cultural gap in text-to-image generation,” arXiv preprint
arXiv:2511.17282, 2025.

[56] Z. Fang, L. Xiang, X. Cai, K. Zhou, and H. Wen, “Flexcon-
trol: Computation-aware conditional control with differen-
tiable router for text-to-image generation,” in Forty-second
International Conference on Machine Learning, 2025. 3

Accelerating Diffusion Model Training under Minimal Budgets:
A Condensation-Based Perspective

Supplementary Material

A. Positioning D2C within Dataset Condensa-

tion Paradigms

While some works equate dataset condensation with
gradient-based pixel-level optimization of synthetic images,
a broader line of literature defines it as constructing compact
training sets that retain the learning efficacy of the origi-
nal data [25], which also includes image-level schemes such
as OD3[32] and RDED[30]. In this broader paradigm, the
key objective is not how the condensed data are obtained,
but whether the resulting small dataset can support training
models that closely match the performance of those trained
on the full dataset. D2C follows this latter view. It con-
denses the dataset by selecting a highly informative subset
guided by diffusion difficulty and then attaching additional
semantic and visual information that enriches each sample
without altering its raw pixels. This design is analogous in
spirit to OD3 and RDED, which also operate at the level of
image selection rather than direct pixel optimization. Con-
sequently, D2C naturally fits within the dataset condensa-
tion family, while being specifically tailored to generative
diffusion models and addressing a gap that is not covered
by existing pixel-level condensation methods.

B. Additional Descriptions of Diffusion Models

This section reviews the fundamentals of the Denoising Dif-
fusion Probabilistic Model (DDPM) [3]. The DDPM frame-
work consists of a fixed forward process that incrementally
perturbs the input data with noise, and a learned reverse pro-
cess trained to iteratively denoise the data, thereby learning
the target distribution. Specific architectural details of our
implementation are summarized in Appendix B.2.

B.1. Denoising Diffusion Probabilistic Model

The DDPM framework models data generation via a
discrete-time Markov chain that progressively adds Gaus-
sian noise to a data sample x0 ∼ p(x). The forward process
is defined as:

q(xt | xt−1) = N (xt; (cid:112)1 − βtxt−1, βtI),

(13)

where βt ∈ (0, 1) are predefined variance schedule pa-
rameters controlling the noise level at each time step t ∈
[1, 2, ..., T ], and I is the identity matrix.

For simplicity, we define αt = 1 − βt, and denote the
i=1 αi. The reverse process,

cumulative product ¯αt = (cid:81)t
which is learned by the model θ, can be defined as:

(cid:16)

(cid:16)

(cid:17)

(cid:17)

1√

xt − βt√

αt

1− ¯αt

xt−1;

ϵθ(xt, t)

, Σθ(xt, t)

pθ(xt−1 | xt) = N

,
(14)
where ϵθ(xt, t) denotes the predicted noise from a neural
t I,
t = βt) or learned through

network. The covariance Σθ(xt, t) is typically set to σ2
where σ2
interpolation σ2

t can be either fixed (σ2

t = (1 − ¯αt−1)/(1 − ¯αt)β.

A simplified training objective minimizes the prediction

error between true and estimated noise:

Lsimple = Ex0,ϵ,t

(cid:2)∥ϵ − ϵθ

(cid:0)√

¯αtx0 +

√

1 − ¯αtϵ, t(cid:1) ∥2(cid:3) .

(15)
In addition to the simple objective, improved variants in-
clude learning the reverse variance Σθ(xt, t) jointly with the
mean, which leads to a variational bound loss of the form:

Lvlb = exp

(cid:16)

v log βt + (1 − v) log ˜βt

(cid:17)

.

(16)

Here, v is an element-wise weight across model out-
put dimensions. When T is sufficiently large and the
noise schedule is carefully chosen, the terminal distribu-
tion p(xT ) approximates an isotropic Gaussian. Sampling
is then performed by iteratively applying the learned reverse
process to recover the data sample from pure noise.

B.2. Diffusion Transformer Architecture

Our model implementation closely follows the design of
DiT [9] and SiT [11], which extend the vision transformer
(ViT) architecture [47] to generative modeling. An input
image is first split into patches, reshaped into a 1D sequence
of length N , and then processed through transformer layers.
To reduce spatial resolution and computational cost, we fol-
low prior work [9, 11] and encode the image into a latent
tensor z = E(x) using a pretrained encoder E from the
stable diffusion VAE.

In contrast to the standard ViT, our transformer blocks
include time-aware adaptive normalization layers known as
adaLN-zero. These layers scale and shift the hidden state
in each attention block according to the diffusion timestep
and conditioning signals. During training, we also add an
auxiliary multilayer perceptron (MLP) head that maps the
hidden state to a semantic target representation space, such
as DINOv2 [41] or CLIP features [48]. This head is used
only for training-time supervision in our alignment loss and
does not affect sampling or inference.

C. Hyperparameters and Implementation De-

tails

Select Phase Settings. In the Select phase, we adopt a pre-
trained DiT-XL/2 model [9] as the scoring network and use
the diffusion loss (w.r.t., mean squared error) as the scoring
metric. To construct subsets of different sizes, we apply
interval sampling with k = 96 for the 10K subset, k = 16
for the 50K subset, and k = 10 for the 100K subset. Each
subset is constructed in a class-wise manner, selecting 10,
50, and 100 samples per class respectively.
Attach Phase Settings. In the Attach phase, we implement
dual conditional embeddings. For textual conditioning, we
use a T5 encoder [40] with captions truncated to 16 tokens,
producing embeddings of dimension 2048. For visual con-
ditioning, we adopt DINOv2-B [41] as the visual encoder.
The number of visual tokens h is set to 256, and each token
has a feature dimension of 768.
Training Settings. In the Training phase, we use the Adam
optimizer with a fixed learning rate of 1e-4 and (β1, β2) =
(0.9, 0.999), without applying weight decay. We employ
mixed-precision (fp16) training with gradient clipping. La-
tent representations are pre-computed using the stable dif-
fusion VAE, and decoded via its native decoder. All ex-
periments are conducted on either 8 NVIDIA A800 80GB
GPUs or 8 NVIDIA RTX 4090 24GB GPUs. We use a
batch size of 256 with a 256 × 256 resolution in Fig. 1, and
a 512 × 512 resolution in Table 2. All other experiments
use a batch size of 128 and a default image resolution of
256 × 256.

D. Evaluation Details

We adopt several widely used metrics to evaluate generation
quality and diversity:

• gFID [44] computes the Fr´echet distance between the
feature distributions of real and generated images. Fea-
tures are extracted using the Inception-v3 network [49].
• sFID [50] extends FID by leveraging intermediate spa-
tial features from the Inception-v3 model to better cap-
ture spatial structure and style in generated images.
• IS [45] evaluates both the quality and diversity of gen-
erated samples by computing the KL-divergence be-
tween the conditional label distribution and the marginal
distribution over predicted classes, using softmax-
normalized logits.

• Precision and Recall [51] respectively measure sample
realism and diversity, quantifying how well generated
samples cover the data manifold and vice versa.

E. Baseline Setting

We evaluate our method against two categories of baselines:

Diffusion models trained on selected or condensed sub-
sets. These include SiT and DiT backbones trained from
scratch on 10K, 50K, and 100K subsets obtained via the
following strategies:

• Random Sampling. A naive baseline that randomly se-
lects a fixed number of real samples without any guid-
ance.

• Herding [36]. A geometry-based method that selects
samples to approximate the global feature mean, ensur-
ing representative coverage.

• K-Center [35]. A diversity-focused algorithm that it-
eratively selects samples maximizing the minimum dis-
tance from the selected set, promoting broad coverage
of the feature space.

• SRe2L [27]. A dataset condensation method that syn-
thesizes class-conditional data through a multi-stage
pipeline. Originally proposed for classification tasks,
we adapt it to the diffusion setting by applying class-
wise condensation to real images and training a diffu-
sion model on the resulting synthetic subset. Visual-
izations of the synthesized samples and corresponding
training results are provided in Appendix K.

Diffusion models trained on the full dataset. These base-
lines are trained with access to the entire training set, with-
out data reduction:

• SiT [11]. A transformer-based diffusion model that re-
formulates denoising as continuous stochastic interpo-
lation, enabling faster training and improved efficiency
under full-data settings.

• REPA [10]. A model-side regularization method that
aligns intermediate features of diffusion transformers
with patch-wise representations from strong pretrained
visual encoders (e.g., DINOv2-B [41], MAE [52], Mo-
Cov3 [53]) using a contrastive loss. It retains the full
dataset and improves convergence and generation qual-
ity via early-layer representation guidance.

F. Framework Design and Implementation

We introduce D2C, a framework for constructing compact
yet effective training subsets for diffusion models under
stringent data budgets. Our approach is motivated by two
complementary intuitions: (1) that the contribution of train-
ing samples is non-uniform, as some are more informative
than others; and (2) that generative training benefits from
semantically enriched conditioning. These insights directly
inform the two core stages of our framework. First, a Se-
lect stage ranks training examples by a difficulty score com-
puted via a pretrained class-conditional diffusion model.
Second, an Attach stage enriches the selected data by in-
jecting textual and visual priors. The complete pipeline is
summarized in Algorithm 1.

Algorithm 1 D2C: Diffusion Dataset Condensation
Require: Full dataset D = {(xi, ci)}N
encoder ftext, visual encoder fvis
// Each xi is an image, and ci ∈ {1, . . . , C} is the class
label.

i=1, interval k, text

1: // Phase 1: Select
2: Compute difficulty score sdiff for all (xi, ci) ∈ D
3: For each class c, sort Dc = {xi

| ci = c} by sdiff

ascending

4: Select every k-th sample (Interval Sampling) in sorted

Dc to form Dselect
5: // Phase 2: Attach
6: for each (x, c) ∈ Dselect do
7:

8:
9:

Generate class prompt P (c) (e.g., “a photo of a
label”)
Extract text embedding: (tc, tmask) ← ftext(P (c))
Extract visual feature: yvis ← fvis(x)
Store triplet (x, c, tc, tmask, yvis) into (cid:101)D

10:
11: end for
12: Return enriched dataset (cid:101)D for diffusion model training

Figure 7. Distribution of diffusion difficulty score computed on
LAION text–image pairs with a pre-trained SDXL model. This
distribution resembles that of C2I, which supports interval sam-
pling for selecting informative training pairs under T2I.

G. Exploration on Text-to-Image Generation

We further examine the applicability of the D2C framework
to text-to-image generation [54–56]. The Select phase re-
quires only a minimal change: replace the class condition in
Eq. 7 with a text condition, i.e., stext
diff (x) = −pθ(x | text).
Using SDXL to score LAION text–image pairs, we observe
a difficulty distribution similar to the class-conditional case
(Fig. 7; see also Fig. 3 and Fig. 8 (right)). Low-score sam-
ples tend to exhibit simple structures, high-score samples
often contain complex or cluttered contexts, and the major-
ity of samples fall in the middle range. Interval sampling
remains effective for identifying informative pairs. The At-

tach phase is also easy to transfer: semantic and visual
representations serve as soft supervisory signals for the se-
lected subset.

As such, while our main experiments focus on class-to-
image tasks for controlled benchmarking like SiT [11], the
framework is generalizable and well suited to text-to-image
generation. We expect it to deliver practical gains in data ef-
ficiency and training speed in this setting, offering a promis-
ing direction for future work.

H. More Discussions about Select

H.1. Detailed Algorithm for Computing Diffusion

Difficulty Score

The diffusion difficulty score, used to rank samples in the
Select phase, is defined as the mean denoising loss over
uniformly sampled timesteps, computed using a frozen pre-
trained diffusion model (see Algorithm 2).

Algorithm 2 Compute Diffusion Difficulty Score
Require: Image dataset D = {(xi, ci)}N

i=1; pretrained
VAE encoder Eϕ; pretrained diffusion model ϵθ;
timestep set T ; batch size n
// Each xi is an image; ci ∈ {1, . . . , C} is the class
label. Timesteps in T are sampled uniformly. Models
are frozen during scoring.
1: Initialize empty map S ← {}
2: for mini-batch {(xi, ci)}n
3:
4:
5:
6:

Encode to latent (if applicable): zi ← Eϕ(xi)
Initialize per-sample accumulator ℓi ← 0
for t ∈ T do

i=1 ⊂ D do

Sample ϵ ∼ N (0, I)
Perturb latent: zt ← αt zi + σt ϵ
Compute loss: ℓi ← ℓi + ∥ϵ − ϵθ(zt, t, ci)∥2
2

end for
si ← ℓi/|T | // Mean denoising loss across timesteps

7:
8:
9:
10:

S[xi] ← si

11:
12: end for
13: Return S // Image-to-score mapping for difficulty-

aware selection

H.2. Practical Insights on Interval Sampling

While Section 4.3 has covered a detailed ablation study on
the choice of interval k in Select phase, we provide addi-
tional insights into how diffusion difficulty score relates to
distributional coverage.

The right panel in Fig. 8 presents the gFID-10K scores
of subsets sampled from different portions of the difficulty-
ranked dataset. We partition the training set into consecu-
tive 10K segments ordered by the diffusion difficulty score

Figure 8. Left: gFID-10K across training steps under different in-
terval values k for a 50K data budget. Moderate intervals (e.g.,
k = 16) achieve superior performance by balancing learnabil-
ity and diversity. Right: Distributional discrepancy (gFID-10K)
between ranked training subsets and the validation set. Both ex-
tremely low and high diffusion difficulty score lead to higher FID,
while mid-range segments show better alignment.

(e.g., the first 10K samples with lowest scores as “Min”,
followed by 10–20K, 20–30K, and so on), and measure
each segment’s discrepancy from the full validation dis-
tribution using gFID. Interestingly, we observe a clear U-
shaped curve: subsets consisting of extremely low or high
difficulty samples exhibit significantly worse distributional
alignment, while those centered around moderate difficulty
levels show substantially lower FID scores. This result
aligns well with our hypothesis that very easy samples (e.g.,
simple textures, clean backgrounds) and extremely hard
samples (e.g., ambiguous, noisy structures) both fail to re-
flect the global data distribution.

These observations provide an empirical justification for
our interval sampling strategy. Specifically, under a 50K
dataset budget with k = 16, each class contributes sam-
ples selected at regular intervals from its difficulty-sorted
list. Given that each class typically contains around 1,200
images, this strategy naturally samples from approximately
the first 800 positions in the ranked list. As a result, the
selected data span both the easy and moderately difficult
regions, while avoiding the extremes at both ends. This bal-
anced coverage across the difficulty spectrum promotes bet-
ter generalization and faster convergence, as evidenced by
the results in Fig. 8 (Left) and discussed in Section 4.3. In
this way, our strategy yields a compact yet effective dataset
that enables the model to converge rapidly while maintain-
ing strong generation quality.

Ablation on interval sampling. As shown in Fig. 9, the
“Medium” variant corresponds to selecting samples from
the center of the difficulty-ranked list rather than apply-
ing interval sampling from low to high diffusion difficulty
scores. Concretely, after sorting each class by diffusion
difficulty, we start from the median position and expand
symmetrically toward both sides until the data budget is
reached. This strategy focuses on medium-difficulty exam-
ples and largely omits easier instances, while still including
a portion of harder ones near the tails. As a result, the se-
lected subset provides less comprehensive coverage of the

Figure 9. T-SNE visualization of class embeddings. Each point
represents a class in the dataset. Left: One-hot class embeddings
show no semantic structure. Right: Text embeddings naturally
cluster semantically related classes. Samples from semantically
related classes, such as different dog breeds, tend to form distinct
clusters in feature space. Leveraging this semantic prior is highly
effective for accelerating diffusion model training.

underlying data distribution, leading to slower convergence
and degraded final performance compared to our proposed
interval sampling scheme.

I. More Discussions about Attach

I.1. Dual Conditional Embedding

Most diffusion models condition on class identifiers repre-
sented as integer IDs or one-hot vectors, which are mapped
to class embeddings trained from scratch. This ignores se-
mantic relationships between categories, resulting in un-
structured embeddings as shown in Fig. 9 (Left).In con-
trast, text embeddings derived from class-specific prompts
(e.g., “a photo of a dog”) via a pre-trained language encoder
naturally encode semantic priors and cluster related classes
(Fig. 9, Right). We propose a dual conditional embedding
that fuses the text embedding with a learnable class embed-
ding (i.e., a traditional class token trained from scratch), as
defined in Eq. 8–9. This hybrid strategy combines semantic
structure with symbolic distinctiveness, and leads to signif-
icantly improved generation quality. As shown in Fig. 6
(Right), using both branches achieves lower FID than using
either one alone.

I.2. Visual Information Injection

Recent studies [10, 19] have shown that relying solely
on diffusion models to learn meaningful representations
from scratch often results in suboptimal semantic features.
In contrast, injecting high-quality visual priors, especially
those derived from strong self-supervised encoders like DI-
NOv2 [41], can significantly improve both training effi-
ciency and generation quality. In our case, we incorporate
a frozen visual encoder to provide external patch-level vi-
sual features during training. These external features serve

gFID-10K(logscale)40K60K160K180K200K8163264128IntervalSamplingAblationStudy(50K)DC-Embedding+Interval=1DC-Embedding+Interval=4DC-Embedding+Interval=8DC-Embedding+Interval=16DC-Embedding+Interval=32Medium Random020Interval80K100K120K140KTrainingSteps2025gFID-10Kat200KiterationsgFID-10K403020100.100.150.200.250.300.350.400.450.50Diffusion Difficulty ScoreminmaxDistributional Discrepancy across Ranked SubsetsText EmbeddingOne-hot Class EmbeddingFigure 10. Top: Images synthesized directly by SRe2L and RDED, two popular dataset condensation methods originally designed for
discriminative tasks. Bottom: Images generated by diffusion model trained on the two synthesized datasets.

Table 7. Ablation of the visual encoder.

Table 8. Comparison of random subset selection and D2C on
CIFAR-10 (reported in gFID-50K).

Vision Encoder

FID↓

N/A (baseline)
MAE-L
MoCov3-L
CLIP-L
DINOv2-L

37.07
9.23
8.78
8.59
7.62

as semantically rich anchors, particularly beneficial at early
layers, allowing the model to focus on generation-specific
details in later stages. Empirically, visual supervision im-
proves feature alignment and accelerates convergence under
limited data, as shown in Tables 1, 2, 5, and 7. All tested en-
coders outperform the no-encoder baseline, indicating that
our method is robust to the choice of visual encoder.

J. Experiments on CIFAR

As shown in Table 8, we further evaluate D2C on CIFAR-10
by selecting 100 images per class to form a 1K data budget
(2% compression rate) and training the diffusion model for
100k steps. Under this highly constrained setting, D2C sig-
nificantly improves gFID from 9.72 with random sampling
to 3.95, demonstrating that our selection and attachment
strategy remains effective beyond ImageNet and transfers
well across datasets.

Method

gFID↓

Random
D2C (Ours)

9.72
3.95

K. Visualization of SRe2L and RDED in Gen-

erative Tasks

As shown in Fig. 10, dataset condensation methods that
excel in classification, such as RDED and SRe2L, trans-
fer poorly to diffusion-based generation. Their objectives
focus on preserving class-discriminative cues, for example
segmentation-guided selection in RDED and gradient-based
image optimization in SRe2L, rather than modeling realistic
global structure and natural image statistics. As a result, dif-
fusion models trained on these synthesized datasets fail to
capture the underlying pixel-level data distribution and pro-
duce severely degraded samples. In contrast, D2C provides
the first dataset condensation framework tailored to diffu-
sion generative modeling and effectively closes this gap.

L. ImageNet 512×512 Experiment

As shown in Table 2, D2C consistently outperforms random
sampling under a strict 10K (0.8%) data budget across both
DiT-L/2 and SiT-L/2 backbones. Visual samples in Fig. 11

Sre²L Generated from Sre²L RDEDGenerated from RDED Figure 11. Generated samples on ImageNet 512×512 from SiT-L/2 trained with D2C using a 10K dataset (CFG=1.5).

further confirm the high fidelity and diversity of generations
at 512×512 resolution, demonstrating that D2C generalizes
effectively to high-resolution settings.

M. Visualization

Figure 12. Generated samples of SiT-L/2 trained with D2C using a 50K dataset (CFG=1.5). Class label = ”macaw”(88)

Figure 13. Generated samples of SiT-L/2 trained with D2C using a 50K dataset (CFG=1.5). Class label = ”arctic wolf”(270)

Figure 14. Generated samples of SiT-L/2 trained with D2C using a 50K dataset (CFG=1.5). Class label = ”jaguar”(290)

Figure 15. Generated samples of SiT-L/2 trained with D2C using a 50K dataset (CFG=1.5). Class label = ”otter”(360)

Figure 16. Generated samples of SiT-L/2 trained with D2C using a 50K dataset (CFG=1.5). Class label = ”lesser panda”(387)

Figure 17. Generated samples of SiT-L/2 trained with D2C using a 50K dataset (CFG=1.5). Class label = ”panda”(388)

Figure 18. Generated samples of SiT-L/2 trained with D2C using a 50K dataset (CFG=1.5). Class label = ”fire truck”(555)

Figure 19. Generated samples of SiT-L/2 trained with D2C using a 50K dataset (CFG=1.5). Class label = ”cheeseburger”(933)

Figure 20. Generated samples of SiT-L/2 trained with D2C using a 50K dataset (CFG=1.5). Class label = ”lake shore”(975)

Figure 21. Generated samples of SiT-L/2 trained with D2C using a 50K dataset (CFG=1.5). Class label = ”volcano”(980)

