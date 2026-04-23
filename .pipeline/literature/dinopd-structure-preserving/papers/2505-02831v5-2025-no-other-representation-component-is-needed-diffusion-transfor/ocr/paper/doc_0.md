6
2
0
2

n
a
J

6
2

]

V
C
.
s
c
[

5
v
1
3
8
2
0
.
5
0
5
2
:
v
i
X
r
a

No Other Representation Component Is Needed:
Diffusion Transformers Can Provide Representation
Guidance by Themselves

Dengyang Jiang1,2 Mengmeng Wang3,2∗ Liuzhuozheng Li2 Lei Zhang1

Haoyu Wang1 Wei Wei1 Guang Dai2 Yanning Zhang1

Jingdong Wang4†
1Northwestern Polytechnical University 2SGIT AI Lab, State Grid Corporation of China
3Zhejiang University of Technology 4Baidu Inc.
Quick overview at: https://vvvvvjdy.github.io/sra
Code is available at: https://github.com/vvvvvjdy/SRA

Abstract

Recent studies have demonstrated that learning a meaningful internal represen-
tation can accelerate generative training. However, existing approaches necessi-
tate to either introduce an off-the-shelf external representation task or rely on a
large-scale, pre-trained external representation encoder to provide representation
guidance during the training process. In this study, we posit that the unique dis-
criminative process inherent to diffusion transformers enables them to offer such
guidance without requiring external representation components. We propose Self-
Representation Alignment (SRA), a simple yet effective method that obtains rep-
resentation guidance using the internal representations of learned diffusion trans-
former. SRA aligns the latent representation of the diffusion transformer in the
earlier layer conditioned on higher noise to that in the later layer conditioned on
lower noise to progressively enhance the overall representation learning during
only the training process. Experimental results indicate that applying SRA to
DiTs and SiTs yields consistent performance improvements, and largely outper-
forms approaches relying on auxiliary representation task. Our approach achieves
performance comparable to methods that are dependent on an external pre-trained
representation encoder, which demonstrates the feasibility of acceleration with
representation alignment in diffusion transformers themselves.

Figure 1: Left: Methods like MaskDiT and SD-DiT use an external representation task to guide
diffusion transformer. Middle: Methods like REPA leverage an external representation foundation
model as guidance. Right (our approach): We do not use any external representation component
but still obtain such guidance through proposed self-representation alignment technique.

∗Corresponding author.
†Project lead.

Technical Report: SRA.

Process (e.g., mask)Diffusion ModelRepresentation Loss (e.g., MAE loss)    RegulationGenerative LossRepresentation Model (e.g., DINOv2) AlignmentGenerative LossDiffusion ModelEMAGenerative LossSelf-Alignment      (a) Representation Training Paradigm Involved   (b) Representation Foundation Model Involved     (c) No Representation Component Involved               latent representationlatent representationlatent representationDiffusion ModelDiffusion Model 
 
 
 
 
 
Figure 2: We empirically investigate the trend of latent representations in the original SiT-XL/2.
Left: Using PCA [1] to visualize the latent features, we observe that the features lead a process
from bad to good when increasing block layers and decreasing noise level. Right: A similar trend
can also be seen in the linear probing results on ImageNet.
Investigation of DiT is provided in
Appendix A, which leads to the similar results as SiT.

1

Introduction

Diffusion transformers [65, 59, 11] and vision transformers [21, 55, 79] have held the dominant
positions in visual generation and representation because of their scalability during pre-training [6,
23, 64, 75] and generalization capacity for downstream tasks [42, 53, 39, 51].

Recently, many works [91, 93, 87, 44] have explored leveraging representation components of vi-
sion transformers for diffusion transformer’s training and have shown that learning a high-quality
internal representation can not only speed up the generative training progress but also improve the
generation quality. These works either utilize the an external discriminative loss in representation
learning (e.g., MAE’s [29], IBOT’s [92]) shown in Figure 1(left) or leverage a large-scale pre-trained
representation foundation model (e.g., DINOv2 [64], CLIP [68]) shown in Figure 1(middle) to give
representation guidance for the diffusion transformer during the original generative training. How-
ever, the former methods require an external representation task, and the latter method relies on a
powerful external pretrained encoder, which limits its usage scenarios when there are no good exter-
nal encoders. Thereout, an intriguing yet underexplored problem has come to light: Can we obtain
such representation guidance without external representation components?

Our observations: Different from the representation model that takes a clean image as input and
then outputs semantically-rich feature, diffusion model often takes a noise latent as input and ob-
tains cleaner one step-by-step. In other words, the generative mechanism by which the diffusion
model operates can be generally considered as a bad to good process. Inspired by this behavior, we
hypothesise that the representations in it also follow such a trend. To testify this, we perform an
empirical analysis with recent diffusion transformers [59, 65]. As shown in Figure 2(left), we first
find out that the latent features in the diffusion transformer are progressively refined, moving from
bad to good, as block layers increase and noise level decreases. Next, akin to the results in previous
studies [87, 83], we observe that the diffusion transformer already learns meaningful discriminative
representations as shown in Figure 2(right). Meanwhile, although the accuracy drops off after reach-
ing a peak at about layer 20 because the model needs to shift away to focus on generating images
with high-frequency details, the quality of the representations basically transfers from bad to good
by increasing block layers and decreasing noise level. These indicate that the diffusion transformer
gets roughly from a bad-to-good discriminative process when only generative training is performed.

Our approach: The observations above motivate us to align the weaker representations in the dif-
fusion transformer to the better ones in the original generative training, thereby enhancing the rep-
resentation learning of the model without involving any external representation component. Driven
from this inspiration, we present Self-Representation Alignment (SRA). As shown in Figure 1(right),
SRA does not need any representation component; in essence, it aligns the output latent represen-
tation in earlier layer conditioned on higher noise to that in later layer conditioned on lower noise
to achieve self-representation alignment. Meanwhile, in order to boost the performance, we obtain

2

Increasing Block LayersDecreasing Noise Levellayer1          layer3          layer8        layer13       layer18       layer23        layer28t=0.1            t=0.3           t=0.5           t=0.7                            (a) PCA Visualization Result                                                        (b) Linear Probing Result the target features from another model that shares the same architecture with the trainable model
but updates weight by weighted moving average (EMA)1. Furthermore, the student’s output latent
feature is first passed through the projection layers to conduct a slight nonlinear transformation for
better representation extraction and then aligned with the target feature output by the teacher. In
a nutshell, our SRA can offer a flexible way to integrate representation guidance without external
component needs and architectural modification.

Finally, we conduct comprehensive experiments to evaluate the effect of SRA. After a series of
component-wise analyses, we show that SRA brings significant performance improvements to both
DiTs [65] and SiTs [59]. Moreover, our ablation study highlights the crucial role of internal repre-
sentations in the success of SRA, which supports our central hypothesis: diffusion transformers can
achieve representation alignment without external components.

In summary, our main contributions are as follows:

• We analyze the representations in diffusion transformers and assume that the unique discriminative

process makes it possible to achieve representation alignment without external components.

• We introduce SRA, a simple yet effective method that aligns the output latent representation of
the diffusion transformers in the earlier layer conditioned on higher noise to that in the later layer
conditioned on lower noise to achieve self-representation alignment.

• With our SRA, both DiTs and SiTs achieve sustained training speed acceleration and nontrivial

generation performance improvement.

2 Method

2.1 Preliminary: Training Object of DiT and SiT

As our method is built upon the denoise-based model (DiT) and flow-based model (SiT), to facilitate
a more seamless introduction of SRA in the subsequent part, we now present a brief overview of the
training object of these two types of models. We omit the class-condition here for simplicity. We
also leave more detailed mathematical descriptions of these types of models in Appendix B.

Denoise-based models learn to transform Gaussian noise into data samples through a step-by-step
denoising process. Given a pre-defined forward process that gradually adds noise, these models
learn the reverse process to recover the original data.

√

For data point x0 from distribution x0 ∼ p(x), the forward process follows: q(xt|xt−1) =
N (xt;
1 − βtx0, βtI). The model learns to reverse this process using a neural network ϵζ(xt, t)
that predicts the noise added at each step. The network is trained using a simple mean squared error
objective that measures how well it can predict the noise:

Lsimple = Ext, ϵ, t

(cid:104)
||ϵ − ϵζ(xt, t)||2
2

(cid:105)

.

(1)

Different from the denoise-based model, the flow-based model aims to learn a velocity field vζ(xt, t)
that governs a probability flow ordinary differential equation (PF ODE). This PF ODE allows the
model to sample data by flowing toward the data distribution. This forward process is defined as:
xt = αtx0 + σtϵ, α0 = σT = 1, αT = σ0 = 0,
(2)
where x0 ∼ p(x) is the data, ϵ ∼ N (0, I) is Gaussian noise, and αt and σt are monotone decreasing
and increasing functions of t ∈ [0, T ], respectively. The PF ODE is given by:

˙xt = vζ(xt, t),
(3)
where the marginal distribution of this ODE at time pt(x) matches that of the forward process. To
learn the velocity field, the model is trained to minimize the following loss function:

(4)
For the sake of simplicity, in the following part, we use generative loss (Lgen) to uniformly represent
the two generative training objectives of denoise-based and flow-based methods.

(cid:2)||vζ(xt, t) − ˙αtx0 − ˙σtϵ||2(cid:3).

Lvelocity = Ex0,ϵ,t

1In the following part, we use term ‘student’ to represent the trainable model and ‘teacher’ to represent the

EMA model in SRA for simplicity.

3

2.2 Self-Representation Alignment

Previous studies have demonstrated that learning good internal representations can both speed up
the diffusion transformer’s training convergence. In SRA, our insight is aligning the students latent
feature in the earlier layer conditioned on higher noise with that in the later layer conditioned on the
lower noise of the teacher to conduct self-representation alignment without requiring any external
representation components. As depicted in Figure 3, the goal of our simple training framework is to
let the diffusion transformer not only predict noise or velocity but also align with better visual rep-
resentations from itself. This operation thereby provides a simple way to enhance the representation
learning in the diffusion transformer during generative training without the need to design complex
representation regulation or introduce external representation foundation models.

Formally, let f be the trainable stu-
dent model and f∗ be the teacher
model. Considering input noise la-
tent, timestep, and condition to be
xt, t, and c. Then, we can ob-
tain the student encoder latent out-
put y = f m(xt, t, c) ∈ RB×N ×D,
where B, N, D > 0 are the batch-
size, number of patches and the em-
bedding dimension for f , and m de-
note the output from the mth layer
in f . Similarly, the output of the
teacher can be expressed as y∗ =
∗ (xt−k, t − k, c) ∈ RB×N ×D. In
f n
our SRA, we set m ≤ n, k ≥ 0, and
2. Hence, we
0 ≤ (t − k) < tmax
can conduct self-representation align-
ment using the teacher’s output y∗
and the student’s output
transform
jψ(y) ∈ RB×N ×D, where jψ(y) is a
projection of the student encoder out-
put y that through a lightweight train-
able MLPs head jψ. Notably, this
projection head can be discarded optionally after training, which enables SRA to provide guidance
without altering any architecture in diffusion transformers.

Figure 3: Overall framework. SRA aligns the student’s la-
tent representation in the earlier layer conditioned on higher
noise (green branch) to that of the teacher in the later layer
conditioned on lower noise (blue branch) to achieve self-
representation alignment. We use a stop-gradient (sg) op-
erator on the teacher to let gradients flow only through the
student, and update the teacher’s parameters with an expo-
nential moving average (ema) of the student’s parameters.

In particular, SRA attains self-alignment by minimizing the patch-wise distance between the
teacher’s output (y∗) and the student’s output variant (jψ(y)):

Lsa(ζs, ψ) = Ext,t,c

(cid:104) 1
N

N
(cid:88)

i=1

dist(y[i]

∗ , jψ(y[i]))

(cid:105)
,

(5)

where [i] is a patch index, dist(·, ·) is a pre-defined distance calculation function, and ζs, ψ is the
parameters of student diffusion transformer and the projection head.

Finally, we add the abovementioned two objectives for joint learning:

L = Lgen + λLsa,
where λ > 0 is a hyperparameter that controls the trade-off between the generation object and the
self-representation alignment object.

(6)

2.3 Teacher Network

In SRA, we do not need an off-the-shelf teacher to give the guidance. Meanwhile, using the output
of the same model as the target to conduct supervision would cause poor results (both found in
previous work [13, 28] and our experiment). Thus, we build the teacher from past iterations of the

2For SiTs, tmax = 1 and for DiTs, tmax = 1000. In practice, we truncate (t − k) to 0 if it is less than 0.

4

student network using an exponential moving average (EMA) on the student weights. In specific, the
updated role of EMA is ζt = αζt + (1 − α)ζs, where α ∈ [0, 1) is the momentum coefficient. Note
that this is widely used in self-supervised learning (SSL) [92, 9, 30]. But we find the default updating
role in SSL does not perform well here, instead, α = 0.9999 unchanged works surprisingly well
in our framework. Experiments and analyses of different α settings are presented in Appendix F.
Moreover, we do not use other operations like clustering constraints [7, 8], batch normalizations [28,
70], and centering [9, 92] in SSL because we find that the training progress is already stable enough
without applying these tricks.

3 Experiment

In this section, we mainly focus on answering the following questions:
• How each design choice and component in SRA influence the performance? (Table 1)
• Does SRA work on different baselines across different model sizes? (Figure 4, and Table 2)
• Can SRA show comparable or superior performance against methods that leverage either the ex-
ternal representation task or the external representation encoder? (Table 3, Table 4, Figure 6)
• Does SRA genuinely enhance the representation capacity of the baseline model, and is the gener-

ation capability indeed strongly correlated with the representation guidance? (Figure 7)

3.1 Experimental Setup

Implementation details. Unless otherwise specified, the training details strictly follow the setup
in DiT [65] and SiT [59], including AdamW [57] with a constant learning rate of 1e-4, no weight
decay, batchsize of 256, using the Stable Diffusion VAE [71] to extract the latent, and etc. For
model configurations, we use the B/2, L/2, and XL/2 architectures introduced in the DiT and SiT
papers, which process inputs with a patch size of 2. Additional experimental details, hyperparameter
settings, linear probing details, and computing resources, are provided in Appendix C.

Evaluation. We report Fr´echet inception distance (FID [31]), sFID [62], inception score (IS [73]),
precision (Pre.) and recall (Rec.) [46]. To ensure a fair comparison with previous methods, we also
use the ADMs TensorFlow evaluation suite [20]with 50K samples and the same reference statistics.
A detailed breakdown of each evaluation metric is included in Appendix D.

Methods for Comparison.

We compare with recent advanced methods based on diffusion models: (a) Pixel diffusion: ADM,
VDM++, Simple diffusion, CDM, (b) Latent diffusion with U-Net: LDM, and (c) Latent diffusion
with transformers: DiT, SiT, SD-DiT, MaskDiT, TREAD, REPA, and MAETok. We give detailed
descriptions of each method in Appendix E. Note that in the diffusion transformers family, we
choose to compare with DiT/SiT and their modifications with representation components involved
but do not compare with works aiming at designing advanced architecture like lightningDiT [86]
and DDT [78].

3.2 Component-Wise Analysis

Below, we provide a detailed analysis of the impact of each component. We use SiT-B/2 and train
with SRA for 400K iterations for evaluation. Results are shown in Table 1. We also provide some
hyperparameter setting principles for more easily transferring SRA to new models in Appendix G.

Block layers for alignment. We begin by analyzing the effect of using different blocks of student
and teacher for alignment. We observe that using the teacher’s last but not least few layers (e.g., 8) to
regulate the student’s first layers (e.g., 3) leads to optimal performance. We assume that the first few
layers need more guidance so they can catch semantically meaningful representation for subsequent
generation. Meanwhile, there is a strong correlation between the quality of the representations of the
teacher’s layers and the performance of the corresponding aligned student (we give the quantitative
results and analysis in the latter Section 3.4). Based on these results, we set alignment layers as
3 −→ 8, 6 −→ 16, and 8 −→ 20 for B, L, and XL models respectively as default3

3For DiTs, we set alignment layers as 3 −→ 7, 6 −→ 14, and 8 −→ 16 by default because DiT’s discriminative

behavior across layers is slightly different from SiT’s (see Figure 2 and Figure 8).

5

Table 1: Component-wise analysis on ImageNet 256×256 without classifier-free guidance. ↓ and
↑ indicate whether lower or higher values are better, respectively. m −→ n denotes aligning features
from mth layer of the student with that from nth layer of the teacher. [0, kmax) denotes time interval
k is chosen randomly from 0 to kmax. PH. denotes whether to use the projection head.
IS↑

Time Interval

Block Layers

FID↓

PH.

λ

SiT-B/2 Baseline [59]

6 −→ 10
4 −→ 8
4 −→ 10
4 −→ 12
2 −→ 6
2 −→ 8
3 −→ 8
3 −→ 3

3 −→ 8
3 −→ 8
3 −→ 8
3 −→ 8
3 −→ 8
3 −→ 8

3 −→ 8
3 −→ 8
3 −→ 8
3 −→ 8

3 −→ 8
3 −→ 8

[0, 0.2)
[0, 0.2)
[0, 0.2)
[0, 0.2)
[0, 0.2)
[0, 0.2)
[0, 0.2)
[0, 0.2)

0.0
0.1
0.2
[0, 0.1)
[0, 0.2)
[0, 0.3)

[0, 0.2)
[0, 0.2)
[0, 0.2)
[0, 0.2)

[0, 0.2)
[0, 0.2)

0.2
0.2
0.2
0.2
0.2
0.2
0.2
0.2

0.2
0.2
0.2
0.2
0.2
0.2

0.1
0.2
0.3
0.4

0.2
0.2

✓
✓
✓
✓
✓
✓
✓
✓

✓
✓
✓
✓
✓
✓

✓
✓
✓
✓

×
✓

33.02

34.85
30.00
30.65
33.19
32.14
29.31
29.10
37.08

31.07
29.55
30.70
29.38
29.10
29.15

30.65
29.10
29.28
29.75

34.23
29.10

43.71

40.28
47.78
46.69
43.30
46.36
50.13
50.20
41.54

47.32
49.01
47.72
49.32
50.20
50.01

48.31
50.20
49.72
49.30

41.07
50.20

Time interval for alignment. We then study the time interval (meaning the same as k in Sec-
tion 2.2) used for alignment. Here, we study fixed and dynamic intervals. Notably, we find that
using the teacher’s features input with a lower noise level than the student’s leads to performance
enhancement, and an interval value of 0.1 or the mean of 0.1 is optimal. We hypothesize that this
is because lower noise levels can offer better representation guidance, but an excessively large time
interval can hinder the model’s learning process, causing it to focus only on optimizing alignment
loss at the expense of neglecting the generative aspects. As dynamic interval shows slightly better
performance, we apply time interval as 0 ∼ 0.2 in our feature experiments4.

Regularization coefficient for alignment. We also examine the effect of the coefficient λ of self-
alignment loss. Note that the results are all better than the baseline, and there is only a little variance
for different λ. We choose 0.2 as default since it achieves overall better performance.

Effect of projection head for alignment. We finally examine the effect of the projection head
for alignment. Surprisingly, We observe that using this simple head to post-possess the student’s
output is much better than directly using it to align. We hypothesize this slight operation enables
the model to learn more effective hidden representations for subsequent projection head to conduct
transformation for final alignment, rather than explicitly aligning the entire latent feature that could
potentially disrupt the original generation field that each layer and timestep is responsible for [90,
89, 26]. Hence, we keep using the projection head in future experiments.

3.3 System-Level Comparison

In this section, a system-level comparative study is performed to assess recent diffusion-based ap-
proaches against diffusion transformers with SRA.

SRA accelerate the convergence of models in different types and sizes. As our method requires
an additional forward pass of the teacher model. For a fair comparison, we report the FID compar-
ison with the baseline under training time. We also give specific training speed and GPU memory
usage in Appendix H, as well as FID vs.
training time against REPA in Figure 6. As shown in
Figure. 4, diffusion transformers trained with SRA demonstrate substantial improvements in perfor-

4For DiTs, we set time interval as ⌊[0, 200)⌋ by default since DiTs adopt the linear variance schedule with

t where t ∈ {0, 1, 2, ..., 999} ∩ Z.

6

Figure 4: FID comparisons with vanilla DiTs and SiTs on ImageNet 256×256 without classifier-
free guidance (CFG). The training time (h) is tested on a single machine with 8 A100 GPUs.

Figure 5: Selected samples on ImageNet 256×256 from the SiT-XL + SRA. We use classifier-free
guidance with w = 4.0. More uncurated samples are provided in Appendix L.

mance at the same training time. Moreover, similar to the observation in some SSL works [64, 24],
we notice that the effect of SRA in a larger size model is more significant, which is probably because
the larger model tends to provide richer guidance. Meanwhile, the benefits of SRA do not saturate
even when the models have already achieved a low FID score. We assume that this is likely due to the
teacher’s constantly improving capacity, which allows it to provide better and better representation
guidance for the student when the training goes on.

Meanwhile, we also conduct
text-to-image experi-
ment following REPA to use COCO2014 [50] and
MMDiT [23] to further demonstrate the effect of SRA,
the results in Table 2 shows that SRA can naturally extend
to text-to-image generation. Our self-representaion align-
ment strategy works, as evidenced by our experiments:
applying SRA to MMDiT without any hyperparameter
tuning improves FID ( from 5.66 to 4.75) and PickScore
(from 20.65 to 21.14) over the baseline, and is compara-
ble to REPA.

Table 2:
FID and Pickscore [43]
comparisons with vanilla MMDiT and
REPA on COCO2014.

Method

FID↓

PickScore↑

ODE, NFE=50, Trained for 150K iter

MMDiT
MMDiT + REPA
MMDiT + SRA

5.86
4.60
4.85

20.05
20.88
21.14

7

Table 3: System-level comparison on ImageNet 256×256 with Classifier-free Guidance (CFG).
The best and second-best results on each metric are highlighted in bold and underlined.

Model

Epochs

Tokenizer

FID↓

sFID↓

IS↑

Pre.↑

Rec.↑

Pixel diffusion
ADM-U
VDM++
Simple diffusion
CDM

Latent diffusion, U-Net

400
560
800
2160

-
-
-
-

3.94
2.40
2.77
4.88

6.14
-
-
-

186.7
225.3
211.8
158.7

0.82
-
-
-

0.52
-
-
-

LDM-4

200

LDM-VAE

3.60

-

247.7

0.87

0.48

Latent diffusion, Transformer

DiT-XL/2
SiT-XL/2
SD-DiT
MaskDiT
DiT + TREAD
SiT + REPA
SiT + MAETok
SiT + SRA (ours)
SiT + SRA (ours)

1400
1400
480
1600
740
800
800
400
800

SD-VAE
SD-VAE
SD-VAE
SD-VAE
SD-VAE
SD-VAE
MAE-Tok
SD-VAE
SD-VAE

2.27
2.06
3.23
2.28
1.69
1.42
1.67
1.85
1.58

4.60
4.50
-
5.67
4.73
4.70
-
4.50
4.65

278.2
270.3
-
276.6
292.7
305.7
311.2
297.2
311.4

0.83
0.82
-
0.80
0.81
0.80
-
0.82
0.80

0.57
0.59
-
0.61
0.63
0.65
-
0.61
0.63

Table 4: System-level comparison on ImageNet 512×512 with Classifier-free Guidance (CFG).

Model

Epochs

FID↓

sFID↓

IS↑

Pre.↑

Rec.↑

Pixel diffusion
VDM++
Simple diffusion

-
800

Latent diffusion, Transformer

DiT-XL/2
SiT-XL/2
MaskDiT
SiT + REPA
SiT + SRA (ours)

600
600
600
200
200

2.65
4.28

3.04
2.62
2.50
2.08
2.17

-
-

5.02
4.18
5.10
4.19
4.15

278.1
171.0

240.8
252.2
256.3
274.6
279.3

-
-

0.84
0.84
0.84
0.83
0.83

-
-

0.54
0.57
0.56
0.58
0.59

SRA demonstrates superior or comparable performance compared to other methods. We also
provide a quantitative comparison between SiT-XL with SRA and other recent methods shown in
Table 3 and Table 4. In ImageNet 256×256, Our method already outperforms the original SiT-XL
model with 1000 fewer epochs, and it is further improved with longer training. At 800 epochs,
SRA achieves FID of 1.58 and IS of 311.4.
It is worth noting that this result is far superior to
methods (e.g., MaskDiT) that rely on external representation task and is comparable with methods
(e.g., REPA) that depend heavily on an external pre-trained representation model.

In higher resolution settings,
the performance of SRA sur-
passes its baseline using 3×
fewer training iterations. Train-
ing with the same iterations,
SRA outperforms REPA in
terms of three metrics (sFID,
IS, and Rec) and on par with
REPA in other metrics.

SRA shows a more sustained
boost than REPA. As shown
in Figure 6, although REPA
converges quickly at
the be-
ginning of training due to its
large-scale, pre-trained repre-
sentation model for guidance, the performance saturates after about 200 epochs. On the contrary, the
progressively higher-quality guidance teacher have throughout training in SRA makes our method

Figure 6: Training time (h) vs. FID plot without CFG.

8

Figure 7: We empirically investigate the effect of representations in SRA. (a) and (b): Linear prob-
ing result of vanilla SiT-XL trained for 1400 epochs and SiT-XL + SRA trained for 800 epochs. (c):
Linear probing vs. FID plot of SiT-XL + SRA with different teacher’s output layers for alignment
(similar plot of DiT + SRA is provided in Appendix J). (d): Linear probing results of layers of the
teacher and the student used for alignment during training.

to provide a more sustained boost without the need for any external models. This observation of sat-
uration effects and potential detrimental impacts of REPA in later stages has also been corroborated
by several contemporary studies [80, 88]. We believe this limitation of REPA can be offset by the
sustained enhancement facilitated by SRA.

3.4 Ablation Study

Since SRA introduces representation guidance in an implicit way, we thus aim at testing whether
representation truly matters in SRA. We give the answer with the following experiments.

Enhanced representation capacity with SRA. We first compare the representation capacity of
vanilla SiT and SiT trained with SRA. As shown in Figure 7((a)) and Figure 7((b)), SRA consistently
improve the quality of latent representation in the diffusion transformer, as indicated by better linear
probing results across different blocks and timesteps.

Tight coupling between generation quality and representation guidance in SRA. We then in-
vestigate the correlation between generation performance and the representation guidance (detailed
experimental setup can be found in Appendix I) in SRA. Figure 7((c)) reveals a strong correlation
between linear probing accuracy and FID scores as the teacher network layers for alignment are
varied. This finding underscores that the model’s generative capabilities are indeed closely tied to
the effectiveness of the self-representation guidance mechanism.

Consistent representation improvement during training in SRA. We finally verify whether the
representational capacity of the student and teacher models is continuously improving during the
training process and whether the teacher can always give better guidance for the student. As shown
in Figure 7((d)), the teacher’s representation quality consistently improves throughout training (from
38.1 at 200K iterations to 54.2 at 800K iterations) and consistently outperforms the student model at
each stage. This empirical evidence directly supports our claim that the teacher’s improving capacity
provides better representation guidance as training proceeds.

4 Related Work

Here, we highlight key related studies and defer a discussion of other relevant studies to Appendix K.

Representations guidance for diffusion transformer’s training. Many recent works have at-
tempted to introduce representation guidance in diffusion transformer training. MaskDiT [91] and
SD-DiT [93] add MAE’s [29] and IBOT’s [92] learning task into the original DiT’s training progress.
TREAD [44] design a token routing strategy with MAE loss to speed up the diffusion model’s train-
ing. REPA [87] utilizes a large-scale data pre-trained representation model for regulating the dif-
fusion model’s latent feature. VA-VAE [86] and MAETok [10] align the latent distribution of the
tokenizer with the external representation foundation model and show this alignment is beneficial to
resulting diffusion models. Different from this work, we aim to look for representation guidance in
the diffusion model itself and its own training paradigm.

9

  (a) Linear Probing Result (t=0.1)       (b) Linear Probing Result of Layer 8                  (c) Validation Acc. vs. FID                    (d)  Linear Probing Comparision               5 Limitation and Future Work

Due to limitations in compuation resources, we are unable to conduct large-scale text-to-video (t2v)
pretraining with SRA. However, we maintain that applying SRA to video domain is conceptually
sound for the following reasons. First, even within the image domain (already exist strong encoder
like DINOv2), our method achieves performance comparable to that of REPA, this supports our
confidence in the effectiveness of approach in text-to-video generation, where currently there is no
well-pretrained strong encoder for open-domain videos (there is some video pretrained model, e.g.,
VideoMAE [77], which is still not strong enough for open-domain video encoder, compared to DI-
NOv2 for image encoder5.). Second, in the video domain, numerous studies [82, 94] have shown
that large-scale pretrained t2v models already capture rich and transferable representations suitable
for various understanding tasks, further reinforcing the potential of SRA in video-related applica-
tions. Moreover, similar to other related works like REPA, our method is also experiment-driven.
Exploring theoretical insights into why learning a good representation is beneficial to generation
will also be an exciting future direction.

6 Conclusion

In this study, we show that diffusion transformers can provide representation guidance by themselves
to boost generation performance with our proposed SRA, which aligns their latent representation in
the earlier layer conditioned on higher noise to that in the later layer conditioned on lower noise
to progressively enhance the representation learning without external components. Considering the
simplicity and effectiveness of SRA, we believe it will facilitate more future research to extend SRA
in other scenarios.

7 Acknowledgment

Our SRA manuscript referred to some of the color schemes and structural options provided in the
REPA [87] manuscript. We sincerely thank REPA’s authors for their fully open-source project. We
also thank Sihyun Yu, Sizhe Dang, and Zanyi Wang for the helpful discussions and suggestions
during the progress of SRA project.

References

[1] Abdi, H., Williams, L.J.: Principal component analysis. Wiley interdisciplinary reviews: com-

putational statistics 2(4), 433–459 (2010)

[2] Albergo, M.S., Vanden-Eijnden, E.: Building normalizing flows with stochastic interpolants.

In: International Conference on Learning Representations (2023)

[3] Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J.,

et al.: Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923 (2025)

[4] Bao, F., Nie, S., Xue, K., Cao, Y., Li, C., Su, H., Zhu, J.: All are worth words: A vit backbone
for diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition. pp. 22669–22679 (2023)

[5] Bao, H., Dong, L., Piao, S., Wei, F.: BEiT: BERT pre-training of image transformers. In:

International Conference on Learning Representations (2022)

[6] Brooks, T., Peebles, B., Holmes, C., DePue, W., Guo, Y., Jing, L., Schnurr, D., Taylor, J.,
Luhman, T., Luhman, E., Ng, C., Wang, R., Ramesh, A.: Video generation models as world
simulators. OpenAI Blog (2024)

5As mentioned by Waver [88], a video generation model uses REPA by employing Qwen2.5-VL [3] to
extract video representations. It is observed that the early training process benefits from the alignment and the
alignment does not bring gain when the training converges. This might stem from the representations extracted
from Qwen2.5-VL are not meaningful enough for video generation. Using internal representations of diffusion
transformer potentially remedies this.

10

[7] Caron, M., Bojanowski, P., Joulin, A., Douze, M.: Deep clustering for unsupervised learning
of visual features. In: Proceedings of the European conference on computer vision (ECCV).
pp. 132–149 (2018)

[8] Caron, M., Misra, I., Mairal, J., Goyal, P., Bojanowski, P., Joulin, A.: Unsupervised learning of
visual features by contrasting cluster assignments. Advances in neural information processing
systems 33, 9912–9924 (2020)

[9] Caron, M., Touvron, H., Misra, I., J´egou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging
properties in self-supervised vision transformers. In: Proceedings of the International Confer-
ence on Computer Vision (2021)

[10] Chen, H., Han, Y., Chen, F., Li, X., Wang, Y., Wang, J., Wang, Z., Liu, Z., Zou, D.,
Raj, B.: Masked autoencoders are effective tokenizers for diffusion models. arXiv preprint
arXiv:2502.03444 (2025)

[11] Chen, J., Yu, J., Ge, C., Yao, L., Xie, E., Wu, Y., Wang, Z., Kwok, J., Luo, P., Lu, H., Li, Z.:
Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In:
International Conference on Learning Representations (2024)

[12] Chen, S., Sun, P., Song, Y., Luo, P.: Diffusiondet: Diffusion model for object detection. In:
Proceedings of the IEEE/CVF international conference on computer vision. pp. 19830–19843
(2023)

[13] Chen, X., He, K.: Exploring simple siamese representation learning. In: Proceedings of the

IEEE/CVF conference on computer vision and pattern recognition. pp. 15750–15758 (2021)

[14] Chen, X., Liu, Z., Xie, S., He, K.: Deconstructing denoising diffusion models for self-

supervised learning. arXiv preprint arXiv:2401.14404 (2024)

[15] Chen, X., Xie, S., He, K.: An empirical study of training self-supervised vision transformers.
In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 9640–9649
(2021)

[16] Dalal, N., Triggs, B.: Histograms of oriented gradients for human detection. In: 2005 IEEE
computer society conference on computer vision and pattern recognition (CVPR’05). vol. 1,
pp. 886–893. Ieee (2005)

[17] Daniel Verd, J.M.: Flux.1 lite: Distilling flux1.dev for efficient text-to-image generation.

https://huggingface.co/Freepik (2024)

[18] Dao, T.: Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv

preprint arXiv:2307.08691 (2023)

[19] Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: Imagenet: A large-scale hierarchi-
cal image database. In: 2009 IEEE conference on computer vision and pattern recognition. pp.
248–255. Ieee (2009)

[20] Dhariwal, P., Nichol, A.: Diffusion models beat gans on image synthesis. Advances in neural

information processing systems 34, 8780–8794 (2021)

[21] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., De-
hghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words:
Transformers for image recognition at scale (2021)

[22] Elfwing, S., Uchibe, E., Doya, K.: Sigmoid-weighted linear units for neural network function

approximation in reinforcement learning. Neural networks 107, 3–11 (2018)

[23] Esser, P., Kulal, S., Blattmann, A., Entezari, R., M¨uller, J., Saini, H., Levi, Y., Lorenz, D.,
Sauer, A., Boesel, F., et al.: Scaling rectified flow transformers for high-resolution image
synthesis. In: Forty-first international conference on machine learning (2024)

[24] Fan, D., Tong, S., Zhu, J., Sinha, K., Liu, Z., Chen, X., Rabbat, M., Ballas, N., LeCun,
Y., Bar, A., Xie, S.: Scaling language-free visual representation learning. arXiv preprint
arXiv:2504.01017 (2025)

11

[25] Fang, G., Li, K., Ma, X., Wang, X.: Tinyfusion: Diffusion transformers learned shallow. arXiv

preprint arXiv:2412.01199 (2024)

[26] Frenkel, Y., Vinker, Y., Shamir, A., Cohen-Or, D.: Implicit style-content separation using b-

lora. In: European Conference on Computer Vision. pp. 181–198. Springer (2024)

[27] Gou, J., Yu, B., Maybank, S.J., Tao, D.: Knowledge distillation: A survey. International Journal

of Computer Vision 129(6), 1789–1819 (2021)

[28] Grill, J.B., Strub, F., Altch´e, F., Tallec, C., Richemond, P., Buchatskaya, E., Doersch, C.,
Avila Pires, B., Guo, Z., Gheshlaghi Azar, M., et al.: Bootstrap your own latent-a new approach
to self-supervised learning. Advances in neural information processing systems 33, 21271–
21284 (2020)

[29] He, K., Chen, X., Xie, S., Li, Y., Doll´ar, P., Girshick, R.: Masked autoencoders are scalable
vision learners. In: Proceedings of the IEEE/CVF conference on computer vision and pattern
recognition. pp. 16000–16009 (2022)

[30] He, K., Fan, H., Wu, Y., Xie, S., Girshick, R.: Momentum contrast for unsupervised visual
representation learning. In: Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition. pp. 9729–9738 (2020)

[31] Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained by a two
time-scale update rule converge to a local nash equilibrium. Advances in neural information
processing systems 30 (2017)

[32] Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. Advances in neural

information processing systems 33, 6840–6851 (2020)

[33] Ho, J., Saharia, C., Chan, W., Fleet, D.J., Norouzi, M., Salimans, T.: Cascaded diffusion
models for high fidelity image generation. Journal of Machine Learning Research 23(47), 1–
33 (2022)

[34] Ho, J., Salimans, T.: Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598

(2022)

[35] Hoogeboom, E., Heek, J., Salimans, T.: simple diffusion: End-to-end diffusion for high res-
olution images. In: International Conference on Machine Learning. pp. 13213–13232. PMLR
(2023)

[36] Hunter, J.S.: The exponentially weighted moving average. Journal of quality technology 18(4),

203–210 (1986)

[37] Jiang, D., Liu, D., Wang, Z., Wu, Q., Li, L., Li, H., Jin, X., Liu, D., Li, Z., Zhang, B., et al.: Dis-
tribution matching distillation meets reinforcement learning. arXiv preprint arXiv:2511.13649
(2025)

[38] Jiang, D., Wang, H., Zhang, L., Wei, W., Dai, G., Wang, M., Wang, J., Zhang, Y.: Unbiased

general annotated dataset generation. arXiv preprint arXiv:2412.10831 (2024)

[39] Jiang, L., Yan, Q., Jia, Y., Liu, Z., Kang, H., Lu, X.: InfiniteYou: Flexible photo recrafting

while preserving your identity. arXiv preprint arXiv:2503.16418 (2025)

[40] Karras, T., Aittala, M., Lehtinen, J., Hellsten, J., Aila, T., Laine, S.: Analyzing and improving
the training dynamics of diffusion models. In: Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition. pp. 24174–24184 (2024)

[41] Kingma, D., Gao, R.: Understanding diffusion objectives as the elbo with simple data augmen-

tation. Advances in Neural Information Processing Systems 36, 65484–65516 (2023)

[42] Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S.,
Berg, A.C., Lo, W.Y., et al.: Segment anything. In: Proceedings of the IEEE/CVF international
conference on computer vision. pp. 4015–4026 (2023)

12

[43] Kirstain, Y., Polyak, A., Singer, U., Matiana, S., Penna, J., Levy, O.: Pick-a-pic: An open
dataset of user preferences for text-to-image generation. Advances in neural information pro-
cessing systems 36, 36652–36663 (2023)

[44] Krause, F., Phan, T., Hu, V.T., Ommer, B.: Tread: Token routing for efficient architecture-

agnostic diffusion training. arXiv preprint arXiv:2501.04765 (2025)

[45] Kynk¨a¨anniemi, T., Aittala, M., Karras, T., Laine, S., Aila, T., Lehtinen, J.: Applying guidance
in a limited interval improves sample and distribution quality in diffusion models. Advances in
Neural Information Processing Systems 37, 122458–122483 (2025)

[46] Kynk¨a¨anniemi, T., Karras, T., Laine, S., Lehtinen, J., Aila, T.: Improved precision and recall
metric for assessing generative models. Advances in neural information processing systems 32
(2019)

[47] Labs, B.F.: Flux. https://github.com/black-forest-labs/flux (2024)

[48] Li, D., Ling, H., Kar, A., Acuna, D., Kim, S.W., Kreis, K., Torralba, A., Fidler, S.:
Dreamteacher: Pretraining image backbones with deep generative models. In: Proceedings
of the IEEE/CVF International Conference on Computer Vision. pp. 16698–16708 (2023)

[49] Li, T., Katabi, D., He, K.: Return of unconditional generation: A self-supervised representation
generation method. Advances in Neural Information Processing Systems 37, 125441–125468
(2024)

[50] Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Doll´ar, P., Zitnick, C.L.:
Microsoft coco: Common objects in context. In: European conference on computer vision. pp.
740–755. Springer (2014)

[51] Lin, W., Wei, X., Zhang, R., Zhuo, L., Zhao, S., Huang, S., Xie, J., Qiao, Y., Gao, P., Li,
H.: Pixwizard: Versatile image-to-image visual assistant with open-language instructions. In:
International Conference on Learning Representations (2025)

[52] Lipman, Y., Chen, R.T., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for generative

modeling. arXiv preprint arXiv:2210.02747 (2022)

[53] Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Jiang, Q., Li, C., Yang, J., Su, H., et al.:
Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In:
European Conference on Computer Vision. pp. 38–55. Springer (2024)

[54] Liu, X., Gong, C., Liu, Q.: Flow straight and fast: Learning to generate and transfer data with

rectified flow. In: International Conference on Learning Representations (2023)

[55] Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin transformer:
Hierarchical vision transformer using shifted windows. In: Proceedings of the IEEE/CVF in-
ternational conference on computer vision. pp. 10012–10022 (2021)

[56] Liu, Z., Mao, H., Wu, C.Y., Feichtenhofer, C., Darrell, T., Xie, S.: A convnet for the 2020s.
In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp.
11976–11986 (2022)

[57] Loshchilov,

I., Hutter, F.: Decoupled weight decay regularization. arXiv preprint

arXiv:1711.05101 (2017)

[58] Luo, S., Tan, Y., Huang, L., Li, J., Zhao, H.: Latent consistency models: Synthesizing high-

resolution images with few-step inference. arXiv preprint arXiv:2310.04378 (2023)

[59] Ma, N., Goldstein, M., Albergo, M.S., Boffi, N.M., Vanden-Eijnden, E., Xie, S.: Sit: Ex-
ploring flow and diffusion-based generative models with scalable interpolant transformers. In:
European Conference on Computer Vision. pp. 23–40. Springer (2024)

[60] Meng, C., Rombach, R., Gao, R., Kingma, D., Ermon, S., Ho, J., Salimans, T.: On distillation
of guided diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition. pp. 14297–14306 (2023)

13

[61] Mukhopadhyay, S., Gwilliam, M., Agarwal, V., Padmanabhan, N., Swaminathan, A., Hegde,
S., Zhou, T., Shrivastava, A.: Diffusion models beat gans on image classification. arXiv
preprint arXiv:2307.08702 (2023)

[62] Nash, C., Menick, J., Dieleman, S., Battaglia, P.W.: Generating images with sparse represen-

tations. arXiv preprint arXiv:2103.03841 (2021)

[63] Nichol, A.Q., Dhariwal, P.: Improved denoising diffusion probabilistic models. In: Interna-

tional conference on machine learning. pp. 8162–8171. PMLR (2021)

[64] Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P.,
Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without
supervision. arXiv preprint arXiv:2304.07193 (2023)

[65] Peebles, W., Xie, S.: Scalable diffusion models with transformers. In: Proceedings of the

IEEE/CVF international conference on computer vision. pp. 4195–4205 (2023)

[66] Preechakul, K., Chatthee, N., Wizadwongsa, S., Suwajanakorn, S.: Diffusion autoencoders:
Toward a meaningful and decodable representation. In: Proceedings of the IEEE/CVF confer-
ence on computer vision and pattern recognition. pp. 10619–10629 (2022)

[67] Qin, Q., Zhuo, L., Xin, Y., Du, R., Li, Z., Fu, B., Lu, Y., Li, X., Liu, D., Zhu, X.,
et al.: Lumina-image 2.0: A unified and efficient image generative framework. arXiv preprint
arXiv:2503.21758 (2025)

[68] Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell,
A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language
supervision. In: International conference on machine learning. pp. 8748–8763. PMLR (2021)

[69] Ren, Y., Xia, X., Lu, Y., Zhang, J., Wu, J., Xie, P., Wang, X., Xiao, X.: Hyper-sd: Trajectory
segmented consistency model for efficient image synthesis. arXiv preprint arXiv:2404.13686
(2024)

[70] Richemond, P.H., Grill, J.B., Altch´e, F., Tallec, C., Strub, F., Brock, A., Smith, S., De,
S., Pascanu, R., Piot, B., et al.: Byol works even without batch statistics. arXiv preprint
arXiv:2010.10241 (2020)

[71] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image syn-
thesis with latent diffusion models. In: Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition. pp. 10684–10695 (2022)

[72] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image syn-
thesis with latent diffusion models. In: Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition. pp. 10684–10695 (2022)

[73] Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., Chen, X.: Improved
techniques for training gans. Advances in neural information processing systems 29 (2016)

[74] Song, Y., Dhariwal, P., Chen, M., Sutskever, I.: Consistency models. In: International Confer-

ence on Machine Learning. pp. 32211–32252. PMLR (2023)

[75] Sun, Q., Wang, J., Yu, Q., Cui, Y., Zhang, F., Zhang, X., Wang, X.: Eva-clip-18b: Scaling clip

to 18 billion parameters. arXiv preprint arXiv:2402.04252 (2024)

[76] Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., Wojna, Z.: Rethinking the inception archi-
tecture for computer vision. In: Proceedings of the IEEE conference on computer vision and
pattern recognition. pp. 2818–2826 (2016)

[77] Wang, L., Huang, B., Zhao, Z., Tong, Z., He, Y., Wang, Y., Wang, Y., Qiao, Y.: Videomae
v2: Scaling video masked autoencoders with dual masking. In: Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition. pp. 14549–14560 (2023)

[78] Wang, S., Tian, Z., Huang, W., Wang, L.: Ddt: Decoupled diffusion transformer. arXiv preprint

arXiv:2504.05741 (2025)

14

[79] Wang, W., Xie, E., Li, X., Fan, D.P., Song, K., Liang, D., Lu, T., Luo, P., Shao, L.: Pyramid
vision transformer: A versatile backbone for dense prediction without convolutions. In: Pro-
ceedings of the IEEE/CVF International Conference on Computer Vision. pp. 568–578 (2021)

[80] Wang, Z., Zhao, W., Zhou, Y., Li, Z., Liang, Z., Shi, M., Zhao, X., Zhou, P., Zhang, K., Wang,
Z., et al.: Repa works until it doesn’t: Early-stopped, holistic alignment supercharges diffusion
training. arXiv preprint arXiv:2505.16792 (2025)

[81] WanTeam: Wan: Open and advanced large-scale video generative models. arXiv preprint

arXiv:2503.20314 (2025)

[82] Wiedemer, T., Li, Y., Vicol, P., Gu, S.S., Matarese, N., Swersky, K., Kim, B., Jaini, P., Geirhos,
R.: Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328
(2025)

[83] Xiang, W., Yang, H., Huang, D., Wang, Y.: Denoising diffusion autoencoders are unified self-
supervised learners. In: Proceedings of the IEEE/CVF International Conference on Computer
Vision. pp. 15802–15812 (2023)

[84] Yang, X., Wang, X.: Diffusion model as representation learner. In: Proceedings of the

IEEE/CVF International Conference on Computer Vision. pp. 18938–18949 (2023)

[85] Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W., Zhang, X.,
Feng, G., et al.: Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv
preprint arXiv:2408.06072 (2024)

[86] Yao, J., Wang, X.: Reconstruction vs. generation: Taming optimization dilemma in latent

diffusion models. arXiv preprint arXiv:2501.01423 (2025)

[87] Yu, S., Kwak, S., Jang, H., Jeong, J., Huang, J., Shin, J., Xie, S.: Representation alignment for
generation: Training diffusion transformers is easier than you think. In: International Confer-
ence on Learning Representations (2025)

[88] Zhang, Y., Yang, H., Zhang, Y., Hu, Y., Zhu, F., Lin, C., Mei, X., Jiang, Y., Peng, B.,
Yuan, Z.: Waver: Wave your way to lifelike video generation. CoRR abs/2508.15761 (2025).
https://doi.org/10.48550/ARXIV.2508.15761, https://doi.org/10.48550/arXiv.2508.
15761

[89] Zhang, Y., Dong, W., Tang, F., Huang, N., Huang, H., Ma, C., Lee, T.Y., Deussen, O., Xu,
C.: Prospect: Prompt spectrum for attribute-aware personalization of diffusion models. ACM
Transactions on Graphics (TOG) 42(6), 1–14 (2023)

[90] Zhang, Z., Zhang, Q., Lin, H., Xing, W., Mo, J., Huang, S., Xie, J., Li, G., Luan, J., Zhao, L.,
et al.: Towards highly realistic artistic style transfer via stable diffusion with step-aware and
layer-aware prompt. arXiv preprint arXiv:2404.11474 (2024)

[91] Zheng, H., Nie, W., Vahdat, A., Anandkumar, A.: Fast training of diffusion models with

masked transformers. arXiv preprint arXiv:2306.09305 (2023)

[92] Zhou, J., Wei, C., Wang, H., Shen, W., Xie, C., Yuille, A., Kong, T.:

ibot: Image bert
pre-training with online tokenizer. In: International Conference on Learning Representations
(2022)

[93] Zhu, R., Pan, Y., Li, Y., Yao, T., Sun, Z., Mei, T., Chen, C.W.: Sd-dit: Unleashing the power
of self-supervised discrimination in diffusion transformer. In: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition. pp. 8435–8445 (2024)

[94] Zhu, Z., Feng, X., Chen, D., Yuan, J., Qiao, C., Hua, G.: Exploring pre-trained text-to-video
diffusion models for referring video object segmentation. In: European Conference on Com-
puter Vision. pp. 452–469. Springer (2024)

15

A Investigation of Representations in DiT

Appendix

We also perform a similar analysis with DiT like those have done in Figure 2(left) (PCA visual-
ization) and Figure 2(right) (linear probing), the results are showed in Figure 8. In short, we also
observe that the representations in DiT basically lead a process from coarse to fine when increasing
block layers and decreasing noise level as SiT’s.

Figure 8: We also empirically investigate the representations in diffusion transformers across differ-
ent blocks and timesteps with the original DiT-XL/2 checkpoint trained for 7M iterations. Similar
to SiT, the latent representations in DiT basically follow the bad to good process, as block layers
increase and noise level decreased.

B Descriptions for Two Types of Baseline Models

In this paper, we use DiT and SiT as our baseline models. We now provide an overview of two types
of generative models that are variants of denoising autoencoders and are used to learn the target
distribution. Specifically, we discuss Denoising Diffusion Probabilistic Models (DDPMs) like DiT
in Section B.1 and Stochastic interpolant models like SiT in Section B.2.

B.1 Denoising Diffusion Probabilistic Models (DiT)

Denoise-based models [32, 63] aim to model the target distribution ( x ∼ p(x) ) by learning a
gradual denoising process that transforms a Gaussian distribution ( N (0, I) ) into p(x). Formally,
diffusion models learn a reverse process ( p(xt−1|xt) ) corresponding to a pre-defined forward
process ( q(xt|x0) ), which incrementally adds Gaussian noise to the data starting from p(x) over a
sequence of time steps ( t ∈ 1, . . . , T ), with T > 0 fixed.

√

For a given x0 ∼ p(x0), the forward process ( q(xt|xt−1) ) is defined as: q(xt|xt−1) =
N (xt;
t I), where βt ∈ (0, 1) are pre-defined small hyperparameters. DDPM [32]
formalizes the reverse process p(xt−1|xt) as:

1 − βtx0, β2

p(xt−1|xt) = N

(cid:16)

xt−1;

1
√
αt

(cid:0)xt −

σ2
t√
1 − ¯αt

(cid:17)
ϵζ(xt, t)(cid:1), Σζ(xt, t)

,

where αt = 1 − βt, ¯αt = (cid:81) i = 1tαi, and ϵζ(xt, t) is parameterized by a neural network.
The model is trained using a simple denoising autoencoder objective:

Lsimple = Ext,ϵ,t

(cid:104)

||ϵ − ϵζ(xt, t)||2
2

(cid:105)
.

16

(7)

(8)

Increasing Block LayersDecreasing Noise Levellayer1          layer3          layer8        layer13       layer18       layer23        layer28t=100          t=300          t=500           t=700                           (a) PCA Visualization Result                                                        (b) Linear Probing Result For the covariance term ( Σζ(xt, t) ), DDPM[32] demonstrated that setting it as ( σ2
t I ) with ( βt =
σ2
t ) is sufficient. Subsequently, Improved-DDPM[63] showed that performance can be enhanced by
jointly learning ( Σζ(xt, t) ) along with ( ϵζ(xt, t) ) in a dimension-wise manner using the following
objective:

Lvlb = exp(v log βt + (1 − v) log ˜βt),

(9)

where v is a component per model output dimension, and ˜βt = 1− ¯αt−1
1− ¯αt

βt.

With a sufficiently large T and an appropriate scheduling of βt, the distribution p(xT ) approaches
an isotropic Gaussian distribution. Thus, sampling is achieved by starting from random Gaussian
noise and iteratively applying the reverse process ( p(xt−1|xt) ) to recover a data sample x0 [32].

B.2 Stochastic Interpolants Models (SiT)

Unlike DDPMs, flow-based models [54, 52] describe a continuous time-dependent process involving
data ( x∗ ∼ p(x) ) and Gaussian noise ( ϵ ∼ N (0, I) ) over t ∈ [0, 1]:

xt = αtx0 + σtϵ, α0 = σ1 = 1, α1 = σ0 = 0,

(10)

with αt and σt being decreasing and increasing functions of t, respectively. The process is governed
by a probability flow ordinary differential equation (PF ODE):

˙xt = v(xt, t),

where the distribution of the ODE at time t matches the marginal distribution pt(x).

and can be approximated by a model vζ(xt, t) trained to minimize the objective:

Lvelocity = Ex0,ϵ,t

(cid:104)
||vζ(xt, t) − ˙αtx0 − ˙σtϵ||2(cid:105)

.

This also corresponds to a reverse stochastic differential equation (SDE) given by:

dxt = v(xt, t)dt −

wts(xt, t)dt +

√

wtd ¯wt,

1
2

As shown in [2], any functions αt and σt that satisfy the following three conditions:

(11)

(12)

(13)

t + σ2

t > 0, ∀t ∈ [0, 1]

1. α2
2. αt and σt are differentiable, ∀t ∈ [0, 1]
3. α1 = σ0 = 0, α0 = σ1 = 1,

lead to an unbiased interpolation process between x0 and ϵ. Example choices include linear inter-
polants (αt = 1 − t, σt = t) or variance-preserving (VP) interpolants (αt = cos( π
2 t))
[59].

2 t), σt = sin( π

An advantage of stochastic interpolants is that the diffusion coefficient ( wt ) can be independently
selected during sampling with the reverse SDE, even after training. This capability simplifies the
design space compared to score-based diffusion models [40].

C Hyperparameter and More Implementation Details

More implementation details. We implement our models based on the original DiT and SiT im-
plementation. To speed up training and save GPU memory, we use mixed-precision (fp16) with a
gradient clipping and FusedAttention [18] operation for attention computation. We also pre-compute
compressed latent vectors from raw pixels via stable diffusion VAE [71] and use these latent vectors.
We do not apply any data augmentation, but we find this does not lead to a big difference, as similarly
observed in MaskDiT [91] and REPA [87]. We also use stabilityai/sd-vae-ft-ema for encod-
ing images to latent vectors and decoding latent vectors to images. For the projection head used
for nonlinear transformation, we use two-layer MLP with SiLU activations [22]. We use smooth-ℓ1
objective for alignment because we find it perform slightly better than ℓ2 and its gradient transitions
are smoother than ℓ1. When using SiT-XL to generate images with classifier-free guidance [34], the

17

Table 5: Default hyperparameter setup. Unless other otherwise specified, we use these sets of
hyperparameters for different models. In our component-wise analysis experiment, settings are also
kept the same except those we point out in Table 1.

SiT-B

SiT-L

SiT-XL

DiT-B

DiT-L

DiT-XL

Architecture
Input dim.
Patch size
Num. layers
Hidden dim.
Num. heads

SRA
Alignment blocks
Alignment time interval
Objective
EMA decay
Using projection head

Optimization
Batch size
Optimizer
lr
(β1, β2)
Interpolants or Denoising
αt
σt
wt
T
Training objective
Sampler
Sampling steps
Guidance Scale

32×32×4
2
12
768
12

3 −→ 8
[0, 0.2)
smooth-ℓ1
0.999
✓

32×32×4
2
24
1024
16

6 −→ 16
[0, 0.2)
smooth-ℓ1
0.999
✓

32×32×4
2
28
1152
16

8 −→ 20
[0, 0.2)
smooth-ℓ1
0.999
✓

32×32×4
2
12
768
12

3 −→ 7
⌊[0, 200)⌋
smooth-ℓ1
0.999
✓

32×32×4
2
24
1024
16

6 −→ 14
⌊[0, 200)⌋
smooth-ℓ1
0.999
✓

32×32×4
2
28
1152
16

8 −→ 16
⌊[0, 200)⌋
smooth-ℓ1
0.999
✓

256
AdamW
0.0001
(0.9, 0.999)

256
AdamW
0.0001
(0.9, 0.999)

256
AdamW
0.0001
(0.9, 0.999)

256
AdamW
0.0001
(0.9, 0.999)

256
AdamW
0.0001
(0.9, 0.999)

256
AdamW
0.0001
(0.9, 0.999)

1 − t
t
σt
-
v-prediction

1 − t
t
σt
-
v-prediction
Euler-Maruyama Euler-Maruyama Euler-Maruyama
250
-

1 − t
t
σt
-
v-prediction

250
1.8 (if used)

250
-

-
-
-
1000
noise-prediction
DDPM
250
-

-
-
-
1000
noise-prediction
DDPM
250
-

-
-
-
1000
noise-prediction
DDPM
250
-

guidance interval introduced in the study [45] with the same setting used in REPA [87] is applied,
which has been demonstrated to yield a slight performance improvement.

Sampler. For DiT, we use the DDPM sampler and set the number of function evaluations (NFE) as
250 by default. For SiT, we use the SDE Euler-Maruyama sampler (for SDE with wt = σt) and set
the NEF as 250 by default. The settings also align with those used in DiT and SiT papers.

Dataset. We use ImageNet [19], where each image is preprocessed to the resolution of 256×256 or
512×512 (denoted as ImageNet 256×256 or ImageNet 512×512), and follow ADM [20] for other
data preprocessing protocols.

Linear probing. We follow the setup used in REPA [87] and I-DAE [14], and use the code-base
of ConvNeXt [56] to conduct the experiment. Specifically, we use AdaptiveAvgPooling and a batch
normalization layer to process the output of the latent feature by the model, then train a linear layer
for 80 epochs. The batch size is set to 4096 with a cosine decay learning rate scheduler, where the
initial learning rate is set to 0.001.

Computing resources. We use 8 NVIDIA A100 80GB GPUs or 8 NVIDIA L40S 48GB GPUs for
training largest model (XL); and use 4 NVIDIA A100 80GB GPUs or 4 NVIDIA L40S 48GB GPUs
for training smaller model (L, B), our training speed is about 2.12 step/s with a global batch size of
256. When sampling, we use either NVIDIA A100 80GB GPUs or NVIDIA L40S 48GB GPUs or
NVIDIA RTX 4090 24GB GPUs to obtain the samples for evaluation.

D Evaluation Metric

In this section, we define the key metrics used to assess the performance of our model. Each metric
is summarized below:

• Frchet Inception Distance (FID) [31]

Purpose: Measures the similarity between the feature distributions of real and generated im-
ages.
Methodology: Uses the Inception-v3 network [76] to extract features. Assumes both feature
distributions are multivariate Gaussian and computes the Frchet distance between them.
Interpretation: Lower FID scores indicate better similarity (and thus higher-quality generated
images).

• Spatial Frchet Inception Distance (sFID) [62]

Purpose: Extends FID by incorporating spatial information to better capture the structural

18

fidelity of generated images.
Methodology: Computes FID using intermediate spatial features (rather than global features)
from the Inception-v3 network.

• Inception Score (IS) [73]

Purpose: Evaluates the quality and diversity of generated images.
Methodology: Uses the Inception-v3 network to compute class probabilities (logits) for gener-
ated images. Measures the KL-divergence between the marginal class distribution of generated
images and the conditional class distribution of a single image (after softmax normalization).
Interpretation: Higher IS values indicate both high image quality (confident predictions) and
diversity (uniform marginal distribution).

• Precision and Recall for Distributions (Precision/Recall) [46]

Purpose: Evaluates the trade-off between sample quality (precision) and distribution coverage
(recall).
Methodology: Precision: Fraction of generated images deemed realistic by a classifier (relative
to real images). Recall: Fraction of the real image manifold covered by generated samples.

E Methods for Comparison

Next, we explain the key ideas behind the methods used for evaluation and comparison.

• ADM [20] enhances U-Net-based architectures for diffusion models and introduces classifier-
guided sampling, a technique used to balance the quality-diversity tradeoff and improve overall
performance.

• VDM++ [41] improves diffusion model training efficiency through an adaptive noise scheduling

mechanism that dynamically adjusts noise levels during optimization.

• Simple Diffusion [35] targets high-resolution synthesis by simplifying both network architec-

tures and noise schedules through systematic exploration of lightweight design choices.

• CDM [33] achieves high-fidelity generation via a cascaded pipeline: training low-resolution dif-
fusion models first, then progressively refining details through super-resolution diffusion stages.

• LDM [72] accelerates training while maintaining generation quality by operating in compressed
latent spaces, modeling image distributions at reduced dimensionality compared to pixel space.

• DiT [65] employs a pure transformer backbone for diffusion models, incorporating AdaIN-zero
modules as an important component modification against vanilla vision transformer [21] of its
architecture.

• SiT [59] provides an extensive analysis of how the training efficiency of DiTs can be improved
by transitioning from discrete diffusion to continuous flow-based modeling, which can be seen
as a flow-based version of DiT.

• SD-DiT [93] leverages IBOT’s [92] training paradigm that combines DINO loss [9] and BEIT

loss [5] for efficiently training diffusion transformers.

• MaskDiT [91] proposes an asymmetric encoder-decoder scheme for efficient training of diffu-
sion transformers, where they train the model with an auxiliary mask reconstruction task similar
to MAE [29].

• TREAD [44] introduces a dynamic token routing strategy combined with the mask reconstruc-
tion task similar to MAE [29] and MaskDiT [91] to accelerate the training of diffusion models.

• REPA [87] achieves significant improvements in both training efficiency and generation quality
by aligning the latent feature of the diffusion model with that of a large-scale data pre-trained
representation model (e.g., DINOv2 [64]).

• MAETok changes the SD-VAE to MAE-Tok, which is trained with auxiliary mask reconstruc-
tion loss and aligns loss with three representation targets (HOG’s [16], DINOv2’s [64], and
CLIP’s [68]) and obtains diffusion transformer with better generation performance.

19

Table 6: Different EMA update rule attempts, other settings are con-
sistent with the default setting of Table 1 in the main paper.

α of EMA

0.0
0.996 −→ 1.0
0.9999

FID↓

35.71
33.17
29.10

IS↑

42.18
44.96
50.20

F Teacher Network Updating Role

In other generative learning studies, the EMA model is often used solely for evaluation. However,
as we need it to provide guidance during training, we study different updating methods. Here, we
investigate three different strategies to build the teacher. First, we find that using teacher copied
from a student (α = 0) would impair the performance, which testifies to our argument in Section 1
. Next, we consider using the strategy widely used in self-supervised learning works [9, 92] that
updates the momentum coefficient α from 0.996 to 1 during training. However, in our framework,
this does not work well. Finally, we use the α of 0.9999 unchanged, which is commonly used
in other generative learning works [20, 65] and find it is also suitable for our framework. In our
analysis, first, student copy (α= 0.0) performs the worst : this is consistent with the observation
in self-supervised learning: student copy leads to unstable training and cannot converge; there is a
diffusion loss in our approach, so the training still converges. Second, using α= 0.9999 is better than
0.996 1.0: In EMA, a larger α indicates a smoother update. We analyze that the reason for keeping
such a very large value of α works well in SRA is that: in our method, the enhanced representation
learning is ultimately for the service of generation. Thus, when the update of teacher model is
relatively intense, the alignment would be less stable, the model may focus more on alignment loss
and overlook diffusion loss, thereby destroying some of the original generation behavior. On the
contrary, maintaining α in a very large value (0.9999) ensures the alignment target is extremely
robust and slow to change, preventing the alignment loss from destabilizing the primary diffusion
loss. Thus, we set the momentum coefficient as 0.9999 as our default.

G Principle of our hyperparameter

Although our method requires some hyperparameter tuning, we believe that some settings are model-
independent, while some settings have selection principles. We now give the detailed explanations,
and we hope these explanations can help to develop SRA to other new models more easily.

• Projection Head.

In generative models, different layers are often believed to have different
generation responsibilities. We hypothesize that using the lightweight projection head instead of
directly conducting the alignment can lead to a relatively soft alignment, and could be the reason
for avoiding disruption to the model’s original generative behavior. We believe this principle and
findings with the design of projection head can be easily transferred to new models.

• α in EMA decay. In EMA, a larger α indicates a smoother update. We analyze that the reason
for keeping such a very large value of α works well in SRA is that: in our method, the enhanced
representation learning is ultimately for the service of generation. Thus, when the update of
teacher model is relatively intense, the alignment would be less stable, the model may focus
more on alignment loss and overlook diffusion loss, thereby destroying some of the original
generation behavior. On the contrary, maintaining α in a very large value (0.9999) ensures the
alignment target is extremely robust and slow to change, preventing the alignment loss from
destabilizing the primary diffusion loss. This can also be indirectly verified by our following
analysis of λ. Given this analysis, we believe the choice of α is will be well-matched to most of
the new model.

• Block layers and time intervals. Principle of the selection of block layers: the block layer for
the teacher is selected so that the corresponding representation has strong semantics (See Figure
2 (b)). The block layer for students is the first few layer as the first few layers are more about
semantics learning, which is similar to REPA, and discussed in REPA. Principle of the time
interval: The intuitive guidance could be that the target representation for the teacher timestep

20

has better semantics, and the teacher timestep is not too far from the current timestep, i.e., a
balance between teacher representation semantics and the semantic distance between teacher
(timestep sampled from the interval) and student (current timestep). These principles are also
applicable to other diffusion transforms (e.g, MMDiT as shown Table 2).

• Coefficient λ. The first principle we adopt λ is to maintain the diffusion loss (approximately
0.7) and the alignment loss (approximately 1.6) in the same scale. Our experimental results
indicate that performance is slightly better when the alignment loss is relatively smaller, when
these two points are satisfied, we consider this little performance variance is not suspicious. We
also conduct additional experiments as follows, suggested that a smaller λ (e.g., 0.02) diminishes
the regularization effect of the alignment loss, while a larger λ (e.g., 2) makes the alignment loss
to dominate training instead of assist generation, which can harm sample quality (this can also
indirectly verifies why EMA performs well when a large fixed αis set.). Hence, once we know
the scale of diffusion loss and the above principles we give, the selection of λ will be very easy
to new model.

H Training Speed and GPU Memory Usage Against Baselines

Table 7: Memory useage, per-epoch training time comparison. “Param” counts only the additional
parameters required by the auxiliary forward pass. These results are tested with total batch size of
256 on a single machine with 8 A100 GPUs.

Method

Mem (GB)

Time (h)

Param

SiT-B
SiT-B + SRA

SiT-L
SiT-L + SRA

SiT-XL
SiT-XL + SRA

DiT-B
DiT-B + SRA

DiT-L
DiT-L + SRA

DiT-XL
DiT-XL + SRA

12.85
13.23

25.68
26.65

29.08
30.24

13.96
14.78

27.85
29.26

30.68
32.31

0.096
0.115

0.282
0.330

0.394
0.476

0.173
0.209

0.503
0.558

0.719
0.804

–
87 M

–
305 M

–
481 M

–
87 M

–
305 M

–
481 M

As our approach needs one more forward pass on the diffusion backbone to compute the target
representation. We now provide a specific comparison on GPU memory usage and training time with
baselines. Noting that the extra pass is only for forward, and no gradient computation is needed, so
we can use half-precision for fast forward. In addition, it is not needed to pass the whole network,
and only a partial pass is needed: 8/12 blocks in SiT-B, 20/28 blocks in SiT-XL. So it can be seen
that the additional cost are not very large, and the results in Figure 4 and Table 3 also indicate that
our method can improve the performance of the model within the same training time.

I Detailed Setup of Ablation Study

We now give the detailed experimental setup of Figure 7c in our ablation study. The accuracy
rate of the horizontal axis in the figure is obtained by using the linear probing results of the original
SiT-XL/2 checkpoint training for 7M iterations, while the vertical axis is the FID evaluation result of
training 400K iterations with SRA without classifier-free guidance (CFG). Since we do not introduce
any representation component and use the representation supervision signal only in the generative
training process, we consider this experiment to validate the effectiveness of our approach.

21

J Ablation Results of DiT

We also perform a similar analysis with DiT like those have done in Figure 7c, the results are
showed in Figure 9. In short, we also observe that the generative capability of DiT with SRA is
indeed strongly correlated with the representation guidance as observed in SiT with SRA.

Figure 9: We also investigate the correlation between generation performance and the representation
guidance of DiT + SRA. A similar tight coupling can also be seen.

K More Discussion on Related Work

We now provide a detailed literature review of other related work.

EMA model as teacher. Unlike traditional knowledge distillation [27] that uses a well pre-trained
model as the teacher, using EMA model [36] as the teacher can be seen as self-distillation because
the weight of the teacher model is obtained by the weighted moving average of the student. This
method often needs a feasible pretext task to succeed. For example, MoCo [30, 15] sets the teacher’s
output as a momentum queue and uses contrastive learning to guide the student model; DINO [9, 64]
feeds two views of images to the teacher, and the trainable student then forces the student’s output
distribution to be close to that of the teacher. Our work also shares some similarities, where we set
aligning the student model’s latent feature in the earlier layer conditioned on higher noise with that
in the later layer conditioned on lower noise of the teacher as our pretext task for training.

Diffusion transformers. Currently, the diffusion model is progressively transitioning from a U-Net-
based architecture to a Transformer-based one, owing to the superior scalability of the latter. At the
beginning, U-ViT [4] shows transformer-based backbones with skip connections can be an effective
backbone for training diffusion models. Then, DiT [65] shows skip connections are not even nec-
essary components, and a pure transformer architecture can be a scalable architecture for training
denoise-based models. Based on DiT, SiT [59] shows the model can be further improved with con-
tinuous stochastic interpolants [2]. Moreover, Stable diffusion 3 [23], Lumina-Image 2.0 [67] and
FLUX 1 [47] show pure transformers can be scaled up for challenging text-to-image generation, and
this characteristic is also verified by Sora [6], CogvideoX [85] and Wan [81] in text-to-video field.
Our work focuses on improving the training of DiT (and SiT) architecture based on our proposed
self-representation alignment technique.

Exploring representation capacity of diffusion models. With the success of diffusion models
to generate detailed images, many works have attempted to test whether discriminative semantic
information can be found in diffusion models. GD [61] and DDAE [83] first observe that the in-
termediate representations of diffusion models have discriminative properties. Driving from this
finding, I-DAE [14] deconstructs diffusion models to be a self-supervised Learner. Moreover, Rep-
fusion [84], DiffusionDet [12], and DreamTeacher [48] use diffusion models to perform various
downstream tasks (e.g., semantic segmentation and object detection). Our work also tries to explore
the representation capacity of the diffusion model, but we focus on leveraging the representations in
diffusion transformers to enhance their generation capacity.

22

Applying representation guidance to other generation tasks. In addition to pre-training of the
class-conditional diffusion model, applying representation guidance can benefit other generation
tasks. For example, lbGen [38] utilizes text features from CLIP [68] as a low-biased reference to
regulate diffusion model for low-biased dataset synthetics. RCG [49] focuses on unconditional gen-
eration, which first use the features from a self-supervised image encoder to serve as the ‘label’ for
image generation by a second generator. Diff-AE [66] uses a learnable encoder for discovering the
high-level semantics and a diffusion model for modeling stochastic; this dual-encoding improves the
realism of the generated images on attribute manipulation and image interpolation tasks. Different
from these works, our study focuses on the class-conditional diffusion transformer’s pre-training
and exploiting representation guidance in itself and its own training paradigm. But we also believe
our plug-and-play method can be easily applied to other tasks with benefits.

Knowledge distillation for diffusion model. Knowledge distillation [27] is also widely used in
diffusion models. Its purposes can roughly be divided into two types: improving generation perfor-
mance and accelerating inference sampling. To achieve the first goal, A more powerful pretrained
model is often used to guide the diffusion model during training [17, 87, 25]. For example, TinyFu-
sion [25] combines progressive distillation with architecture-specific optimizations for U-Net back-
bones, enabling deployment on edge devices. REPA [87] distills the knowledge from a large-scale,
pretrained representation foundation model [64, 68] and finds that this distillation can improve both
training efficiency and generation quality. Meanwhile, to achieve the second goal, the student and
teacher models will be initialized by pre-trained models (here it can be one model or two different
models), then the training target is to make the prediction of the student model with fewer steps
conform to that of the teacher model with multiple steps [58, 69, 60, 74, 37], thereby achieving
the result of accelerated sampling and generation speed. For example, CM [74] constrains the stu-
dent’s outputs of adjacent points on a sampling path to be consistent with teacher’s. LCM [58]
leverages this idea in the latent space. Our study also have some similarities, while we do not need
a pretrained model and improve the generation performance of diffusion transformers by applying
proposed SRA.

L More Qualitative Results (CFG 4.0)

Below we show some uncurated generation results on ImageNet 256×256 from the SiT-XL + SRA.
We use classifier-free guidance with w = 4.0.

23

Figure 10: Uncurated samples of loggerhead turtle (class label: 33).

Figure 11: Uncurated samples of sulphur-crested cockatoo (class label: 89).

24

Figure 12: Uncurated samples of golden retriever (class label: 207).

Figure 13: Uncurated samples of white fox (class label: 279).

25

Figure 14: Uncurated samples of tiger (class label: 292).

Figure 15: Uncurated samples of red panda (class label: 387).

26

Figure 16: Uncurated samples of acoustic guitar (class label: 402).

Figure 17: Uncurated samples of balloon (class label: 417).

27

Figure 18: Uncurated samples of baseball (class label: 429).

Figure 19: Uncurated samples of fire truck (class label: 555).

28

Figure 20: Uncurated samples of laptop (class label: 620).

Figure 21: Uncurated samples of ice cream (class label: 928).

29

Figure 22: Uncurated samples of cheeseburger (class label: 933).

Figure 23: Uncurated samples of cliff drop-off (class label: 972).

30

Figure 24: Uncurated samples of coral reef (class label: 973).

Figure 25: Uncurated samples of lakeside (class label: 975).

31

