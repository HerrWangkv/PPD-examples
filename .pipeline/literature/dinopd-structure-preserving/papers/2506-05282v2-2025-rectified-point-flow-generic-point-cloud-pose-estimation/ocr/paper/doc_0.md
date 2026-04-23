5
2
0
2

t
c
O
4
2

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
8
2
5
0
.
6
0
5
2
:
v
i
X
r
a

Rectified Point Flow:
Generic Point Cloud Pose Estimation

Tao Sun∗
Stanford University

Liyuan Zhu∗
Stanford University

Shengyu Huang
NVIDIA Research

Shuran Song
Stanford University

Iro Armeni
Stanford University

Abstract

We present Rectified Point Flow, a unified parameterization that formulates pair-
wise point cloud registration and multi-part shape assembly as a single conditional
generative problem. Given unposed point clouds, our method learns a continuous
point-wise velocity field that transports noisy points toward their target positions,
from which part poses are recovered. In contrast to prior work that regresses part-
wise poses with ad-hoc symmetry handling, our method intrinsically learns assem-
bly symmetries without symmetry labels. Together with an overlap-aware encoder
focused on inter-part contacts, Rectified Point Flow achieves a new state-of-the-art
performance on six benchmarks spanning pairwise registration and shape assembly.
Notably, our unified formulation enables effective joint training on diverse datasets,
facilitating the learning of shared geometric priors and consequently boosting accu-
racy. Our code and models are available at https:// rectified-pointflow.github.io/ .

1

Introduction

Estimating the relative poses of rigid parts from 3D point clouds for alignment is a core task in
computer vision and robotics, with applications spanning pairwise registration [1] and complex
multi-part shape assembly [2]. In many settings, the input consists of an unordered set of part-
level point clouds–without known correspondences, categories, or semantic labels–and the goal
is to infer a globally consistent configuration of poses, essentially solving a multi-part (two or
more) point cloud pose estimation problem. While conceptually simple, this problem is technically
challenging due to the combinatorial space of valid assemblies and the prevalence of symmetry and
part interchangeability in real-world shapes [3, 4, 5].

Despite sharing the goal of recovering 6-DoF transformations, different 3D reasoning tasks—such as
object pose estimation, part registration, and shape assembly—have historically evolved in silos, treat-
ing each part independently and relying on task-specific assumptions and architectures. For instance,
object pose estimators often assume known categories or textured markers [6, 7], while part assembly
algorithms may require access to a canonical target shape or manual part correspondences [8]. This
fragmentation has yielded solutions that perform well in narrow domains but fail to generalize across
tasks, object categories, or real-world ambiguities.

Among these tasks, multi-part shape assembly presents especially unique challenges. The problem is
inherently under constrained: parts are often symmetric [9], interchangeable [10], or geometrically
ambiguous, leading to multiple plausible local configurations. As a result, conventional part-wise
registration can produce flipped or misaligned configurations that are locally valid but globally
inconsistent with the intended assembly. Overcoming such ambiguities requires a model that can

∗Equal contribution.

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

 
 
 
 
 
 
reason jointly about part identity, relative placement, and overall shape coherence—without relying
on strong supervision or hand-engineered heuristics.

In this work, we revisit 3D pose regression and propose a generative approach for generic point cloud
pose estimation that casts the problem as learning a continuous point-wise flow field over the input
geometry, effectively capturing priors over assembled shapes. Our method, Rectified Point Flow,
models the motion of points from random Gaussian noise in Euclidean space toward the point clouds
of assembled objects. This learned flow implicitly encodes part-level transformations, enabling both
discriminative pose estimation and generative shape assembly within a single framework. Rectified
Point Flow consists of an encoder that extracts point-wise features, and a flow model that, given the
features, predicts the final assembled positions.

To instill geometric awareness of inter-part relationships, we pretrain the encoder on large-scale
3D shape datasets: predicting point-wise overlap across parts, formulated as a binary classification
task. While GARF [11] also highlights the value of encoder pretraining for a flow model, it relies
on mesh-based physical simulation [12] to generate fracture-based supervision signals. In contrast,
we introduce a lightweight and scalable alternative that constructs pretraining data by computing
geometric overlap between parts. Our data generation is agnostic to data sources tailored for different
tasks—including part segmentation [13, 14, 15], shape assembly [12, 16, 17], and registration [18,
19]—without requiring watertight mesh or simulation, an important step towards scalable pretraining
for pose estimation.

Our flow-based pose estimation departs from traditional pose-vector regression in three ways: (i)
Joint shape-pose reasoning: We cast the registration and assembly tasks as one unified task that
reconstructs the complete shape while simultaneously enabling the estimation of part poses; (ii)
Scalable shape prior learning: By training to predict the final assembled point cloud, our model
learns from heterogeneous datasets and part definitions, yielding scalable training and transferable
geometric knowledge across standard pairwise registration, fracture reassembly, and complex furniture
assembly tasks; and (iii) Intrinsic symmetry handling: Rather than regressing pose vectors directly
in SE(3) space, we operate in Euclidean space over dense point clouds. This makes the model
inherently robust to symmetries, part interchangeability, and spatial ambiguities that often challenge
conventional methods. Our main contributions are summarized as follows:

• We propose Rectified Point Flow, a generative approach for generic point cloud pose estimation
that addresses both pairwise registration and multi-part assembly tasks and achieves state-of-the-
art performances on all the tasks.

• We propose a generalizable pretraining strategy with geometric awareness of inter-part relation-

ships across several 3D shape datasets, and formulate it as point-wise overlap prediction.

• We show that our parameterization supports joint training across different registration tasks,

boosting the performance on each individual task.

2 Related Work

Parametrization for Pose Estimation. Euler angles and quaternions are the predominant
parametrization of rotation in various pose regression tasks [20, 21, 22, 23, 24, 25, 11, 26] due
to their simplicity and usability. As Euler angles and quaternions are discontinuous representations,
Zhou et al. [27] proposed to represent 3D rotation with a continuous representation for neural net-
works using 6D and 7D vectors. In contrast to directly regressing pose vectors, other methods train
networks to find sparse correspondences between image pairs or point cloud pairs and extract pose
vectors using Singular Value Decomposition (SVD) [28, 29, 8, 30, 31, 32]. More recently, RayDif-
fusion [33] proposed to represent camera poses as ray bundles, naturally suited for coupling image
features and transformer architectures. Huang et al. [34] adopted a point cloud generative model for
policy learning in robot pick-and-place tasks, then recovered relative poses between the object and
gripper via SVD. DUSt3R [35] directly regresses the pointmap of each camera in a global reference
frame and then extracts the camera pose using RANSAC-PnP [36, 37]. Our proposed Rectified Point
Flow, extends the dense point cloud or point map representation for learning generalizable pose
estimation on point cloud registration and shape assembly tasks.

Learning-based 3D Registration. 3D registration aims to align point cloud pairs in the same
reference frame by solving the relative transformation from source to target. The first line of work

2

Figure 1: Rectified Point Flow’s Pose-from-Shape Pipeline. Our formulation supports shape
assembly (first row) and pairwise registration (second row) tasks in a single framework. Given a set
of unposed part point clouds
Ω, Rectified Point Flow predicts each part’s point cloud at the
Ω. Subsequently, we solve Procrustes problem via SVD between
target assembled state
∈
the condition point cloud ¯Xi and the estimated point cloud ˆXi(0) to recover the rigid transformation
ˆTi for each non-anchored part.

¯Xi}i
{
ˆXi(0)
}i
{

∈

focuses on correspondence-based methods [38, 1, 39, 40] that first extract correspondences between
point clouds, followed by robust estimators to recover the transformation. Subsequent works [29,
41, 42, 32] advance the performance by learning more powerful features with improved architecture
and loss design. The second line of work comprises direct registration methods [31, 43, 30, 22] that
directly compute a score matrix and apply differentiable weighted SVD to solve for the transformation.
Correspondence-based methods can fail in extremely low-overlap scenarios in shape assembly and
direct methods fall short in terms of pose accuracy. Our method, which directly regresses the
coordinates of each point in the source point cloud, is agnostic and more generalizable to varying
overlap ratios compared to direct methods.

Multi-Part Registration and Assembly. Multi-part registration and shape assembly generalize
pairwise relative pose estimation to multiple parts, with applications in furniture assembly [17]
and shape reassembly [12]. Methods [44, 45, 26, 46, 47] tackle the multi-part registration problem
by estimating the transformation for each rigid part in the scene (multi-source and multi-target).
Multi-part shape assembly differs as a task from registration because it has multi-source input and
a canonical target, and each part has almost ‘zero’ overlap w.r.t. each other. Chen et al. [48] adopt
an adversarial learning scheme to examine the plausibility for different shape configurations. Wu
et al. [49] leverage SE(3) equivariant representation to handle pose variations in shape assembly.
DiffAssembly [50] and PuzzleFusion [24, 25] leverage diffusion models to predict the transformation
for each part. GARF [11] combines fracture-aware pretraining with a flow matching model to predict
per-part transformation. These methods, however, do not handle interchangeability and symmetry as
well as ours does. Moreover, Rectified Point Flow is the first solution for furniture assembly of 3D
shapes on the PartNet-Assembly [15] and IKEA-Manual [17] datasets.

3 Pose Estimation via Rectified Point Flow

Rectified Point Flow addresses the multi-part point cloud pose estimation problem, defined in Sec. 3.1.
The overall pipeline consists of two consecutive stages: overlap-aware point encoding (Sec. 3.2)
and conditional Rectified Point Flow (Sec. 3.3). Finally, we explain how our formulation inherently
addresses the challenges posed by symmetric and interchangeable parts in Sec. E.

3

ConditionPoint Clouds<latexit sha1_base64="xj6I7sAlbpURHEmGug9u+jzzxHE=">AAADOnicbZLNattAEMdX6leqfjntsZelxpBAa6QQ0h4NvfjWFOzEoBVitVrZS7SS2B2VmkXP05foK/Ta0mtupdc+QFe2k8hOBoRmf/Mf5mM3qXKhwfd/O+69+w8ePtp77D15+uz5i97+yzNd1orxKSvzUs0SqnkuCj4FATmfVYpTmeT8PLn42MbPv3ClRVlMYFnxSNJ5ITLBKFgU7zujQUgSaWZN7B/A4VtMWFqCtv81HFsYeYPQrM9WdSO5ZuPII8ZmUHWNBCZNbAQRBfkk+Zw23oDkPIOQZIoyQySFhZImba6KNx0GTaePu/XjbT1RYr4A2+gKJpn5aluwreN3eIv4h3YWsqBwM49lnWrd0NiGoiv9pImDHd2knRzHvb4/9FeGbzvBxumjjZ3GvUuSlqyWvACWU63DwK8gMlSBYDlvPFJrXlF2Qec8tG5BJdeRWV11gweWpDgrlf0KwCvazTBUar2UiVW2c+vdWAvvjCVypzJkHyIjiqoGXrB14azOMZS4fUc4FYozyJfWoUwJ2ztmC2rvCuxr8+xSgt0V3HbOjobByfDk83F/dLxZzx56jd6gAxSg92iExugUTRFzvjk/nJ/OL/e7e+n+cf+upa6zyXmFtsz99x9k0gpX</latexit>{¯Xi}i→!Per-Part SVD<latexit sha1_base64="cekwvzQgCp+8wzr3Ve6u2Hghp6Y=">AAADPXicbZJNa9swGMdl763z3tJtt13ESiCBLdhjdDsWevFtHSRtIDJGluVE1LKN9KQ0CH+hfYl9hV2742C3seuuU+xmc9I+IPTo9/zF8yIlVS40+P6V4965e+/+g72H3qPHT54+6+0/P9XlUjE+YWVeqmlCNc9FwScgIOfTSnEqk5yfJefH6/jZBVdalMUYVhWPJJ0XIhOMgkXxvnPcn5FEmmkd+wMYvsGEpSVou7cwtDDy+jPTnq3qv+QfCyOPGEwWFDZo4A9jgUkdG0FEQT5JPqe11yc5z2BGMkWZIZLCQkmT1pv8dYdB3Snldn24rSdKzBdga21gkpnLOha2evwWbxF/aNvp1mob97uNd0OhDUUb/biOgx3deN08jnsH/shvDN90gmvn4Ohl1thJ3PtB0pItJS+A5VTrWeBXEBmqQLCc1x5Zal5Rdk7nfGbdgkquI9O8do37lqQ4K5VdBeCGdm8YKrVeycQq133r3dga3hpL5E5myD5GRhTVEnjB2sTZMsdQ4vVXwqlQnEG+sg5lStjaMVtQ+1ZgP5xnhxLsjuCmc/puFByODj/b6bxHre2hV+g1GqAAfUBHKEQnaIKY88X55lw5392v7k/3l/u7lbrO9Z0XaMvcP38BWocNsg==</latexit>{ˆX(0)i}i→!<latexit sha1_base64="Nzrr+hWJcPCjQsE7fk/YPg/IyaA=">AAADN3icbZLPa9swFMdl71fn/Uq34y5iIZDAFuxSul4GhV1yWwdJG4iMkRU5EbVsIz2PBqH/Zv/E/oVdt8tOgx3GrvsPpiTumqR9YPz0ed/H+yGlVS40hOEPz79z9979B3sPg0ePnzx91tp/fqbLWjE+YmVeqnFKNc9FwUcgIOfjSnEq05yfpxfvl/HzT1xpURZDWFQ8lnRWiEwwCg4l+967zoSk0oxtEnah9xoTNi1Bu/8aDhyMg87ErM9OdS35zwZxQEyT0Y16icDEJkYQUZAPks+oDTok5xlMSKYoM0RSmCtppvaqst1gYDeauF0/2NYTJWZzcF2uYJqZS5sI1zd+g7dI2HODkDmF62Ec26i2GRq4UHylH9ok2tENl2PjpNUO++HK8E0napw2auw0af0i05LVkhfAcqr1JAoriA1VIFjObUBqzSvKLuiMT5xbUMl1bFb3bHHHkSnOSuW+AvCKbmYYKrVeyNQpl3Pr3dgS3hpL5U5lyI5jI4qqBl6wdeGszjGUePmI8FQoziBfOIcyJVzvmM2puytwTy1wS4l2V3DTOTvoR0f9o4+H7ZPDZj176CV6hbooQm/RCRqgUzRCzPvsffW+ed/9L/5P/7f/Zy31vSbnBdoy/+8/ioAIMg==</latexit>{X(1)i}i→!Noised Point CloudsEstimatedPoint Clouds<latexit sha1_base64="aa3enPU7gaEB+YPZGyTGTBdLInE=">AAADUXicbVJNb9NAEF0nfJQUaArcuKyoIrUSRDZCLcdKXHyjSEkTKWtZ6806WdVrW7sT1Gjlv8af4MQVcYU7N8ZOG5K0I1k7fu+t5s3sJGWmLPj+D6/VfvDw0eO9J539p8+eH3QPX1zaYmGEHIoiK8w44VZmKpdDUJDJcWkk10kmR8nVp5offZXGqiIfwLKUkeazXKVKcEAoPvRGvQlLtBtXsX8MJ28pE9MCLJ4rMEQw6vQmbvWPqv+SNRaigjnK5hzcoIoVZVXsFFM5+6zljFcd5tbcLtVjmUxhwlLDhWOaw9xoN61uPVUbGFQb9u7Xh9t6ZtRsDrW7GkxSd40OsCP6jm4h/gm22Fhct4nYRrVNKkQqutVjS8GOblAPhMbdI7/vN0HvJsFNcnT+Km3iIu7+ZNNCLLTMQWTc2knglxA5bkCJTOIMF1aWXFzxmZxgmnMtbeSaDahoD5EpTQuDXw60QTdvOK6tXeoElXXfdperwXu5RO9UhvRj5FReLkDmYlU4XWQUClqvF50qIwVkS0y4MAq9UzHn+FaAS9jBoQS7I7ibXL7vB6f90y84nQ9kFXvkNXlDjklAzsg5CckFGRLhffN+eb+9P63vrb9t0m6tpC3v5s5LshXt/X/8RROZ</latexit>{ˆTi}i→!RecoveredPosesEstimatedPartCondition PartCondition PartEstimatedPartAnchoredAnchoredRectified Point FlowPer-Part SVDShape AssemblyPairwise RegistrationFigure 2: Encoder pre-training via overlap points prediction. Given unposed multi-part point
clouds, our encoder with a point-wise overlap prediction head performs a binary classification to
identify overlapping points. Predicted overlap points are shown in blue. For comparison, the ground-
truth overlap points are visualized on the assembled object for clarity (target overlap).

3.1 Problem Definition

Ω, where Ω is the
Consider a set of unposed point clouds of multiple object parts,
}i
is the number of parts, and Ni is the number of points in part i. The goal
part index set, H :=
Ω
|
is to solve for a set of rigid transformations
Ω that align each part in the unposed
Ti ∈
multi-part point cloud X to form a single, assembled object Y in a global coordinate frame, where

Xi ∈

SE(3)

}i

{

{

×

∈

∈

|

R3

Ni

X :=

R3

N , Y :=

×

Xi ∈

TiXi ∈

R3

N ,

×

and N :=

Ni.

(1)

Ω
(cid:91)i
∈

Ω
(cid:91)i
∈

Ω
(cid:88)i
∈

To eliminate global translation and rotation ambiguity, we set the first part (i = 0) as the anchor and
define its coordinate frame as the global frame. All other parts are registered to this anchor.

3.2 Overlap-aware Point Encoding

Pose estimation relies on geometric cues from mutually overlapping regions among connected
parts [29, 32, 11]. In our work, we address this challenge through a pretraining module that develops
a task-agnostic, overlap-aware encoder capable of producing pose-invariant point features. As
illustrated in Fig. 2, we train an encoder F to identify overlapping points in different parts. Given
SE(3) and compose
a set of unposed parts
transformed point clouds ˜Xi = ˜TiXi as input to the encoder. These data augmentations enable the
encoder to learn more robust pose-invariant features. The encoder then computes per-point features
Rd for the j-th point on part i, after which an MLP overlap prediction head estimates the
Ci,j ∈
overlap probability ˆpi,j. The binary ground-truth label pi,j is 1 if point ˜xi,j falls within radius ϵ of at
least one point in other parts.

Ω, we first apply random rigid transforms ˜Ti ∈

Xi}i

{

∈

We train both the encoder and the overlap head using binary cross-entropy loss. For objects without
predefined part segmentation, we employ off-the-shelf 3D part segmentation methods to generate the
necessary labels. The features extracted by our trained encoder subsequently serve as conditioning
input for our Rectified Point Flow model.

3.3 Generative Modeling for Pose Estimation

The overlap-aware encoder identifies potential overlap regions between parts but cannot determine
their final alignment, particularly in symmetric objects that allow multiple valid assembly configu-
rations. To address this limitation, we formulate the point cloud pose estimation as a conditional
generation task. With this approach, Rectified Point Flow leverages the extracted point features to
sample from the conditional distribution of all feasible assembled states across multi-part point clouds,
generating estimates that maximize the likelihood of the conditional input point cloud. By recasting
pose estimation as a generative problem, we naturally accommodate the inherent ambiguities arising
from symmetry and part interchangeability in the data.

Preliminaries. Rectified Flow (RF) [51, 52] is a score-free generative modeling framework that
learns to transform a sample X(0) from a source distribution, into X(1) from a target distribution.
The forward process is defined as linear interpolation between them with a timestep t as

X(t) = (1

−

t)X(0) + tX(1),

t

[0, 1].

∈

(2)

4

Overlap HeadFeaturesMulti-part Point CloudsOverlap Points Prediction….Overlap Points TargetEncoderConditionPoint Clouds<latexit sha1_base64="xj6I7sAlbpURHEmGug9u+jzzxHE=">AAADOnicbZLNattAEMdX6leqfjntsZelxpBAa6QQ0h4NvfjWFOzEoBVitVrZS7SS2B2VmkXP05foK/Ta0mtupdc+QFe2k8hOBoRmf/Mf5mM3qXKhwfd/O+69+w8ePtp77D15+uz5i97+yzNd1orxKSvzUs0SqnkuCj4FATmfVYpTmeT8PLn42MbPv3ClRVlMYFnxSNJ5ITLBKFgU7zujQUgSaWZN7B/A4VtMWFqCtv81HFsYeYPQrM9WdSO5ZuPII8ZmUHWNBCZNbAQRBfkk+Zw23oDkPIOQZIoyQySFhZImba6KNx0GTaePu/XjbT1RYr4A2+gKJpn5aluwreN3eIv4h3YWsqBwM49lnWrd0NiGoiv9pImDHd2knRzHvb4/9FeGbzvBxumjjZ3GvUuSlqyWvACWU63DwK8gMlSBYDlvPFJrXlF2Qec8tG5BJdeRWV11gweWpDgrlf0KwCvazTBUar2UiVW2c+vdWAvvjCVypzJkHyIjiqoGXrB14azOMZS4fUc4FYozyJfWoUwJ2ztmC2rvCuxr8+xSgt0V3HbOjobByfDk83F/dLxZzx56jd6gAxSg92iExugUTRFzvjk/nJ/OL/e7e+n+cf+upa6zyXmFtsz99x9k0gpX</latexit>{¯Xi}i→!Noised Point Clouds<latexit sha1_base64="nyiRznOjr29uLxEohrfp8R5X8DI=">AAADbXicbVLbattAEF3LvSTuzenlqaUsDaYJpEYqJe1joC9+KDSFODF4hVitV/aSXUnsjkvMog9tH/sD/YWOZCe1nQwIjc45uzNnNGmplYMw/NUK2vfuP3i4s9t59PjJ02fdvefnrphbIYei0IUdpdxJrXI5BAVajkoruUm1vEgvv9b8xU9pnSryM1iUMjZ8mqtMCQ4IJXst1xuz1PhRlYQHcHhEmZgU4PC9BAcIxp3e2C+/UfVfcoMNUME8ZTMO/qxKFGVV4hVTOftu5JRXHbaSJgqv2yZ7TMsMxiyzXHhmOMys8ZPquqtqDYNqrcG79YNNPbNqOoO6vxpMM3+1bIJ+oBtIeIgmGwM3RhFbq7ZODZCKr/VoONrSndUjoauagmv/DS0zkFfgZS7wgu5+2A+boLeTaJXsn7zMmjhNun/YpBBzI3MQmjs3jsISYs8tKKElznfuZMnFJZ/KMaY5N9LFvtmOivYQmdCssPjkQBt0/YTnxrmFSVFZ9+u2uRq8k0vNVmXIvsRe5eUcaptN4WyuKRS0Xj06UVYK0AtMuLAKe6dixvEvAi5oB4cSbY/gdnL+sR8d949/4HQ+kWXskNfkHTkgEflMTsiAnJIhEa3fAQl2g07wt/2q/ab9dikNWqszL8hGtN//A0BxG18=</latexit>{Xi(t)}i→!Flow ModelCEncoderPositionalEncoding<latexit sha1_base64="osqNiA0SU/M+puiYsDC9PMkuj8I=">AAACqnicbZFfq9MwGMbTqsdZPceqF154ExyDHTiM9jCmN8JA0F1OcX+kLSNN0y2cpC3JW2GUfig/jrd+EtNtQvfnhZCH3/skb3gSF4Jr8Lw/lv3o8ZOrp51nzvMX1zcv3Vev5zovFWUzmotcLWOimeAZmwEHwZaFYkTGgi3ih89Nf/GLKc3z7AdsCxZJss54yikBg1bu714QxrJa1iuvD7d3OKRJDtrsezgxMHIaD1HVnhln29bik8gJBUshCFNFaBVKAhslq6T+P6FuMahbt1z2T479oeLrDUQrt+sNvF3hc+EfRBcdarpy/4ZJTkvJMqCCaB34XgFRRRRwKljthKVmBaEPZM0CIzMimY6qXbQ17hmS4DRXZmWAd7R9oiJS662MjbN5qj7tNfBiL5YnkyH9GFU8K0pgGd0PTkuBIcfNv+GEK0ZBbI0gVHHzdkw3xOQG5ncdE4p/GsG5mN8P/NFg9G3YHQ8P8XTQO/Qe9ZGPPqAxmqApmiFqvbU+WV+sr/ad/d3+aQd7q20dzrxBR2Un/wBzNNRX</latexit>!dX0dt,···,dXHdt"NoisedPointsTarget PointCloudsVelocity Field˜Xi}i

Figure 3: Learning Rectified Point Flow. The input to Rectified Point Flow are the condition point
clouds
Ω at timestep t. They are first encoded by the
}i
pre-trained encoder and the positional encoding, respectively. The encoded features are concatenated
Ω and
and passed through the flow model, which predicts per-point velocity vectors
defines the flow used to predict the part point cloud in its assembled state.

Ω and noised point clouds
∈

dXi(t)/dt

Xi(t)

}i

{

{

{

∈

∈

The reverse process is modeled as a velocity field
V (t, X(t)

X) conditioned on X and trained using conditional flow matching (CFM) loss [53],

∇tX(t), which is parameterized as a network

|

LCFM(V ) = Et,X

∥
(cid:104)

V (t, X(t)

X)

|

− ∇tX(t)

2
∥

.

(cid:105)

(3)

R3

Rectified Point Flow.
In our method, we directly apply RF to the 3D Euclidean coordinates of the
Mi denote the time-dependent point cloud for part i, where
multi-part point clouds. Let Xi(t)
Mi is number of sampled points. At t = 0,
Ω is uniformly sampled from the assembled
Xi(0)
object Y , while at t = 1, points on each part are independently sampled from a Gaussian, i.e.,
(0, I). Then, we define the continuous flow for each part as straight-line interpolation in
Xi(1)
3D Euclidean space between the points in noised and assembled states. Specifically, for each part i,
(4)
t)Xi(0) + tXi(1),

Xi(t) = (1

[0, 1].

∼ N

}i

∈

{

×

∈

t

The velocity field of Rectified Point Flow is therefore,

−

∈

dXi(t)
dt

= Xi(1)

Xi(0).

−

(5)

We fix the anchored part (i = 0) by setting X0(t) = X0(0) for all t
[0, 1], implemented via a
mask that zeros out the velocity for its points. Once the model predicts the assembled point cloud of
each part ˆXi(0), we recover its pose Ti in a Procrustes problem,
ˆXi(0)

(6)

∈

ˆTi = arg min
ˆTi

SE(3) ∥

ˆTiXi −

∈

∥F .

Solving poses ˆTi for all non-anchored parts via SVD completes the pose estimation task in Eq. 1.

{

Xi(t)

Learning Objective. We train a flow model V to recover the velocity field in Eq. 5, taking the
noised point clouds
Ω and conditioning on unposed multi-part point cloud X, as shown
∈
in Fig. 3. First, we encode X using the pre-trained encoder F . For each noised point cloud, we
apply a positional encoding to its 3D coordinates and part index, concatenate these embeddings with
the point features, and feed the result into the flow model. We denote its predicted velocity field by
M . We optimize the flow model V by
the flow model for all points by V (t,
Ω; X)
∈
minimizing the conditional flow matching loss in Eq. 3.

Xi(t)

R3

}i

}i

{

×

∈

3.4

Invariance Under Rotational Symmetry and Interchangeability

In our method, the straight-line point flow and point-cloud sampling, while simple, guarantee that
every flow realization and its loss in Eq. (3) remain invariant under an assembly symmetry group
:
G
, we have the learning
Theorem 1 (
G
objective in Eq. 3 following

-invariance of the learning objective). For every element g

The formal definition of
G
result, the flow model learns all the symmetries in
hand-made data augmentation or heuristics on symmetry and interchangeability.

LCFM(V ) =
and the proof of Theorem 1 appear in the supplementary material. As a
during training, without the need for additional

LCFM(g(V (t,

Ω; g(X)))).

Xi(t)

∈ G

}i

G

{

∈

5

Multi-PartPoint Clouds<latexit sha1_base64="xj6I7sAlbpURHEmGug9u+jzzxHE=">AAADOnicbZLNattAEMdX6leqfjntsZelxpBAa6QQ0h4NvfjWFOzEoBVitVrZS7SS2B2VmkXP05foK/Ta0mtupdc+QFe2k8hOBoRmf/Mf5mM3qXKhwfd/O+69+w8ePtp77D15+uz5i97+yzNd1orxKSvzUs0SqnkuCj4FATmfVYpTmeT8PLn42MbPv3ClRVlMYFnxSNJ5ITLBKFgU7zujQUgSaWZN7B/A4VtMWFqCtv81HFsYeYPQrM9WdSO5ZuPII8ZmUHWNBCZNbAQRBfkk+Zw23oDkPIOQZIoyQySFhZImba6KNx0GTaePu/XjbT1RYr4A2+gKJpn5aluwreN3eIv4h3YWsqBwM49lnWrd0NiGoiv9pImDHd2knRzHvb4/9FeGbzvBxumjjZ3GvUuSlqyWvACWU63DwK8gMlSBYDlvPFJrXlF2Qec8tG5BJdeRWV11gweWpDgrlf0KwCvazTBUar2UiVW2c+vdWAvvjCVypzJkHyIjiqoGXrB14azOMZS4fUc4FYozyJfWoUwJ2ztmC2rvCuxr8+xSgt0V3HbOjobByfDk83F/dLxZzx56jd6gAxSg92iExugUTRFzvjk/nJ/OL/e7e+n+cf+upa6zyXmFtsz99x9k0gpX</latexit>{¯Xi}i→!Noised Point Clouds<latexit sha1_base64="nyiRznOjr29uLxEohrfp8R5X8DI=">AAADbXicbVLbattAEF3LvSTuzenlqaUsDaYJpEYqJe1joC9+KDSFODF4hVitV/aSXUnsjkvMog9tH/sD/YWOZCe1nQwIjc45uzNnNGmplYMw/NUK2vfuP3i4s9t59PjJ02fdvefnrphbIYei0IUdpdxJrXI5BAVajkoruUm1vEgvv9b8xU9pnSryM1iUMjZ8mqtMCQ4IJXst1xuz1PhRlYQHcHhEmZgU4PC9BAcIxp3e2C+/UfVfcoMNUME8ZTMO/qxKFGVV4hVTOftu5JRXHbaSJgqv2yZ7TMsMxiyzXHhmOMys8ZPquqtqDYNqrcG79YNNPbNqOoO6vxpMM3+1bIJ+oBtIeIgmGwM3RhFbq7ZODZCKr/VoONrSndUjoauagmv/DS0zkFfgZS7wgu5+2A+boLeTaJXsn7zMmjhNun/YpBBzI3MQmjs3jsISYs8tKKElznfuZMnFJZ/KMaY5N9LFvtmOivYQmdCssPjkQBt0/YTnxrmFSVFZ9+u2uRq8k0vNVmXIvsRe5eUcaptN4WyuKRS0Xj06UVYK0AtMuLAKe6dixvEvAi5oB4cSbY/gdnL+sR8d949/4HQ+kWXskNfkHTkgEflMTsiAnJIhEa3fAQl2g07wt/2q/ab9dikNWqszL8hGtN//A0BxG18=</latexit>{Xi(t)}i→!Flow ModelCEncoderPositionalEncoding<latexit sha1_base64="osqNiA0SU/M+puiYsDC9PMkuj8I=">AAACqnicbZFfq9MwGMbTqsdZPceqF154ExyDHTiM9jCmN8JA0F1OcX+kLSNN0y2cpC3JW2GUfig/jrd+EtNtQvfnhZCH3/skb3gSF4Jr8Lw/lv3o8ZOrp51nzvMX1zcv3Vev5zovFWUzmotcLWOimeAZmwEHwZaFYkTGgi3ih89Nf/GLKc3z7AdsCxZJss54yikBg1bu714QxrJa1iuvD7d3OKRJDtrsezgxMHIaD1HVnhln29bik8gJBUshCFNFaBVKAhslq6T+P6FuMahbt1z2T479oeLrDUQrt+sNvF3hc+EfRBcdarpy/4ZJTkvJMqCCaB34XgFRRRRwKljthKVmBaEPZM0CIzMimY6qXbQ17hmS4DRXZmWAd7R9oiJS662MjbN5qj7tNfBiL5YnkyH9GFU8K0pgGd0PTkuBIcfNv+GEK0ZBbI0gVHHzdkw3xOQG5ncdE4p/GsG5mN8P/NFg9G3YHQ8P8XTQO/Qe9ZGPPqAxmqApmiFqvbU+WV+sr/ad/d3+aQd7q20dzrxBR2Un/wBzNNRX</latexit>!dX0dt,···,dXHdt"NoisedPointsAssembled StateVelocity Field4 Experiments

Implementation Details. We use PointTransformerV3 (PTv3) [54] as the backbone for point cloud
encoder, and use Diffusion Transformer (DiT) [55] as our flow model. Each DiT layer applies two
self-attention stages: (i) part-wise attention to consolidate part-awareness, and (ii) global attention
over all part tokens to fuse information. We stabilize the attention computation by applying RMS
Normalization [56, 57] to the query and key vectors per head before attention operations. We sample
the time steps from a U-shaped distribution following [58]. We pre-train the PTv3 encoder on all
datasets with an additional subset of Objaverse [14] meshes, where we apply PartField [13] to obtain
annotations. After pretraining, we freeze the weights of the encoder. We train our flow model on
8 NVIDIA A100 80GB GPUs for 400k iterations with an effective batch size of 256. We use the
4 which is halved every 25k iterations
AdamW [59] optimizer with an initial learning rate 5
after the first 275k iterations.

10−

×

Table 1: Dataset statistics. We train our flow model on six datasets with varying sizes, part definitions,
and complexities. The encoder is pre-trained on these datasets with an extra Objaverse dataset.

Dataset

Task

Part Definition

Train & Val

Test

# Samples

# Parts

# Samples

# Parts

IKEA-Manual [17] Assembly
Assembly
TwoByTwo [16]
Assembly
PartNet-Assembly
Assembly
BreakingBad [12]
Registration
TUD-L [18]
Registration
ModelNet-40 [19]

Reusability and packing
Insertable parts
Semantics and functions
Fracture simulation
RGB-D sensor scans
Random partition

Objaverse 1.0 [14]

Pre-training

From PartField [13]

84
308
23755
35114
19138
19680

63199

[2, 19]
[2, 2]
[2, 64]
[2, 49]
[2, 2]
[2, 2]

[3, 12]

18
144
261
265
300
260

6794

[2, 19]
[2, 2]
[2, 64]
[2, 49]
[2, 2]
[2, 2]

[3, 12]

4.1 Experimental Setting

Datasets. For the multi-part shape assembly task, we experiment on the BreakingBad [12],
TwoByTwo [16], PartNet [15], and IKEA-Manual [17] datasets. The PartNet dataset has been
processed for the shape assembly task following the same procedure as [17] but includes all object
categories; we refer to this version as PartNet-Assembly. Evaluation of the pairwise registration is
performed on the TUD-L [18] and ModelNet-40 [19] datasets. We follow [22] for prepossessing
the TUD-L dataset. We split all datasets into train/val/test sets following existing literature for fair
comparisons. These datasets define parts at distinct levels, ranging from random partitions (e.g.,
ModelNet-40 and BreakingBad) to human-labeled (e.g., semantically meaningful parts in PartNet
and IKEA-Manual). The statistics and information of all datasets are summarized in Tab. 1.

Evaluation Protocols. We evaluate the pose accuracy following the convention of each benchmark,
with Rotation Error (RE), Translation Error (TE), Rotation Recall at 5◦ (Recall @ 5◦), and Translation
Recall at 1 cm (Recall @ 1 cm). For the shape assembly task, we measure Part Accuracy (Part Acc)
by computing per object the fraction of parts with Chamfer Distance under 1 cm, and then averaging
those per-object scores across the dataset, following [25, 11, 30, 17].

Following [11], we select the largest-volume part as the anchor and fix it during inference. However,
this effectively provides the model with anchor pose in the object’s CoM (center of mass) frame, an
unrealistic assumption for real-world assembly applications. Therefore, we also train our model in an
anchor-free setting (see Appendix B: Anchor-free Models), and argue that anchor-free evaluation
should be the standard protocol for shape assembly tasks.

Baseline Methods. We evaluated our method against state-of-the-art methods for pairwise registra-
tion and shape assembly. For pairwise registration, we compare against DCPNet [31], RPMNet [30],
GeoTransformer [32], and Diff-RPMNet [22]. For shape assembly, we compare against MSN [48],
SE(3)-Assembly [49], Jigsaw [60], PuzzleFussion++ [25], and GARF [11]. We report our per-
formances under two training configurations: dataset-specific training where models are trained
independently for each dataset (denoted Ours (Single)), and joint training where a single model is
trained across all datasets (denoted Ours (Joint)).

6

Figure 4: Qualitative Results on PartNet-Assembly. Columns show objects with increasing number
of parts (left to right). Rows display (1) colored input point clouds of each part, (2) GARF outputs
(dashed boxes indicate samples limited to 20 by GARF’s design, selecting the top 20 parts by volume),
(3) Rectified Point Flow outputs, and (4) ground-truth assemblies. Compared to GARF, our method
produces more accurate pose estimation on most parts, especially as the number of parts increases.

Table 2: Multi-Part Assembly Results. Rectified Point Flow (Ours) achieves the best performance
across all metrics on BreakingBad-Everyday, TwoByTwo, and PartNet-Assembly datasets.

Methods

BreakingBad-Everyday [12]
Part Acc ↑
RE ↓
[%]
[deg]

TE ↓
[cm]

TwoByTwo [16]
TE ↓
RE ↓
[cm]
[deg]

MSN [48]
SE(3)-Assembly [49]
Jigsaw [60]
PuzzleFusion++ [25]
GARF [11]

Ours (Single)
Ours (Joint)

85.6
73.3
42.3
38.1
9.9

9.6
7.4

15.7
14.8
10.7
8.0
2.0

1.8
2.0

16.0
27.5
68.9
76.2
93.0

93.5
91.1

70.3
52.3
53.3
58.2
22.1

18.7
13.2

28.4
23.3
36.0
34.2
7.1

4.1
3.0

PartNet-Assembly

RE ↓
[deg]

–
–
–
–
66.9

24.8
21.8

TE ↓
[cm]

–
–
–
–
21.9

15.4
14.8

Part Acc ↑
[%]

–
–
–
–
25.7

50.2
53.9

4.2 Evaluation

We report pose accuracy for shape assembly and pairwise registration in Tab. 2 2 and Tab. 3, re-
spectively. Our model outperforms all existing approaches by a substantial margin. For multi-part
assembly, the closest competitor is GARF [11], which formulates per-part pose estimation as 6-DoF
pose regression; see Figs. 4 and 5. We attribute our superior results to two key advantages of Rectified
Point Flow: (i) in contrast to our closest competitor GARF [11] which performs 6-DoF pose regres-
sion, our dense shape-and-pose parametrization helps the model learn better global shape prior and
fine-grained geometric details more effectively; and (ii) our generative formulation natively handles

2We found that the BreakingBad benchmark [12, 11, 25] originally computed rotation error (RE) using
the RMSE of Euler angles, which is not a proper metric on SO(3). To ensure consistency, we re-evaluate all
baselines using the geodesic distance between rotation matrices via the Rodrigues formula [61, 31, 29, 32]. For
Ours (Single) on TwoByTwo, we used encoder pretrained in the Ours (Joint) setting but trained the flow model
only on TwoByTwo, due to its limited size.

7

GARFOursInputGT5 parts7 parts8 parts12 parts24 parts48 partsFigure 5: Qualitative Results Across Registration and Assembly Tasks. From left to right: pairwise
registration on ModelNet 40 and TUD-L, shape assembly on BreakingBad-Everyday. From top to
bottom: Colored input point clouds, GARF results, ours, and ground truth (GT). Our single model
performs the best across registration and assembly tasks.

Table 3: Pairwise Registration Results. Rectified Point Flow (Ours) outperforms all baselines on
both TUD-L and ModelNet 40, achieving the highest accuracy and lowest errors across all metrics.

Methods

DCPNet [31]
RPMNet [30]
GeoTransformer [32]
GARF [11]
Diff-RPMNet [22]

Ours (Single)
Ours (Joint)

TUD-L [18]

Recall @5° ↑
[%]

Recall @1cm ↑
[%]

ModelNet 40 [19]
TE ↓
RE ↓
[unit]
[deg]

23.0
73.0
88.0
53.1
90.0

97.0
97.7

4.0
89.0
97.5
52.5
98.0

98.7
99.0

11.98
1.71
1.58
42.5
–

1.37
0.93

0.171
0.018
0.018
0.063
–

0.003
0.002

part symmetries and interchangeability. For pairwise registration, GARF–despite being retrained on
target datasets–fails to generalize beyond the original task. In contrast, our method achieves a new
state-of-the-art performance on registration benchmarks, outperforming methods designed solely for
registration (e.g., GeoTransformer [32] and Diff-RPMNet [22]) and demonstrating strong generaliza-
tion across different datasets (Fig 5). We also achieve the strongest shape assembly performance on
IKEA-Manual [17]; for more details on evaluation and visualizations, see supplementary.

Joint Training. By recasting pairwise registration as a two-part assembly task, our unified for-
mulation enables joint training of the flow model on all six datasets—including very small sets like
TwoByTwo (308 samples) and IKEA-Manual (84 samples)—and the additional pretraining data from
Objaverse. Ours (Joint) consistently matches or outperforms the dataset-specific (Ours (Single))
models. For example, on TwoByTwo the rotation error drops from 18.7◦ to 13.2◦ (
30%), and on
23%), while on ModelNet-40, the rotation error is reduced from
BreakingBad from 9.6◦ to 7.4◦ (
1.37° to 0.93°. These results demonstrate that joint training enables the model to learn shared geomet-
ric priors from datasets with diverse part segmentation, symmetries, and common pose distributions,
which substantially boosts performance, particularly on datasets with limited training samples.

≈

≈

8

GARFOursInputGTModelNet 40TUD-LBreakingBad-EverydaySymmetry Handling. We demonstrate our model’s ability to handle symmetry (Sec. E) on IKEA-
Manual [17], a dataset with symmetric assembly configurations. As shown in Fig. 6, while being only
trained on a single configuration (left), Rectified Point Flow samples various reasonable assembly
configurations (right), conditioned on the same input unposed point clouds. Note how our model
permutes the 12 repetitive vertical columns and swaps the 2 middle baskets, yet always retains the
non-interchangeable top and footed bottom baskets in their unique positions.

Figure 6: Learning from a symmetric assembly. Left
to right:
(1) a single training sample from IKEA-
Manual [17], and (2–4) three independent samples gen-
erated, conditioned on the same inputs. Parts are color-
coded consistently across plots. (Best viewed in color.)

Figure 7: Two common failure types.
First column: Assemblies that are geomet-
rically plausible but mechanically non-
functional. Second column: Objects with
high geometric complexity.

Table 4: Ablation on Encoder Pre-training. We ablate the impact of different pre-training tasks on
the shape assembly performance. Our overlap detection pre-training yields the best results.

Dataset

Encoder

Pre-training Task

BreakingBad-Everyday [12]

MLP
PTv3 [54]
PTv3 [54]
PTv3 [54]

No Pre-training
No Pre-training
Instance Segmentation
Overlap Detection (ours)

PartNet-Assembly [15]

Point-BERT [62]
PTv3 [54]

Point Cloud Completion
Overlap Detection (ours)

RE ↓
[deg]

41.7
18.5
16.7
9.6

27.4
24.8

TE ↓
[cm]

12.3
4.9
4.4
1.8

23.3
15.4

Part Acc ↑
[%]

68.3
79.5
80.9
93.5

45.2
50.2

Ablation on Overlap-aware Pretraining. The first block of Tab. 4 compares four pretraining
strategies for our flow-based assembly model on BreakingBad–Everyday [12]. The first two encoders
(MLP and PTv3 without pre-training) are trained jointly with the flow model. The last two encoders
are PTv3 pretrained on instance segmentation and our overlap-aware prediction tasks, respectively.
Their pretrained weights are frozen during flow model training. We find that PTv3 is a more powerful
feature encoder compared to the MLP, and pretraining on instance segmentation can already extract
useful features for pose estimation, while our proposed overlap-aware pretraining leads to the best
accuracy. We hypothesize that, although the segmentation backbone provides strong semantic
features, only our overlap prediction task explicitly encourages the encoder to learn fine-grained part
interactions and pre-assembly cues, critical for precise assembly and registration.

To further compare against autoencoder-based pretraining, we substitute our encoder with Point-
BERT [62]’s encoder, which is pre-trained on ShapeNet [63], a superset of PartNet. We then train
the flow model on PartNet under the same protocol. As reported in Tab. 4, Point-BERT encoder
reduces the Part Accuracy from 50.2% to 45.2%, and worsens the accuracy in both TE and RE. We
attribute this performance drop to: (i) Point-BERT is pre-trained for masked point cloud completion,
which does not explicitly encourage the encoder to capture inter-part relations (e.g., contacts) that our
overlap-aware pretraining targets; and (ii) PointBERT’s default 64-group tokenization aggregates
points into relatively coarse groups, losing fine-grained geometric details for accurate pose estimation.

9

GeneratedResultsTrainingExamplePredictionGTShape Prior Learning. To probe whether our model learns
the shape priors of assembled objects better than pose-vector
methods, we construct a cylindrical toy dataset. We train using
a single part partition scheme and then evaluate on the same
cylinder shapes but with different partition schemes.

Table 5: Part Accuracy [%] on Test-
ing Part Schemes. Our model shows
much less drop on OOD partitions.

Partition Scheme GARF Ours

Specifically, we generated 6,000 training cylinders with
heights and diameters uniformly sampled from [0.2, 1.0] m,
each cut into two parts by a horizontal plane at a random height.
For testing, 600 new cylinders were cut under three schemes:
(i) Horizontal (in-distribution, ID), (ii) Axial: through the cen-
tral axis at a random orientation (out-of-distribution, OOD), and (iii) Random: through random 3D
plane (OOD). As shown in Tab. 5, our method achieved part accuracies of 100.0%, 97.0%, and
97.8% on three partition schemes, respectively. While GARF has comparable performance on the ID
scheme, it degrades on two OOD schemes, verifying that our model learns a transferable shape prior
over the whole assembled object rather than overfitting to the part partition-specific patterns.

Horizontal (ID)
Axial (OOD)
Random (OOD)

100.0
97.0
97.8

99.5
89.5
87.5

98.3

92.1

All

Table 6: Zero-shot Evaluation on the Unseen FRACTURA Testset. We report Part Accuracy (%)
for GARF and ours and include the supervised performance for reference.

Setting

Method

Supervised

Zero-shot

GARF

GARF
Ours

Leg

89.7

70.5
79.9

Hip

80.8

72.8
63.4

Rib

74.8

62.9
76.2

Vertebra

Pig Bones

60.8

37.7
42.0

79.0

53.4
63.2

All

77.3

57.7
64.4

Generalization to Unseen Dataset. We evaluated the out-of-domain generalization of our model
by performing zero-shot tests on unseen bone fracture on the test split of the FRACTURA dataset [11],
covering human bones (Leg, Hip, Vertebra, Rib) and pig bones. We report Part Accuracy for our
method and GARF [11] in Tab. 6. Our model achieves strong zero-shot performance, surpassing
GARF in most categories, especially Leg and Rib, and higher overall accuracy. While there remains
a gap to fully supervised training, we expect that pretraining and/or fine-tuning on medical datasets
will further improve our model by instilling bone-specific shape priors and fracture geometry cues.

5 Conclusion

We introduce Rectified Point Flow, a unified flow-based framework for point cloud pose estimation
across registration and assembly tasks. By modeling part poses as velocity fields, it captures fine
geometry, handles symmetries and part interchangeability, and scales to varied part counts via
joint training on 100K shapes. Our two-stage pipeline—overlap-aware encoding and rectified flow
training—achieves state-of-the-art results on six benchmarks. Our work opens up new directions for
robotic manipulation and assembly by enabling precise, symmetry-aware motion planning.

Limitations and Future Work. While our experiments focus on object-centric point clouds, real-
world scenarios often involve cluttered scenes and partial observations. Moreover, while our model
can generate multiple plausible assemblies, some of these may not be mechanically functional;
see Fig. 7 (first column). Also, our model cannot handle objects that exceed a certain geometric
complexity; see Fig. 7 (second column). Another limitation arises from the number of points our
model can handle, which may restrict its usage on large-scale objects. Future work will extend
Rectified Point Flow to robustly handle occlusion, support scene-level and multi-body registration,
incorporate object-function reasoning, and scale to objects with larger point clouds.

Broader Impact. Rectified Point Flow makes it easier to build reliable 3D alignment and assembly
systems straight from raw scans, benefiting robotics, digital manufacturing, AR, and heritage recon-
struction. Given its performance and speed, it reduces the barrier for applying 3D part reasoning
in resource-constrained settings. However, the model can still produce incorrect, hallucinated, or
nonfunctional assemblies. For safety, further work on assembly verification and assembly error
recovery will be essential.

10

Acknowledgments and Disclosure of Funding

Tao Sun is supported by the Stanford Graduate Fellowship (SGF). Liyuan Zhu is partially supported
by SPIRE Stanford Student Impact Fund Grant. Shuran Song is supported by the NSF Award
#2037101. We appreciate the authors of GARF [11] for providing the FRACTURA dataset. We also
thank Stanford Marlowe Cluster [64] for providing GPU resources.

Supplementary Material

In this supplementary material, we provide the following:

• Model Details (Sec. A): Description of the DiT architecture and positional encoding scheme.
• Additional Evaluation (Sec. B):

– Runtime analysis.
– Evaluation on the preservation of rigidity at the part level.
– Comparison against category-specific assembly models on PartNet and IKEA-Manual.
– Analysis of different generative formulations,
– Evaluation of the anchor-free version of our model.

• Randomness in Assembly Generation (Sec. C): Investigation of the assemblies generated through

noise sampling and linear interpolation in the noise space.

• Generalization Ability (Sec. D): Qualitative results on unseen assemblies with same- or cross-

category parts to test the model’s generalization ability.

• Proof of Theorem 1 (Sec. E): Formal definition of the assembly symmetry group

and complete

proof of the flow invariance under the group

.
• Generalization Bounds (Sec. F): Derivation of the generalization risk guarantees and comparison

G

G

with that of existing 6-DoF methods.

A Model Details

Figure 8: Details of the DiT Block. Our flow model consists of an Encoder and a position embedding
(Pos. Emb.), and sequential DiT blocks (N = 6). Each block comprises Part-wise Attention, Global
Attention, MLP, and AdaLayerNorm layers.

DiT Architecture. Our flow model consists of 6 sequential DiT [55] blocks, each with a hidden
dimension of 512. For the multi-head self-attention in the DiT block, we set the number of attention
heads to 8, resulting in a head dimension of 64. As illustrated in Figure 8, inspired by [65], we apply
separated Part-wise Attention and Global Attention operations in each DiT block to capture both
intra-part and inter-part context:

• Part-wise Attention: Points within each part independently undergo a self-attention operation,

improving the model’s ability to capture local geometric structures.

11

Head<latexit sha1_base64="OwOm4vIo/GYfoLdK5L81uJcGkb0=">AAACOXicbVDLSgMxFM3UVx1fVZdugqVQQcqMlOqy0E2XFewD2qFk0kwbmswMyR2hDP0df8JfcKvgUlfi1h8wfSxs64GQwznncpPjx4JrcJx3K7O1vbO7l923Dw6Pjk9yp2ctHSWKsiaNRKQ6PtFM8JA1gYNgnVgxIn3B2v64NvPbj0xpHoUPMImZJ8kw5AGnBIzUz1UL3Z4v08607xTh6hr36CACbe6FWDeiZ88jNRNZ9Y1S9/q5vFNy5sCbxF2SPFqi0c999gYRTSQLgQqiddd1YvBSooBTwaZ2L9EsJnRMhqxraEgk0146/+kUF4wywEGkzAkBz9W/EymRWk+kb5KSwEivezPxX8+Xa5shuPNSHsYJsJAuFgeJwBDhWY14wBWjICaGEKq4eTumI6IIBVO2bUpx1yvYJK2bklspVe7L+Wp5WU8WXaBLVEQuukVVVEcN1EQUPaEX9IrerGfrw/qyvhfRjLWcOUcrsH5+AYinq0c=</latexit>[C0,···,CH]Latent FeaturesMLPPart-wise AttentionGlobal AttentionMLPAda LayerNormAda LayerNormAda LayerNorm× N+++Noised Point Clouds<latexit sha1_base64="QynWrU6SbOmh4UHuPkef0ODHU1U=">AAACG3icbZDLSgMxFIYz9VbrbdSlm9AiVJAyI6W6LLjpsoK9QGcomTRtQ5OZITkjlKF7X8JXcKt7d+LWhVufxLSdhbYeCPn5/nM4yR/EgmtwnC8rt7G5tb2T3y3s7R8cHtnHJ20dJYqyFo1EpLoB0UzwkLWAg2DdWDEiA8E6weR27ncemNI8Cu9hGjNfklHIh5wSMKhvF3teINPurO+U4eISe3QQgTb3EjYM9Pt2yak4i8Lrws1ECWXV7Nvf3iCiiWQhUEG07rlODH5KFHAq2KzgJZrFhE7IiPWMDIlk2k8Xf5nhc0MGeBgpc0LAC/p7IiVS66kMTKckMNar3hz+6wVyZTMMb/yUh3ECLKTLxcNEYIjwPCg84IpREFMjCFXcvB3TMVGEgomzYEJxVyNYF+2rilur1O6qpXo1iyePzlARlZGLrlEdNVATtRBFj+gZvaBX68l6s96tj2VrzspmTtGfsj5/AIn3oCE=</latexit>[X0(t),···,XH(t)]ConditionPoint Clouds<latexit sha1_base64="VPsAlC5xES6Btc5Q18qlz0cxvLE=">AAACRXicbZDLagIxFIYz9mbtzbbLbkJFsFBkpojtUujGRRcW6gWcQTIxajCZGZIzBRl8qb5EX6Fd1n13pds26iys9kDIz3f+w0l+PxJcg22/W5mt7Z3dvex+7uDw6Pgkf3rW0mGsKGvSUISq4xPNBA9YEzgI1okUI9IXrO2P7+f99jNTmofBE0wi5kkyDPiAUwIG9fIPxa7ry6Qz7dkluLrGLu2HoM29hHUDvZyxEJUskTGuulZ43evlC3bZXhTeFE4qCiitRi8/c/shjSULgAqiddexI/ASooBTwaY5N9YsInRMhqxrZEAk016y+PUUFw3p40GozAkAL+jqREKk1hPpG6ckMNLrvTn8t+fLtc0wuPMSHkQxsIAuFw9igSHE80hxnytGQUyMIFRx83ZMR0QRCib4nAnFWY9gU7Ruyk61XH2sFGqVNJ4sukCXqIQcdItqqI4aqIkoekFv6APNrFfr0/qyvpfWjJXOnKM/Zf38Aru8sPs=</latexit>[¯X0,···,¯XH]Velocity Field<latexit sha1_base64="osqNiA0SU/M+puiYsDC9PMkuj8I=">AAACqnicbZFfq9MwGMbTqsdZPceqF154ExyDHTiM9jCmN8JA0F1OcX+kLSNN0y2cpC3JW2GUfig/jrd+EtNtQvfnhZCH3/skb3gSF4Jr8Lw/lv3o8ZOrp51nzvMX1zcv3Vev5zovFWUzmotcLWOimeAZmwEHwZaFYkTGgi3ih89Nf/GLKc3z7AdsCxZJss54yikBg1bu714QxrJa1iuvD7d3OKRJDtrsezgxMHIaD1HVnhln29bik8gJBUshCFNFaBVKAhslq6T+P6FuMahbt1z2T479oeLrDUQrt+sNvF3hc+EfRBcdarpy/4ZJTkvJMqCCaB34XgFRRRRwKljthKVmBaEPZM0CIzMimY6qXbQ17hmS4DRXZmWAd7R9oiJS662MjbN5qj7tNfBiL5YnkyH9GFU8K0pgGd0PTkuBIcfNv+GEK0ZBbI0gVHHzdkw3xOQG5ncdE4p/GsG5mN8P/NFg9G3YHQ8P8XTQO/Qe9ZGPPqAxmqApmiFqvbU+WV+sr/ad/d3+aQd7q20dzrxBR2Un/wBzNNRX</latexit>!dX0dt,···,dXHdt"CNoisedPointsAssembled StateFlowEncoderPos. Emb.DiT Blocks• Global Attention: Subsequently, global self-attention operation is applied to all points across

parts, facilitating inter-part information exchange.

As discussed in Sec. 4, we apply RMS normalization individually to the query and key features in
each attention head before both attention operations to enhance numerical stability during training.
Additionally, every DiT block employs AdaLayerNorm, a layer normalization whose scaling and
shifting parameters are modulated by the time step t, following [55].

X, we construct a 10-dimensional vector, which comprises:

Positional Encoding. We adopt a multi-frequency Fourier feature mapping [66], to encode spatial
information in both the condition and noised point clouds. For the j-th point in the i-th part in the
condition point cloud, xi,j ∈
• The 3D absolute coordinates of xi,j.
• The 3D surface normal ni,j at that point xi,j in the condition point cloud.
• The 3D absolute coordinates of the noised point cloud X(t) at the index (i, j).
• The scalar part index i.
Each of these vectors is mapped through sinusoidal embeddings at multiple frequencies and then
concatenated with the point-wise feature output of encoder F .

Inference. At inference time, we recover the assembled point cloud of each part by numerically
integrating the predicted velocity fields V (t,
X) from t = 1 to t = 0. In practice, we
Xi(t)
perform K uniform Euler steps as,
∆t) = ˆX(t)
ˆX(t

}i
After K iterations, the resulting ˆX(0) approximates the point clouds of all parts in the assembled
state. For all evaluations, we set K = 20.

X)∆t, where ∆t = 1/K.

Xi(t)

V (t,

Ω |

Ω |

}i

−

−

{

{

∈

∈

B Additional Evaluation

Runtime Analysis. The number of sampling steps of the flow model is an important factor that
affects both accuracy and runtime. In Tab. 7, we vary the sampling steps and report the Part Accuracy,
Chamfer Distance, and the runtime per sample in PartNet-Assembly, measured on a single RTX 4090
GPU. Increasing steps consistently improves accuracy and geometric precision, with diminishing
returns beyond 20 steps. We use K = 20 sampling steps for all evaluations in the paper. In this
setting, our model achieves 4.3 samples per second, making it practical for robotic assembly and
localization tasks that require frequent online inference.

Table 7: Trade-offs of Sampling Steps between Accuracy and Runtime. Metrics and runtime
evaluated on PartNet-Assembly dataset with a single NVIDIA RTX 4090 GPU.

Sampling steps

Part Accuracy [%] ↑
Chamfer Distance [cm] ↓
Runtime / sample [s] ↓

1

25.2
3.23
0.072

2

38.1
1.70
0.081

5

46.7
0.90
0.108

10

52.1
0.75
0.148

20

53.9
0.73
0.232

50

54.6
0.71
0.483

Part-level Rigidity Preservation. As a dense point map prediction framework, Rectified Point
Flow is not explicitly trained to preserve the rigidity of each part. To quantify how well it preserves
the rigidity of the parts, we first align each predicted part ˆXi(0) with the part in assembled state
Xi(0) using the Kabsch algorithm.
We then measure two rigidity preservation errors using (1) the Root Mean Square Error (RMSE)
over all points and (2) the Overlap Ratio (OR) over all points at varying thresholds τ
cm. Specifically, for part i with Mi points, we compute
0.1, 0.2, 0.5, 1, 2

∈

{

}

RMSE =

1
Mi

(cid:118)
(cid:117)
(cid:117)
(cid:116)

Mi

T ′i ˆxi,j(0)

j=1
(cid:88)

(cid:13)
(cid:13)

12

xi,j(0)

2

and

−

(cid:13)
(cid:13)

Table 8: Part-level Rigidity Preservation Evaluation. Rectified Point Flow demonstrates low
shape discrepancy between condition and predicted part point clouds, measured by the Root Mean
Square Error (RMSE), Relative RMSE, and Overlap Ratios (ORs) across datasets. D represents the
average object scale of each dataset. (Abbr: BreakingBad-E = BreakingBad-Everyday; PartNet-A =
PartNet-Assembly; IKEA-M = IKEA-Manual.)

Metric

Object Scale D [cm] –
[cm] ↓
RMSE
[%] ↓
Relative RMSE
[%] ↑
[%] ↑
[%] ↑
[%] ↑
[%] ↑

OR (τ = 0.1 cm)
OR (τ = 0.2 cm)
OR (τ = 0.5 cm)
OR (τ = 1 cm)
OR (τ = 2 cm)

Shape Assembly

Pairwise Registration

BreakingBad-E TwoByTwo PartNet-A IKEA-M TUD-L ModelNet-40

52.1
0.76
1.5

52.3
61.7
74.9
81.4
89.5

107.7
2.46
2.3

63.8
70.8
76.8
78.7
81.9

89.0
1.04
1.2

33.1
48.6
66.8
77.9
87.4

61.4
0.66
1.1

46.7
57.7
69.8
81.0
92.0

40.8
0.16
0.4

96.9
97.1
97.4
97.7
98.2

70.0
0.30
0.4

95.0
96.0
96.3
96.6
97.1

OR (τ ) =

1
Mi

j

{

| ∥

T ′i ˆxi,j(0)

xi,j(0)

< τ

∥

−

.

}
(cid:12)
(cid:12)

(cid:12)
(cid:12)

SE(3) denotes the optimal rigid transform returned by Kabsch; xi,j and ˆxi,j denote the
Here, T ′i ∈
j-th point on Xi(0) and on ˆXi(0), respectively. Because each part is first rigidly aligned to the
ground-truth assembled state, these metrics intentionally ignore pose errors, and only measure the
shape difference between the predicted and ground truth point parts. To factor in the variations in
object size across datasets, we compute the average scale of an object, denoted by D, as twice the
average distance from the object’s center of gravity to all its points. Then, we define the Relative
RMSE as RMSE / D, i.e., the RMSE normalized by the average object scale. We report these metrics
averaged for all parts in each dataset in Tab. 8.

For the pairwise registration task, Rectified Point Flow demonstrates strong rigidity preservation. On
TUD-L, we obtain a Relative RMSE of 0.4% and ORs above 96.9% even at the strictest τ = 0.1 cm
threshold; on ModelNet-40, we achieve the same Relative RMSE of 0.4% with similar high ORs above
95.0%. Specifically, on TUD-L we record ORs of 96.9% (τ = 0.1 cm), 97.1% (τ = 0.2 cm), 97.4%
(τ = 0.5 cm), 97.7% (τ = 1 cm) and 98.2% (τ = 2 cm); on ModelNet-40 the corresponding ORs
are 95.0%, 96.0%, 96.3%, 96.6% and 97.1%, demonstrating consistently strong rigidity preservation.

In the more challenging shape assembly task, rigidity errors remain low. Across the four datasets, the
Relative RMSE ranges from 1.1% to 2.3%. At a strict threshold of τ = 0.1 cm, overlap ratios (ORs)
span 33.1 % (PartNet-Assembly) up to 63.8 % (TwoByTwo); By τ = 1 cm, the ORs exceed 77.9% in
the four datasets (77.9%-81.4%), increasing further to 81.9%-92.0% in the more relaxed τ = 2 cm.
The highest Relative RMSE and lower averaged ORs are observed in TwoByTwo, probably due to its
limited training samples and lower shape similarity to other datasets, and the fact that TwoByTwo
has the largest overall object scale of 107.7 cm among all datasets. In contrast, IKEA-Manual,
despite having fewer training samples, benefits from shared priors in furniture objects in joint training,
delivering the lowest RMSE and high ORs at all thresholds. These results demonstrate robust rigidity
preservation of Rectified Point Flow even in complex shape assembly scenarios.

Furthermore, please note that the subsequent pose recovery stage in Rectified Point Flow further
refines part poses via an SVD-based global optimization, which fits optimal poses under noises.
Overall, we empirically confirm that Rectified Point Flow generates point clouds that reliably respect
the rigid structure of the conditioning parts.

Comparison with Category-specific Models. We compare against category-specific point-cloud
assembly methods in Tab. 9. All baselines are trained separately for each category, and the category
labels are assumed to be known at inference time. RGL-Net [67] additionally assumes a top-to-bottom
ordering of the input parts. In contrast, Rectified Point Flow is class-agnostic and performs inference
without any class label or part ordering. We evaluated both shape Chamfer Distance (CD) and Part
Accuracy (PA) in PartNet-Assembly and IKEA-Manual, following the protocol of Huang et al. [47].

13

Table 9: Comparison with Category-specific Models. We report Shape Chamfer Distance (CD)
and Part Accuracy (PA) on the PartNet-Assembly and IKEA-Manual. All baselines are trained per
category, whereas Rectified Point Flow is trained over all categories. (∗RGL-Net additionally requires
a top-to-bottom part ordering.)

Method

Known
Category

PartNet-Assembly

IKEA–Manual [17]

Chair

Table

Lamp

All

Chair

All

CD ↓ PA ↑ CD ↓ PA ↑ CD ↓ PA ↑ CD ↓ PA ↑ CD ↓ PA ↑ CD ↓ PA ↑
[%]
[cm]

[cm]

[cm]

[cm]

[cm]

[cm]

[%]

[%]

[%]

[%]

[%]

B-LSTM [68]
B-Global [68]
RGL-Net* [67]
Huang et al. [68]

Ours (Joint)

✓
✓
✓
✓

×

1.31
1.46
0.87
0.91

21.8
15.7
49.2
39.0

1.25
1.12
0.48
0.50

28.6
15.4
54.2
49.5

0.77
0.79
0.72
0.93

20.8
22.6
37.6
33.3

–
–
–
–

–
–
–
–

1.81
1.95
5.08
1.51

3.5
0.9
4.0
6.9

–
–
–
–

–
–
–
–

0.71

44.1

0.36

49.4

0.49

70.0

0.48

53.9

1.49

29.9

0.48

33.2

Table 10: Generative Formulation Comparison. We compare Rectified Flow (RF) with Denoising
Diffusion Probabilistic Model (DDPM) in our method, with both using the same DiT architecture and
pretrained encoder. RF achieves superior performance on Rotation Error (RE) and Translation Error
(TE) across all datasets. (Abbr: BreakingBad-E = BreakingBad-Everyday; PartNet-A = PartNet-
Assembly; IKEA-M = IKEA-Manual.)

Metric

Generative
Formulation

RE [deg] ↓

TE [cm] ↓

DDPM
RF

DDPM
RF

Shape Assembly

Pairwise Registration

BreakingBad-E TwoByTwo PartNet-A IKEA-M TUD-L ModelNet-40

13.0
7.4

3.5
2.0

17.2
13.2

10.1
3.0

29.5
21.8

21.3
14.8

21.4
10.8

19.2
17.2

2.6
1.4

0.5
0.3

3.4
0.9

0.7
0.2

Without category or ordering assumptions like the baseline methods, our joint model still achieves the
lowest CD and matches or exceeds the PA of category-specific baselines optimized for each category
(chair, table, lamp). In particular, we observe a relative improvement of 110.2% on Lamps PA over
the strongest baseline. In IKEA-Manual, we observe that all category-specific models collapse to PA
6.9% for the Chair category. We hypothesize that the baselines’ architecture and hyperparameter
≤
are largely tailored to PartNet. In contrast, our joint model achieves 29.9% PA for the Chair category
and 33.2% PA for all categories, over 4 times higher than any baselines. Those observations confirm
that our category-agnostic cross-dataset training improves the learning of shared geometric priors far
beyond any single category or dataset.

Ablation on Generative Formulation. As an alternative to the generative formulation of Rectified
Flow (RF) in our method, we also evaluate a Denoising Diffusion Probabilistic Model (DDPM) [69]
using an identical DiT architecture and the pre-trained encoder. In this setup, the forward noising
4 to 0.02 over T = 1000
process employs constant variances (β) that increase linearly from 10−
timesteps. As shown in Tab. 10, the RF-based model consistently outperforms the DDPM variant
on both shape assembly and pairwise registration tasks, with 35.3% lower rotation error (RE) and
11.63% lower translation error (TE). This result is in line with the findings of GARF [11]. We
hypothesize that the straight-line flow in RF reduces the learning difficulty in our tasks. DDPM’s
frequency-based generation—which works well for images—may not be as effective as RF for 3D
point cloud synthesis in Euclidean space.

In the anchor-free setting, we do not fix the anchor’s pose. Instead, during
Anchor-free Models.
training, we treat the anchor exactly like any other part in the conditioning: its point cloud is centered
to its own CoM frame and then randomly rotated. The model, therefore, never receives the true
anchor pose from condition. The flow target is defined in the anchor frame: we put the completed
assembly point cloud in the anchor’s frame and then re-center the whole assembled point cloud. At
inference, we first estimate the rigid transform between the predicted anchor and the ground-truth

14

Table 11: Evaluation of the Anchor-free Model on Shape Assembly Datasets.

Dataset

BreakingBad-E [12]
TwoByTwo [16]
PartNet-Assembly [15]
IKEA-Manual [17]

Anchor-fixed

Anchor-free (Average)

Anchor-free (Best-of-3)

RE ↓
[deg]

7.4
13.2
21.8
20.7

TE ↓
[cm]

2.0
3.0
14.8
24.7

PA ↑
[%]

93.5
–
53.9
33.2

RE ↓
[deg]

17.4
15.2
47.3
54.7

TE ↓
[cm]

8.0
24.2
40.5
51.5

PA ↑
[%]

90.2
–
45.3
19.5

RE ↓
[deg]

13.0
9.0
38.2
39.0

TE ↓
[cm]

5.9
16.7
32.9
40.5

PA ↑
[%]

92.2
–
54.3
28.6

Table 12: Evaluation of the Anchor-free Model on Pairwise Registration Datasets.

Setting

Anchor-fixed
Anchor-free (Average)
Anchor-free (Best-of-3)

ModelNet-40 [19]

TUD-L [18]

RE ↓
[deg]

0.93
2.05
1.18

TE ↓
[unit]

0.002
0.012
0.007

Recall @ 5◦ ↑
[%]

Recall @ 1cm ↑
[%]

97.7
96.6
97.3

99.0
96.3
97.3

anchor point cloud via ICP, and then apply this transform to all predicted parts. Metrics (RE, TE, and
Part Acc.) are computed after this anchor alignment.

Across shape assembly datasets (Tab. 11) and pairwise registration (Tab. 12), anchor-free (average)
underperforms anchor-fixed as expected due to (i) error propagation from anchor misalignment and
(ii) ambiguity induced by symmetric anchor parts. However, despite the lower absolute performance,
our anchor-free model preserves the same relative ordering among baselines on all shape-assembly
datasets; the only exception is GARF, which is evaluated in the anchor-fixed protocol. In particular,
our model significantly improves anchor-free SOTA on BreakingBad-Everyday’s Part Accuracy from
76.2% (PuzzleFussion++ [25]) to 90.2%. For pairwise registration, our model is also achieving
the lowest TE on ModelNet and the highest Recall@5◦ on TUD-L among baselines, while remain
competitive on other metrics.

Furthermore, we observe that the anchor-free (best-of-3) evaluation, which selects the best prediction
out of 3 different random seeds per sample, largely narrows the gap to anchor-fixed. This indicates
that a small stochastic budget may find a more reliable anchor-free prediction. The effect is most
pronounced on datasets with higher part count and weaker geometric cues (e.g., PartNet-Assembly
and IKEA-Manual), where anchor ambiguity is stronger and small anchor errors propagate severely.

C Randomness in Assembly Generation

Diversity via Noise Sampling. To evaluate the diversity of assembly configurations generated by
Rectified Point Flow, we sample the Gaussian noise vector Z multiple times for the same conditional
(unposed) point cloud inputs. At inference time, we set X(1) = Z and run the model to obtain
prediction ˆX(0). In Figure 9, each row corresponds to a single final assembly: the first column
shows the ground-truth assembly, and the next four columns display outputs produced by four
different Gaussian noises. All generated assemblies preserve the part structure, yet exhibit meaningful
variations in the parts’ placement and orientation, and overall geometry of the object. As expected, the
model produces diverse configurations for symmetric or interchangeable parts, such as the armrests
and the chair base. This shows that Rectified Point Flow effectively captures a diverse conditional
distribution of valid assemblies.

Linear Interpolation in Noise Space. We illustrate Rectified Point Flow’s learned mapping from
random Gaussian noise to plausible assembly configurations. In Figure 10, each row uses the
same condition (unposed) point cloud, with the left and right columns showing the outputs of two
randomly sampled noise vectors Z0 and Z1, respectively. The three columns in between display

15

Figure 9: Sampling in Noise Space. For each fixed condition input point clouds, we sample four
independent Gaussian noise vectors to generate distinct assembly outputs (shown in columns 2–5).
While all samples preserve the object’s structure, they show meaningful variation in part placement,
orientation, and overall geometry, particularly for symmetric parts (e.g., armrests and chair bases).
For comparison, the first column shows the ground-truth assemblies.

Figure 10: Linear Interpolation in Noise Space. For different objects in each row, we fix the same
conditional input and decode two independently sampled Gaussian noise vectors, Z0 (leftmost) and
Z1 (rightmost), into plausible part configurations. The three center columns show outputs from the
linearly interpolated noises between Z0 and Z1. We observe a continuous, semantically meaningful
mapping from Gaussian noise to valid assemblies.

16

Generated Results by Different ZGTPartInterchangingStructural ChangingNoise Z0 Noise Z1 Interpolated Noise Z(s) = (1 - s)Z0 + sZ1 Figure 11: Generalization to Unseen Assemblies Within the Same Category: We select parts from
two objects of the same category in the PartNet-Assembly test set. Parts from Object 1 are shown in
blue, and parts from Object 2 in red; unselected parts are shown in gray. The results demonstrate that
the model comprehends the underlying geometric structure of the category and can re-target parts to
construct the final shape.

results generated by Z(s) which linearly interpolates between Z0 and Z1 in noise space, i.e.,

Z(s) := (1

−

s)Z0 + sZ1, where s

0.25, 0.5, 0.75

.

}

∈ {

(7)

At each interpolation step s, we run inference with X(1) = Z(s). As s increases, the predicted
shapes smoothly morph from the configuration induced by Z0 toward that of Z1. As shown in the first
2 rows in Figure 10, we observe smooth transitions among interchangeable parts in both examples.
The 2 bottom rows in Figure 10 visualize the transitions in the overall structure of objects. In the
table example, we observe a gradual reduction in overall height, a lowering of the horizontal beams,
and a more centralized positioning where the four legs meet. In the shelf example, the transformation
is more drastic: two vertical boards become horizontal and two diagonal cables are rearranged to a
new vertical configuration. The above transitions across various assemblies confirm that Rectified
Point Flow learns a continuous mapping from Gaussian noise to a semantically meaningful geometry
space. Note that most of the interpolated configurations are physically plausible assemblies, creating
functional objects that can stand in real-world.

D Generalization Ability

We test the generalization ability of our model for novel assemblies under two different settings:
between objects from the same (in-category) and different (cross-category) categories. Given two
objects in PartNet-Assembly, we select certain parts from each of them as the input to Rectified Point
Flow to test if the model can generate novel and plausible assemblies.

In-category Test. As shown in Fig. 11, parts selected from Object 1 are rendered in blue and
those from Object 2 in red. Our model then synthesizes novel assemblies that blend and reconfigure

17

Generated ResultsObject 1Object 2Figure 12: Generalization to Unseen Assemblies Across Categories: We select parts from two
objects of different categories in the PartNet-Assembly test set. Parts from Object 1 are shown in
blue, and parts from Object 2 in red; unselected parts are shown in gray. The results demonstrate that
the model can reason about part compositionality and re-target parts to construct a plausible final
shape even if some of them originate in completely different objects.

these parts in a coherent and category-consistent manner. For example, in the chair category (first
two rows), the model successfully retains a functional and plausible seat-back-leg structure while
creatively mixing parts. In the lamp category (third row), even though the base and shade style differ
significantly between objects, generated results exhibit sensible combinations that maintain structural
integrity. Similarly, in the table category (last row), our method combines parts from a flat-top table
and lattice-style base to produce hybrid yet coherent table designs.

Cross-category Test. Fig. 12 highlights Rectified Point Flow’s ability to generalize to unseen part
combinations across categories. This is a particularly challenging test, since such part combination
may not even be possible to be assembled into a meaningful object. Nevertheless, our method still
demonstrates a certain degree of generalization. We show two input objects from different categories,
for example, a monitor and a chair, a chair and a lamp, or a wall sconce and a spray bottle. The results
on the right demonstrate that our model can reconfigure these parts into plausible new assemblies,
preserving geometric coherence. This suggests that the model has learned a strong understanding of
part relationship, allowing it to reason about compositionality even across category boundaries.

E Proof of Theorem 1

A key advantage of Rectified Point Flow is that it learns both rotational symmetries of individual
parts and the interchangeability of a set of identical parts, without any labels of symmetry parts.
Below, we first formally define an assembly symmetry group
that characterizes the symmetry and
interchangeability of the parts in the multi-part point cloud.
Definition 1 (Assembly symmetry group). For each part i
SO(3) be the (finite)
be the set of
stabilizer of its assembled shape, i.e., RXi(0) = Xi(0) for all R
permutations that only permute indices of identical parts. We define the assembly symmetry group as
the semidirect product

Ω, let Gi ⊆
Gi. Let S
∈

S

⊆

∈

G

Ω

|

|

(8)

=

G

G1 × · · · ×

G
|

Ω

|

(cid:0)

(cid:1)

18

⋊ S.

Generated ResultsObject 1Object 2acts on every realization of the Rectified Point
A group element g = (R1, . . . , R
Flow by g(Xi(t)) := Ri Xσ−1(i)(t), and on network outputs of the i-th part (denoted as Vi)
by g(Vi(t, g(X))) := Ri Vσ−1(i)(t, g(X)).

∈ G

, σ)

Ω

|

|

Now, we show the following result that a single point’s flow distribution is invariant under any g
Lemma 1 (
point cloud X, we sample a flow realization:

-invariance of the flow distribution). For every element g

.
∈ G
and a given multi-part

∈ G

G

x(t) = tx(1) + (1

−

t)x(0), where x(1)

(0, I), x(0)

X.

∼

∼ N

then, we have

p

g

{

·

x(t)

[0,1]

}t

∈

= p(
{

x(t)

}t

∈

[0,1]).

(cid:0)

(cid:1)

Proof. Recall that, in Rectified Point Flow, a flow of a single point is x(t) := (1
t)x(0) + tx(1),
−
where x(0)
(0, I). Because the
end–points of the linear interpolation are sampled independently, the PDF of the path distribution
factorizes as

X is drawn uniformly from the assembled shape and x(1)

∼ N

∼

(9)
= p
which indicates the randomness resides by the states t = 0 and t = 1 only. Because the perturbation
SO(3). For p(x(0)) we
x(1)
distinguish two cases:

}t
(0, I) is isotropic, p(x(1)) is invariant under every rotation R

∼ N

x(1)

x(0)

x(t)

[0,1]

∈

p

p

{

(cid:0)

(cid:1)

(cid:0)

(cid:0)

(cid:1)

(cid:1)

∈

,

• Rotational symmetry: If R

∈

Gi, then RXi(0) = Xi(0) point-wise, so p(x(0)) = p(Rx(0)).

• Interchangeability. If parts i and j are identical, sampling first a part index with probability
Xj(0)).

p(i) = Ni/N and then a point uniformly inside it implies p(x(0)
Therefore exchanging the indices (σ(i) = j, σ(j) = i) leaves p(x(0)) unchanged.

Xi(0)) = p(x(0)

∈

∈

By composing the above two properties for all parts, we complete the proof.

Lemma 1 can directly lift from single points to the full multi–part flow
us to the Theorem 1: For every element g
LCFM(g(V (t,
LCFM(V ) =

∈ G
Ω; g(X)))).

Ω. This leads
, we have the learning objective in Eq. 3 following

Xi(t)

Xi(t)

}i

}i

{

{

∈

∈

F Generalization Bounds

While the Rectified Point Flow predicts a much higher-dimensional space (3Mi coordinates per
part), we find that its Rademacher complexity scales exactly the same rate as the 6-DoF methods,
O(1/√m), where m is the number of samples in the training set.

Below, we compute their Rademacher complexities and empirical risks, respectively. Without loss
of generality, we use the reconstruction error for the evaluation of poses, i.e., ℓ( ˆR, ˆt; R⋆, t⋆) =
( ˆR

. First, we define hypothesis classes for both methods:

R⋆)X ⋆ + ˆt

t⋆

−

−

F

(cid:13)
(cid:13)
ˆXi(0; θ)
Ci (cid:55)→

(cid:13)
• Our Rectified Point Flow:
(cid:13)

Fi =
• Pose vector-based flow:

{

θ

Θ
}

∈

|

, where ˆXi(0; θ) := Xi(1)

1

−

0
(cid:90)

Vi(t; C, θ)dt.

Gi =
Rademacher Complexity of Rectified Point Flow. With m i.i.d.
(C (k), R⋆(k), t⋆(k))
{
cial risk

m
k=1, we write the population risk

(h) = E

Ci (cid:55)→

R

( ˆRi, ˆti)ϕ |

Φ

∈

ϕ

}

{

}

ℓ

.

training objects D =
and empri-

h(C), R⋆, t⋆

m

ˆ
RD(h) =

1
m

(cid:88)k=1

(cid:0)

ℓ

h(C (k)), R⋆(k), t⋆(k)

.

(cid:1)(cid:3)

(cid:2)

(cid:0)

(cid:1)

Since our Rectified Point Flow method estimates the part pose by the Procrustes operator, i.e.,
( ˆR, ˆt) = Pr
SE(3) is the Procrustes operator, we have following
Lipschitz contracting property.

, where Pr : R3N

ˆX(0; θ)

→

(cid:0)

(cid:1)

19

Property 1 (Lipschitz Contracting). Let X ⋆
single part, and denote σmin = σmin((X ⋆)⊤X ⋆). If
solution ( ˆR, ˆt) = P ( ˆX(0)) satisfies

∈

R3N be the centralized ground-truth point set of a
ε, the optimal Procrustes

X ⋆

ˆX(0)
∥

−

∥F ≤

( ˆR

∥

−

R⋆, ˆt

−

t⋆)

∥ ≤

ε
√σmin

.

(10)

This property directly follows from Davis–Kahan perturbation bounds [70] for the Top-3 singular
vectors. Crucially, σmin = Ω(N )3 for well-spread point clouds, so Pr is a 1
√N

-Lipschitz map.

Let Rm(
L-Lipschitz map contracts Rademacher complexity

H

) denote the empirical Rademacher complexity on S. Because composition with a

Rm(Pr

)

◦ F

≤

1
√N

Rm(

)

F

≤

LΘ√3N
√N

1
√m

= O

(cid:16)

LΘ
√m

.

(cid:17)

(11)

Rademacher Complexity of 6DoF-based Methods. For the baseline we need only regress d = 6
numbers, hence

Rm(

)

G

≤

LΦ√d
√m

= O

(cid:16)

LΦ
√m

.

(cid:17)

(12)

Comparison of Generalization Bounds. Applying Bartlett Theorem and using (10), we obtain,
with probability at least 1

δ over the samples from D,

−

P

ˆf

◦

R

ˆ
RD

P

ˆf

◦

≤

+ 2 Rm(P

) + 3

◦F

(cid:0)

(cid:0)

(cid:1)
(ˆg)

(cid:1)
ˆ
RD(ˆg) + 2 Rm(
are the empirical-risk minimizers on S.

) + 3

(cid:114)

R

≤

G

log(2/δ)
2m

,

log(2/δ)
2m

,

(cid:114)

(FLOW)

(6DOF)

∈ F

and ˆg

where ˆf
In conclusion, while Rectified Point Flow predicts a much higher-dimensional space, the contraction
of the SVD stage cancels this apparent over-parameterization, producing a complexity term that
scales at the same rate of O(1/√m) as the 6-DoF baseline; (FLOW)–(6DOF).

∈ G

As a result, our method enjoys at least same generalization risk guarantees despite operating in an
over-parameterized prediction space, while retaining the

-invariance benefits proven in Sec. E.

G

References

[1] A. Zeng, S. Song, M. Nießner, M. Fisher, J. Xiao, and T. Funkhouser, “3dmatch: Learning local geometric

descriptors from rgb-d reconstructions,” in CVPR, 2017.

[2] Y. Li, A. Zeng, and S. Song, “Rearrangement planning for general part assembly,” in Conference on Robot

Learning, 2023.

[3] Y. Li, L. Jiang, Y. Liu, Y. Nie, H. Zhu, and D. Lin, “Gapartnet: Graph-structured assembly from part

segments,” in ECCV, 2022.

[4] J. Zhang, M. Wu, and H. Dong, “Generative category-level object pose estimation via diffusion models,”

NeurIPS, vol. 36, 2024.

[5] H. Zhao, S. Wei, D. Shi, W. Tan, Z. Li, Y. Ren, X. Wei, Y. Yang, and S. Pu, “Learning symmetry-aware

geometry correspondences for 6d object pose estimation,” in ICCV, 2023.

[6] T. Hodan et al., “Bop challenge 2020 on 6d object localization,” in ECCV Workshops, 2020.

[7] B. Tekin, S. Sinha, and P. Fua, “Real-time seamless single shot 6d object pose prediction,” in CVPR, 2018.

[8] H. Wang, S. Sridhar, J. Huang, J. Valentin, S. Song, and L. J. Guibas, “Normalized object coordinate space

for category-level 6d object pose and size estimation,” in CVPR, 2019.

[9] K. A. Murphy, C. Esteves, V. Jampani, S. Ramalingam, and A. Makadia, “Implicit-pdf: Non-parametric

representation of probability distributions on the rotation manifold,” in ICML, 2021.

3Here, Ω(·) denote the asymptotic rate, instead of part index set.

20

[10] Y. Li, K. Mo, Y. Duan, H. Wang, J. Zhang, and L. Shao, “Category-level multi-part multi-joint 3d shape

assembly,” in CVPR, pp. 3281–3291, 2024.

[11] S. Li, Z. Jiang, G. Chen, C. Xu, S. Tan, X. Wang, I. Fang, K. Zyskowski, S. P. McPherron, R. Iovita,
C. Feng, and J. Zhang, “Garf: Learning generalizable 3d reassembly for real-world fractures,” arXiv
preprint arXiv:2504.05400, 2025.

[12] S. Sellán, Y.-C. Chen, Z. Wu, A. Garg, and A. Jacobson, “Breaking bad: A dataset for geometric fracture

and reassembly,” NeurIPS, 2022.

[13] M. Liu, M. A. Uy, D. Xiang, H. Su, S. Fidler, N. Sharp, and J. Gao, “Partfield: Learning 3d feature fields

for part segmentation and beyond,” in arxiv, 2025.

[14] M. Deitke, D. Schwenk, J. Salvador, L. Weihs, O. Michel, E. VanderBilt, L. Schmidt, K. Ehsani, A. Kemb-
havi, and A. Farhadi, “Objaverse: A universe of annotated 3d objects,” arXiv preprint arXiv:2212.08051,
2022.

[15] K. Mo, S. Zhu, A. X. Chang, L. Yi, S. Tripathi, L. J. Guibas, and H. Su, “Partnet: A large-scale benchmark

for fine-grained and hierarchical part-level 3d object understanding,” in CVPR, 2019.

[16] Y. Qi, Y. Ju, T. Wei, C. Chu, L. L. Wong, and H. Xu, “Two by two: Learning multi-task pairwise objects

assembly for generalizable robot manipulation,” CVPR, 2025.

[17] R. Wang, Y. Zhang, J. Mao, R. Zhang, C.-Y. Cheng, and J. Wu, “Ikea-manual: Seeing shape assembly step

by step,” in NeurIPS Datasets and Benchmarks Track, 2022.

[18] T. Hodan, F. Michel, E. Brachmann, W. Kehl, A. GlentBuch, D. Kraft, B. Drost, J. Vidal, S. Ihrke,

X. Zabulis, et al., “Bop: Benchmark for 6d object pose estimation,” in ECCV, 2018.

[19] Z. Wu, S. Song, A. Khosla, F. Yu, L. Zhang, X. Tang, and J. Xiao, “3d shapenets: A deep representation

for volumetric shapes,” in CVPR, 2015.

[20] J. L. Schönberger and J.-M. Frahm, “Structure-from-motion revisited,” in CVPR, 2016.

[21] Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M. R. Oswald, and M. Pollefeys, “Nice-slam: Neural

implicit scalable encoding for slam,” in CVPR, 2022.

[22] H. Jiang, M. Salzmann, Z. Dang, J. Xie, and J. Yang, “Se (3) diffusion model-based point cloud registration

for robust 6d object pose estimation,” in NeurIPS, 2023.

[23] L. Zhu, Y. Li, E. Sandström, S. Huang, K. Schindler, and I. Armeni, “Loopsplat: Loop closure by registering

3d gaussian splats,” in 3DV, 2025.

[24] S. Hosseini, M. A. Shabani, S. Irandoust, and Y. Furukawa, “Puzzlefusion: Unleashing the power of

diffusion models for spatial puzzle solving,” in NeurIPS, 2023.

[25] Z. Wang, J. Chen, and Y. Furukawa, “Puzzlefusion++: Auto-agglomerative 3d fracture assembly by denoise

and verify,” in ICLR, 2025.

[26] L. Zhu, S. Huang, and I. A. Konrad Schindler, “Living scenes: Multi-object relocalization and reconstruc-

tion in changing 3d environments,” in CVPR, 2024.

[27] Y. Zhou, C. Barnes, J. Lu, J. Yang, and H. Li, “On the continuity of rotation representations in neural

networks,” in CVPR, 2019.

[28] P.-E. Sarlin, D. DeTone, T. Malisiewicz, and A. Rabinovich, “Superglue: Learning feature matching with

graph neural networks,” in CVPR, 2020.

[29] S. Huang, Z. Gojcic, M. Usvyatsov, and K. S. Andreas Wieser, “Predator: Registration of 3d point clouds

with low overlap,” in CVPR, 2021.

[30] Z. J. Yew and G. H. Lee, “Rpm-net: Robust point matching using learned features,” in CVPR, 2020.

[31] Y. Wang and J. M. Solomon, “Deep closest point: Learning representations for point cloud registration,” in

ICCV, 2019.

[32] Z. Qin, H. Yu, C. Wang, Y. Guo, Y. Peng, and K. Xu, “Geometric transformer for fast and robust point

cloud registration,” in CVPR, 2022.

[33] J. Y. Zhang, A. Lin, M. Kumar, T.-H. Yang, D. Ramanan, and S. Tulsiani, “Cameras as rays: Pose

estimation via ray diffusion,” in ICLR, 2024.

[34] H. Huang, K. Schmeckpeper, D. Wang, O. Biza, Y. Qian, H. Liu, M. Jia, R. Platt, and R. Walters,
“Imagination policy: Using generative point cloud models for learning manipulation policies,” arXiv
preprint arXiv:2406.11740, 2024.

[35] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud, “Dust3r: Geometric 3d vision made easy,” in

CVPR, 2024.

[36] M. A. Fischler and R. C. Bolles, “Random sample consensus: a paradigm for model fitting with applications

to image analysis and automated cartography,” Communications of the ACM, 1981.

21

[37] V. Lepetit, F. Moreno-Noguer, and P. Fua, “Ep n p: An accurate o (n) solution to the p n p problem,” IJCV,

2009.

[38] C. Choy, J. Park, and V. Koltun, “Fully convolutional geometric features,” in ICCV, pp. 8958–8966, 2019.

[39] H. Deng, T. Birdal, and S. Ilic, “Ppfnet: Global context aware local features for robust 3d point matching,”

in CVPR, 2018.

[40] Z. Gojcic, C. Zhou, J. D. Wegner, and A. Wieser, “The perfect match: 3d point cloud matching with

smoothed densities,” in CVPR, pp. 5545–5554, 2019.

[41] X. Bai, Z. Luo, L. Zhou, H. Fu, L. Quan, and C.-L. Tai, “D3feat: Joint learning of dense detection and

description of 3d local features,” arXiv:2003.03164 [cs.CV], 2020.

[42] H. Wang, Y. Liu, Z. Dong, and W. Wang, “You only hypothesize once: Point cloud registration with

rotation-equivariant descriptors,” in ACM International Conference on Multimedia, 2022.

[43] Y. Wang and J. M. Solomon, “Prnet: Self-supervised learning for partial-to-partial registration,” NeurIPS,

2019.

[44] J. Huang, H. Wang, T. Birdal, M. Sung, F. Arrigoni, S. Hu, and L. J. Guibas, “Multibodysync: Multi-body

segmentation and motion estimation via 3d scan synchronization,” in CVPR, 2021.

[45] C. Deng, J. Lei, B. Shen, K. Daniilidis, and L. Guibas, “Banana: banach fixed-point network for pointcloud

segmentation with inter-part equivariance,” in NeurIPS, 2023.

[46] M. Atzmon, J. Huang, F. Williams, and O. Litany, “Approximately piecewise e(3) equivariant point

networks,” in ICLR, 2024.

[47] S. Huang, Z. Gojcic, J. Huang, A. Wieser, and K. Schindler, “Dynamic 3d scene analysis by point cloud

accumulation,” in ECCV, 2022.

[48] Y.-C. Chen, H. Li, D. Turpin, A. Jacobson, and A. Garg, “Neural shape mating: Self-supervised object

assembly with adversarial shape priors,” in CVPR, 2022.

[49] R. Wu, C. Tie, Y. Du, Y. Zhao, and H. Dong, “Leveraging se (3) equivariance for learning 3d geometric

shape assembly,” in ICCV, 2023.

[50] G. Scarpellini, S. Fiorini, F. Giuliari, P. Morerio, and A. Del Bue, “Diffassemble: A unified graph-diffusion

model for 2d and 3d reassembly,” arXiv preprint arXiv:2402.19302, 2024.

[51] X. Liu, C. Gong, and Q. Liu, “Flow straight and fast: Learning to generate and transfer data with rectified

flow,” arXiv preprint arXiv:2209.03003, 2022.

[52] Q. Liu, “Rectified flow: A marginal preserving approach to optimal transport, 2022,” URL https://arxiv.

org/abs/2209.14577.

[53] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,”

arXiv preprint arXiv:2210.02747, 2022.

[54] X. Wu, L. Jiang, P.-S. Wang, Z. Liu, X. Liu, Y. Qiao, W. Ouyang, T. He, and H. Zhao, “Point transformer

v3: Simpler, faster, stronger,” in CVPR, 2024.

[55] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” arXiv preprint arXiv:2212.09748,

2022.

[56] B. Zhang and R. Sennrich, “Root mean square layer normalization,” NeurIPS, vol. 32, 2019.

[57] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. Müller, H. Saini, Y. Levi, D. Lorenz, A. Sauer, F. Boesel,

et al., “Scaling rectified flow transformers for high-resolution image synthesis,” in ICML, 2024.

[58] S. Lee, Z. Lin, and G. Fanti, “Improving the training of rectified flows,” vol. 37, pp. 63082–63109, 2024.

[59] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” arXiv preprint arXiv:1711.05101,

2017.

[60] J. Lu, Y. Sun, and Q. Huang, “Jigsaw: Learning to assemble multiple fractured objects,” NeurIPS, 2023.

[61] Wikipedia contributors, “Rodrigues’ rotation formula—Wikipedia, The Free Encyclopedia.” https:
//en.wikipedia.org/wiki/Rodrigues%27_rotation_formula, 2025. [Online; accessed 11 May
2025].

[62] X. Yu, L. Tang, Y. Rao, T. Huang, J. Zhou, and J. Lu, “Point-bert: Pre-training 3d point cloud transformers
with masked point modeling,” in Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), 2022.

[63] A. X. Chang, T. Funkhouser, L. Guibas, P. Hanrahan, Q. Huang, Z. Li, S. Savarese, M. Savva, S. Song,
H. Su, J. Xiao, L. Yi, and F. Yu, “ShapeNet: An Information-Rich 3D Model Repository,” Tech. Rep.
arXiv:1512.03012 [cs.GR], Stanford University — Princeton University — Toyota Technological Institute
at Chicago, 2015.

22

[64] C. Kapfer, K. Stine, B. Narasimhan, C. Mentzel, and E. Candes, “Marlowe: Stanford’s gpu-based

computational instrument,” Jan. 2025.

[65] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny, “Vggt: Visual geometry grounded
transformer,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2025.

[66] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing
scenes as neural radiance fields for view synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106,
2021.

[67] A. Narayan, R. Nagar, and S. Raman, “Rgl-net: A recurrent graph learning framework for progressive part

assembly,” in WACV, 2022.

[68] G. Zhan, Q. Fan, K. Mo, L. Shao, B. Chen, L. J. Guibas, H. Dong, et al., “Generative 3d part assembly via

dynamic graph learning,” NeurIPS, 2020.

[69] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information

processing systems, vol. 33, pp. 6840–6851, 2020.

[70] C. Davis and W. M. Kahan, “The rotation of eigenvectors by a perturbation. iii,” SIAM Journal on

Numerical Analysis, vol. 7, no. 1, pp. 1–46, 1970.

23

