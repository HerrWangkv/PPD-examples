Variational Rectified Flow Matching

Pengsheng Guo Alexander G. Schwing

5
2
0
2

b
e
F
3
1

]

G
L
.
s
c
[

1
v
6
1
6
9
0
.
2
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

We study Variational Rectified Flow Match-
ing, a framework that enhances classic rectified
flow matching by modeling multi-modal velocity
vector-fields. At inference time, classic rectified
flow matching ‘moves’ samples from a source
distribution to the target distribution by solving
an ordinary differential equation via integration
along a velocity vector-field. At training time,
the velocity vector-field is learnt by linearly in-
terpolating between coupled samples one drawn
from the source and one drawn from the target dis-
tribution randomly. This leads to “ground-truth”
velocity vector-fields that point in different di-
rections at the same location, i.e., the velocity
vector-fields are multi-modal/ambiguous. How-
ever, since training uses a standard mean-squared-
error loss, the learnt velocity vector-field averages
“ground-truth” directions and isn’t multi-modal.
In contrast, variational rectified flow matching
learns and samples from multi-modal flow di-
rections. We show on synthetic data, MNIST,
CIFAR-10, and ImageNet that variational recti-
fied flow matching leads to compelling results.

1. Introduction

Diffusion models (Ho et al., 2020; Song et al., 2021a;b)
and flow matching (Liu et al., 2023; Lipman et al., 2023;
Albergo & Vanden-Eijnden, 2023; Albergo et al., 2023)
have been remarkably successful in recent years. These
techniques have been applied across domains from computer
vision (Ho et al., 2020) and robotics (Kapelyukh et al., 2023)
to computational biology (Guo et al., 2024) and medical
imaging (Song et al., 2022).

Flow matching (Lipman et al., 2023; Liu et al., 2023; Al-
bergo & Vanden-Eijnden, 2023) can be viewed as a continu-
ous time generalization of classic diffusion models (Albergo
et al., 2023; Ma et al., 2024). Those in turn can be viewed
as a variant of a hierarchical variational auto-encoder (Luo,

Correspondence

to:
sheng guo@apple.com>.

Pengsheng

Guo

<peng-

1

2022). At inference time, flow matching ‘moves’ a sample
from a source distribution to the target distribution by solv-
ing an ordinary differential equation via integration along
a velocity vector-field. To learn this velocity vector-field,
at training time, flow matching regresses to a constructed
vector-field/flow connecting any sample from the source
distribution — think of the data-domain positioned at time
zero — to any sample from the target distribution attained
at time one. Notably, in a ‘rectified flow,’ the samples
from the source and target distribution are connected via a
straight line as shown in Figure 1(a). Inevitably, this leads
to multi-modality/ambiguity, i.e., flows pointing in different
directions at the same location in the data-domain-time-
domain, as illustrated for a one-dimensional data-domain in
Figure 1(a). Since classic rectified flow matching employs
a standard squared-norm loss to compare the predicted ve-
locity vector-field to the constructed velocity vector-field, it
does not capture this multi-modality. Hence, rectified flow
matching aims to match the source and target distribution in
alternative ways. This is illustrated in Figure 1(b).

To enable rectified flow matching to capture this multi-
modality in the data-domain-time-domain, we study varia-
tional rectified flow matching. Intuitively, variational recti-
fied flow matching introduces a latent variable that permits
to disentangle multi-modal/ambiguous flow directions at
each location in the data-domain-time-domain. This ap-
proach follows the classic variational inference paradigm
underlying expectation maximization or variational auto-
encoders. Indeed, as shown in Figure 1(c), variational recti-
fied flow matching permits to model flow trajectories that
intersect. This leads to learned trajectories that more closely
align with the ground truth flow. The latent variable can
also be used to disentangle different directions.

Note that flow matching, diffusion models, and variational
auto-encoders are all able to capture multi-modality in the
data-domain, as one expects from a generative model. Im-
portantly, variational rectified flow matching differs in that it
also models multi-modality in the data-domain-time-domain.
This enables different flow directions at the same data-
domain-time-domain point, allowing the resulting flows
to intersect at that location.

We demonstrate the benefits of variational rectified flow
matching across various datasets and model architectures.

 
 
 
 
 
 
Variational Rectified Flow Matching

(a) Ground Truth

(b) Rectified FM (Baseline)
Figure 1: Intuition and motivation: Rectified flow matching randomly couples source data and target data samples, as
illustrated in panel (a). This leads to velocity vector-fields with ambiguous directions. Panel (b) shows that the classic
rectified flow matching averages ambiguous targets, which leads to curved flows. In contrast, the proposed variational
rectified flow matching is able to successfully model ambiguity which leads to less curved flows as depicted in panel (c).

(c) Variational Rectified FM (Ours)

On synthetic data, our method more accurately models data
distributions and better captures velocity ambiguity. On
MNIST, it enables controllable image generation with im-
proved quality. On CIFAR-10, our approach outperforms
classic rectified flow matching across different integration
steps. Lastly, on ImageNet, our method consistently im-
proves the FID score of SiT-XL (Ma et al., 2024).

In summary, our contribution is as follows: we study the
properties of variational rectified flow matching, and, along
the way, offer an alternative way to interpret the flow match-
ing procedure.

2. Preliminaries

Given a dataset D = {(x1)} consisting of data samples
x1, e.g., an image, generative models learn a distribution
p(x1), often by maximizing the likelihood. In the following
we discuss how this distribution is learnt with variational
auto-encoders and rectified flow matching, and why the
corresponding modeled data distribution is multi-modal.

2.1. Variational Auto-Encoders (VAEs)

Variational
inference generally and variational auto-
encoders (VAEs) (Kingma & Welling, 2014) specifically
have been shown to learn multi-modal distributions. This
is achieved by introducing a latent variable z. At inference
time, a latent z is obtained by sampling from the prior distri-
bution p(z), typically a zero mean unit covariance Gaussian.
A decoder which characterizes a distribution p(x1|z) over
the output space is then used to obtain an output space sam-
ple x1.

At training time, variational auto-encoders use an encoder
to compute an approximate posterior distribution qϕ(z|x1)
over the latent space. As the approximate posterior distri-
bution is only needed at training time, the data x1 can be
leveraged. Note, the approximate posterior distribution is
often a Gaussian with parameterized mean and covariance.
A sample from this approximate posterior distribution is
then used as input in the distribution pθ(x1|z) characterized

by the decoder. The loss encourages a high probability of
the output space samples while favoring an approximate
posterior distribution qϕ(z|x1, c) that is similar to the prior
distribution p(z). To achieve this, formally, VAEs maximize
a lower-bound on the log-likelihood, i.e.,

Ex1∼D log p(x1)
≥ Ex1∼D

(cid:2)Ez∼qϕ [log pθ(x1|z)] − DKL(qϕ(·|x1)|p(·))(cid:3) .

2.2. Rectified Flow Matching

For flow matching, at inference time, a source distribution
p0(x0) is queried to obtain a sample x0. This is akin to sam-
pling of a latent variable from the prior in VAEs. Different
from VAEs which perform a single forward pass through the
decoder, in flow matching, the source distribution sample
x0 is used as the boundary condition for an ordinary differ-
ential equation (ODE). This ODE is ‘solved’ by pushing
the sample x0 forward from time zero to time one via in-
tegration along a trajectory specified via a learned velocity
vector-field vθ(xt, t) defined at time t and location xt, and
commonly parameterized by deep net weights θ. Note, the
velocity vector-field is queried many times during integra-
tion. The likelihood of a data point x1 can be assessed via
the instantaneous change of variables formula (Chen et al.,
2018; Song et al., 2021b; Lipman et al., 2023),

log p1(x1) = log p0(x0) +

(cid:90) 0

1

div vθ(xt, t)dt,

(1)

which is commonly (Grathwohl et al., 2018) approximated
via the Skilling-Hutchinson trace estimator (Skilling, 1989;
Hutchinson, 1990). Here, div denotes the divergence vector
operator.

Intuitively, by pushing forward samples x0, randomly drawn
from the source distribution p0(x0), ambiguity in the data
domain is captured as one expects from a generative model.

At training time the parametric velocity vector-field vθ(xt, t)
needs to be learnt. For this, coupled sample pairs (x0, x1)
are constructed by randomly drawing from the source and
the target distribution, often independently from each other.

2

Variational Rectified Flow Matching

A coupled sample (x0, x1) and a time t ∈ [0, 1] is then
used to compute a time-dependent location xt at time t
via a function ϕ(x0, x1, t) = xt. Recall, rectified flow
matching uses xt = ϕ(x0, x1, t) = (1 − t)x0 + tx1. In-
terpreting xt as a location, intuitively, the “ground-truth”
velocity vector-field v(x0, x1, t) is readily available via
v(x0, x1, t) = ∂ϕ(x0, x1, t)/∂t, and can be used as the
target to learn the parametric velocity vector-field vθ(xt, t).
Concretely, flow matching learns the parametric velocity
vector field vθ(xt, t) by matching the target via an ℓ2 loss,
i.e., by minimizing w.r.t. trainable parameters θ the objective

Et,x0,x1

(cid:2)∥vθ(xt, t) − v(x0, x1, t)∥2

2

(cid:3) .

Consider two different couplings that lead to different
“ground-truth” velocity vectors at the same data-domain-
time-domain point (xt, t). The parametric velocity vector-
field vθ(xt, t) is then asked to match/regress to a differ-
ent target given the same input (xt, t). This leads to av-
eraging and the optimal functional velocity vector-field
v∗(xt, t) = E{(x0,x1,t):ϕ(x0,x1,t)=xt} [v(x0, x1, t)]. Hence,
multi-modality in the data-domain-time-domain is not cap-
tured. In the following we discuss and study a method that
is able to model this multi-modality.

3. Variational Rectified Flow Matching

Our goal is to capture the multi-modality inherent in
“ground-truth” velocity vector-fields obtained from typically
used couplings (x0, x1) that connect source distribution
samples x0 ∼ p0 with target data samples x1 ∈ D. Here, p0
is a known source distribution and D is a considered dataset.
This differs from classic rectified flow matching which does
not capture this multi-modality even for simple distributions
as shown in Figure 1 and as discussed in Section 2. The
struggle to capture multi-modality leads to velocity vec-
tor fields that may be more curve and consequently more
difficult to integrate at inference time. In turn, this leads
to distributions that may not fit the data as well. We will
show evidence for both, more difficult integration and less
accurately captured data distributions in Section 4.

To achieve our goal we combine rectified flow matching and
variational auto-encoders. In the following we first discuss
the objective before detailing training and inference.

3.1. Objective

The goal of flow matching is to learn a velocity vector-
field vθ(xt, t) that transports samples from a known source
distribution p0 at time t = 0 to samples from a commonly
unknown probability density function p1(x1) at time t = 1.
The probability densities p0, p1 and the velocity vector-field
vθ are related to each other via the transport problem

∂ log pt(xt)
∂t

= − div vθ(xt, t),

(2)

or its integral form given in Equation (1).

Solving the partial differential equation given in Equa-
tion (2) in general analytically is challenging, even when
assuming availability of the probability density functions,
i.e., when addressing a classic boundary value problem.

However, if we assume the probability density functions to
be Gaussians and if we restrict the velocity vector-field to be
constant, i.e., of the simple parametric form vθ(xt, t) = θ,
we can obtain an analytic solution. This is expressed in the
following claim:
Claim 1. Consider two Gaussian probability density func-
tions ˜p0 = N (ξ0; x0, I) and ˜p1 = N (ξ1; x1, I) with mean
x0 and x1 respectively. Assume a constant velocity vector-
field vθ(ξt, t) = θ. Then θ = x1 − x0 solves the partial
differential equation given in Equation (2) and its integral
form given in Equation (1) and xt = (1 − t)x0 + tx1.

Proof: Given the constant velocity vector-field vθ(ξt, t) =
θ, we have (cid:82) 0
1 div vθ(ξt, t)dt ≡ 0. Plugging this and
both probability density functions into Equation (1) yields
(ξ0 − x0)2 − (ξ1 − x1)2 ≡ 0 ∀ξ0, ξ1. Using ξ1 = ξ0 +
(cid:82) 1
0 vθ(ξt, t)dt = ξ0+θ leads to (ξ0−x0)2−(ξ0−x1+θ)2 ≡
0 ∀ξ0 which is equivalent to (x1 − x0 − θ)(2ξ0 − x0 − x1 +
θ) ≡ 0 ∀ξ0. This can only be satisfied ∀ξ0 if θ = x1 − x0,
leading to xt = x0 + tθ = (1 − t)x0 + tx1, which proves
■
the claim.

The arguably very simple setup in Claim 1 provides intuition
for the objective of classic rectified flow matching and offers
an alternative way to interpret the flow matching procedure.
Specifically, instead of two Gaussian probability density
functions ˜p0 and ˜p1, we assume the real probability den-
sity functions for the source and target data are composed
of Gaussians centered at given data points x0 and x1 re-
spectively, e.g., p0(ξ0) = (cid:80)
x0∈S N (ξ0; x0, I)/|S|. More-
over, importantly, let us assume that the velocity vector-field
vθ(xt, t) at a data-domain-time-domain location (xt, t) is
characterized by a uni-modal standard Gaussian

p(v|xt, t) = N (v; vθ(xt, t), I)

with a parametric mean vθ(xt, t). Maximizing the log-
likelihood of the empirical “velocity data” is equivalent
to the following objective

Et,x0,x1 [log p(x1 − x0|xt, t)]
∝ −Et,x0,x1

(cid:2)∥vθ(xt, t) − x1 + x0∥2

2

(3)

(cid:3) .

Note that this objective is identical to classic rectified flow
matching. Moreover, note our use of the standard rectified
flow velocity vector-field, also derived in Claim 1.

This derivation highlights a key point: because the vector
field is parameterized via a Gaussian at each data-domain-
time-domain location, multi-modality cannot be captured:

3

Variational Rectified Flow Matching

the Gaussian distribution is uni-modal. Hence, classic recti-
fied flow matching averages the “ground-truth” velocities.

As mentioned before, this can be sub-optimal. To capture
multi-modality, we study the use of a mixture model over
velocities at each data-domain-time-domain location. For
this, we assume an unobserved continuous random variable
z, drawn from a prior distribution p(z), governs the mean of
the conditional distribution of the velocity vector-field, i.e.,

p(v|xt, t, z) = N (v; vθ(xt, t, z), I).

Note, this model captures multi-modality as p(v|xt, t) =
(cid:82) p(v|xt, t, z)p(z)dz is a Gaussian mixture.
We now derive the variational flow matching objective.
Since the random variable z is not observed, at training
time, we introduce a recognition model qϕ(z|x0, x1, xt, t)
a.k.a. an encoder. It is parameterized by ϕ and approximates
the intractable true posterior.

Using this setup, the marginal likelihood of an individual
data point can be lower-bounded by

log p(v|xt, t) ≥ Ez∼qϕ [log p(v|xt, t, z)]

− DKL(qϕ(·|x0, x1, xt, t)|p(·)).

(4)

Replacing the log-probability of the Gaussian in the deriva-
tion of Equation (3) with the lower bound given in Equa-
tion (4) immediately leads to the variational rectified flow
matching objective Et,x0,x1 [log p(x1 − x0|xt, t)] ≥
(cid:2)∥vθ(xt, t, z) − x1 + x0∥2
(cid:3)

Et,x0,x1 [−Ez∼qϕ
− DKL(qϕ(·|x0, x1, xt, t)|p(·))].

(5)

2

We note that this objective could be extended in a number of
ways: for instance, the prior p(z) could be a trainable deep
net conditioned on x0 and/or t. Note however that this leads
to a more complex optimization problem with a moving
target. We leave a study of extensions to future work.

In Appendix A, we provide a theoretical proof demonstrat-
ing that the distribution learned by the variational objective
preserves the marginal data distribution, as previously estab-
lished for classic rectified flow matching (Liu et al., 2023).

In the following we first discuss optimization of this objec-
tive before detailing the inference procedure.

3.2. Training

To optimize the objective given in Equation (5),
we follow the classic VAE setup.
Specifically,
the prior p(z) = N (z; 0, I) and we let
we let
=
the
N (z; µϕ(x0, x1, xt, t), σϕ(x0, x1, xt, t)).
This enables
analytic computation of the KL-divergence in Equation (5).
Note that the mean of the approximate posterior is obtained

qϕ(z|x0, x1, xt, t)

approximate

posterior

Algorithm 1 Variational Rectified Flow Matching Training
Data: source distribution p0 and target sample dataset D
while stopping conditions not satisfied do

sample x0 ∼ p0, x1 ∈ D {we use a mini-batch}
sample t ∼ U (0, 1) {different t for each mini-batch
sample}
xt = (1 − t)x0 + tx1
get latent z = µϕ(x0, x1, xt, t) + ϵσϕ(x0, x1, xt, t)
with ϵ ∼ N (0, 1) {reparameterization trick}
compute loss following Equation (5)
perform gradient update on θ, ϕ

end while

Algorithm 2 Variational Rectified Flow Matching Inference

Data: source distribution p0
sample x0 ∼ p0
get latent z ∼ p(z)
ODE integrate x0 from t = 0 to t = 1 using velocity
vector-field vθ(xt, t, z)

from the deep net µϕ(x0, x1, xt, t) and the standard
deviation is obtained from σϕ(x0, x1, xt, t). Further, we
use the re-parameterization trick to enable optimization
of the objective w.r.t. the trainable parameters θ and
ϕ. Moreover, we use a single-sample estimate for the
expectation over the unobserved variable z. We summarize
the training procedure in Algorithm 1. Note, it’s more
effective to work with a mini-batch of samples rather than a
single data point, which was merely used for readability in
Algorithm 1.

Note that variational rectified flow matching training differs
from training of classic rectified flow matching in only a
single step: computation of a latent sample z in Algorithm 1.
From a computational point of view we add a deep net
forward pass to obtain the mean µϕ and standard deviation
σϕ of the approximate posterior, and a backward pass to
obtain the gradient w.r.t. ϕ. Also note that the velocity
vector-field architecture vθ(xt, t, z) might be more complex
as the latent variable z needs to be considered. However, the
additional amount of computation is likely not prohibitive.

We provide implementation details for the deep nets
vθ(xt, t, z), µϕ(x0, x1, xt, t), and σϕ(x0, x1, xt, t) in Sec-
tion 4, as their architecture depends on the data.

3.3. Inference

We summarize the inference procedure in Algorithm 2. Note
that we sample a latent variable only once prior to classic
ODE integration of a random sample x0 ∼ p0 drawn from
the source distribution p0. To obtain the latent z we sample
from the prior z ∼ p(z) = N (z; 0, I). Subsequently, we
ODE integrate the velocity field vθ(xt, t, z) from time t = 0
to time t = 1 starting from a random sample x0 drawn from
the source distribution.

4

Variational Rectified Flow Matching

Figure 2: Quantitative evaluation on synthetic 1D data for
varying evaluation steps. Metrics are averaged over three
runs. For True and Parzen Window Log-Likelihood, higher
values are better.

4. Experiments

We evaluate the efficacy of variational rectified flow match-
ing and compare to the classic rectified flow (Lipman et al.,
2023; Liu et al., 2023; Albergo & Vanden-Eijnden, 2023)
across multiple datasets and model architectures. Our exper-
iments show that variational rectified flow matching is able
to capture the multi-modal velocity in the data-domain-time-
domain, leading to compelling evaluation results. Moreover,
we demonstrate that explicitly modeling multi-modality
through a conditional latent z can enhance the interpretabil-
ity of flow matching models, leading to controllability. Im-
plementation details for all experiments are provided in
Appendix D.

4.1. Synthetic 1D Data

For synthetic 1D experiments, the source distribution is a
zero-mean, unit-variance Gaussian, while the target distri-
bution is bimodal, with modes centered at −1.0 and 1.0.

For the rectified flow baseline, we use a multi-layer MLP
network vθ to model the velocity. The network operates on
inputs xt and t and predicts the velocity through a series
of MLP layers. We follow this structure in our variational
rectified flow matching, but add an encoding layer for the
latent variable z. The posterior model qϕ follows a similar
design as vθ, outputting µϕ and σϕ. At inference time,
qϕ isn’t used. Instead, we sample directly from the prior
distribution p(z) = N (z; 0, I). The KL loss weight is 1.0.

We assess the performance using the Euler ODE solver and
vary the evaluation steps. Results are presented in Figure 2.
Across both metrics, i.e., True Log-Likelihood and Parzen
Window Log-Likelihood, and most evaluation steps, our
method outperforms the baseline. Notably, as the model
handles multi-modality in the data-domain-time-domain, it
produces reasonable results even for 2 or 5 evaluation steps.
Qualitative visualizations of flow trajectories are provided
in Appendix C.3.

To better understand the multi-modality of the velocity and
to assess the efficacy of our model in handling it, we ran-
domly sample different trajectories and plot the velocity
range standard deviation across predefined bins in the data-

(a)

(b)

(c)

Figure 3: 1D velocity ambiguity analysis with various con-
ditioning options and sampling strategies. (a) Ground Truth,
(b) Baseline (Rectified Flow), (c) Ours (Variational Rectified
Flow) . The heatmap illustrates the velocity standard devia-
tion for sampled bins in data-domain-time-domain, along
with histograms of the velocity at four sampled locations.
Our method effectively models velocity ambiguity, while
the baseline produces deterministic outputs.

domain-time-domain, as shown in Figure 3. The ground-
truth flow in Figure 3(a) shows that the standard deviation
increases with time, peaking at (x = 0.0, t = 0.75). The ve-
locity distribution transitions from a bi-modal distribution at
early times t to a uni-modal distribution at later times t. Fig-
ure 3(b) shows that the rectified flow baseline, which uses an
MSE loss, fails to model the velocity distribution faithfully,
collapsing to a Dirac-delta distribution as expected. In con-
trast, Figure 3(c) demonstrates that our model captures the
distribution with higher velocity standard deviation range,
matching the ground-truth reasonably, albeit not perfectly.
The complete ablation study on various conditioning options
is provided in Appendix C.2.

4.2. Synthetic 2D Data

We further test efficacy using synthetic 2D data. Following
Liu et al. (2023), we model the source distribution as a mix-
ture of Gaussian components positioned at six equidistant
points along a circle with a radius of 1/3, shown in red in
Figure 4(a). The target distribution follows a similar struc-
ture, but with components arranged along a larger circle
with a radius of 1, shown in purple.

For the architecture we follow Section 4.1 and condition the
posterior on [x0, x1, xt]. We report the True Log-Likelihood
and the Parzen Window Log-Likelihood for various evalu-

5

Variational Rectified Flow Matching

(a) Ground Truth

(b) Rectified FM

(c) Variational Rectified FM (Ours)

Figure 4: Flow visualization for synthetic 2D data using the Euler solver with 20 function evaluations. Sampled points from
the source distribution are shown in red, and points from the target distribution in purple. Different from Rectified FM,
which predicts flow trajectories with sharp curvature and U-turns to avoid crossings, our model captures velocity ambiguity
and predicts flows that intersect.

input to qϕ and set the KL loss weight to 1e−3.

Following Kingma & Welling (2014), we set the latent vari-
able z to be 2-dimensional. During inference, we sample
linearly spaced coordinates on the unit square, transforming
them through the inverse CDF of the Gaussian to generate
latents z. Using these latents, we integrate the samples with
an ODE solver and plot the generated samples in Figure 6.
To show the effects of the source distribution sample x0
and the latent z, we visualize the learned MNIST manifold
for two randomly sampled x0 values in Figure 6(a,b). The
results demonstrate that the latent space z enables smooth in-
terpolation between different digits within the 2D manifold,
providing control over the generated images. By adjusting z,
we can transition between various shapes and styles. The ini-
tial noise x0 enhances the generation process by introducing
additional variations in character styles, allowing the model
to better capture the target data distribution. Our method
achieves better FID scores than the baseline, even with a
2-dimensional conditional latent space. Further details and
results can be found in Appendix C.4.

4.4. CIFAR-10

Next, we evaluate on CIFAR-10, a widely used benchmark
in prior work (Lipman et al., 2023; Tong et al., 2024).
For a fair comparison, we use the architecture and train-
ing paradigm of Tong et al. (2024), but train the UNet
model with the variational rectified flow loss detailed in
Equation (5). The UNet consists of downsampling and
upsampling residual blocks with skip connections, and a
self-attention block added after the residual block at 16 × 16
resolution and in the middle bottleneck layer. The model
takes both xt and t as input, with the time embedding t used
to regress learnable scale and shift parameters γ and β for
adaptive group norm layers.

The posterior model qϕ shares a similar encoder structure
as vθ: image space inputs are chosen from [x0, x1, xt] and
concatenated along the channel dimension, while time t

Figure 5: Quantitative evaluation on synthetic 2D data for
varying evaluation steps. Metrics are averaged over three
runs with different random seeds.

(a) x0
0

(b) x1
0

Figure 6: Visualization of learned MNIST manifold with
different random noise x0.

ation steps of the Euler ODE solver, as shown in Figure 5.
Compared to the 1D data, our model shows a more signif-
icant performance boost here. We hypothesize that this is
due to the increased complexity of the task: explicitly mod-
eling multi-modality avoids the need for curved trajectories,
making it easier to fit the target distribution. The qualitative
flow visualization in Figure 4 supports this hypothesis: the
rectified flow requires a U-turn to avoid collisions, while our
model, aided by the variational training objective, moves in
trajectories that intersect and aren’t as curved.

4.3. MNIST

Modeling multi-modality also enables more explicit control
without additional conditioning signals. To show this we
use variational rectified flow matching to train a vanilla
convolutional net with residual blocks (He et al., 2015) on
MNIST data (LeCun et al., 1998). We use (x0, x1, xt) as

6

Variational Rectified Flow Matching

NFE / sample

# Params.

2

5

10

50

100

1000 Adaptive

OT-FM (Lipman et al., 2023)
I-CFM (Tong et al., 2024)

36.5M 166.655 36.188 14.396 5.557 4.640 3.822
36.5M 168.654 35.489 13.788 5.288 4.461 3.643

V-RFM (adaptive norm, x1, 2e-3)
V-RFM (adaptive norm, x1, 5e-3)

1
2
3 V-RFM (adaptive norm, x1 + t, 5e-3)
4 V-RFM (bottleneck sum, x1 + t, 2e-3)

37.2M 135.275 28.912 13.226 5.382 4.430 3.642
37.2M 159.940 35.293 14.061 5.265 4.349 3.582
37.2M 117.666 27.464 13.632 5.512 4.484 3.614
37.0M 104.634 25.841 13.508 5.618 4.540 3.596

3.655
3.659

3.545
3.561
3.478
3.520

Table 1: Following Tong et al. (2024), we train the same UNet model and reported the FID scores for our method and the
baselines using both fixed-step Euler and adaptive-step Dopri5 ODE solvers. The baselines checkpoint was directly taken
from Tong et al. (2024). We present four model variants of our V-RFM, which differ in fusion mechanism, posterior model
input, and KL loss weight.

is conditioned using adaptive group normalization. The
network predicts µϕ and σϕ with dimensions 1 × 1 × 768.
During training, the conditional latent z is sampled from
the predicted posterior, and at test time, from a standard
Gaussian prior. The latent is processed through two MLP
layers and serves as a conditional signal for the velocity
network vθ. We identify two effective approaches as con-
ditioning mechanisms: adaptive normalization, where z is
added to the time embedding before computing shift and
offset parameters, and bottleneck sum, which fuses the la-
tent with intermediate activations at the lowest resolution
using a weighted sum before upsampling.

We evaluate results using FID scores computed for varying
numbers of function evaluations, as shown in Table 1. Four
model variants were tested, differing in fusion mechanisms,
posterior model qϕ inputs, and KL loss weighting. Com-
pared to prior work (Lipman et al., 2023; Liu et al., 2023;
Tong et al., 2024), model 1 achieves superior FID scores
with fewer function evaluations and performs comparably at
higher evaluations. Using the adaptive Dopri5 solver further
improves scores, highlighting the importance of capturing
flow ambiguity. Model 2 increases the KL loss weight,
improving performance at higher function evaluations but
reducing effectiveness at lower evaluations, likely due to
reduced information from latent z. Model 3, with additional
time conditioning, significantly improves FID at low eval-
uations and performs best with the adaptive solver. Model
4, incorporating bottleneck sum fusion, delivers robust FID
scores across evaluation settings, demonstrating the flexibil-
ity of the variational rectified flow objective with different
fusion strategies.

Similar to the MNIST results in Section 4.3, we observe
clear patterns in color and content for the generated samples
x1, demonstrating a degree of controllability. Figure 7 visu-
alizes three sets of images (a)–(c). Each set is conditioned
on a different latent z, while the starting noise x0 varies
across individual images within each set. The same noise
x0 is applied to images at the same grid location across all
subplots. Images conditioned on the same latent exhibit con-
sistent color patterns, while images at the same grid location
display similar content, as highlighted in the last row.

(a) z0

(b) z1
Figure 7: Varying x0 while keeping the latent z fixed. Im-
ages at the same position across panels share the same x0,
while images within a panel share the same latent sampled
from the prior distribution.

(c) z2

4.5. ImageNet

To assess efficacy on large-scale data, we use ImageNet
256 × 256 data and SiT-XL (Ma et al., 2024), a recent
transformer-based model that has shown strong results in
image generation. For a fair comparison, we strictly follow
the original training recipe in the open-source SiT repository
and replicate the training process from the SiT paper, while
introducing our model, V-SiT-XL, by substituting the classic
rectified flow loss with the variational rectified flow loss in
Equation (5). The posterior model qϕ also utilizes an SiT
transformer architecture but with half the number of blocks.
In the final layer, the features are average pooled and passed
through an MLP layer to predict µϕ and σϕ. We sample the
latent variable z from the posterior during training and from
the prior distribution during inference. This latent variable
z is then processed by two MLP layers and fused with the
velocity network vθ via adaptive normalization. By default,
we use the Euler-Maruyama sampler with the SDE solver
and 250 integration steps, as described by Ma et al. (2024).

Following the evaluation protocol of Ma et al. (2024), we
randomly generate 50K images from the models and report
the FID scores in Table 2. V-SiT-XL consistently outper-
forms both DiT-XL and SiT-XL, achieving gains under the
same training conditions, with and without classifier-free
guidance. These results underscore the importance of mod-
eling multi-modality in the velocity vector field, which con-
tributes to a substantial improvement in generation quality,
particularly in the large-scale high-resolution data domain.
Additionally, we analyze the model’s performance across
different training iterations and varying numbers of function

7

Model

Params (M) Training Steps FID ↓

Variational Rectified Flow Matching

DiT-XL
SiT-XL
V-SiT-XL

SiT-XL
V-SiT-XL

SiT-XLcfg=1.5
V-SiT-XLcfg=1.5

SiT-XLcfg=1.5
V-SiT-XLcfg=1.5

675
675
677

675
677

675
677

675
677

400K
400K
400K

800K
800K

400K
400K

800K
800K

19.5
17.2
14.6

13.1
10.6

5.40
4.91

3.43
3.22

Table 2: FID-50K score evaluation of class-conditional gen-
eration on ImageNet 256 × 256, comparing the baselines
(DiT-XL, SiT-XL) with our proposed model V-SiT-XL.

evaluations, presenting the findings in Figure 8. The results
reveal a consistent performance boost, further highlighting
the effectiveness of our approach.

5. Related Work

Generative modeling has advanced significantly in the last
decade, thanks in part due to seminal works like genera-
tive adversarial nets (Goodfellow et al., 2014), variational
auto-encoders (Kingma & Welling, 2014), and normalizing
flows (Rezende & Mohamed, 2015).

More recently, score matching (Song & Ermon, 2019) and
diffusion models (Ho et al., 2020) were introduced. They
can be viewed as augmenting variational auto-encoders hi-
erarchically (Luo, 2022) while restricting involved distribu-
tions to be Gaussian. Notably, and analogously to classic
discrete normalizing flows, the number of hierarchy levels,
i.e., the number of time steps, remained discrete, which
introduced complications.

Flow matching (Lipman et al., 2023) was introduced re-
cently as a compelling alternative to avoid some of these
complications. It formulates an ordinary differential equa-
tion (ODE) in continuous time. This ODE connects a source
distribution to a target distribution. Solving the ODE via
forward integration through time permits to obtain samples
from the target distribution, essentially by ‘moving’ samples
from the known source distribution to the target time along
a learned velocity field.

To learn the velocity field, various mechanisms to interpolate
between the source distribution and the target distribution
have been considered (Lipman et al., 2023; Liu et al., 2023;
Tong et al., 2024). Rectified flow matching emerged as
a compelling variant, which linearly interpolates between
samples from the two distributions. For instance, it was used
to attain impressive results on large scale data (Ma et al.,
2024; Esser et al., 2024). Different from other techniques,
linear interpolation encourages somewhat straight flows,
which simplifies numerical solving of the ODE.

The importance of straight flows was further studied in

8

Figure 8: FID-50K score over training iterations and number
of function evaluations. Our model, V-SiT-XL, consistently
achieves a better FID score compared to SiT-XL trained
with classic rectified flow matching.

ReFlow (Liu et al., 2023), which sequentially formulates
multiple ODEs and learns velocity fields by adjusting the
interpolations and ‘re-training.’ Consistency models (Song
et al., 2023; Kim et al., 2023; Yang et al., 2024) strive
for straight flows by modifying the loss to encourage self-
consistency across timesteps. More details and comparisons
are provided in Appendix B and C.1.

While the aforementioned works aim to establish straight
flows either via ‘re-training’ or ‘re-parameterizing’ of an
already existing flow, differently, in this work we study the
results of enabling a rectified flow to capture the ambiguity
inherent in the usually employed ground-truth flow fields.

Structurally similar to this idea is work by Preechakul et al.
(2022). In a first stage, an autoencoder is trained to com-
press images into a latent space. The resulting latents then
serve as a conditioning signal for diffusion model training
in a second stage. Note, this two-stage approach doesn’t
directly model ambiguity in the data-domain-time-domain.
In similar spirit is work by Pandey et al. (2022). A VAE and
a diffusion model are trained in two separate stages, with the
goal to enable controllability of diffusion models. Related is
also work by Eijkelboom et al. (2024) which focuses on flow
matching only for categorical data, achieving compelling
results on graph generation tasks.

Concurrently, Zhang et al. (2025) also study a method to
model multi-modal velocity vector fields. In this paper, we
discuss how to use a lower-dimensional latent space to en-
able modeling of the velocity distribution via a variational
approach. Differently, Zhang et al. (2025) study use of a
hierarchy of ordinary differential equations. The variational
approach enables to capture semantics while use of a hi-
erarchy of ordinary differential equations permits to more
accurately model the velocity distribution.

6. Conclusion

We study Variational Rectified Flow Matching, a framework
which enables to model the multi-modal velocity vector
fields induced by the ground-truth linear interpolation be-
tween source and target distribution samples. Encouraging
results can be obtained on low-dimensional synthetic and
high-dimensional image data.

200K300K400K500K600K700K800KTraining Steps10.012.515.017.520.022.525.0FID-50KSiT-XL (Baseline)V-SiT-XL (Ours)50100150200250Number of Function Evaluations1015202530FID-50KSiT-XL (Baseline)V-SiT-XL (Ours)Variational Rectified Flow Matching

References

Albergo, M. and Vanden-Eijnden, E. Building normalizing
flows with stochastic interpolants. In Proc. ICLR, 2023.

Albergo, M., Boffi, N., and Vanden-Eijnden, E. Stochastic
Interpolants: A unifying framework for flows and diffu-
sions. In arXiv preprint arXiv:2303.08797, 2023.

Chen, R., Rubanova, Y., Bettencourt, J., and Duvenaud, D.
Neural ordinary differential equations. In Proc. NeurIPS,
2018.

Eijkelboom, F., Bartosh, G., Naesseth, C., Welling, M., and
van de Meent, J.-W. Variational Flow Matching for Graph
Generation. In arXiv preprint arXiv:2406.04843, 2024.

Esser, P., Kulal, S., Blattmann, A., Entezari, R., M¨uller,
J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F.,
Podell, D., Dockhorn, T., English, Z., Lacey, K., Good-
win, A., Marek, Y., and Rombach, R. Scaling rectified
flow transformers for high-resolution image synthesis. In
Proc. ICML, 2024.

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B.,
Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y.
Generative adversarial nets. In Proc. NeurIPS, 2014.

Grathwohl, W., Chen, R., Bettencourt, J., Sutskever, I., and
Duvenaud, D. FFJORD: Free-form continuous dynamics
for scalable reversible generative models. In Proc. ICLR,
2018.

Guo, Z., Liu, J., Wang, Y., Chen, M., Wang, D., Xu, D.,
and Cheng, J. Diffusion models in bioinformatics and
computational biology. Nature reviews bioengineering,
2024.

He, K., Zhang, X., Ren, S., and Sun, J.

Residual Learning for
https://arxiv.org/abs/1512.03385, 2015.

Image Recognition.

Deep
In

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion proba-

bilistic models. In Proc. NeurIPS, 2020.

Hutchinson, M. A stochastic estimator of the trace of the
influence matrix for Laplacian smoothing splines. Com-
munications in Statistics-Simulation and Computation,
1990.

Kapelyukh, I., Vosylius, V., and Johns, E. Dall-e-bot: In-
troducing web-scale diffusion models to robotics. IEEE
Robotics and Automation Letters, 2023.

Kim, D., Sony, A., Lai, C.-H., Liao, W.-H., Murata, N.,
Takida, Y., He, Y., Mitsufuji, Y., and Ermon, S. Consis-
tency trajectory models: Learning probability flow ode
trajectory of diffusion. In Proc. NeurIPS, 2023.

Kingma, D. and Welling, M. Auto-Encoding Variational

Bayes. In Proc. ICLR, 2014.

LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. Gradient-
based learning applied to document recognition. Proceed-
ings of the IEEE, 1998.

Lipman, Y., Chen, R., Ben-Hamu, H., Nickel, M., and Le,
M. Flow Matching for Generative Modeling. In Proc.
ICLR, 2023.

Liu, X., Gong, C., and Liu, Q. Flow straight and fast:
Learning to generate and transfer data with rectified flow.
In Proc. ICLR, 2023.

Luo, C. Understanding diffusion models: A unified perspec-

tive. In arXiv preprint arXiv:2208.11970, 2022.

Ma, N., Goldstein, M., Albergo, M., Boffi, N., Vanden-
Eijnden, E., and Xie, S. SiT: Exploring Flow and
Diffusion-based Generative Models with Scalable Inter-
polant Transformers. In arXiv preprint arXiv:2401.08740,
2024.

Nguyen, B., Nguyen, B., and Nguyen, V. A. Bellman opti-
mal stepsize straightening of flow-matching models. In
The Twelfth International Conference on Learning Repre-
sentations, 2024.

Pandey, K., Mukherjee, A., Rai, P., and Kumar, A. Dif-
fuseVAE: Efficient, controllable and high-fidelity gener-
ation from low-dimensional latents. In arXiv preprint
arXiv:2201.00308, 2022.

Preechakul, K., Chatthee, N., Wizadwongsa, S., and Suwa-
janakorn, S. Diffusion autoencoders: Toward a meaning-
ful and decodable representation. In Proc. CVPR, 2022.

Rezende, D. and Mohamed, S. Variational inference with

normalizing flows. In Proc. ICML, 2015.

Skilling, J. The eigenvalues of mega-dimensional matrices.

Maximum Entropy and Bayesian Methods, 1989.

Song, J., Meng, C., and Ermon, S. Denoising diffusion

implicit models. In Proc. ICLR, 2021a.

Song, Y. and Ermon, S. Generative modeling by estimating
gradients of the data distribution. In Proc. NeurIPS, 2019.

Song, Y., Sohl-Dickstein, J., Kingma, D., Kumar, A., Ermon,
S., and Poole, B. Score-Based Generative Modeling
Through Stochastic Differential Equations. In Proc. ICLR,
2021b.

Song, Y., Shen, L., Xing, L., and Ermon, S. Solving inverse
problems in medical imaging with score-based generative
models. In Proc. ICLR, 2022.

9

Variational Rectified Flow Matching

Song, Y., Dhariwal, P., Chen, M., and Sutskever, I. Consis-
tency models. In International Conference on Machine
Learning, pp. 32211–32252. PMLR, 2023.

Tong, A., Fatras, K., Malkin, N., Huguet, G., Zhang, Y.,
Improv-
Rector-Brooks, J., Wolf, G., and Bengio, Y.
ing and generalizing flow-based generative models with
minibatch optimal transport. TMLR, 2024.

Yan, H., Liu, X., Pan, J., Liew, J. H., Liu, Q., and Feng,
J. Perflow: Piecewise rectified flow as universal plug-
and-play accelerator. arXiv preprint arXiv:2405.07510,
2024.

Yang, L., Zhang, Z., Zhang, Z., Liu, X., Xu, M., Zhang,
W., Meng, C., Ermon, S., and Cui, B. Consistency flow
matching: Defining straight flows with velocity consis-
tency. arXiv preprint arXiv:2407.02398, 2024.

Zhang, Y., Yan, Y., Schwing, A., and Zhao, Z. Towards

Hierarchical Rectified Flow. In Proc. ICLR, 2025.

10

Appendix: Variational Rectified Flow Matching

Variational Rectified Flow Matching

This appendix is structured as follows: in Appendix A we show that our approach maintains the marginal distribution; in
Appendix B we discuss additional related work; in Appendix C we provide additional experimental analysis; in Appendix D
we provide more implementation details; in Appendix E we list additional qualitative results.

A. On Preserving the Marginal Data Distribution

We obtain samples by numerically solving the ordinary differential equation

dut = vθ(xt, t, z)dt with z ∼ p(z) = N (z; 0, I).

This differs slightly from Theorem 3.3 of Liu et al. (2023) because the velocity vθ depends on a latent variable z drawn from
a standard Gaussian. However, Theorem 3.3 of Liu et al. (2023) can be extended to fit this setting as follows.
First, note that we have v∗(xt, t, z) = E[ ˙Xt|Xt, Z] where Xt and Z are random variables corresponding to instances xt
and z.

Incorporating the velocity field depending on the latent variable z into the transport problem defined in Equation (2) and
taking an expectation over the latent variable, we obtain the continuity equation

˙pt + div(EZ[vθ(xt, t, z)]pt) = 0.

(6)

Following Liu et al. (2023), one can show equivalence to the following equality, which uses any compactly supported
continuously differentiable test function h:

d
dt

E[h(Xt)] = E[∇h(Xt)T ˙Xt] = E[∇h(Xt)T v∗(Xt, t)] = EX [∇h(Xt)T EZ[v∗(Xt, t, Z)]].

Concretely, equivalence can be shown via

0 = EZ

(cid:18)(cid:90)

xt

h( ˙pt + div(v∗(Xt, t, Z)pt)

(cid:19)

=

d
dt

E[h(Xt)] − EX [∇h(Xt)T EZ[v∗(Xt, t, Z)]].

Note, different from Liu et al. (2023), in our case Ut is driven by a velocity field v(xt, t, z) that depends on a latent variable.
Averaging over instantiations of the random latent variable Z leads to the same marginal velocity that appears in the
continuity equation (Equation (6)). Therefore, we solve the same equation with the same initial condition (X0 = U0).
Equivalence follows if the solution to Equation (6) is unique.

B. Additional Related Work Discussion

Here, we discuss related work aimed at improving the sample efficiency of diffu-
sion and flow matching models, either via consistency modeling or via distillation.

Consistency models. Consistency models, such as those by Song et al. (2023) and
Yang et al. (2024), enforce self-consistency across timesteps, ensuring trajectories
map back to the same initial point. Moreover, Kim et al. (2023) ensure consistent
trajectories for probability flow ODEs. While consistency models focus on im-
proving results via trajectory alignment if few function evaluations are used, they
don’t model the multi-modal ground-truth velocity distribution, which is our goal.

To illustrate this, we train the recently developed consistency flow matching model
proposed by Yang et al. (2024) (which improves upon work by Song et al. (2023)
and Kim et al. (2023); both are not flow matching based; it also improves upon
distillation work by Nguyen et al. (2024)) on the data for which V-RFM results
are presented in Figures 3 and 14. Specifically, we used the publicly available
baseline.1 We obtain the results illustrated in Figure 9. As expected, we observe
that classic consistency modeling does not capture the multi-modal velocity distribution, unlike the proposed V-RFM.

Figure 9: Velocity distribution of con-
sistency flow matching (Yang et al.,
2024).

1https://github.com/YangLing0818/consistency flow matching

11

Variational Rectified Flow Matching

Figure 10: Additional quantitative evaluation with the consistency flow matching baseline on synthetic 1D data. Higher
values are better for True and Parzen Window Log-Likelihood, while lower values are preferred for Wasserstein Distance.

Figure 11: Additional quantita-
tive evaluation with the consis-
tency flow matching baseline on
synthetic 2D data. Metrics are av-
eraged over three runs with differ-
ent random seeds.

Furthermore, we conduct additional experiments with consistency flow matching across multiple datasets, summarizing the
results in Appendix C.1. We observe that the consistency flow matching method performs well in the low function evaluation
regime (i.e., NFE = 2 or 5), but its performance degrades as the NFEs increase. Most notably, its best performance across all
NFEs does not surpass that of classic rectified flow matching or our proposed variational rectified flow matching. Based
on the empirical evidence and the key differences in capturing multi-modal velocity distributions, we believe consistency
models are orthogonal to our proposed variational formulation. Therefore, we find it exciting to explore future research on
combining variational flow matching with consistency models, which is beyond the scope of this paper.

Distillation. Nguyen et al. (2024) perform distillation by optimizing step sizes in pretrained flow-matching models to refine
trajectories and improve training dynamics. Moreover, Yan et al. (2024) perform distillation by introduceing a piecewise
rectified flow mechanism to accelerate flow-based generative models. Note, both methods distill useful information from
a pretrained model, either by using dynamic programming to optimize the step size or by applying reflow to straighten
trajectories, i.e., they focus on distilling already learned models. In contrast, our V-RFM focuses on learning via single-stage
training, directly from ground-truth data, and without use of a pre-trained deep net, a flow-matching model, which captures
a multi-modal velocity distribution. More research on the distillation of a V-RFM model is required to assess how multi-
modality can be maintained in the second distillation step. We think this is exciting future research, which is beyond the
scope of this paper.

C. Additional Experimental Results and Analysis

C.1. Comparison to Consistency Flow Matching

We conduct additional experiments to compare our approach with consistency models across multiple datasets. For this,
we use the recently developed consistency flow matching model from Yang et al. (2024) as a representative baseline, as it
advances earlier consistency modeling efforts by Song et al. (2023); Kim et al. (2023) and distillation work by Nguyen et al.
(2024). Specifically, we used the publicly available implementation.2

The results are summarized as follows: Synthetic 1D data in Figure 10, Synthetic 2D data in Figure 11, MNIST data in
Figure 12, and CIFAR-10 data in Table 3. These results demonstrate that V-RFM outperforms the consistency flow matching
baseline across various evaluation steps for synthetic data, with V-RFM showing superior performance when the number of
evaluation steps exceeds 2 for MNIST and 5 for CIFAR-10. Importantly, while consistency flow matching achieves strong
performance for a low number of evaluation steps, its best performance still does not surpass that of classic rectified flow
matching or our proposed variational rectified flow matching with a high number of evaluation steps. This highlights its
distinct nature as an orthogonal research direction to our method. As discussed in Appendix B, we believe that combining
variational formulations with consistency models presents an exciting avenue for future research, though it is beyond the
scope of this paper.

2https://github.com/YangLing0818/consistency flow matching

12

258101620Evaluation Steps43210True Log-LikelihoodRectified FMConsistency FMVRFM (Ours)258101620Evaluation Steps43210Parzen WindowLog-LikelihoodRectified FMConsistency FMVRFM (Ours)258101620Evaluation Steps102101100Wasserstein DistanceRectified FMConsistency FMVRFM (Ours)2458101620Evaluation Steps10.07.55.02.50.0True Log-LikelihoodRectified FMConsistency FMVRFM (Ours)2458101620Evaluation Steps10.07.55.02.50.0Parzen WindowLog-LikelihoodRectified FMConsistency FMVRFM (Ours)Variational Rectified Flow Matching

Figure 12: FID score evaluation for the MNIST experiment,
including the additional consistency flow matching baseline.
Our model with a latent dimension of 2 outperforms the base-
lines, except at 2 evaluation steps where Consistency FM
performs best. Note, the latent dimension of 2 is chosen for
a controllability analysis rather than being optimized for FID
score improvement.

NFE / sample

# Params.

2

5

10

50

100

1000 Adaptive

OT-FM
(Lipman et al., 2023; Tong et al., 2024)
I-CFM
(Liu et al., 2023; Tong et al., 2024)

36.5M 166.655 36.188 14.396

5.557

4.640

3.822

3.655

36.5M 168.654 35.489 13.788

5.288

4.461

3.643

3.659

Consistency-FM
(Yang et al., 2024)
Consistency-FM-XL
(Yang et al., 2024)

36.5M

15.758

14.588 24.107 36.800 38.675 40.486

40.711

61.8M

5.323

11.412 23.948 36.652 38.680 40.402

40.677

V-RFM (adaptive norm, x1, 2e-3)
V-RFM (adaptive norm, x1, 5e-3)

1
2
3 V-RFM (adaptive norm, x1 + t, 5e-3)
4 V-RFM (bottleneck sum, x1 + t, 2e-3)

37.2M 135.275 28.912 13.226
37.2M 159.940 35.293 14.061
37.2M 117.666 27.464 13.632
37.0M 104.634 25.841 13.508

5.382
5.265
5.512
5.618

4.430
4.349
4.484
4.540

3.642
3.582
3.614
3.596

3.545
3.561
3.478
3.520

Table 3: Additional quantitative evaluation of the consistency flow matching baseline on CIFAR-10. The consistency flow
matching method performs well in the low function evaluation regime (NFE = 2 or 5), but its performance degrades as NFEs
increase. Notably, its best performance across all NFEs does not surpass that of classic rectified flow matching (OT-FM,
I-CFM) or our proposed variational rectified flow matching (V-RFM).

C.2. 1D Velocity Ambiguity Analysis

As discussed in Section 3.2, the posterior qϕ can be conditioned in different ways. To understand the implications, we
performed ablation studies and visualize the velocity distribution maps in Figure 13 (c)-(f). For x0 conditioning (d),
the model struggles to predict the bi-modal distribution at early timesteps (xt = 0.0, t = 0.0) due to the absence of x1
information. However, when t is sufficiently large, the model can infer x1 from xt, enabling it to predict a bi-modal
distribution again at (x = 0.0, t = 0.5). Conversely, with x1 conditioning (e), the model fails to capture the ground-truth
distribution at later timesteps (x = −1.0, t = 0.95) as the influence of x1 diminishes. With xt conditioning (f), the ambiguity
plot follows the baseline as no extra data is provided to the posterior.

C.3. Qualitative Results of Synthetic 1D Experiment

We provide qualitative flow visualizations from the synthetic 1D experiment in Figure 14. Our method effectively captures
velocity ambiguity and predicts crossing flows, whereas the baselines produce deterministic outputs.

C.4. Quantitative Results of MNIST Experiment

We evaluate the FID scores of our method using this 2-dimensional conditional latent space and report the results in Figure 12.
Despite the small latent dimension, it still enables the velocity model vθ to achieve better FID scores than the baselines,
except at 2 evaluation steps where consistency flow matching (Yang et al., 2024) performs best.

C.5. Inception Score Evaluation of CIFAR-10 Experiment

We evaluate the Inception Score of our model trained on CIFAR-10 data and present results in Table 4. This score quantifies
the distribution of predicted labels for the generated samples. Compared to the vanilla rectified flow baseline, our method
consistently achieves higher Inception Scores, reflecting improved diversity in the generated samples.

13

101102Evaluation Steps0255075100FIDRectified FMConsistency FMVRFM (Ours)Variational Rectified Flow Matching

(a)

(b)

(c)

(d)

(e)

(f)

Figure 13: 1D velocity ambiguity analysis with various conditioning options and sampling strategies. (a) Ground Truth (GT),
(b) Baseline (Rectified Flow), (c) Ours (x0 + x1 + xt), (d) Ours (x0), (e) Ours (x1), (f) Ours (xt). The heatmap illustrates
the velocity standard deviation for sampled bins in data-domain-time-domain, along with histograms of the velocity at four
sampled locations. Our method effectively models velocity ambiguity, while the baseline produces deterministic outputs.

C.6. Ablation on Posterior Model Size

We conducted ablations to study the impact of varying the size of the encoder qϕ, reducing it to 6.7% and 17.5% of its
original size. The results reported in Table 5 demonstrate that our model maintains comparable performance across these
variations, highlighting the flexibility and robustness of our approach.

C.7. Reconstruction Loss Visualizations

We present the reconstruction loss curves for our model and the baseline trained on MNIST and CIFAR-10 data in Figure 15.
We observe better reconstruction losses of our model compared to vanilla rectified flow, indicating that the predicted
velocities more accurately approximate the ground-truth velocities.

D. Implementation Details

D.1. Synthetic data

In the rectified flow baseline, the velocity network vθ features separate encoders for time t and data x. Each encoder
consists of a sinusoidal positional encoding layer followed by two MLP layers with GeLU activation. The resulting time and
data embeddings are concatenated and passed into a four-layer MLP, also utilizing GeLU activations. Both the positional
embedding and hidden dimensions of the encoder and decoder are set to 64. The training batch size is 1000, and we employ
the standard rectified flow objective, i.e., we compute the current data via xt = (1 − t)x0 + tx1, the ground truth velocity
via v(x0, x1, t) = x1 − x0, and we use the L2 loss for supervision.

For consistency flow matching, we adopt the same velocity network vθ and modify the loss function to incorporate the

14

Variational Rectified Flow Matching

(a) Ground Truth

(b) Rectified FM

(c) Consistency FM

(d) Ours (x0)

(e) Ours (x1)

(f) Ours (xt)

(g) Ours (x0 + x1 + xt)

Figure 14: 1D flow visualization for uni-modal Gaussian to bi-modal Gaussian.

velocity consistency loss proposed by Yang et al. (2024). We find the hyperparameter settings suggested by the publicly
available codebase to work best. Specifically, we use ∆t = 1 × 10−3, Nsegments = 2, and boundary = 0.0 for the first
training stage, transitioning to boundary = 0.9 in the second stage. Additionally, the loss weighting factor α is set to
1 × 10−5. For complete implementation details, we kindly direct readers to the open-source repository which we used to
obtain the reported results.3

In both cases, the AdamW optimizer is used with the default weight decay and a learning rate of 1 × 10−3, over a total of
20,000 training iterations.

In our variational flow matching approach, the velocity network vθ incorporates an additional latent encoding module
comprising three MLP layers with a hidden dimension of 128. The conditional latent embedding z is concatenated with the
embeddings for time t and data x. The decoder maintains the same structure as the baseline, with the first MLP layer adjusted
to accommodate the increased channel input. For the posterior model qϕ, we employ a similar architecture, designing
a separate encoder for each possible input selected from [x0, x1, xt, t]. Each encoder consists of a sinusoidal positional
encoder layer followed by two MLP layers with GeLU activation. The output embeddings are concatenated along the
channel dimension and processed through three MLP layers to produce the predicted µϕ and σϕ. The latent dimension of z
is set to 4 for 1D experiments and 8 for 2D experiments. During training, we utilize the reparameterization trick to sample z
from the predicted posterior distribution; during inference, the posterior model qϕ is omitted, and sampling is performed
from a unit variance Gaussian prior distribution. The loss is defined as the sum of the rectified flow reconstruction loss and

3https://github.com/YangLing0818/consistency flow matching

15

Variational Rectified Flow Matching

NFE / sample

2

5

10

50

100

1000 Adaptive

I-CFM
(Liu et al., 2023; Tong et al., 2024)

2.786 7.143 8.326 8.770 8.872 9.022

9.041

V-RFM (adaptive norm, x1, 2e-3)
V-RFM (adaptive norm, x1, 5e-3)

1
3.943 7.728 8.499 8.973 9.050 9.168
2
3.083 7.202 8.342 8.868 8.997 9.166
3 V-RFM (adaptive norm, x1 + t, 5e-3)
4.460 7.930 8.583 9.007 9.104 9.220
3 V-RFM (bottleneck sum, x1 + t, 2e-3) 4.831 7.996 8.529 9.062 9.150 9.293

9.171
9.183
9.238
9.308

Table 4: Inception Score evaluation of our method compared to the baseline on CIFAR-10, using fixed-step Euler and
adaptive-step Dopri5 ODE solvers. Higher scores indicate better performance.

NFE / sample

2

5

10

50

100

1000 Adaptive

OT-FM
(Lipman et al., 2023; Tong et al., 2024)
I-CFM
(Liu et al., 2023; Tong et al., 2024)

166.655 36.188 14.396 5.557 4.640 3.822

3.655

168.654 35.489 13.788 5.288 4.461 3.643

3.659

1
2
3

V-RFM-L (100% Posterior Model)
V-RFM-M (17.5% Posterior Model)
V-RFM-S (6.7% Posterior Model)

135.275 28.912 13.226 5.382 4.430 3.642
135.983 30.106 13.783 5.486 4.500 3.697
144.676 31.224 13.406 5.289 4.398 3.699

3.545
3.607
3.639

Table 5: We use the same flow matching model vθ and pair it with different sizes of encoders qϕ during training while
maintaining the exact same hyper-parameters. We report the FID scores for our method and the baseline using both fixed-step
Euler and adaptive-step Dopri5 ODE solvers.

the KL divergence loss, with the KL loss weighted at 1.0 for the 1D experiments and 0.1 for the 2D experiments. We employ
AdamW as the optimizer with a learning rate of 1 × 10−3 and train the two networks qϕ and vθ jointly for 20,000 iterations.

D.2. MNIST

In the rectified flow baseline, the velocity network vθ uses separate encoders for time t and data x. The time t encoder
consists of a sinusoidal positional encoding layer followed by two MLP layers with SiLU activation. The data x encoder
includes a convolutional in-projection layer, five consecutive ResNet (He et al., 2015) blocks (each consisting of two
convolutional layers with a kernel size of 3, group normalization, and SiLU activation), followed by a convolutional
out-projection layer. The time and data embeddings are concatenated and passed to a decoder composed of a convolutional
in-projection layer, five consecutive ResNet blocks, and a convolutional out-projection layer with a kernel size of 1 and an
output channel of 1. The hidden dimension is set to 64. MNIST data is normalized to the [−1, 1] range. We adopted the
consistency velocity loss from the consistency flow matching baseline used for synthetic data experiments. We train the
network for 100,000 iterations using the AdamW optimizer with a learning rate of 1 × 10−3 and batch size of 256.

In our variational flow matching approach, the velocity network vθ includes an additional latent encoding module consisting
of a sinusoidal positional encoding layer followed by two MLP layers with SiLU activation. The conditional latent
embedding z is concatenated with the embeddings for time t and data x. The decoder structure mirrors the baseline, with the
first in-projection layer adjusted to handle the increased channel input. The posterior model qϕ follows a similar architecture,
with separate encoders for each input [x0, x1, xt]. The resulting embeddings are concatenated and passed through a decoder
consisting of a convolutional in-projection layer, followed by three consecutive interleaving ResNet blocks and average
pooling layers. The final hidden activation is flattened and processed by two linear MLP layers to predict the 1D latent
z with a dimension of 2. The two networks are trained jointly for 100,000 iterations using the AdamW optimizer with a
learning rate of 1 × 10−3 and a batch size of 256. The KL loss weight is set to 1 × 10−3.

D.3. CIFAR-10

For the rectified flow baseline, we directly use the OT-FM and I-CFM models from (Tong et al., 2024) and evaluate their
performance under different NFEs. For the consistency flow matching model, we take the public implementation from
(Yang et al., 2024) and integrate the consistency loss into the same I-CFM model, naming it Consistency-FM. Additionally,
we evaluate the original model from (Yang et al., 2024) with a larger parameter count, referring to it as Consistency-FM-XL.

For our V-RFM model variants, we adopt the I-CFM model from (Tong et al., 2024) and add modules to incorporate

16

Variational Rectified Flow Matching

Figure 15: Reconstruction loss for MNIST (left) and CIFAR-10 (right). We observe lower reconstruction losses for the
variational formulation, indicating a better fit.

conditional signals from a 1D latent z. For both conditioning mechanisms discussed in Section 4.4, the sampled latent is
processed through two MLP layers with SiLU activation, with both hidden and output dimensions set to 512.

In the adaptive norm variant, the latent embedding z is combined with the time embedding from vθ to regress the learnable
scale and shift parameters γ and β for the adaptive group norm layers. For the bottleneck sum variant, the latent is added to
the bottleneck feature of vθ. Since the lowest spatial resolution of the baseline network is 4 × 4, the 1D latent is spatially
repeated and fused with the bottleneck feature via a weighted sum. To ensure effective use of the latent, we assign a
weighting of 0.9 to the latent and 0.1 to the original velocity feature.

The posterior model qϕ shares a similar encoder structure to vθ but omits the decoder. To achieve greater spatial compression,
we increase the number of downsampling blocks, predicting features at a 1 × 1 spatial resolution. The base channel size is
set to 16. Both networks are trained jointly for 600,000 iterations using the Adam optimizer with a learning rate of 2 × 10−4
and a batch size of 128. The KL loss weighting is presented alongside the results in Table 1.

D.4. ImageNet

We build upon the open-source SiT-XL model (Ma et al., 2024) by incorporating additional modules to integrate conditional
signals from the sampled 1D latent variable z. The sampled latent is processed through two MLP layers with SiLU
activation, with both the hidden and output dimensions set to 1152. The processed latent is then directly added to the original
conditional latent c, which contains timestep and class label information. The resulting conditional feature is used to predict
the learnable scale and shift parameters, γ and β, for the adaptive group normalization layers.

The posterior model qϕ shares the SiT-XL architecture but uses only half the number of transformer blocks. To achieve
greater spatial compression, we apply an average pooling layer to compress the latent representation into a 1D vector, which
is then processed by an MLP layer to predict µϕ and σϕ. The base channel size is set to 1152, the patch size to 2, and the
number of heads to 16. Both networks are trained jointly for 800,000 iterations using the AdamW optimizer with a learning
rate of 1 × 10−4 and a global batch size of 256. The KL loss weight is set to 2 × 10−3, and the posterior model qϕ takes x1
as input. To ensure a fair comparison, we strictly adhere to the original training recipe of SiT (Ma et al., 2024), i.e., we
don’t tune learning rate, decay or warm-up schedules, AdamW parameters, or employ additional data augmentation or
gradient clipping during training.

E. Qualitative Results

E.1. CIFAR-10

We present qualitative results of our model trained on CIFAR-10 data in Figure 16.

E.2. ImageNet

We present qualitative results of our model trained on ImageNet data in Figure 17.

17

020000400006000080000100000iterations1.4×1011.6×1011.8×1012×1012.2×101reconstruction lossMNISTBaselineOurs100000200000300000400000iterations1.55×1011.6×1011.65×1011.7×1011.75×101reconstruction lossCIFAR-10BaselineOursVariational Rectified Flow Matching

Figure 16: Randomly selected samples generated from our model trained on CIFAR-10 data.

18

Variational Rectified Flow Matching

Figure 17: Randomly selected samples generated from our model trained on ImageNet data.

19

