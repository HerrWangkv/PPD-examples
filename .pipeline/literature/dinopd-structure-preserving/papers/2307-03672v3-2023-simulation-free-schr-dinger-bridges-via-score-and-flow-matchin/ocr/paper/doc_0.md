Simulation-Free Schrödinger Bridges via Score and Flow Matching

4
2
0
2

r
a

M
1
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
2
7
6
3
0
.
7
0
3
2
:
v
i
X
r
a

Alexander Tong†
Mila – Québec AI Institute
Université de Montréal

Nikolay Malkin†
Mila – Québec AI Institute
Université de Montréal

Kilian Fatras†
Mila – Québec AI Institute
McGill University

Lazar Atanackovic
University of Toronto
Vector Institute

Yanlei Zhang
Mila – Québec AI Institute
Université de Montréal

Guillaume Huguet
Mila – Québec AI Institute
Université de Montréal

Guy Wolf
Mila – Québec AI Institute
Université de Montréal
Canada CIFAR AI Chair

Yoshua Bengio
Mila – Québec AI Institute
Université de Montréal
CIFAR Senior Fellow

Abstract

1

INTRODUCTION

We present simulation-free score and flow
matching ([SF]2M), a simulation-free objec-
tive for inferring stochastic dynamics given
unpaired samples drawn from arbitrary source
and target distributions. Our method general-
izes both the score-matching loss used in the
training of diffusion models and the recently
proposed flow matching loss used in the train-
ing of continuous normalizing flows. [SF]2M
interprets continuous-time stochastic genera-
tive modeling as a Schrödinger bridge prob-
lem.
It relies on static entropy-regularized
optimal transport, or a minibatch approx-
imation, to efficiently learn the SB with-
out simulating the learned stochastic pro-
cess. We find that [SF]2M is more efficient
and gives more accurate solutions to the
SB problem than simulation-based methods
from prior work. Finally, we apply [SF]2M
to the problem of learning cell dynamics
from snapshot data. Notably, [SF]2M is the
first method to accurately model cell dy-
namics in high dimensions and can recover
known gene regulatory networks from sim-
ulated data. Our code is available in the
TorchCFM package at https://github.com/
atong01/conditional-flow-matching.

† Equal contribution. Proceedings of the 27th International
Conference on Artificial Intelligence and Statistics (AIS-
TATS) 2024, Valencia, Spain. PMLR: Volume 238. Copy-
right 2024 by the author(s).

Score-based generative models (SBGMs), including
diffusion models, are a powerful class of generative
models that can represent complex distributions over
high-dimensional spaces (Sohl-Dickstein et al., 2015;
Song and Ermon, 2019; Ho et al., 2020; Nichol and
Dhariwal, 2021; Dhariwal and Nichol, 2021). SBGMs
typically generate samples by simulating the evolu-
tion of a source density – nearly always a Gaussian –
according to a stochastic differential equation (SDE)
(Song et al., 2021b). Despite their empirical success,
SBGMs are restricted by their assumption of a Gaus-
sian source, which is essential for optimization with the
simulation-free denoising objective. This assumption
is often violated in the temporal evolution of physical
or biological systems, such as in the case of single-cell
gene expression data, which prevents the use of SBGMs
for learning the underlying dynamics.

An approach of choice in such problems has been to use
flow-based generative models, synonymous with con-
tinuous normalizing flows (CNFs) (Chen et al., 2018;
Grathwohl et al., 2019; Finlay et al., 2020). Flow-based
models assume a deterministic continuous-time gener-
ative process and fit an ordinary differential equation
(ODE) that transforms the source density to the target
density. Flow-based models were previously limited
by inefficient simulation-based training objectives that
require an expensive integration of the ODE at training
time. However, recent work has introduced simulation-
free training objectives that make CNFs competitive
with SBGMs when a Gaussian source is assumed (Lip-
man et al., 2023; Liu et al., 2023b; Pooladian et al.,

 
 
 
 
 
 
Simulation-Free Schrödinger Bridges via Score and Flow Matching

Figure 1: Left: ODE and SDE paths from 8gaussians to moons, sampled from a model trained using [SF]2M. [SF]2M
makes it possible to vary the diffusion schedule at inference time and thus interpolate between ODEs and SDEs that have
the same marginal densities. Right: Illustration of the stochastic regression objective in [SF]2M. Given a source point
x0 and target point x1 sampled from an entropic OT plan between marginals, an intermediate point xt is sampled from
the Brownian bridge (marginal in light blue) in a simulation-free way. Neural networks are regressed to the ODE drift
u◦
t (xt|x0, x1) and to the conditional score ∇ log pt(xt|x0, x1). The regression objective is stochastic, as the same point
xt may appear on different conditional paths, e.g., the dotted path from x′
1. The stochastic regression recovers
dynamics that transform the marginal at time 0 to that at time 1.

0 to x′

2023) and extended these objectives to the case of ar-
bitrary source distributions (Liu, 2022; Albergo and
Vanden-Eijnden, 2023; Tong et al., 2024). However,
these objectives do not yet apply to learning stochastic
dynamics, which can be beneficial both for generative
modeling and for recovering the dynamics of systems.

The Schrödinger bridge (SB) problem – the canoni-
cal probabilistic formulation of stochastically mapping
between two arbitrary distributions – considers the
most likely evolution between a source and target prob-
ability distributions under a given reference process
(Schrödinger, 1932; Léonard, 2014b). The SB prob-
lem has been applied in a wide variety of problems,
including generative modeling (De Bortoli et al., 2021;
Vargas et al., 2021; Chen et al., 2022; Wang et al., 2021;
Song and Ermon, 2019), modeling natural stochastic
dynamical systems (Schiebinger et al., 2019; Holdijk
et al., 2022; Koshizuka and Sato, 2023), and mean field
games (Liu et al., 2022). Except for a small number
of special cases (e.g., Gaussian (Mallasto et al., 2022;
Bunne et al., 2022a)), the SB problem typically does
not have a closed-form solution, but can be approxi-
mated with iterative algorithms that require simulating
the learned stochastic process (De Bortoli et al., 2021;
Chen et al., 2022; Bunne et al., 2022a). While theo-
retically sound, these methods present numerical and
practical issues that limit their scalability to high di-
mensions (Shi et al., 2023).

This paper introduces a simulation-free objec-
tive for the Schrödinger bridge problem called
simulation-free score and flow matching ([SF]2M).
[SF]2M simultaneously generalizes (1) the simulation-
free objectives for CNFs (Tong et al., 2024; Liu et al.,

2023b) to the case of stochastic dynamics and (2) the
denoising training objective for diffusion models to the
case of arbitrary source distributions (Fig. 1). Our
algorithm uses a connection between the SB problem
and entropic optimal transport (OT) to express the
Schrödinger bridge as a mixture of Brownian bridges
(De Bortoli et al., 2021; Léonard, 2014b). In contrast
to dynamic SB algorithms that require simulating an
SDE at every iteration, [SF]2M takes advantage of
static entropic OT maps between source and target
distributions, which are efficiently computed by the
Sinkhorn algorithm (Sinkhorn, 1964).

We demonstrate the effectiveness of [SF]2M on both
synthetic and real-world datasets. On synthetic data,
we show that [SF]2M performs better than related
prior work and finds a better approximation to the
true Schrödinger bridge. As an application to real
data, we consider modeling sequences of cross-sectional
measurements (i.e., unpaired time series observations)
by a sequence of Schrödinger bridges. While there are
many prior methods on modeling cells with Schrödinger
bridges in the static setting (Schiebinger et al., 2019;
Huguet et al., 2022b; Lavenant et al., 2021; Nolan et al.,
2023) or low-dimensional dynamic setting (Bunne et al.,
2022b,a; Koshizuka and Sato, 2023), [SF]2M is the first
method able to scale to thousands of gene dimensions,
as its training is completely simulation-free. We also in-
troduce a static manifold geodesic map which improves
cell interpolations in the dynamic setting, demonstrat-
ing one of the first practical applications of Schrödinger
bridge approximations with non-Euclidean costs. Fi-
nally, we show that unlike with static optimal transport,
we are able to directly model and recover the gene-gene

xtu◦t(xt|x0,x1)∇logpt(xt|x0,x1)x0x1x′0x′1tA. Tong, N. Malkin, K. Fatras, L. Atanackovic, Y. Zhang, G. Huguet, G. Wolf, Y. Bengio

interaction network driving the cell dynamics.

entropic OT problem is defined as follows:

We summarize our main contributions below:
• We present [SF]2M, the first simulation-free objec-
tive for the Schrödinger bridge problem, and prove
its correctness.

• We study effective empirical and minibatch approx-
imations to the entropic OT plan used in [SF]2M.
• We validate our proposed method on synthetic distri-
butions and in several single-cell dynamics problems.

2 PRELIMINARIES

We consider a pair of compactly supported distributions
over Rd with (unknown) densities q(x0) and q(x1) (also
denoted q0, q1). We assume access to a finite dataset
of samples from q0 and q1. The problem of continuous-
time stochastic generative modeling, or SDE inference,
consists in finding a stochastic mapping f that trans-
forms q0 to q1. Samples from q1 can then be generated
by drawing a sample from q0 and applying f to obtain
a sample from q1.

2.1 Schrödinger bridges via entropic OT

The Schrödinger bridge problem asks to find most
likely evolution between two probability measures q0
and q1 with respect to a reference stochastic process
Q. Formally, the Schrödinger bridge is the solution of:

P⋆ = arg min

P:p0=q0,p1=q1

KL(P

Q),

∥

(1)

→

where P is a stochastic process (distribution over con-
tinuous paths [0, 1]
Rd) with law p (with marginals
denoted pt).
SDEs and diffusion processes. The stochastic pro-
cesses we consider can be represented as Itô SDEs
of the form dx = ut(x) dt + g(t) dwt, where ut is a
smooth vector field and dwt a Brownian motion. A
density p(x0) evolved according to a SDE induces
marginal distributions pt(xt), viewed as a function
R+. They are characterized by the
p : [0, 1]
initial conditions p0 and the Fokker-Planck equation
pt)
∂tpt =
is the Laplacian.

2 ∆pt, where ∆pt =

(ptut) + g2(t)

(
∇

−∇ ·

∇ ·

Rd

→

×

In this work, we consider Q = σW, where W is the
standard Brownian motion defined by the SDE dx =
dwt, in which case the solution to (1) is known as the
diffusion Schrödinger bridge (De Bortoli et al., 2021;
Bunne et al., 2022a). We refer to Léonard (2014a,b)
for a full discusssion on Schrödinger bridges.

Entropically-regularized optimal transport. The

π∗

ε(q0, q1) =
(cid:90)

arg min
π∈U (q0,q1)

d(x0, x1)2 dπ(x0, x1) + ε KL(π

q0
∥

⊗

q1),

(2)

where U (q0, q1) is the set of admissible transport plans
(joint distributions over x0 and x1 whose marginals
are equal to q0 and q1), d(
) is the ground cost, ε
,
·
·
q1 is the
is the regularization parameter, and q0
joint distribution over x0, x1 in which x0 and x1 are
independent. When ε
0, we recover exact optimal
transport. We now recall a cornerstone theorem that
connects the SB problem to the entropic OT plan:
Proposition 2.1 (Föllmer (1988)). Let the reference
process be a Brownian motion ( i.e., Q = σW). Then
the Schrödinger bridge problem admits a unique solution
P∗ having the form of a mixture of Brownian bridges
weighted by an entropic OT plan:

→

⊗

P∗((xt)t∈[0,1]) =

(cid:90)

Q((xt)t

|

x0, x1) dπ⋆

2σ2(x0, x1)

(3)

where Q((xt)t∈(0,1) |
between x0 and x1 with diffusion rate σ.

x0, x1) is the Brownian bridge

Motivated by this result, the algorithm we propose
stochastically regresses the parameters defining an un-
conditional SDE to those defining Brownian bridges.

2.2 Neural SDEs and probability flows

In this section, we consider an SDE dx = ut(x) dt +
g(t) dwt. We review some important properties and
discuss the approximation of ut by a neural network.
Score and flow parametrization. In the degenerate
0, an SDE becomes an ODE and the Fokker-
case g(t)
Planck equation recovers the continuity equation ∂p
∂t =
(ptut). From the Fokker-Plank and continuity

−∇ ·
equations, it can easily be derived that the ODE

≡

(cid:20)
ut(x)

dx =

−

g(t)2

(cid:124)

2 ∇
(cid:123)(cid:122)
t (x)

(cid:21)
log pt(x)

dt,

(4)

(cid:125)

u◦
together with a distribution over initial conditions
p(x0), induces the same marginal distributions pt(
)
·
as the SDE; therefore, (4) is called the probability flow
ODE of the stochastic process. Conversely, if the prob-
t (x), the diffusion schedule
ability flow ODE’s drift u◦
log pt(x) are known, then
), and the score function
g(
the SDE’s drift term can be recovered via

∇

·

ut(x) = u◦

t (x) +

g(t)2

2 ∇

log pt(x).

(5)

Therefore, specifying an SDE is tantamount to
specifying the probability flow ODE and its

Simulation-Free Schrödinger Bridges via Score and Flow Matching

in the
score function. By reversing the sign of u◦
t
ODE (4) and converting it to an SDE using (5), we also
get the time reversal formula from Anderson (1982):

dx =

(cid:20)

−

u◦
t (x) +

g(t)2

2 ∇

= (cid:2)

−

ut(x) + g(t)2

∇

(cid:21)
log pt(x)
dt + g(t) dwt
log pt(x)(cid:3) dt + g(t) dwt,

The Schrödinger bridge approximation algorithm we
will propose leverages fast solutions to the entropy-
regularized OT problem (2) and the closed-form u◦
t
and

log pt of Brownian bridges (8).

∇

(6)

3 SIMULATION-FREE SDE

TRAINING

∇

which induces the same distribution on x1−t as the
original SDE does on xt.
Approximating SDEs with neural networks. If
the marginal pt(x) can be tractably sampled and one
knows the probability flow ODE’s drift u◦
t (x) and
log pt(x), both can be approximated by neu-
score
ral networks. Specifically, time-varying vector fields
Rd
Rd and sθ(
vθ(
can be trained with the (unconditional) score and flow
matching objective
LU[SF]2M(θ) =
E(cid:2)
2
u◦
t (x)
vθ(t, x)
∥
−
(cid:125)
(cid:123)(cid:122)
flow matching loss

− ∇
(cid:123)(cid:122)
score matching loss

sθ(t, x)
∥
(cid:124)

) : [0, 1]
·

log pt(x)

+λ(t)2

) : [0, 1]

2
∥
(cid:125)

Rd

Rd

→

→

×

×

∥
(cid:124)

,

,

·

·

·

(7)
(cid:3),

∼ U

(0, 1), x

where the expectation is over t
pt(x) and
∼
) is some choice of positive weights. (In practice, it
λ(
·
can be more stable to approximate g(t)2
log pt(x)
log pt(x), a simple parametrization
rather than
change that does not change the learning problem or
the objective.) Once trained, vθ and sθ can be used
to simulate the SDE from source samples x0. This
procedure is described in Alg. 2.

∇

∇

·

Remarkably, with a separate parametrization of the
probability flow ODE and the score, we can simulate the
SDE at inference time with an arbitrary diffusion rate
) that need not match the one used at training time.
g(
If the global optimum of (7) is attained (Fig. 1), we are
ensured to obtain samples from the same marginals for
). For example, we can
any arbitrary diffusion rate g(
·
simulate the probability flow ODE by setting g(t)
0.
Similarly, the backward SDE can be simulated starting
at samples x1 using the time reversal formula (6).
ODEs and SDEs for Brownian bridges. For pro-
cesses whose marginals pt(x) are Gaussian, Theorem 3
of Lipman et al. (2023) or Theorem 2.1 of Tong et al.
(2024) yield expressions for the flow and score. The
main case of interest is the Brownian bridge from x0 to
x1 with constant diffusion rate g(t) = σ. The marginals
are given by pt(x) =
t)),
and the ODE and score are computed using the afore-
mentioned result:

t)x0, σ2t(1

(x; tx1 + (1

N

≡

−

−

u◦
t (x) =

1
t(1

2t
t)

−
−

log pt(x) =

∇

(x

(tx1 + (1

−
tx1 + (1

t)x0)) + (x1

x0),

−

−

−
σ2t(1

x

.

−

t)x0
t)

−

(8)

We next describe our simulation-free method to learn
SDEs through score and flow matching, summarized
in Alg. 1. We present the general case in §3.1, then
consider the Schrödinger bridge case in §3.2.

3.1 Matching the conditional flow and score

Tong et al. (2024) described a simulation-free stochastic
regression objective, conditional flow matching (CFM),
that fits an ODE generating marginal distributions
given by mixtures of simpler probability paths. We
generalize CFM to matching stochastic dynamics.

Suppose the stochastic process P((xt)t∈[0,1]), with
marginals pt(x), is a mixture over a latent variable
z with density q(z), i.e.,

P((xt)t∈[0,1]) =

(cid:90)

P((xt)
z)q(z) dz.
|

(9)

|

z) is defined by the SDE dx =
Suppose that P((xt)
z), and
z) dt+g(t) dwt with initial conditions p0(x
ut(x
|
|
let u◦
z) be drift of the corresponding probability
t (x
|
flow ODE given by (4). One then has expressions
for the probability flow ODE and score that generate
the marginals of the process P given initial conditions
p0(x) = (cid:82)

z)q(z) dz:

z p0(x
|

t (x) = Eq(z)
u◦

log pt(x) = Eq(z)

∇

z)
|

,

u◦
t (x

z)pt(x
|
pt(x)
(cid:20) pt(x
z)
|
pt(x) ∇

log pt(x

(cid:21)

.

z)
|

(10)

To be precise, we generalize Theorem 3.1 from Tong
et al. (2024) to stochastic settings:

Theorem 3.1. Under mild regularity conditions, the
t (x) dt generates the marginals pt of P
ODE dx = u◦
from initial conditions p0, and the score is given by (10).
The SDE dx = [u◦
2 g(t)2
log pt(x)] dx + g(t) dt
∇
generates the Markovization of P.

t (x) + 1

We emphasize that, in general, the SDE in Theorem 3.1
does not recover P, but only its Markovization (i.e., the
process with the same infinitesimal transition kernel).
A process P of the form (9) is not necessarily Markovian
and may not be generated by any SDE.

A stochastic regression objective. The marginal
ODE drift and score expressions in (10) motivate ob-
log pt(x) with neural
jectives for fitting u◦

t (x) and

∇

A. Tong, N. Malkin, K. Fatras, L. Atanackovic, Y. Zhang, G. Huguet, G. Wolf, Y. Bengio

Algorithm 1 Score and Flow Matching Training

z), com-

z), initial networks vθ and sθ.
|

Input: Efficiently samplable q(z), pt(x
|
putable ut(x
while Training do
z
q(z);
t
∼
∼ U
vθ(t, x)
L[SF]2M ← ∥
+ λ(t)2
θ

x
pt(x
∼
|
u◦
2
z)
t (x
∥
|
x log pt(x

∥
Update(θ,

sθ(t, x)
θ

(0, 1);

z)

z)

|

−
− ∇
L[SF]2M)

∇

←
return vθ, sθ

2 ▷ see (11)
||

networks when only the conditional ODEs and scores
are known. Generalizing (7), we define the (condi-
tional) simulation-free score and flow matching objec-
tive ([SF]2M) for neural networks vθ(
) approximating
the ODE drift and sθ(
L[SF]2M(θ) = EQ′

,
·
) approximating the score:
·
vθ(t, x)
∥
(cid:124)

−
(cid:123)(cid:122)
conditional flow matching loss

u◦
t (x

2
∥
(cid:125)

z)
|

(11)

,
·

·

+ EQ′λ(t)2

,
z)
sθ(t, x)
∥
∥
− ∇
(cid:125)
(cid:123)(cid:122)
(cid:124)
conditional score matching loss

log pt(x

|

2

|

⊗

⊗

∼ U

pt(x

) is some choice of positive weights
where, as in (7), λ(
·
and Q′ = (t
z). This objective
q(z)
(0, 1))
can be used to approximate the quantities defined in
(10), provided the conditional ODEs and scores are
known and pt(x
z) can be tractably sampled. Correct-
|
ness is guaranteed by the following Theorem:
Theorem 3.2 (Equality of conditional gradients).
[0, 1], then
If pt(x) > 0 for all x
∈
LU[SF]2M(θ) is
LU[SF]2M(θ) =

Rd and t
L[SF]2M(θ), where

∇
the unconditional score and flow matching loss (7).

∇

∈

θ

θ

This result generalizes Theorem 3.2 of Tong et al.
(2024). It provides a simulation-free way to train neu-
ral networks sufficient to simulate an SDE generating
marginals pt(x) with arbitrary diffusion rate g(
) (cf.
·
the discussion following (7)). The training and infer-
ence algorithms are summarized in Alg. 1, Alg. 2.

In our approach, the SDE recovered from the ODE
and score defined via (10) is the Markovization of the
mixture of stochastic processes indexed by z.

Sources of conditional ODEs and scores. Al-
though the [SF]2M framework can handle general con-
ditioning information z, in this paper we consider the
case where z is identified with a pair (x0, x1) of a source
and target point. For a given z = (x0, x1), we will take
the conditional probability path pt(x
z) to be a Brow-
|
nian bridge with constant diffusion scale σ, so that
z) are given by (8). To avoid
u◦
t (x
|
numerical issues for t close to 0 or 1, we add a small
smoothing constant to the variance. The conditional
distributions are thus peaky at x0 and x1 at t = 0 and
t = 1. (An extension to nonconstant diffusion scale is
described in Appendix E.)

z) and
|

log pt(x

∇

For the resulting marginal pt(x) to satisfy the boundary
conditions p0(x) = q0(x) and p1(x) = q1(x), q(x0, x1)
must be a coupling of q0 and q1 (i.e., a transport plan).
This is formalized in the following theorem:
Theorem 3.3 ([SF]2M recovers marginals from
bridges). If q(
θ globally min-
θ + 1
2 g(t)2s∗
imize
θ]
and diffusion g, and initial conditions p0 = q0, is the
Markovization of the mixture of Brownian bridges from
x0 to x1 over q(x0, x1). In particular, if the SDE gen-
erates marginals pt, then p1 = q1.

θ , s∗
L[SF]2M(θ), the SDE with drift [v∗

U (q0, q1) and v∗

)
·

,
·

∈

This theorem tells us that as long as our joint distri-
bution q(x0, x1) has the correct marginals, [SF]2M will
recover a valid generative model which pushes q0 to q1.

3.2 Building Schrödinger bridges via [SF]2M

and entropic optimal transport

In the previous section, we showed that our method,
[SF]2M, can approximate the marginal probability pt of
a mixture of processes of the form (9). In this section,
we explain how our [SF]2M approximates the SB.

[SF]2M approximates the Schrödinger bridge.
In order to achieve an efficient approximation of the
SB, we leverage Proposition 2.1. The SB can be ex-
pressed as a mixture of Brownian bridges weighted by
an entropic optimal transport plan (3). Therefore, to
approximate the SB with [SF]2M, we set the distri-
bution q(x0, x1) to be equal to the entropic OT plan
2σ2(q0, q1) and train the networks vθ and sθ using
π⋆
Alg. 1. We show that this procedure recovers the SB:
Proposition 3.4 ([SF]2M with entropic OT recovers
the SB process). Let P∗ be the Schrödinger bridge be-
tween q0 and q1 with respect to Q = σW. If v⋆
θ , s⋆
θ
2σ2 (q0, q1),
globally minimize
then P∗ is defined by the SDE with drift [v∗
2 g(t)2s∗
θ],
diffusion g, and initial conditions p0 = q0.

L[SF]2M, with coupling π⋆
θ + 1

Empirical approximation. Unfortunately, the real
distributions q0 and q1 are usually unknown and we
only have access to i.i.d. samples forming empirical
distributions ˆq0 and ˆq1 of size n. Therefore, we can only
approximate the true entropic OT plan by computing
the entropic OT plan π⋆
2σ2(ˆq0, ˆq1) between the empirical
distributions (Cuturi, 2013; Altschuler et al., 2017).
This empirical OT plan can be used in [SF]2M to
construct an empirical Schrödinger bridge.

Fortunately, the true entropic OT can be efficiently
approximated using empirical distributions, even in
high-dimensional spaces (Genevay et al., 2019; Mena
and Niles-Weed, 2019), and it was recently shown that
the Schrödinger bridge inherits this property (Stromme,
2023, Theorem 5). In turn, the entropic OT plan be-
tween empirical distributions can be efficiently com-

Simulation-Free Schrödinger Bridges via Score and Flow Matching

Table 1: Comparison of SB algorithms (see §6). [SF]2M is
the first algorithm that does not assume paired samples,
require SDE integration during training, or use an IPF
outer loop. DSBM uses simulation only in the outer loop.
Algorithm

I2SB/ASB DSB/NLSB DSBM/IDBM [SF]2M

→

Unpaired samples
Bridge matching
Single loop
Sim.-free training
No explicit (x0, x1) pairing

✗
✓
✓
✓
✓

✓
✗
✗
✗
✓

✓
✓
✗
✗/✓
✓

✓
✓
✓
✓
✗

O

puted using the Sinkhorn algorithm (Cuturi, 2013),
which has
(n2) computational complexity (Altschuler
et al., 2017), or using stochastic algorithms (Genevay
et al., 2016; Seguy et al., 2018). However, if this cost
is to high (e.g., if n is too large or if one has the true
generative process, as in the Gaussian-to-data setting),
the plan can be further approximated using minibatch
OT (Fatras et al., 2020, 2021a); see Appendix A.

The use of an entropic OT plan and marginalization
via stochastic regression distinguishes [SF]2M from
existing neural SB algorithms (Table 1). Such past
approaches include mean-matching (DSB and NLSB,
De Bortoli et al., 2021; Koshizuka and Sato, 2023), and
bridge-matching approaches (DSBM and IDBM, Shi
et al., 2023; Peluchetti, 2023), both of which require
an outer iterative proportional fitting loop with an
inner training loop. Others have studied the problem
assuming paired source and target data (I2SB and ASB,
Liu et al., 2023a; Somnath et al., 2023); SF2M can be
thought of as inferring the pairing while jointly fitting
the SDE.

See §C.3 for further discussion of the implications of
these choices and practical recommendations.

4 LEARNING CELL DYNAMICS

WITH [SF]2M

Modeling cell dynamics is a major open problem in
single-cell data science, as it is important for under-
standing – and eventually intervening in – cellular pro-
grams of development and disease (Lähnemann et al.,
2020). In this section, we show how [SF]2M can be
used for and adapted to modeling single-cell dynamics.

The cellular dynamics between time-resolved snap-
shot data, representing observations of cells lying in
the space of gene activations, are commonly modeled
using Schrödinger bridges (Hashimoto et al., 2016;
Schiebinger et al., 2019; Bunne et al., 2022a; Koshizuka
and Sato, 2023). The applicability of the SB formula-
tion to cell dynamics relies upon the principle of least
action, which is thought to hold for cellular systems
over short timescales (Schiebinger, 2021), and moti-
vates our choice to apply [SF]2M to these problems.

Figure 2: Visualization of learned Waddington’s landscape
W with a bifurcating trajectory for one Gaussian to two
Gaussians (left) and for the Embryoid Body (EB) data
(Moon et al., 2019) (right). The dimensions are space
(left-right), time (forward-back), and potential (up-down).

Learning flows on cell manifolds. Cells are thought
to lie on a low-dimensional manifold in the space of
gene expressions (Moon et al., 2018), which has moti-
vated work on density-adhering regularizations (Tong
et al., 2020; Koshizuka and Sato, 2023) and manifold
embeddings (Huguet et al., 2022a). Because [SF]2M
can use a coupling between marginals q(x0, x1) defined
by entropic OT with an arbitrary cost function, we
can take advantage of these embeddings to compute
the pairing using a ground cost that is adapted to
the geometry of the manifold. Specifically, we use
the Geodesic Sinkhorn method (Huguet et al., 2022b),
which computes the entropic OT plan with cost

cgeo(x0, x1) = (cid:112)

log

t(x0, x1).

(12)

−

H

H
t approximates the heat kernel defined
The matrix
via the Laplace-Beltrami operator on the manifold,
efficiently approximated using a k-nearest-neighbour
graph. We find using this cost leads to more accurate
trajectories in high dimensions (see Table 5).

Learning developmental landscapes. A common
model of cell development, known as Waddington’s epi-
genetic landscape (Waddington, 1942), assumes that
cells evolve and differentiate in the space of gene expres-
sions by following (noisy) gradient ascent on an energy
function. While a few heuristic methods have been
proposed to approximate this energy function from
single-cell data before (Tang, 2017; Qin et al., 2023),
we propose a novel approach to model the landscape
directly. In our approach, the negative energy can be
directly interpreted as an action potential, inspired by
the modeling in Neklyudov et al. (2023).

−∇

To do this, we impose a Langevin dynamics
parametrization on the flow and score in [SF]2M:
xEs(t, x),
xEv(t, x) and sθ(t, x) =
vθ(t, x) =
−∇
where Ev and Es are neural networks. We can define
2 g(t)2Es.
the Waddington’s landscape by W := Ev + 1
The drift of the SDE is then ut(x) =
xW (t, x),
meaning that the time-evolution of a cell follows gra-
dient dynamics on W with added Gaussian noise of

−∇

A. Tong, N. Malkin, K. Fatras, L. Atanackovic, Y. Zhang, G. Huguet, G. Wolf, Y. Bengio

Table 2: Two-dimensional data: generative modeling performance (W2) and dynamic OT optimality (NPE) divided into
SDE methods (top) and ODE methods (bottom). [SF]2M performs the best on 3 of 4 datasets and is similar to OT-CFM,
which is equivalent to [SF]2M as g(t) → 0. *Indicates results taken from Shi et al. (2023).
Metric

)

8gaussians moons

2 (

)
W
↓
8gaussians

→
Algorithm

| Dataset

↓

→

[SF]2M-Exact
[SF]2M-I
DSBM-IPF (Shi et al., 2023)*
DSBM-IMF (Shi et al., 2023)*
DSB (De Bortoli et al., 2021)*

OT-CFM (Tong et al., 2024)
SB-CFM (Tong et al., 2024)
RF (Liu, 2022)
I-CFM (Tong et al., 2024)
FM (Lipman et al., 2023)

N→
0.275±0.058
0.393±0.054
0.315±0.079
0.338±0.091
0.411±0.084
0.303±0.043
2.314±2.112
0.421±0.071
0.373±0.103
0.343±0.058

→

0.726±0.137
1.482±0.151
0.812±0.092
0.838±0.098
0.987±0.324

0.601±0.027
—
1.525±0.330
1.557±0.407
—

8gaussians moons

Normalized Path Energy (
↓
moons

8gaussians

moons

N→
0.124±0.023
0.185±0.028
0.140±0.006
0.144±0.024
0.190±0.049
0.130±0.016
0.434±0.594
0.283±0.045
0.178±0.014
0.209±0.055

scurve

N→
0.128±0.005
0.201±0.062
0.140±0.024
0.145±0.037
0.272±0.065 —
0.144±0.028
0.341±0.468
0.345±0.079
0.242±0.141
0.198±0.037

N→
0.016±0.012
0.160±0.019
0.022±0.020
0.029±0.017

0.031±0.027
1.000±0.000
0.044±0.031
0.202±0.055
0.190±0.054

→

0.045±0.031
2.577±0.323
0.244±0.027
0.345±0.049
—

N→
0.053±0.038
0.855±0.130
0.383±0.034
0.230±0.028
—

0.015±0.010
—
0.203±0.090
2.680±0.292
—

0.083±0.009
0.995±0.000
0.130±0.078
0.891±0.120
0.762±0.099

scurve
N→
0.034±0.024
0.845±0.106
0.297±0.036
0.286±0.033
—

0.027±0.012
0.745±0.039
0.099±0.066
0.856±0.031
0.743±0.116

scale g(t). We visualize these landscapes in Fig. 2 with
further details in §F.6.

Learning gene regulatory networks. Finally, we
use [SF]2M to learn gene regulatory networks from
population snapshots of gene expressions, a persisting
challenge in cellular biology (Pratapa et al., 2020). Fol-
lowing previous work in discovering sparse interaction
structure from continuous-time systems (Tank et al.,
2021; Aliee et al., 2021; Bellot and Branson, 2022; Aliee
et al., 2022; Atanackovic et al., 2023), we define the
gene regulatory network as the directed graph whose
vertices are genes (dimensions of the space) and an
j is present if and only if ∂(vθ(t,x))j
edge i
= 0. This
∂xi
directed graph is expected to be sparse.

→

Previous work resorted to performing inference of tra-
jectories in a low-dimensional (and dense) representa-
tion (Tong et al., 2023; Bunne et al., 2022b), which
complicated the discovery of the sparse graph structure
in gene space. [SF]2M is the first Schrödinger bridge
method to scale to high dimensions. This allows us to
learn a dynamic directly in the gene space and recover
the sparse gene interactions. To accomplish this, we use
a specialized parametrization of vθ, inspired by Bellot
and Branson (2022), which enables the graph structure
to be read out from the sparsity pattern of the initial
layer of the trained model (see §F.7 for details).

5 RELATED WORK

Stochastic continuous-time modeling. Our frame-
work is related to both flow-based (Chen et al., 2018;
Grathwohl et al., 2019; Albergo et al., 2023; Albergo
and Vanden-Eijnden, 2023; Neklyudov et al., 2022; Liu,
2022) and score-based (Sohl-Dickstein et al., 2015; Song
and Ermon, 2019, 2020; Song et al., 2021b; Ho et al.,
2020; Winkler et al., 2023; Dhariwal and Nichol, 2021;
Watson et al., 2023) generative modeling. Both have
drawn attention due to their stability and efficiency in
training and high quality of generated samples. See
Appendix C for further discussion.

Schrödinger bridge approximation methods.
While there is significant theoretical work on the SB
problem (Léonard, 2014b; Stromme, 2023; Albergo
et al., 2023), practical solutions have assumed paired
samples from the Schrödinger bridge or required simu-
lation during training. Algorithms based on iterative
proportional fitting (DSB and DSBM; De Bortoli et al.,
2021; Shi et al., 2023) have the advantage of yielding
the exact Schrödinger bridge if trained to optimality
on each iteration, but may accumulate error with each
outer-loop step due to underfitting and function ap-
proximation. On the other hand, our proposed [SF]2M
requires neither training-time integration nor outer-
loop iteration and therefore will converge to the exact
SB – if the neural network function class and learn-
ing algorithm so allow – but, unlike DSB and DSBM,
require knowledge of the entropic OT plan and the
conditional paths (see Table 1).

The relative advantages of these algorithms and prac-
tical recommendations are further discussed in §3.2,
§C.3. Furthermore, in Appendix D we show that the
simulation-based outer loop in Peluchetti (2023); Shi
et al. (2023) can be combined with [SF]2M to improve
the SB marginals at the cost of generative performance.

Applications to cell dynamics. When the observer
seeks to recover dynamics from multiple snapshots with
scRNA-seq data, the machinery of optimal transport
can be used (Schiebinger et al., 2019; Yang et al., 2020;
Tong et al., 2020; Bunne et al., 2022b; Huguet et al.,
2022a; Bunne et al., 2022a; Koshizuka and Sato, 2023).
However, these methods all require simulation during
training, which scales poorly to high dimensions.

6 EXPERIMENTS

In this section we empirically evaluate [SF]2M with
respect to optimal transport, generative modeling, and
single-cell interpolation criteria. We compare:

• Minibatch [SF]2M with exact OT minibatches
([SF]2M-Exact), with entropic OT (Sinkhorn) mini-

̸
Simulation-Free Schrödinger Bridges via Score and Flow Matching

Table 3: Gaussian-to-Gaussian Schrödinger bridges with 104
datapoints between a Gaussian with parameters estimated
from empirical samples (pt) with error to the continuous
Schrödinger bridge marginals (p∗
t ) either at the target dis-
tribution (left) or averaged across 21 timepoints (right).
Metric

KL(p1, p∗
1)
20

5

50

5

Mean KL(pt, p∗
t )
20

50

0.007±0.000
0.015±0.005
8.757±—

0.001±0.000

0.029±0.002
0.132±0.004
49.963±—
0.034±0.003

0.006±0.000
0.005±0.002

0.124±0.003
0.528±0.013
221.213±— 8.757±—
0.170±0.002

0.008±0.000

0.028±0.001
0.050±0.002
49.963±—
0.086±0.002

0.258±0.001
0.221±0.004
221.213±—
0.447±0.003

→
| Dim.

Alg.

↓

[SF]2M-Exact
DSBM-IPF
DSB

→

SB-CFM

Table 4: Single-cell comparison over three datasets, aver-
aged over leaving out different intermediate timepoints on 5
PCs. For each left-out point, we measure the 1-Wasserstein
distance between the imputed and ground truth distribu-
tions at the left-out time point, following Tong et al. (2020).
*Indicates values taken from aforementioned work.
Algorithm

| Dataset

Multi

Cite

EB

→

↓
[SF]2M-Geo
[SF]2M-Exact
[SF]2M-Sink
DSBM (Shi et al., 2023)
DSB (De Bortoli et al., 2021)

1.017±0.104
0.920±0.049
1.054±0.087
1.705±0.160
0.953±0.140

OT-CFM (Tong et al., 2024)
I-CFM (Tong et al., 2024)
SB-CFM (Tong et al., 2024)

0.882±0.058
0.965±0.111
1.067±0.107
—
Reg. CNF (Finlay et al., 2020)*
TrajectoryNet (Tong et al., 2020)* —
NLSB (Koshizuka and Sato, 2023) —

0.879±0.148
0.793±0.066
1.198±0.342
1.775±0.429
0.862±0.023

0.790±0.068
0.872±0.087
1.221±0.380
0.825
0.848
0.970

1.255±0.179
0.933±0.054
1.098±0.308
1.873±0.631
1.079±0.117

0.937±0.054
1.085±0.099
1.129±0.363
—
—
—

batches (-Sink), with independent couplings (-I), and
with Geodesic OT (-Geo) when applicable.

• A variety of (ODE) flow-based models, including
optimal transport conditional flow matching (OT-
CFM, Tong et al., 2024), rectified flow (RF, Liu,
2022), and flow matching (FM, Lipman et al., 2023).
• Schrödinger bridge models: diffusion Schrödinger
bridges (DSB, De Bortoli et al., 2021) and diffu-
sion Schrödinger bridge matching (DSBM, Shi et al.,
2023), which is equivalent to work on iterated diffu-
sion mixture transport (IDBM, Peluchetti, 2023).
• Single-cell dynamics models: Neural Lagrangian
Schrödinger bridges (NLSB, Koshizuka and Sato,
2023), TrajectoryNet (Tong et al., 2020).

See Appendix F for all experiment details. All results
are presented as mean

std. over five seeds.

±

[SF]2M is a competitive generative model for
low-dimensional data. We first evaluate in Table 2
how well various methods approximate dynamic opti-
mal transport on low-dimensional datasets (8gaussians,
moons, and scurve). We train Schrödinger bridges
between a Gaussian and each dataset, and between
8gaussians and moons, using [SF]2M. We report the
2-Wasserstein distance between the predicted distribu-
tion and the target distribution with samples of size
10,000. Following Tong et al. (2024), we also report the
Normalized Path Energy relative to the 2-Wasserstein
distance, defined as NPE(p, q) :=

(cid:82)

−
2 (p, q). This metric is equal to zero if and
2

|

vθ(t, x)
∥

2dt
∥

2
2 (p, q)

W

|

/
W

Table 5: Leave-one-timepoint-out testing of dynamics inter-
polation methods measuring the error between the predicted
and ground truth left out timepoint using the 1-Wasserstein
distance. We test on 50 and 100 principal components as
well as 1000 highly variable genes.
Dim.

1000

100

50

Alg.

→
| Dataset

↓
[SF]2M-Geo
[SF]2M-Exact
DSBM

OT-CFM
I-CFM

→

Cite

Multi

Cite

Multi

Cite

Multi

38.52±0.29
40.01±0.78
53.81±7.74
38.76±0.40
41.83±3.28

44.80±1.91
45.34±2.83
66.43±14.39
47.58±6.62
49.78±4.43

44.50±0.42
46.53±0.43
58.99±7.62
45.39±0.42
48.28±3.28

52.20±1.96
52.89±1.99
70.75±14.03
54.81±5.86
57.26±3.86

40.09±1.53
43.66±0.72
50.09±4.81
43.25±0.73
44.12±0.52

51.29±0.09
53.15±1.86
61.71±13.90
52.29±1.55
52.99±1.50

only if vθ solves the dynamic optimal transport prob-
lem. Table 2 summarizes our results, showing that
[SF]2M outperforms all methods, both stochastic (top)
and deterministic (bottom). Despite minibatch OT
bias (which can be seen as a form of regularization
like the entropic regularization (Fatras et al., 2020)),
we find [SF]2M-Exact approximates the Schrödinger
bridge best, with the OT computation accounting for
only 1% of the training time on batch sizes of 512.

[SF]2M recovers the SB. Next we evaluate how
well [SF]2M can model Schrödinger bridge marginals.
We use a Gaussian-to-Gaussian Schrödinger bridge be-
cause it has closed-form Gaussian marginals (Mallasto
et al., 2022; Bunne et al., 2022a) following De Bortoli
et al. (2021). After training all methods, we evaluate
the quality of empirical marginals with respect to the
ground truth by sampling trajectories using Alg. 2.
We compute the KL divergence between a Gaussian
approximation of the empirical marginal and the Gaus-
sian marginal of the ground truth Schrödinger bridge
at multiple timepoints. This evaluation is shown in
Table 3 at just the last timepoint (t = 1) and an aver-
age over 21 equally spaced timepoints. We train each
method for an equal number of steps, using 20 outer
loops for DSB and DSBM, which require iterative op-
timization of forward and backward models. We find
that in low dimensions SB-CFM, which corresponds
to [SF]2M’s probability ODE flow, performs the best,
closely followed by [SF]2M-Exact. In high dimensions,
[SF]2M-Exact better matches the target distribution,
and performs similarly to DSBM and significantly bet-
ter than DSB on the intermediate marginals.

[SF]2M accurately models high-dimensional sin-
gle cell dynamics. We train our method [SF]2M on
single cell dynamics, as described in §4, on three real-
world datasets in the setup established by Tong et al.
(2020) (see §F.6) and gathered our results for different
dimensions in Table 4 and Table 5. Given K unpaired
data distributions representing a cell population at K
different timepoints, we solve a SB problem between
every two successive time points, sharing parameters
between the models. To test the interpolation ability of
the trained models, we perform leave-one-out interpo-
lation, predicting timepoint k using a model trained on

A. Tong, N. Malkin, K. Fatras, L. Atanackovic, Y. Zhang, G. Huguet, G. Wolf, Y. Bengio

Table 6: GRN recovery from simulated time-lapsed single-
cell gene expression. Shows structure predictive perfor-
mance in terms of area under the receiver operator charac-
teristic (AUC-ROC) and average precision (AP).
Bifurcating System
GRN

Trifurcating System

) AP (
AUC-ROC (
↑

Alg.

→
| Metric

↓

→
NGM-[SF]2Mσ=0
NGM-[SF]2Mσ=0.1
Spearman
Pearson
DREMI
Granger
SCODE

0.786±0.081
0.723±0.014
0.755±0.003
0.744±0.000
0.594±0.017
0.664±0.013
0.570±0.036

)
↑
0.521±0.160
0.444±0.030
0.438±0.002
0.415±0.000
0.293±0.011
0.421±0.04
0.370±0.028

AUC-ROC (

0.764±0.066
0.731±0.077
0.718±0.00
0.710±0.00
0.419±0.02
0.613±0.04
0.570±0.06

) AP (
↑

)
↑
0.485±0.105
0.453±0.091
0.413±0.005
0.405±0.002
0.205±0.007
0.343±0.039
0.332±0.077

2014), and pairwise Granger causality (Granger, 1969).
Higher values of σ do not perform as well, but still
outperform the baselines on most metrics.

7 CONCLUSION

We have introduced a novel class of simulation-free
objectives for learning continuous-time stochastic gen-
erative models between general source and target dis-
tributions. For sources and targets with finite sup-
port, we can directly approximate the continuous-time
Schrödinger bridge without simulation by computing
the entropic OT plan via efficient algorithms. We have
shown how our method can be applied to learn cell
dynamics and extract the gene regulatory structure. Fu-
ture work can consider how to train [SF]2M-like models
with interventional data to improve GRN inference.

Limitations. The main limitation of [SF]2M is that
it requires knowledge of conditional path distributions
(Brownian bridges). These distributions are not avail-
able in closed form if one considers more general refer-
ence processes (Fernandes et al., 2022), which may be
useful to encode biological priors (Koshizuka and Sato,
2023), or on general Riemannian manifolds.

Acknowledgments

We would like to thank Hananeh Aliee, Paul Bertin,
Valentin de Bortoli, Stefano Massaroli, and Austin
Stromme for productive conversations. The authors
acknowledge funding from CIFAR, Genentech, Sam-
sung, IBM, Microsoft, and Google. We are also grateful
to the anonymous reviewers for suggesting numerous
improvements. This research was enabled in part by
compute resources provided by Mila (mila.quebec) and
NVIDIA Corporation. In addition, K.F. acknowledges
funding from NSERC (RGPIN-2019-06512) and G.W.
acknowledges funding from NSERC Discovery grant
03267 and NIH grant R01GM135929.

Figure 3: Simulation of trajectories from a given cell on 2D
EB data. Left: Probability flow ODE trajectory, approxi-
mated by SB-CFM (Tong et al., 2024). Right: Five SDE
trajectories from [SF]2M; more target samples (20) in blue.

−

timepoints [1, . . . , k
1, k + 1, . . . , K]. We consider four
data representations of different dimensionality: using
the first 5, 50, or 100 whitened principal components
and using the 1000 dimensions corresponding to the
most highly variable genes (Wolf et al., 2018). [SF]2M
performs the best among the Schrödinger bridge meth-
ods and similarly to the ODE-based OT-CFM on the
low-dimensional data (Table 4), but is better in higher
dimensions (Table 5). As the number of dimensions
grows, the advantage of geodesic interpolation ([SF]2M-
Geo) becomes apparent.

We also compare the stochastic process modeled by
[SF]2M to its probability flow ODE SB-CFM. We show
in Fig. 3 that [SF]2M models a stochastic dependence
of the output on the input, unlike SB-CFM, despite
the two algorithms sharing marginal densities. This is
important in the EB data as the initial stem cells are
thought to be pluripotent and should evolve stochas-
tically into differentiated cell types over time. Such
differentiation cannot be modelled by ODE-based meth-
ods, thus motivating the use of a stochastic process for
modeling single-cell dynamics.

[SF]2M can be used to recover gene regula-
tory networks. We demonstrate the use-case of
[SF]2M for recovering gene regulatory networks (GRNs)
from single-cell gene expression using the algorithm
described at the end of §4. We show how we can use
[SF]2M to simultaneously learn dynamics and GRN
structure from single-cell gene expression data. We use
BoolODE (Pratapa et al., 2020) to simulate two single-
cell systems given ground truth GRNs: (1) a system
with bifurcating trajectories (7 genes), and (2) a system
with trifurcating trajectories (9 genes). We summarize
our results in Table 6. We also measure how accurately
the ground truth GRN is recovered using the standard
AUC-ROC and average precision (AP) metrics. We
find that [SF]2M-Exact with no noise (corresponding to
OT-CFM (Tong et al., 2024)) performs best at inferring
the underlying GRN as compared to correlation base-
lines (Pearson and Spearman correlation), a mutual
information baseline DREMI (Krishnaswamy et al.,

Simulation-Free Schrödinger Bridges via Score and Flow Matching

References

Albergo, M. S., Boffi, N. M., and Vanden-Eijnden,
E. (2023).
Stochastic interpolants: A unifying
framework for flows and diffusions. arXiv preprint
2303.08797.

Albergo, M. S. and Vanden-Eijnden, E. (2023). Build-
ing normalizing flows with stochastic interpolants.
International Conference on Learning Representa-
tions (ICLR).

Aliee, H., Richter, T., Solonin, M., Ibarra, I., Theis,
F., and Kilbertus, N. (2022). Sparsity in continuous-
depth neural networks. Neural Information Process-
ing Systems (NeurIPS).

Aliee, H., Theis, F. J., and Kilbertus, N. (2021). Be-
yond predictions in neural ODEs: Identification and
interventions. arXiv preprint 2106.12430.

Altschuler, J., Niles-Weed, J., and Rigollet, P. (2017).
Near-linear time approximation algorithms for opti-
mal transport via Sinkhorn iteration. Neural Infor-
mation Processing Systems (NIPS).

Anderson, B. D. (1982). Reverse-time diffusion equa-
tion models. Stochastic Processes and their Applica-
tions, 12(3):313–326.

Atanackovic, L., Tong, A., Hartford, J., Lee, L. J.,
Wang, B., and Bengio, Y. (2023). DynGFN: Bayesian
dynamic causal discovery using generative flow net-
works. Neural Information Processing Systems
(NeurIPS).

Bellot, A. and Branson, K. (2022). Neural Graphical
Modelling in Continuous Time: Consistency Guar-
antees and Algorithms. International Conference on
Learning Representations (ICLR).

Bunne, C., Hsieh, Y.-P., Cuturi, M., and Krause,
A. (2022a). The Schrödinger bridge between gaus-
sian measures has a closed form. arXiv preprint
2202.05722.

Bunne, C., Meng-Papaxanthos, L., Krause, A., and
Cuturi, M. (2022b). Proximal optimal transport
modeling of population dynamics. Artificial Intelli-
gence and Statistics (AISTATS).

Burkhardt, D., Bloom, J., Cannoodt, R., Luecken,
M. D., Krishnaswamy, S., Lance, C., Pisco, A. O.,
and Theis, F. J. (2022). Multimodal single-cell in-
tegration across time, individuals, and batches. In
NeurIPS Competitions.

Chen, R. T. Q., Rubanova, Y., Bettencourt, J., and
Duvenaud, D. (2018). Neural ordinary differential
equations. Neural Information Processing Systems
(NeurIPS).

forward-backward SDEs theory. International Con-
ference on Learning Representations (ICLR).

Cuturi, M. (2013). Sinkhorn distances: Lightspeed com-
putation of optimal transport. Neural Information
Processing Systems (NIPS).

Damodaran, B. B., Kellenberger, B., Flamary, R., Tuia,
D., and Courty, N. (2018). DeepJDOT: Deep joint
distribution optimal transport for unsupervised do-
main adaptation. European Conference on Computer
Vision (ECCV).

De Bortoli, V., Thornton, J., Heng, J., and Doucet,
A. (2021). Diffusion Schrödinger bridge with appli-
cations to score-based generative modeling. Neural
Information Processing Systems (NeurIPS).

Dhariwal, P. and Nichol, A. (2021). Diffusion models
beat GANs on image synthesis. Neural Information
Processing Systems (NeurIPS).

Fatras, K., Sejourne, T., Flamary, R., and Courty, N.
(2021a). Unbalanced minibatch optimal transport;
applications to domain adaptation. International
Conference on Machine Learning (ICML).

Fatras, K., Zine, Y., Flamary, R., Gribonval, R., and
Courty, N. (2020). Learning with minibatch Wasser-
stein: Asymptotic and gradient properties. Artificial
Intelligence and Statistics (AISTATS).

Fatras, K., Zine, Y., Majewski, S., Flamary, R., Gribon-
val, R., and Courty, N. (2021b). Minibatch optimal
transport distances; analysis and applications. arXiv
preprint 2101.01792.

Fernandes, D., Vargas, F., Ek, C. H., and Campbell, N.
D. F. (2022). Shooting Schrödinger’s cat. Advances
in Approximate Bayesian Inference.

Finlay, C., Jacobsen, J.-H., Nurbekyan, L., and Ober-
man, A. M. (2020). How to train your neural
ode: The world of jacobian and kinetic regulariza-
tion. International Conference on Machine Learning
(ICML).

Flamary, R., Courty, N., Gramfort, A., Alaya, M. Z.,
Boisbunon, A., Chambon, S., Chapel, L., Corenflos,
A., Fatras, K., Fournier, N., Gautheron, L., Gayraud,
N. T. H., Janati, H., Rakotomamonjy, A., Redko, I.,
Rolet, A., Schutz, A., Seguy, V., Sutherland, D. J.,
Tavenard, R., Tong, A., and Vayer, T. (2021). POT:
Python Optimal Transport. Journal of Machine
Learning Research (JMLR), 22.

Föllmer, H. (1988). Random fields and diffusion pro-
cesses.
In Hennequin, P.-L., editor, École d’Été
de Probabilités de Saint-Flour XV–XVII, 1985–87,
pages 101–203, Berlin, Heidelberg. Springer Berlin
Heidelberg.

Chen, T., Liu, G.-H., and Theodorou, E. A. (2022).
Likelihood training of Schrödinger bridge using

Genevay, A., Chizat, L., Bach, F., Cuturi, M., and
Peyré, G. (2019). Sample complexity of Sinkhorn di-

A. Tong, N. Malkin, K. Fatras, L. Atanackovic, Y. Zhang, G. Huguet, G. Wolf, Y. Bengio

vergences. Artificial Intelligence and Statistics (AIS-
TATS).

Genevay, A., Cuturi, M., Peyré, G., and Bach, F.
(2016). Stochastic optimization for large-scale opti-
mal transport. Neural Information Processing Sys-
tems (NIPS).

Genevay, A., Peyre, G., and Cuturi, M. (2018). Learn-
ing generative models with sinkhorn divergences. Ar-
tificial Intelligence and Statistics (AISTATS).

Granger, C. W. J. (1969). Investigating causal relations
by econometric models and cross-spectral methods.
Econometrica, 37(3):424–438.

Grathwohl, W., Chen, R. T. Q., Bettencourt, J.,
Sutskever, I., and Duvenaud, D. (2019). Ffjord: Free-
form continuous dynamics for scalable reversible gen-
erative models. International Conference on Learn-
ing Representations (ICLR).

Hashimoto, T. B., Gifford, D. K., and Jaakkola, T. S.
(2016). Learning population-level diffusions with gen-
erative recurrent networks. International Conference
on Machine Learning (ICML).

Ho, J., Jain, A., and Abbeel, P. (2020). Denoising
diffusion probabilistic models. Neural Information
Processing Systems (NeurIPS).

Holdijk, L., Du, Y., Hooft, F., Jaini, P., Ensing, B.,
and Welling, M. (2022). Path integral stochastic
optimal control for sampling transition paths. arXiv
preprint 2207.02149.

Huguet, G., Magruder, D. S., Tong, A., Fasina, O.,
Kuchroo, M., Wolf, G., and Krishnaswamy, S.
(2022a). Manifold interpolating optimal-transport
flows for trajectory inference. Neural Information
Processing Systems (NeurIPS).

Huguet, G., Tong, A., Zapatero, M. R., Wolf, G., and
Krishnaswamy, S. (2022b). Geodesic Sinkhorn: Op-
timal transport for high-dimensional datasets. arXiv
preprint 2211.00805.

Kester, L. and van Oudenaarden, A. (2018). Single-cell
transcriptomics meets lineage tracing. Cell Stem
Cell, 23(2):166–179.

Klambauer, G., Unterthiner, T., Mayr, A., and Hochre-
iter, S. (2017). Self-normalizing neural networks.
Neural Information Processing Systems (NIPS).
Klein, D., Palla, G., Lange, M., Klein, M., Piran,
Z., Gander, M., Meng-Papaxanthos, L., Sterr, M.,
Bastidas-Ponce, A., Tarquis-Medina, M., Lickert, H.,
Bakhti, M., Nitzan, M., Cuturi, M., and Theis, F. J.
(2023). Mapping cells through time and space with
moscot. bioRxiv preprint 2023.05.11.540374.

Koshizuka, T. and Sato, I. (2023). Neural Lagrangian
Schrödinger bridge. Internationcal Conference on
Learning Representations (ICLR).

Krishnaswamy, S., Spitzer, M. H., Mingueneau, M.,
Bendall, S. C., Litvin, O., Stone, E., Pe’er, D., and
Nolan, G. P. (2014). Conditional density-based anal-
ysis of T cell signaling in single-cell data. Science,
346(6213):1250689.

Lähnemann, D., Köster, J., Szczurek, E., McCarthy,
D. J., Hicks, S. C., Robinson, M. D., Vallejos, C. A.,
Campbell, K. R., Beerenwinkel, N., Mahfouz, A.,
Pinello, L., Skums, P., Stamatakis, A., Attolini, C.
S.-O., Aparicio, S., Baaijens, J., Balvert, M., Bar-
banson, B. D., Cappuccio, A., Corleone, G., Dutilh,
B. E., Florescu, M., Guryev, V., Holmer, R., Jahn,
K., Lobo, T. J., Keizer, E. M., Khatri, I., Kielbasa,
S. M., Korbel, J. O., Kozlov, A. M., Kuo, T.-H.,
Lelieveldt, B. P., Mandoiu, I. I., Marioni, J. C.,
Marschall, T., Mölder, F., Niknejad, A., Raczkowski,
L., Reinders, M., Ridder, J. D., Saliba, A.-E., So-
marakis, A., Stegle, O., Theis, F. J., Yang, H., Ze-
likovsky, A., McHardy, A. C., Raphael, B. J., Shah,
S. P., and Schönhuth, A. (2020). Eleven grand chal-
lenges in single-cell data science. Genome Biology,
21(1):31.

Lavenant, H., Zhang, S., Kim, Y.-H., and Schiebinger,
G. (2021). Towards a mathematical theory of trajec-
tory inference.

Léonard, C. (2014a). Some properties of path measures.
In Donati-Martin, C., Lejay, A., and Rouault, A.,
editors, Séminaire de Probabilités XLVI, pages 207–
230. Springer.

Léonard, C. (2014b). A survey of the Schrödinger
problem and some of its connections with optimal
transport. Discrete and Continuous Dynamical Sys-
tems, 34(4):1533–1574.

Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel,
M., and Le, M. (2023). Flow matching for generative
modeling.
International Conference on Learning
Representations (ICLR).

Liu, G.-H., Chen, T., So, O., and Theodorou, E. A.
(2022). Deep generalized Schrödinger bridge. Neural
Information Processing Systems (NeurIPS).

Liu, G.-H., Vahdat, A., Huang, D.-A., Theodorou,
E. A., Nie, W., and Anandkumar, A. (2023a). I2sb:
Image-to-image schrödinger bridge. International
Conference on Machine Learning (ICML).

Liu, Q. (2022). Rectified flow: A marginal preserv-
ing approach to optimal transport. arXiv preprint
2209.14577.

Liu, X., Gong, C., and Liu, Q. (2023b). Flow straight
and fast: Learning to generate and transfer data with
rectified flow. International Conference on Learning
Representations (ICLR).

Simulation-Free Schrödinger Bridges via Score and Flow Matching

Loshchilov, I. and Hutter, F. (2019). Decoupled weight
decay regularization. International Conference on
Learning Representations (ICLR).

Mallasto, A., Gerolin, A., and Ha Quang, M. (2022).
Entropy-regularized 2-Wasserstein distance between
Gaussian measures. Information Geometry, 5.

Mena, G. and Niles-Weed, J. (2019). Statistical bounds
for entropic optimal transport: sample complexity
and the central limit theorem. Neural Information
Processing Systems (NeurIPS).

Moon, K. R., Stanley, J. S., Burkhardt, D., van Dijk,
D., Wolf, G., and Krishnaswamy, S. (2018). Mani-
fold learning-based methods for analyzing single-cell
rna-sequencing data. Current Opinion in Systems
Biology, 7:36–46.

Moon, K. R., van Dijk, D., Wang, Z., Gigante, S.,
Burkhardt, D. B., Chen, W. S., Yim, K., van den
Elzen, A., Hirn, M. J., Coifman, R. R., Ivanova,
N. B., Wolf, G., and Krishnaswamy, S. (2019). Visu-
alizing structure and transitions in high-dimensional
biological data. Nature Biotechnology, 37(12):1482–
1492.

Neklyudov, K., Brekelmans, R., Severo, D., and
Makhzani, A. (2023). Action matching: Learning
stochastic dynamics from samples.
International
Conference on Machine Learning (ICML).

Neklyudov, K., Severo, D., and Makhzani, A. (2022).
Action matching: A variational method for learning
stochastic dynamics from samples. arXiv preprint
2210.06662.

Nichol, A. and Dhariwal, P. (2021). Improved denoising
diffusion probabilistic models. International Confer-
ence on Machine Learning (ICML).

Nolan, T. M., Vukašinović, N., Hsu, C.-W., Zhang, J.,
Vanhoutte, I., Shahan, R., Taylor, I. W., Greenstreet,
L., Heitz, M., Afanassiev, A., Wang, P., Szekely, P.,
Brosnan, A., Yin, Y., Schiebinger, G., Ohler, U.,
Russinova, E., and Benfey, P. N. (2023). Brassinos-
teroid gene regulatory networks at cellular resolution
in the arabidopsis root. Science, 379(6639):eadf4721.
Peluchetti, S. (2023). Diffusion bridge mixture trans-
ports, Schrödinger bridge problems and generative
modeling. Journal of Machine Learning Research
(JMLR), 24:1–51.

Pooladian, A.-A., Ben-Hamu, H., Domingo-Enrich, C.,
Amos, B., Lipman, Y., and Chen, R. T. (2023).
Multisample flow matching: Straightening flows with
minibatch couplings. International Conference on
Learning Representations (ICLR).

Pratapa, A., Jalihal, A. P., Law, J. N., Bharadwaj, A.,
and Murali, T. (2020). Benchmarking algorithms for
gene regulatory network inference from single-cell
transcriptomic data. Nature methods, 17.

Qin, X., Rodriguez, F. C., Sufi, J., Vlckova, P., Claus,
J., and Tape, C. J. (2023). A single-cell perturbation
landscape of colonic stem cell polarisation. Preprint,
Cancer Biology.

Salimans, T., Zhang, H., Radford, A., and Metaxas, D.
(2018). Improving GANs using optimal transport. In-
ternational Conference on Learning Representations
(ICLR).

Schiebinger, G. (2021). Reconstructing developmen-
tal landscapes and trajectories from single-cell data.
Current Opinion in Systems Biology, 27:100351.
Schiebinger, G., Shu, J., Tabaka, M., Cleary, B., Sub-
ramanian, V., Solomon, A., Gould, J., Liu, S., Lin,
S., Berube, P., Lee, L., Chen, J., Brumbaugh, J.,
Rigollet, P., Hochedlinger, K., Jaenisch, R., Regev,
A., and Lander, E. S. (2019). Optimal-transport
analysis of single-cell gene expression identifies de-
velopmental trajectories in reprogramming. Cell,
176(4):928–943.e22.

Schrödinger, E. (1932). Sur la théorie relativiste de
l’électron et l’interprétation de la mécanique quan-
tique. Annales de l’Institut Henri Poincaré, 2(4):269–
310.

Seguy, V., Damodaran, B. B., Flamary, R., Courty, N.,
Rolet, A., and Blondel, M. (2018). Large scale opti-
mal transport and mapping estimation. International
Conference on Learning Representations (ICLR).
Shi, Y., De Bortoli, V., Campbell, A., and Doucet,
A. (2023). Diffusion Schrödinger bridge matching.
arXiv preprint 2303.16852.

Shi, Y., De Bortoli, V., Deligiannidis, G., and Doucet,
A. (2022). Conditional simulation using diffusion
Schrödinger bridges. Uncertainty in Artificial Intel-
ligence (UAI).

Sinkhorn, R. (1964). A relationship between arbitrary
positive matrices and doubly stochastic matrices.

Sohl-Dickstein, J., Weiss, E. A., Maheswaranathan, N.,
and Ganguli, S. (2015). Deep unsupervised learning
using nonequilibrium thermodynamics. International
Conference on Machine Learning (ICML).

Somnath, V. R., Pariset, M., Hsieh, Y.-P., Martinez,
M. R., Krause, A., and Bunne, C. (2023). Aligned dif-
fusion Schrödinger bridges. Uncertainty in Artificial
Intelligence (UAI).

Song, J., Meng, C., and Ermon, S. (2021a). Denoising
diffusion implicit models. International Conference
on Learning Representations (ICLR).

Song, Y. and Ermon, S. (2019). Generative modeling by
estimating gradients of the data distribution. Neural
Information Processing Systems (NeurIPS).

A. Tong, N. Malkin, K. Fatras, L. Atanackovic, Y. Zhang, G. Huguet, G. Wolf, Y. Bengio

D. (2023). Broadly applicable and accurate protein
design by integrating structure prediction networks
and diffusion generative models. Nature.

Winkler, L., Ojeda, C., and Opper, M. (2023). A score-
based approach for training Schrödinger bridges for
data modelling. Entropy, 25(2):316.

Wolf, F. A., Angerer, P., and Theis, F. J. (2018).
SCANPY: Large-scale single-cell gene expression
data analysis. Genome Biology, 19(1):15.

Yang, K. D., Damodaran, K., Venkatachalapathy, S.,
Soylemezoglu, A. C., Shivashankar, G. V., and Uhler,
C. (2020). Predicting cell lineages using autoencoders
and optimal transport. PLOS Computational Biology,
16:1–20.

Song, Y. and Ermon, S. (2020). Improved techniques
for training score-based generative models. Neural
Information Processing Systems (NeurIPS).

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar,
A., Ermon, S., and Poole, B. (2021b). Score-based
generative modeling through stochastic differential
equations.
International Conference on Learning
Representations (ICLR).

Stromme, A. (2023). Sampling from a Schrödinger
bridge. Artificial Intelligence and Statistics (AIS-
TATS).

Tang, Y. (2017). Potential landscape of high dimen-
sional nonlinear stochastic dynamics with large noise.
Scientific Reports.

Tank, A., Covert, I., Foti, N., Shojaie, A., and Fox, E.
(2021). Neural Granger Causality. IEEE Transac-
tions on Pattern Analysis and Machine Intelligence.
Tong, A., Fatras, K., Malkin, N., Huguet, G., Zhang,
Y., Rector-Brooks, J., Wolf, G., and Bengio, Y.
(2024). Improving and generalizing flow-based gen-
erative models with minibatch optimal transport.
Transactions on Machine Learning Research.

Tong, A., Huang, J., Wolf, G., van Dijk, D., and Kr-
ishnaswamy, S. (2020). TrajectoryNet: A dynamic
optimal transport network for modeling cellular dy-
namics. International Conference on Machine Learn-
ing (ICML).

Tong, A., Kuchroo, M., Gupta, S., Venkat, A., Perez
San Juan, B., Rangel, L., Zhu, B., Lock, J. G.,
Chaffer, C., and Krishnaswamy, S. (2023). Learn-
ing transcriptional and regulatory dynamics driving
cancer cell plasticity using neural ode-based optimal
transport. bioRxiv preprint 2023.03.28.534644.
Vargas, F., Thodoroff, P., Lawrence, N. D., and
Lamacraft, A. (2021). Solving Schrödinger bridges
via maximum likelihood. Entropy, 23(9).

Waddington, C. H. (1942). The epigenotype. Endeav-

our, 1:18–20.

Wagner, D. E. and Klein, A. M. (2020). Lineage tracing
meets single-cell omics: Opportunities and challenges.
Nature Reviews Genetics, 21(7):410–427.

Wang, G., Jiao, Y., Xu, Q., Wang, Y., and Yang, C.
(2021). Deep generative learning via Schrödinger
bridge. International Conference on Machine Learn-
ing (ICML).

Watson, J. L., Juergens, D., Bennett, N. R., Trippe,
B. L., Yim, J., Eisenach, H. E., Ahern, W., Borst,
A. J., Ragotte, R. J., Milles, L. F., Wicky, B. I. M.,
Hanikel, N., Pellock, S. J., Courbet, A., Sheffler, W.,
Wang, J., Venkatesh, P., Sappington, I., Torres, S. V.,
Lauko, A., De Bortoli, V., Mathieu, E., Barzilay, R.,
Jaakkola, T. S., DiMaio, F., Baek, M., and Baker,

Simulation-Free Schrödinger Bridges via Score and Flow Matching

(a) The full text of instructions given to partici-
pants and screenshots. [Not Applicable]
(b) Descriptions of potential participant risks,
with links to Institutional Review Board (IRB)
approvals if applicable. [Not Applicable]
(c) The estimated hourly wage paid to partici-
pants and the total amount spent on partici-
pant compensation. [Not Applicable]

Checklist

1. For all models and algorithms presented, check if

you include:

(a) A clear description of the mathematical set-
ting, assumptions, algorithm, and/or model.
[Yes]

(b) An analysis of the properties and complexity
(time, space, sample size) of any algorithm.
[Yes]

(c) Source code, with specification of all depen-
dencies, including external libraries. [Yes]

2. For any theoretical claim, check if you include:

(a) Statements of the full set of assumptions of

all theoretical results. [Yes]

(b) Complete proofs of all theoretical results.

[Yes]

(c) Clear explanations of any assumptions. [Yes]

3. For all figures and tables that present empirical

results, check if you include:

(a) The code, data, and instructions needed to re-
produce the main experimental results (either
in the supplemental material or as a URL).
[Yes]

(b) All the training details (e.g., data splits, hy-
perparameters, how they were chosen). [Yes]
(c) A clear definition of the specific measure or
statistics and error bars (e.g., with respect to
the random seed after running experiments
multiple times). [Yes]

(d) A description of the computing infrastructure
used. (e.g., type of GPUs, internal cluster, or
cloud provider). [Yes]

4. If you are using existing assets (e.g., code, data,
models) or curating/releasing new assets, check if
you include:

(a) Citations of the creator if your work uses

existing assets. [Yes]

(b) The license information of the assets, if appli-

cable. [Not Applicable]

(c) New assets either in the supplemental material
or as a URL, if applicable. [Not Applicable]
from data

(d) Information about

consent

providers/curators. [Not Applicable]

(e) Discussion of sensible content if applicable,
e.g., personally identifiable information or of-
fensive content. [Not Applicable]

5. If you used crowdsourcing or conducted research

with human subjects, check if you include:

A. Tong, N. Malkin, K. Fatras, L. Atanackovic, Y. Zhang, G. Huguet, G. Wolf, Y. Bengio

Simulation-Free Schrödinger Bridges via Score and Flow Matching:
Supplementary Materials

Our code can be found at https://github.com/atong01/conditional-flow-matching. The supplementary
material is structured as follows:

• Appendix A gives more background on optimal transport.

• Appendix B presents the proofs of our different results.

• Appendix C gives more background on the related work.

• Appendix D describes additional results and experiments.

• Appendix E discusses Schrödinger bridges with varying diffusion rate.

• Appendix F presents the experimental details of our experiments in the main paper.

A BACKGROUND ON OPTIMAL TRANSPORT

In this section, we review optimal transport and its application in machine learning.

Algorithm 2 Simulation-Free Score and Flow Matching Inference (with Euler-Maruyama integration)

Input: Source distribution q0, flow and score networks vθ and sθ, diffusion schedule g(
∆t.
x0
for t in [0, 1/∆t) do

q0(x)

∼

), integration step size
·

ut
←
xt+∆t

vθ(t, xt) + g(t)2

2 sθ(t, xt)

(x + ut∆t, g(t)2∆t)

return Samples x1

∼ N

Figure 4: Optimal transport couplings for different OT costs and batch sizes on a 2D example. The top row represent the
OT matching between samples while the bottom row represent the minibatch OT plan. We can see that coupling entropic
OT with minibatches lead to a uniform plan contrary to using only entropic regularization or minibatch approximation.

OTmatrixOTofsamplesMEOT(m=4)MEOT(m=4,λ=0.001)MEOT(m=6)MEOT(m=6,λ=0.001)Entropic(λ=0.002)OT0.0000.0250.0500.0750.1000.0000.0250.0500.0750.100Simulation-Free Schrödinger Bridges via Score and Flow Matching

A.1 Minibatch OT

In the context of generative modeling, the source distribution is a Gaussian distribution and the target distribution
is the real data distribution. This scenario corresponds to a semi-discrete optimal transport problem. Therefore,
the Sinkhorn algorithm cannot be used to compute the entropic OT plan between distributions. It is nonetheless
possible to compute it with stochastic algorithms (Genevay et al., 2016). Unfortunately, these stochastic algorithms
are slow to converge and it is prohibitive for large scale datasets. Therefore, we chose to rely on a minibatch optimal
transport approximation (Fatras et al., 2020, 2021b). Minibatch OT computes the OT between minibatches of
samples and thus corresponds to the discrete-discrete optimal transport setting. It is known to have quadratic
computational and memory costs in the number of samples (see §A.2 for a longer discussion).

The minibatch optimal transport approximation has been successfully before used in generative modeling (Genevay
et al., 2018; Salimans et al., 2018). As it is different from the original optimal transport problem, we want to
make sure that the basic properties from Schrödinger Bridges are conserved. We use the minibatch optimal
transport plans definitions from Fatras et al. (2020), namely
Definition A.1 (minibatch transport plan (Fatras et al., 2020)). Consider αn and βn two empirical probability
distributions. For each A =
m(βn) we denote by ΠA,B the
m(αn) and B =
a1, . . . , am
b1, . . . , bm
{
{
n matrix where all entries are zero except those
optimal plan between the random variables, considered as a n
indexed in A

} ∈ P
B. We define the averaged minibatch transport matrix :

} ∈ P

×

×

Πm(αn, βn) =

(cid:19)−2

(cid:18) n
m

(cid:88)

(cid:88)

ΠA,B.

A∈Pm(αn)

B∈Pm(βn)

Following the subsampling idea, we define the subsampled minibatch transportation matrix:

Πk(αn, βn) := k−1 (cid:88)

ΠA,B

(A,B)∈Dk

(13)

(14)

where Dk is a set of cardinality k whose elements are drawn at random from the uniform distribution on
Γ :=

).

m(
{

P

x1
0,

· · ·

, xn
)
0 }

x1
1,
m(
{

· · ·

, xn
1 }

× P

Note that Πk converges exponentially fast to Πm as k grows (Fatras et al., 2020, Theorem 2). In practice, it is
enough to set k equal to 1 to get good performance in deep learning applications (Genevay et al., 2018; Damodaran
et al., 2018; Fatras et al., 2021a). Therefore, the subsampling estimator does not have the correct marginals in
general, contrary to Πm which has always the right marginals (Fatras et al., 2020, Proposition 1). Minibatch
optimal transport has been shown to implicitly regularize the transport plan (Fatras et al., 2020, 2021b). Indeed,
as we draw uniformly at random sample to build the minibatches, we create non optimal connections. This is
similar to the entropic optimal transport which densifies the transport plan. Therefore, coupling the minibatch
approximation with the entropic OT cost might lead to an extremely dense plan that is close to the uniform
transport plan. Unfortunately such a transport plan looses all geometric insights from data and is also far from
the original entropic OT cost. We illustrate this phenomenon in Fig. 4 on a toy example between two 2D data
distributions. Notably, the minibatch OT plan is closer to the entropic OT plan than the minibatch entropic
OT plan. That is why in our experiments, we observed that the minibatch approximation with exact optimal
transport outperforms the minibatch approximation with the entropic OT cost. We leave the question of closeness
between minibatch OT and entropic OT as future work. While it would have been possible to decrease these
non-optimal connections with OT variants (Fatras et al., 2021a), it would have added extra hyperparameters that
would have led to more compute.

A.2 Computational complexity

The batch size we use is 512 for all experiments except for the Gaussian-to-Gaussian experiments, which use batch
size 500 to match the DSBM baseline. The complexity of static discrete OT is cubic in the batch size m and
linear in the dimension d: O(m3 + m2d). This can be reduced to quadratic complexity in the entropy-regularized
discrete OT case. However, for batches <10k, the exact minibatch OT is often actually faster than the Sinkhorn
solver for the entropic OT problem thanks to the optimized simplex solver code in POT (Flamary et al., 2021).
Note that the memory complexity is quadratic in the number of samples due to the storage of the ground cost
matrix.

A. Tong, N. Malkin, K. Fatras, L. Atanackovic, Y. Zhang, G. Huguet, G. Wolf, Y. Bengio

The asymptotic complexity does not tell the full story, as these discrete algorithms run much faster than the
(theoretically linear time) neural network training. In practice, for high dimensional settings we find a < 1%
overhead in including the OT solution per batch (much of which is in the transfer from GPU to CPU and back).
We note that this is significantly faster than contemporary works such as DSBM and IBDM, which must train a
bridge matching model for every iterative proportional fitting (IPF) step.

B PROOFS OF THEOREMS

Theorem 3.1. Under mild regularity conditions, the ODE dx = u◦
initial conditions p0, and the score is given by (10). The SDE dx = [u◦
generates the Markovization of P.

t (x) dt generates the marginals pt of P from
log pt(x)] dx + g(t) dt

t (x) + 1

2 g(t)2

∇

z), and their first derivatives are continuous in x, uniformly
Proof of Theorem 3.1. We assume ut(x
|
in z. The first statement is Theorem 3.1 from Tong et al. (2024), but we reproduce the key derivation here for
completeness:

log pt(x

z),

∇

|

d
dt

pt(x) =

=

=

=

=

(cid:90)

d
dt
(cid:90) d
dt
(cid:90)

−

∇ ·
(cid:18)(cid:90)

−∇ ·

−∇ ·

pt(x

z)q(z)dz
|

(pt(x

|

z)q(z)) dz

(u◦

t (x

z)pt(x
|

z)q(z)) dz
|

u◦
t (x

|

z)pt(x

z)q(z)dz
|

(cid:19)

(u◦

t (x)pt(x)) ,

showing that pt and u◦
t
generates pt from initial conditions p0.
To show the score is given by the expression in (10), using that pt(x) = Eq(z)pt(x

satisfy the continuity equation d

dt pt(x) +

(u◦

∇ ·

t (x)pt(x)) = 0, which implies that u◦
t

z), we have

|

log pt(x) = ∇

∇

Eq(z)[pt(x
pt(x)

z)]
|

as desired.

= Eq(z)

Eq(z)[

=

pt(x

z)]
|

∇
pt(x)
(cid:20) pt(x

z)
|

log pt(x

∇
pt(x)

z)
|

(cid:21)

,

The Markovization of a mixture of SDEs P with a common diffusion rate g(t) is an SDE dx = ut(x) dt + g(t) dw,
where

ut(xt) = lim
∆t→0

Ext+∆t∼P(xt+∆t|xt)

(cid:20) xt+∆t
∆t

−

(cid:21)

xt

.

Using the definition of P as a mixture in (9), we decompose this expectation over the posterior p(z
pt(xt|z)q(z)
pt(xt)

:

|

xt) =

lim
∆t→0

Ext+∆t∼P(xt+∆t|xt)

(cid:20) xt+∆t
∆t

−

(cid:21)

xt

= lim
∆t→0

Ez∼p(z|xt)Ext+∆t∼P(xt+∆t|xt,z)

Ext+∆t∼P(xt+∆t|xt,z)

= Ez∼p(z|xt) lim
∆t→0
= Ez∼p(z|xt) [ut(xt

z)] .

|

−

(cid:20) xt+∆t
∆t
(cid:20) xt+∆t
∆t

−

(cid:21)

xt

(cid:21)

xt

The score and probability flow defined in (10) can equivalently be expressed as
log pt(xt) = Ez∼p(z|xt)∇

t (xt) = Ez∼p(z|xt)u◦
u◦

t (xt

z),

∇

|

log pt(xt

z),

|

Simulation-Free Schrödinger Bridges via Score and Flow Matching

which, together with ut(x

z) = u◦

t (x

z) + 1

2 g(t)2

|

log pt(x

|

∇

|

z), implies that

ut(x) = Ez∼p(z|x) [ut(x

z)] = u◦

t (x) +

|

1
2

g(t)2

∇

log pt(x),

as desired.

Theorem 3.2 (Equality of conditional gradients). If pt(x) > 0 for all x

Rd and t

[0, 1], then

∈

∈

θ

LU[SF]2M(θ) =

∇

θ

L[SF]2M(θ), where

LU[SF]2M(θ) is the unconditional score and flow matching loss (7).
∇
Proof of Theorem 3.2. We show this individually for the flow matching and score matching parts of the losses.
(Note that the equality of gradients of the flow matching parts of U[SF]2M and [SF]2M is equivalent to Theorem
3.2 of Tong et al. (2024).)

We claim that for any conditional vector field w◦
vector field wθ(t, x), under some regularity conditions on wt(x
2(cid:3) =
z)
∥
|

θEz∼q(z),x∼pt(x|z)

wθ(t, x)

(cid:2)
∥

t (x

t (x

w◦

∇

−

z) and w◦

∇

|

|

Assuming this claim, the theorem would follow from applying the claim to w◦

pt(x

z) for every value of t, noting that in these cases w◦

t (x) = u◦

∇
Theorem 3.1, and integrating over t.

|

t (x) := Eq(z)
z), we have

pt(x|z)
pt(x) w◦

t (x

z), and approximating
|

θEx∼pt(x)

w◦

(cid:2)
wθ(t, x)
∥

−
z) = u◦
t (x
|
t (x) and w◦
t (x) =

(15)

2(cid:3) .
t (x)
∥
z) and to w◦
|

z) =
pt(x), respectively, by

t (x

t (x

|

∇

We proceed to prove the claim. We drop the distributions in the expectations for conciseness, noting that because
pt(x) = Ez∼q(z)p(x
z), no ambiguity is caused: the marginal distribution over x that stands under the expectation
is the same in Ex∼pt(x) and in Ez∼q(z),x∼pt(x|z).

|

(cid:2)

(cid:0)Ez,x
θ
θ (

w◦
wθ(t, x)
t (x
∥
2Ez,x
wθ(t, x), w◦
t (x

−

−

⟨

2(cid:3)

z)
∥
−
+ 2Ex

|
z)
⟩

|

⟨

Ex

(cid:2)
wθ(t, x)
∥
wθ(t, x), w◦

−
t (x)

=

∇

∇

2(cid:3)(cid:1)
∥

w◦
t (x)
)
⟩
z), w◦

where we rewrote the squared norms as inner products and used that w◦
that wθ(t, x) is independent of z. To conclude, we compute

t (x

|

t (x) are independent of θ and

Ex

wθ(t, x), w◦
⟨

t (x)
⟩

=

=

=

(cid:90) (cid:90)

(cid:90) (cid:90)

wθ(t, x), w◦
⟨
(cid:90) (cid:28)

t (x)
⟩
(cid:90) pt(x

wθ(t, x),

pt(x) dx

z)
|
pt(x)

(cid:29)

z)q(z) dz

pt(x) dx

w◦

t (x

|

wθ(t, x), w◦
⟨

wθ(t, x), w◦

t (x

z)
|
t (x

⟩
z)
⟩

|

= Ez,x

⟨

pt(x

z)q(z) dx dz
|

,

showing that the difference of gradients vanishes. Note that the above derivation required exchanging the order
of integration, which requires some assumptions of regularity at infinity. (Absolute convergence of the integrals is
sufficient, and in particular, guaranteed by polynomial growth of in x of wθ w◦
z) and exponential decay of
pt(x

z) uniformly in z.)

t (x
|

Theorem 3.3 ([SF]2M recovers marginals from bridges). If q(
θ globally minimize
L[SF]2M(θ), the SDE with drift [v∗
θ] and diffusion g, and initial conditions p0 = q0, is the Markovization
of the mixture of Brownian bridges from x0 to x1 over q(x0, x1). In particular, if the SDE generates marginals pt,
then p1 = q1.

U (q0, q1) and v∗

2 g(t)2s∗

θ + 1

θ , s∗

)
·

,
·

∈

|

and score

Proof of Theorem 3.3. The probability flow drift u◦
t
Theorem 3.1 shows. By Theorem 3.2, assuming sufficient regularity, optimization of
to optimization of
for (Lebesgue,pt)-almost all t
[0, 1] and x
∈
continuous in t and x and pt has full support.
By hypothesis, the vector field u◦
assumed to satisfy p0(x) = q0(x) and p1(x) = q1(x). By the algebraic derivation in §2, u◦

LU[SF]2M(θ), which is globally minimized when vθ(t, x) = u◦

t (x) satisfies the continuity equation jointly with pt(x) = Eq(z)pt(x
t (x) + 1

log pt of the mixture in question are given by
L[SF]2M(θ) is equivalent
log pt(x)
log pt, vθ, sθ are

Rd. Moreover, ‘almost all’ implies ‘all’ if u◦
t ,

z), which was
log pt(x)

t (x) and sθ(t, x) =

∇

∇

∇

∈

|
2 g(t)2

∇

A. Tong, N. Malkin, K. Fatras, L. Atanackovic, Y. Zhang, G. Huguet, G. Wolf, Y. Bengio

and pt(x) jointly satisfy the Fokker-Planck equation. Assuming all derivatives appearing in the Fokker-Planck
equation exist and are continuous everywhere, the given SDE generates marginal probabilities pt from initial
conditions p0.

Proposition 3.4 ([SF]2M with entropic OT recovers the SB process). Let P∗ be the Schrödinger bridge between
2σ2(q0, q1), then P∗ is
q0 and q1 with respect to Q = σW. If v⋆
defined by the SDE with drift [v∗

L[SF]2M, with coupling π⋆

θ], diffusion g, and initial conditions p0 = q0.

θ globally minimize

θ + 1

θ , s⋆

2 g(t)2s∗

Proof of Proposition 3.4. This is an immediate consequence of Theorem 3.3, which shows that the SDE learned
by [SF]2M is the Markovization of a mixture of Brownian bridges, and of Proposition 2.1, which characterizes the
SB as a mixture of Brownian bridges with mixture weights given by the entropic OT plan.

C FURTHER DISCUSSION ON RELATED WORK

Next we discuss similarities and differences in related algorithms, including deterministic flow models and
stochastic score-based generative models.

C.1 Flow models

Recently, there has been significant advances in simulation-free training of flow models, which were originally
trained using maximum likelihood training in what is termed as continuous normalizing flows (Chen et al., 2018;
Grathwohl et al., 2019). In this section, we discuss the more recent flow matching techniques, which allow for
simulation-free training of flow models. In this paper we show that any deterministic flow model can be converted
into a stochastic model with the addition of a conditional score matching loss. This generalizes flow matching
ideas to SDEs and provides a simple link between score-based generative modelling and flow-based generative
modeling.

Conditional flow matching, terminology introduced by Lipman et al. (2023), and extended to dynamic optimal
transport in Tong et al. (2024); Pooladian et al. (2023), trains with the conditional flow matching loss,

L

CFM(θ) = Et∼U (0,1),z∼q(z),x∼pt(x|z)∥
for some predefined conditional paths z, q(z), ut(x
z). Depending on the choice of conditioning and
z), and pt(x
|
|
probability paths we can recover most known flow matching techniques, such as the originally described flow
matching (Lipman et al., 2023), action matching (Neklyudov et al., 2022), stochastic interpolants (Albergo and
Vanden-Eijnden, 2023; Albergo et al., 2023), the 1-rectified flow (Liu, 2022), optimal transport conditional flow
matching (OT-CFM) (Tong et al., 2024) also known as multisample flow matching (Pooladian et al., 2023), and
its deterministic Schrödinger bridge counterpart, Schrödinger bridge conditional flow matching (SB-CFM) (Tong
et al., 2024).

vθ(t, x)

u◦
t (x

2.
∥

z)
|

(16)

−

SDEs vs. ODEs. Flow models are a powerful way to learn deterministic dynamics which are often faster to
sample from Song et al. (2021a), particularly with ideas from dynamic optimal transport (Tong et al., 2024;
Pooladian et al., 2023). However, recent work (Liu et al., 2023a; Shi et al., 2023) has noted advantages of
stochastic dynamics which we also observe (see Table 2 and Table 3), particularly in high dimensional settings
and in terms of generative performance. In this work, we also seek to model an inherently stochastic system,
where there is randomness introduced based on a variety of external factors. In particular, the fact that a
single-cell develops into a whole organism, and similarly, the fact that a single population of stem cells develops
into a multitude of different cell types, requires the modelling with stochastic dynamics that can model complex
conditional distributions. While it is possible to approximate the conditional distribution with lineage-tracing
techniques (Kester and van Oudenaarden, 2018; Wagner and Klein, 2020; Klein et al., 2023), these are biologically
complex, and we leave their analysis to future work.

C.2 Learning Schrödinger bridges

Schrödinger bridge models have been used as the source-conditioned variation of score-based generative models,
which can be conditioned on a variety of source distributions including dirac (Wang et al., 2021), and from data
or noised data (Somnath et al., 2023; Liu et al., 2023a). Most algorithms are based on mean-matching, which

Simulation-Free Schrödinger Bridges via Score and Flow Matching

is simulation-based and requires simulating, storing, and matching whole trajectories (De Bortoli et al., 2021),
with similar algorithms introduced by Vargas et al. (2021); Chen et al. (2022). In particular, these algorithms
parameterize a forward drift vf
, simulating trajectories in one direction and training the
θ
reverse direction to match them. Recent work extends this to Markovian bridges (termed bridge matching) (Shi
et al., 2023; Peluchetti, 2023), which greatly improves performance.

and backwards drift vb
θ

C.3 Practical implications of Schrödinger bridge modeling choices

Approximation error. All known algorithms for learning Schrödinger bridges for general distributions can only
approximate the Schrödinger bridge. Approximation error accumulates from a number of sources, which can
roughly be categorized into the following areas:

1. Error in the static joint optimization ˆπ(x0, x1) approximating the true joint π⋆(x0, x1). Since there is no closed
form for this optimization, all approximations are iterative and discrete. Error in this distribution accumulates
in two places, and can be divided into error on the marginals
, and error on
|
the joint

ˆπ(x0,

q(x0)

q(x1)

ˆπ(
|

, x1)

,
|

−

−

)

·

·

|

:

ˆπ(x0, x1)

|

π⋆(x0, x1)
|

−

(a) Marginal and Joint error from using a finite number of iterations (IPF underfitting).

IPF both in
continuous space (as used by mean-matching and bridge matching) and IPF in discrete space (as used
by [SF]2M) are run for a finite number of iterations. Continuous space IPF is much more expensive
(involving fitting and simulating neural networks) and is thus is used with tens of iterations, often L = 20
(De Bortoli et al., 2021; Shi et al., 2022). In contrast, discrete space IPF uses many more iterations. We
use the default python optimal transport (POT) parameters with Lmax = 1000 with early stopping if the
marginal error is < 10−9 (Flamary et al., 2021). The discrete OT computation adds a negligible (< 1%)
computational overhead in [SF]2M.

(b) Joint error from minibatch approximation (discrete IPF only). By using discrete OT solvers, we accumulate
discretization error between by using finite batch sizes. This affects the recovery of the true Schrödinger
bridge, but does not affect the marginal error, which is the important error for generative modeling
performance at the endpoints. As batch size
, this error goes to zero; for a finite dataset of small
enough size, [SF]2M can use full-batch discrete entropic OT, where the batch size used for OT can be
larger than that used for neural network training.

→ ∞

In practice, [SF]2M, by using discrete OT solvers, trades off increased error in the joint via minibatch
approximation error for reduced error from IPF underfitting. It is difficult to bound these errors theoretically,
however, our experiments show that this tradeoff is useful in practice. We leave further theoretical investigation
to future work.

2. Finite sample error. Datasets are often finite (i.e., discrete). Approximating the continuous densities from

discrete samples is challenging and accumulates error.

3. Neural network fitting approximation (applicable to all neural SDE approximations to Schrödinger bridges).
4. Discretization error. Simulation-free objectives like [SF]2M treat time as a continuous variable, but, at inference
time, error accumulates from integration in discrete time (e.g., using an Euler-Maruyama scheme). We note
that simulation-based objectives, such as that of DSB (De Bortoli et al., 2021), also train in a fixed time
discretization, inducing a further approximation error.

Iterative fitting. Both mean-matching and bridge matching Schrödinger bridge methods require an outer IPF
step, which samples data to train for each inner neural network optimization step (Table 1). At each step, new
data is resampled to determine the boundary distributions of the next iteration (Alg. 3). This multiplies the
training time by L (the number of outer loops), and cannot guarantee that that the neural networks match the
right endpoint marginals until the outer IPF step has converged.

In contrast, [SF]2M is always optimizing for the correct marginals, but at the cost of a biased Schrödinger bridge
due to bias in the minibatch OT. We hypothesize that this is one of the reasons that [SF]2M has better generative
modeling performance than mean-matching or bridge-matching methods.

In summary, we recommend using DSBM / IDBM when approximating the true Schrödinger bridge is important
and the computational budget is large. We recommend using [SF]2M when the computational budget is small or
when the generative performance is more important than matching the true Schrödinger bridge process.

A. Tong, N. Malkin, K. Fatras, L. Atanackovic, Y. Zhang, G. Huguet, G. Wolf, Y. Bengio

Table 7: Gaussian-to-Gaussian Schrödinger bridges with 10,000 datapoints. Here we text Sinkhorn-Exact which uses the
exact OT (default), against the Sinkhorn algorithm for the static OT within batches, and using outer loops where we
simulate 10,000 trajectories for further training 20 times following Shi et al. (2023).

Dim [SF]2M-Sinkhorn

[SF]2M-Exact

[SF]2M-Exact-Looped

KL(p1, q1)

Mean KL(pt, qt)

2
5
20
50
100

2
5
20
50
100

0.002
0.004
0.029
0.122
0.493

0.001
0.004
0.042
0.276
1.000

0.000
0.001
0.001
0.003
0.010

0.000
0.000
0.001
0.001
0.007

±
±
±
±
±

±
±
±
±
±

0.003
0.007
0.029
0.124
0.486

0.004
0.006
0.028
0.258
0.977

0.000
0.000
0.002
0.003
0.005

0.000
0.000
0.001
0.001
0.003

±
±
±
±
±

±
±
±
±
±

0.032
0.088
0.293
0.610
1.578

±
±
±
±
±

0.009
0.011
0.021
0.033
0.026

0.006
0.019
0.080
0.243
0.792

±
±
±
±
±

0.002
0.003
0.006
0.012
0.008

D ADDITIONAL RESULTS AND ABLATIONS

Figure 5: Learned ODE (top) and SDE (bottom) simulations for σ ∈ [0.1, 1, 2, 3] from left to right for trained [SF]2M
model. The marginals match regardless of the chosen σ. Trajectory initializations are matched between runs.

D.1 Looped [SF]2M

As previously discussed, the majority of Schrödinger bridge algorithms to date have used outer iterations to
perform iterative proportional fitting on the continuous distributions. This can create marginals closer to the
true Schrödinger bridge marginals at the cost of additional computation, and potentially worse generative
modelling performance.
[SF]2M is the first Schrödinger bridge method to approximate Schrödinger bridges
without performing the iterative proportional fitting in continuous time and space, instead using much more
efficient iterations in the static, discrete OT setting.

However, [SF]2M is compatible with outer looping. In Table 7 we see that outer looping (with 20 outer loops
following Shi et al. (2023)) produces better Schrödinger bridge marginals, but worse marginals at time 1 indicating
worse generative performance. We note that outer looping takes much longer to train as [SF]2M effectively takes a

Simulation-Free Schrödinger Bridges via Score and Flow Matching

Figure 6: From top to bottom vθ(t, x), sθ(t, x) and ut(x) inferred through (5) for the 8-Gaussians to moons dataset.

[SF]2M I-CFM OT-CFM DSBM FM RF

4.145

4.381

4.456

4.511 4.611 6.01

Table 8: CIFAR-10 FID using 100-step Euler integration. First 4 models trained with batch size 128 for 16 A100-hours.
Last 2 models from the DSBM paper, trained for ∼200 A100-hours. [SF]2M performs better than DSBM (with 1/12 the
compute) and the deterministic methods (I-CFM, OT-CFM, FM). Better performance can be obtained by considering
higher-order or adaptive step solvers.

single outer loop, and may have advantages even over a single outer loop as shown in Tong et al. (2024); Pooladian
et al. (2023) where the static solution sped up training in the deterministic setting.

Here we accomplish outer looping by simulating 5,000 (stochastic) trajectories from the backwards SDE and 5,000
trajectories from the forwards SDE. We then use the start and end points of these trajectories as samples from
the approximate static OT matrix in the next iteration. This algorithm is detailed in Alg. 3. We always set n
(the number of samples per outer loop) to the size of the empirical dataset, fix the number of outer loops to 20,
and the number inner loops to the [SF]2M without outer loops divided by 20. This gives the same number of
gradient descent steps for all methods, but we note that the outer loop methods require simulation for each outer
loop. This adds additional computation cost.

D.2 On the choice of static OT method: Sinkhorn vs. Exact

As the minibatch size gets large, [SF]2M with entropic OT with ϵ = 2σ2 is the correct choice. However, with
minibatching, we want a smaller ϵ. In practice we often use ϵ = 0 corresponding to exact (unregularized) optimal
transport. We test this on the Gaussian-to-Gaussian experiment in Table 7. Here we see that [SF]2M with
sinkhorn OT works better in low dimensions, but struggles in high dimensions. We believe this is because
minibatch-OT effectively adds more entropy in higher dimensions. More experimentation and theory is needed to
determine the optimal setting of ϵ for a given dataset and minibatch size.

D.3 Extra experiments on Cifar10

We have conducted experiments on Cifar 10 using the code from Tong et al. (2024) and the experimental evaluation
from Shi et al. (2023). Results are gathered in Table 8.

A. Tong, N. Malkin, K. Fatras, L. Atanackovic, Y. Zhang, G. Huguet, G. Wolf, Y. Bengio

Algorithm 3 Looped Minibatch Simulation-Free Score and Flow Matching Training

Input: Samplable source and target q0, q1, number of outer loop iterations L, inner loop iterations I, cache
size n, noise term σ, weighting schedule λ(t), initial networks vθ and sθ.
for Outer loop l

[1, . . . , L] do

[1, . . . , I] do

∈
for Inner loop i
if l = 0 then
x0, x1
π
←
x0, x1

∈
q⊗m
0

π⊗m

, q⊗m
1
∼
Sinkhorn(x0, x1, 2σ2)

▷ Or OT(x0, x1) see §A.1
▷ Resample OT pairs from π

t))

← N
x0, x1)
|

vθ(t, x)

t)x0, σ2t(1

−
2 + λ(t)2

(x; tx1 + (1

−

u◦
t (x
−
L[SF]2M)

θ

x0, x1)
∥

|

sθ(t, x)
∥

x log pt(x

x0, x1)
|

2
||

− ∇

▷ where ˆx1 is sampled according to Alg. 2
▷ where ˆx0 is sampled according to the backwards analog of Alg. 2

T⊗m

∼

else

x0, x1

∼
(0, 1)
t
∼ U
x0, x1)
pt(x
|
x
pt(x
L[SF]2M ← ∥
θ

∼

Update(θ,
←
∇
(x0, ˆx1)⊗n//2
(ˆx0, x1)⊗n//2
[Tf , Tb]

Tf
Tb
T

←
←
←
return vθ, sθ

Table 9: GRN recovery and leave-last-timepoint-out testing using single-cell gene expression over two simulated datasets.
We measure performance of predicting the distribution of the final left-out timepoints (2-Wasserstein and radial basis
function maximum mean discrepancy) as well accuracy of structure recovery (AUC-ROC and AP).

Bifurcating System

Trifurcating System

OT-CFM
[SF]2M
NGM-[SF]2Mσ=0
NGM-[SF]2Mσ=0.1
NGM-[SF]2Mσ=0.01
NGM-[SF]2Mσ=0.001
Spearman
Pearson
DREMI (Krishnaswamy et al., 2014)
Granger (Granger, 1969)

)
↓
0.105
0.097
0.112
0.089
0.101
0.063

W
0.782
0.791
0.783
0.835
0.813
0.844

2 (

±
±
±
±
±
±
—
—
—
—

)
↑

)

AP (
↑

—
—

) AUC-ROC (
RBF-MMD (
↓
0.004
0.005
0.006
0.011
0.008
0.018

0.054
0.056
0.055
0.064
0.064
0.082

—
—

0.786
0.723
0.715
0.699
0.755
0.744
0.594
0.664

±
±
±
±
±
±
±
±

0.081
0.014
0.047
0.043
0.003
0.000
0.017
0.013

0.521
0.444
0.442
0.418
0.438
0.415
0.293
0.421

±
±
±
±
±
±
±
±

0.160
0.030
0.033
0.060
0.002
0.000
0.011
0.043

±
±
±
±
±
±
—
—
—
—

)
↓
0.142
0.159
0.161
0.195
0.121
0.150

W
0.921
0.932
0.912
1.049
0.956
1.005

2 (

±
±
±
±
±
±
—
—
—
—

)
↑

)

AP (
↑

—
—

) AUC-ROC (
RBF-MMD (
↓
0.006
0.006
0.005
0.014
0.005
0.014

0.068
0.062
0.064
0.080
0.069
0.094

—
—

0.764
0.731
0.726
0.725
0.718
0.710
0.419
0.613

±
±
±
±
±
±
±
±

0.066
0.077
0.081
0.080
0.005
0.002
0.021
0.048

0.485
0.453
0.451
0.445
0.413
0.405
0.205
0.343

±
±
±
±
±
±
±
±

0.105
0.091
0.094
0.082
0.005
0.002
0.007
0.039

±
±
±
±
±
±
—
—
—
—

E SCHRÖDINGER BRIDGES WITH VARYING DIFFUSION RATE

While we consider constant diffusion for the majority of this paper, the theory also applies for time varying
diffusion with the variation as specified in the following Lemma.
Lemma E.1 (Brownian bridge with time-varying diffusion). Suppose xt is a stochastic process with values in
Rd, defined by initial conditions x0 = a and SDE dxt = σ(t) dwt, where σ(t) is continuous and
functions [0, 1]
→
positive on (0, 1). Define F (t) = (cid:82) t

0 σ2(s) ds. Then xt satisfies

xt

x1 = b

xt

|

(a, F (t))
(cid:18)

a + (b

∼ N

∼ N

a)

F (t)
F (1)

−

, F (t)

(cid:19)

.

F (t)2
F (1)

−

Proof. The constraints on σ guarantee that F has a unique inverse F −1 on [0, F (1)]. Consider the process

yt =

xF −1(F (1)t) −
(cid:112)F (1)

a

,

which is equivalently characterized by

xt = a + (cid:112)F (1)yF (t)/F (1).
A straightforward computation using the chain rule shows that yt satisfies y0 = 0 and dyt = dwt, i.e., yt is
Brownian motion.

Simulation-Free Schrödinger Bridges via Score and Flow Matching

By standard facts about Brownian bridges, we have

yt

∼ N

y1 =

yt

|

b
a
(cid:112)F (1) ∼ N

−

(0, t)
(cid:32)

b
a
−
(cid:112)F (1)

(cid:33)

t)

.

t, t(1

−

The result follows by applying the reverse transformation to obtain the marginals of xt.

Because Markovization commutes with time reparametrization, and the SB is the Markovization of a mixture of
Brownian bridges, this immediately implies:

Corollary E.2. The solution p to the SB problem with reference process dxt = σ(t) dwt has marginal p(x0, x1) =
π2F (1)(x0, x1).

F EXPERIMENTAL DETAILS

Algorithm 4 Minibatch Optimal Transport Simulation-Free Score and Flow Matching Training

Input: Samplable source and target q0, q1, noise term σ, weighting schedule λ(t), initial networks vθ and sθ.
while Training do

q⊗m
0

, q⊗m
1
∼
Sinkhorn(x0, x1, 2σ2)

π⊗m

(0, 1)

t
∼ U
x0, x1
π
←
x0, x1
∼
pt(x
x0, x1)
|
x
pt(x
L[SF]2M ← ∥
θ

∼

|

Update(θ,

← N
x0, x1)

(x; tx1 + (1

−

t)x0, σ2t(1

−
2 + λ(t)2
∥

x0, x1)

vθ(t, x)

ut(x
−
|
L[SF]2M)

θ

∇

←
return vθ, sθ

t))

▷ Or OT(x0, x1) see §A.1
▷ Resample OT pairs from π

sθ(t, x)
∥

x log pt(x

x0, x1)

2
||

|

− ∇

▷ see (8, 11)

F.1

Implementation details and settings

Throughout we use networks of three layers of width 64 with SeLU activations (Klambauer et al., 2017) except
for the 1000 dimensional experiment where we use width 256, and for the neural graphical model (NGM) model
used in the gene regulatory network recovery task. For optimization we use ADAM-W (Loshchilov and Hutter,
2019) with learning rate 10−3 and weight decay 10−5. We train for 1,000 epochs unless otherwise noted. The flow
network and score networks always have exactly the same structure. We provide a more detailed picture of the
algorithm used in Alg. 4. For sampling we always take 100 integration steps with either the Euler integrator for
ODEs or Euler-Maruyama integrator for SDEs, except for the Gaussian experiments where we take 20 steps to
match the setup of De Bortoli et al. (2021); Shi et al. (2023). When there are multiple timepoints (e.g., single-cell
trajectories) we take 100 steps between each timepoint for a total of 100k steps for all methods. We use σ = 1
unless otherwise noted.

F.1.1 Weighting schedule λ(t)

log pt(x

Similar to other score and diffusion-based losses, the [SF]2M loss is defined with a weighting schedule λ(t). Since
z) goes to infinity as t tends to zero or one, we must standardize the loss to be roughly even over time.
∇
We set λ(t) such that the target has zero mean and unit variance, i.e., we predict the noise added in sampling x
from µt before multiplying by σ(cid:112)t(1

t). This leads to the setting:

|

−

this weighting schedule ensures that the regression target for sθ is distributed

(0, 1).

N

λ(t) = σt = σ(cid:112)t(1

t)

−

(17)

A. Tong, N. Malkin, K. Fatras, L. Atanackovic, Y. Zhang, G. Huguet, G. Wolf, Y. Bengio

We also experiment with directly regressing sθ against the scaled target 1
weighting function

2 σ2

x log pt(x

∇

z), with a different
|

2
σ2 σt =
which also ensures the regression target for σ is distributed
Gaussian pt(x

ϵ/σt where ϵ

λ(t) =

z) can be simplified to
|

t)

−

2(cid:112)t(1
σ
(0, 1). With this λ, and since
(0, 1) we have the simplified objective:

∇

N

x log pt(x

λ(t)2

sθ(t, x)

∥

− ∇

−
x log pt(x

x0, x1)

|

∼ N
2 =
∥

=

λ(t)sθ(t, x)
∥
λ(t)sθ(t, x) + ϵt
∥

λ(t)
2
∥

−

x log pt(x

x0, x1)
|

2
∥

∇

(18)

z) with a

|

(19)

(20)

This is also numerically stable as it avoids dividing by anything that approaches zero. We leave improved
weighting schedules to future work. In practice we use the schedule in (18) and the simplified objective in (20).

F.1.2 Static optimal transport

As discussed in §A.1 it is sometimes preferable to use exact optimal transport instead of entropic optimal transport.
In addition to the “entropy” added by minibatching, there are also numerical and practical considerations.
In practice, we find that for batches of size < 10000, implementations of exact OT through the Python
Optimal Transport package (POT) (Flamary et al., 2021) are often faster than implementations of the Sinkhorn
algorithm (Cuturi, 2013), as the Sinkhorn algorithm is known to have numerical difficulties and needs many
iterations for good approximation of the true entropic transport for small values of σ.

F.2 Computational resources

All experiments were performed on a shared heterogeneous high-performance-computing cluster. This cluster is
primarily composed of GPU nodes with RTX8000, A100, and V100 Nvidia GPUs, and CPU nodes with 32 and
64 CPUs.

F.3 Two-dimensional experimental details

For the two-dimensional experiments we follow the setup from Shi et al. (2023). This is adapted from the setup
of Tong et al. (2024) except using larger test sets for lower variance in the empirical estimation of the Wasserstein
distance. We use a training set size of 10,000, a validation set size of 10,000, and a test set size of 10,000 for all
methods and models.

We evaluate the empirical 2-Wasserstein distance for 10,000 forward samples from our model pushing the source
distribution to the target. The number reported for

2 is then

W
(cid:90)

(cid:18)

2 =

W

min
π∈U ( ˆp1,q1)

(cid:19)1/2

2
2dπ(x, y)
∥

y

x

∥

−

(21)

where ˆp1 is sampled via Alg. 2, and q1 is the test set.
As mentioned in the main text, we also measure the Normalized Path Energy. We note that this is only defined
for ODE integration, hence for stochastic methods, (i.e., [SF]2M, DSBM) we measure the normalized path energy
of the probability flow ODE (see (4)). The normalized path energy measures the relative deviation of the path
energy of the model ((cid:82)
2 (q0, q1)), which is equivalent
2) to the path energy of the optimal paths (
2
∥
to the squared 2-Wasserstein distance between the test set source and the test set target. More formally, the
normalized path energy can be calculated as

vθ(t, x)
∥

W

N P E(q0, q1, vθ) = |

Ex(0)∼q0

(cid:82)

2dt
vθ(t, xt)
∥
∥
2
2
W

2
2 (q0, q1)

|

− W

(22)

where xt is the solution of the probability flow ODE with dx = vθ(t, x)dt with initial condition x0. This measures
how close the paths defined by vθ(t, x) are to the optimal transport paths in terms of average energy. We note
that measuring the path energy rather than the length (
1) has the additional benefit that the
energy differentiates between models that follow the same paths, but at different rates, encouraging constant rate
models.

instead of

2
2
W

W

Simulation-Free Schrödinger Bridges via Score and Flow Matching

F.4 Gaussian-to-Gaussian experimental details

(
−

0.1, I) source to 10,000 points sampled from a

Similar to experiments in De Bortoli et al. (2021); Shi et al. (2023), we train on a sample of 10,000 points from a
(0.1, I) target, for dimensions 5, 20, and 50. We could not
N
find all the details of those previous experiments but we match what we can here. Since we know the closed-form
solution to the Schrödinger bridge from Mallasto et al. (2022); Bunne et al. (2022a) qt, we can compare our
estimate of the marginal at time t (ˆpt) with the true distribution at time t. In particular, we compare a Gaussian
approximation of ˆpt, ˜pt with mean and covariance estimated from 10,000 samples of the model with qt which has
the form

N

with the KL divergence.

qt(x) =

N

(cid:16)

x; 0.2t

0.1, (t(1

−

−

t)

4 + σ4 + (1

t)2 + t2)I

−

(cid:112)

(cid:17)

We compare either the average KL over 21 timepoints (including the start and end timepoints) i.e.,

KL =

1
20

20
(cid:88)

k=0

KL(˜pk/20||

qk/20)

(23)

(24)

to measure how closely the learned flow matches the true Schrödinger bridge marginals, and we also compare the
KL divergence at time t = 1, to measure the performance as a generative model.

KL(˜p1

q1)

||

(25)

F.5 Waddington’s landscape experimental details

We use two Waddington landscapes, one Gaussian to two Gaussians and cross-sectional measurements of Embryoid
Body (EB) data, to demonstrate the versatility of [SF]2M for trajectory inference. The three dimensions of the
landscape are space, time and potential.

xW (t, x). The
More specifically, the space dimension is evolved according to the drift of the SDE by ut(x) =
xEv(t, x) and
2 g(t)2Es, where vθ(t, x) =
potential dimension is the Waddington’s landscape by W := Ev + 1
xEs(t, x) are the flow and score for Langevin dynamics. Es and Ev are both parameterized by three
sθ(t, x) =
layer neural networks of width 64, the only difference with our standard implementation of vθ and sθ is that they
have output dimension width of one instead of d.

−∇
−∇

−∇

For the experiment of one Gaussian to two Gaussians, we train on a sample of 256 points from the one-dimensional
source
(1, 0.1) for 10,000 steps.
We then plot 20 trajectories from the source to the target with the potentials following from the gradient descend
of W .

(0, 0.1) to 256 points sampled from the one-dimensional target

1, 0.1)

(
−

∪ N

N

N

For the cross-sectional measurements from the embryoid body (EB) data, we first embed the data in one dimension
with the non-linear dimensionality reduction technique PHATE (Moon et al., 2019), which we then whiten to
ensure the data is at a reasonable scale for the neural network initialization. We train the [SF]2M model following
Alg. 5 for 50,000 steps and plot 100 stochastic trajectories along with the height of W (t, x) normalized so to
gradually descend over time.

F.6 Single-cell interpolation experimental details

Here we perform two comparisons, the first matching the setup of Tong et al. (2020) in low dimensions and the
second exploring higher dimensional single-cell interpolation. Following Huguet et al. (2022b), we repurpose the
CITE-seq and Multiome datasets from a recent NeurIPS competition for this task (Burkhardt et al., 2022) as well
as the Embryoid-body data from Moon et al. (2019); Tong et al. (2020), which has 5 population measurements
over 30 days.

For the Embryoid body (EB) data, we use the same processed artifact which contains the first 100 principal
components of the data. For our tests in Table 4, we truncate to the first five dimensions, then whiten each
dimension following Tong et al. (2020) before interpolation. For the Embryoid body (EB) dataset which consists
of 5 timepoints collected over 30 days we train separate models leaving out times 1, 2, 3 in turn. During testing we

A. Tong, N. Malkin, K. Fatras, L. Atanackovic, Y. Zhang, G. Huguet, G. Wolf, Y. Bengio

Algorithm 5 Trajectory Simulation-Free Score and Flow Matching Training

Input: Samplable source and target Q =
vθ and sθ.
while Training do

q0,
{

· · ·

, qK−1

}

, noise term σ, weighting schedule λ(t), initial networks

for k

[0,
∈
xk, xk+1
π
Xk

, K
2] do
· · ·
−
q⊗m
, q⊗m
k
k+1
∼
Sinkhorn(xk, xk+1, 2σ2)
←
π⊗m
∼
(1, K)⊗m
k
∼ U
(0, 1)⊗m
t
∼ U
Xi
xi
0, xi
1 ←
k
x0, x1)
pt(x
|
pt(x
x
L[SF]2M ← ∥
θ

vθ(t + k, x)
θ

(x; tx1 + (1

← N
x0, x1)

Update(θ,

ut(x

∼

−

|

−
L[SF]2M)

∇

←
return vθ, sθ

t)x0, σ2t(1

t))

−
2 + λ(t)
∥

x0, x1)
∥
|

▷ Or OT(xk, xk+1) see §A.1
▷ Resample OT pairs from π

sθ(t + k, x)

x log pt(x

x0, x1)

2
||

|

− ∇

push forward all observed points Xt−1 to time t then measure the 1-Wasserstein distance between the predicted
and true distribution.

For the Cite and Multi datasets these are sourced from the Multimodal Single-cell Integration challenge at
NeurIPS 2022, a NeurIPS challenge hosted on Kaggle where the task was multi-modal prediction (Burkhardt
et al., 2022). Here, we repurpose this data for the task of time series interpolation. Both of these datasets consist
of four timepoints from CD34+ hematopoietic stem and progenitor cells (HSPCs) collected on days 2, 3, 4, and 7.
For more information and the raw data see the competition site.1 We preprocess this data slightly to remove
patient specific effects by focusing on a single donor (donor 13176).

Since these data have the full (pre-processed) gene level single-cell data, we try interpolating on higher-dimensional
unwhitened principle components, and on the first 1000 highly variable genes, which is a standard preprocessing
step in single-cell data analysis. To our knowledge, [SF]2M is the first method to scale to the gene space of
single-cell data. In Table 5, we again measure the 1-Wasserstein distance between the push forward predicted
distribution and the ground truth distribution.

F.6.1 Geodesic ground costs

We also introduce the Geodesic Sinkhorn method from Huguet et al. (2022b) for dynamic Schrödinger bridge
interpolation. Here the cost is a geodesic cost based on a k-nearest-neighbour graph between cells.

F.7 Gene regulatory network recovery experimental details

Using the neural graphical model (NGM), we can parameterize the gene-gene interaction graph directly within
the ODE drift model vθ(t, x). To do so, following from Bellot and Branson (2022) we can define:

vθj (t, x) = ϕ(

· · ·

ϕ(ϕ(xθ(1)

j )θ(2)
j )

)θ(K)
j

,

· · ·

j = 1, . . . , d,

(26)

j ∈
−

1 are parameters of each proceeding hidden layer, θ(K)

Rd×h represents a continuous adjacency matrix of the gene-gene interactions, θ(k)

where θ(1)
Rh×h, k =
) is an activation
2, . . . , K
function. Then we can consider vθ(t, x) = (vθ1(t, x), . . . , vθd (t, x)) as h ensembles over structure θ(1). We can
then use Algorithm Alg. 1 to train the NGM model with the addition of an L1 penalty over structure to enforce
sparsity on gene-gene interactions, i.e λ1
Using BoolODE (Pratapa et al., 2020), we generate simulated single-cell gene expression trajectories for a
bifurcating system and a trifurcating system. For the bifurcating system we consider 7 synthetic genes and
generate trajectories over 1000 cells using a simulation time of 5 and an initial condition on gene 1 at a value of 1.

1. We include bias terms in our implementation of (26).

Rd, and ϕ(

Rh×1, x

j ∈

θ(1)

∈

∈

∥

∥

·

j

1https://www.kaggle.com/competitions/open-problems-multimodal/data

Simulation-Free Schrödinger Bridges via Score and Flow Matching

Figure 7: Simulated single-cell trajectories given synthetic GRNs emulating bifurcating (top) and trifurcating (bottom)
systems. GRNs contain directed edges of Boolean relationships between genes. For example, a red edge between gene 3
and gene 0 (top left) indicates that the rule for gene 0 is (not gene 3). Likewise, a blue edge between gene 6 and gene 4
indicates that rule for gene 6 is (gene 4 and gene 6). This follows from the procedure for defining synthetic GRNs using
the BoolODE framework (Pratapa et al., 2020). The color bar indicates the scale of temporal progression.

We post-process the data and sub-sample to 55 timepoints and scramble the cell pairing to emulate real-world
data. For the trifurcating system we consider 9 synthetic genes and generate trajectories over 800 cells using a
simulation time of 6 and an initial condition on gene 1 at a value of 1. We post-process the data and sub-sample to
66 timepoints and scramble the cell pairing to emulate real-world data. We use a train-test data split of
0.8, 0.2
}
{
respectively, and leave out the end timepoints for trajectory prediction evaluation. We the show underlying
synthetic GRNs and simulated single-cell trajectories in Fig. 7.

For OT-CFM (i.e., [SF]2M with σ = 0), we parameterize the NGM model with two hidden layers where θ(1)
j ∈
with h = 100 and d represents the number of input genes. Then the second layer (i.e., k = 2) is θ(2)
Rh×1. We
use this parameterization for both the bifurcating system and trifurcating systems. For [SF]2M with σ > 0, we use
two heads stemming from θ(1)
for the flow matching model and score matching model, respectively. Specifically,
we use an additional layer ˜θ(2)
j ), j = 1, . . . , d. We use the SeLU
activation functions for both models. To train [SF]2M and NGM-[SF]2M models on the bifurcating system, we
use the Adam optimizer with a learning rate of 0.01 and batch size of 128 and use λ1 = 10−5. On the trifurcating
system, we use the Adam optimizer with a learning rate of 0.01 and batch size of 64 and use λ1 = 10−6. We
generate results for 5 model seeds. For baseline methods (i.e., Spearman, Pearson, DREMI, and Granger) we
generate results over 5 cell-pair scramble seeds. To evaluate GRN recovery performance, we compute the area

Rh×1 such that sθj (x, t) = ϕ(ϕ(xθ(1)

j )θ(2)

j ∈

j ∈

Rd×h

j

A. Tong, N. Malkin, K. Fatras, L. Atanackovic, Y. Zhang, G. Huguet, G. Wolf, Y. Bengio

under the receiver operator characteristic (AUC-ROC) and average precision (AP) scores of the predicted GRNs
compared to the ground truth GRNs used for generating the simulated data. We mask out the diagonal elements
(self regulation loops) of the predicted and ground truth GRNs for computing the AUC-ROC and AP. We provide
the full results of the GRN recovery experiments in Table 9.

