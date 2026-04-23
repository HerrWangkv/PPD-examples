5
2
0
2

g
u
A
7

]

V
C
.
s
c
[

4
v
8
3
9
4
0
.
1
1
3
2
:
v
i
X
r
a

Improved DDIM Sampling with Moment Matching Gaussian
Mixtures

Prasad Gabbur
Apple

pgabbur@apple.com

Abstract

We propose using a Gaussian Mixture Model (GMM) as reverse transition operator (ker-
nel) within the Denoising Diﬀusion Implicit Models (DDIM) framework, which is one of the
most widely used approaches for accelerated sampling from pre-trained Denoising Diﬀusion
Probabilistic Models (DDPM). Speciﬁcally we match the ﬁrst and second order central mo-
ments of the DDPM forward marginals by constraining the parameters of the GMM. We see
that moment matching is suﬃcient to obtain samples with equal or better quality than the
original DDIM with Gaussian kernels. We provide experimental results with unconditional
models trained on CelebAHQ and FFHQ, class-conditional models trained on ImageNet,
and text-to-image generation using Stable Diﬀusion v2.1 on COYO700M datasets respec-
tively. Our results suggest that using the GMM kernel leads to signiﬁcant improvements in
the quality of the generated samples when the number of sampling steps is small, as mea-
sured by FID and IS metrics. For example on ImageNet 256x256, using 10 sampling steps,
we achieve a FID of 6.94 and IS of 207.85 with a GMM kernel compared to 10.15 and 196.73
respectively with a Gaussian kernel. Further, we derive novel SDE samplers for rectiﬁed
ﬂow matching models and experiment with the proposed approach. We see improvements
using both 1-rectiﬁed ﬂow and 2-rectiﬁed ﬂow models.

1 Introduction

Diﬀusion models (Song & Ermon, 2019; Ho et al., 2020; Song et al., 2021), also known as Score based gen-
erative models (Sohl-Dickstein et al., 2015; Song et al., 2020), have demonstrated great success in model-
ing data distributions in various domains including images (Dhariwal & Nichol, 2021; Nichol & Dhariwal,
2021; Saharia et al., 2022; Rombach et al., 2022), videos (Ho et al., 2022; Blattmann et al., 2023), speech
(Kong et al., 2021) and 3D (Poole et al., 2022; Watson et al., 2022). This is due to their ﬂexibility in model-
ing complex multimodal distributions and ease of training relative to other competitive approaches such as
VAEs (Kingma & Welling, 2014; Rezende et al., 2014), GANs (Goodfellow et al., 2014; Salimans et al., 2016;
Karras et al., 2018; Brock et al., 2019), autoregressive models (van den Oord et al., 2016b;a) and normalizing
ﬂows (Rezende & Mohamed, 2015; Dinh et al., 2017), which do not exhibit both of the advantages simulta-
neously. In spite of their success, the main bottleneck to their adoption is the slow sampling speed, usually
requiring hundreds to thousands of denoising steps to generate a sample. A number of accelerated sampling
approaches have emerged, notably Denoising Diﬀusion Implicit Models (Song et al., 2021), pseudo numerical
methods (Liu et al., 2022), distillation (Salimans & Ho, 2022; Luhman & Luhman, 2021; Meng et al., 2023;
Yin et al., 2024; Sauer et al., 2024), consistency models (Song et al., 2023; Kim et al., 2023) and various
approximations to solving the reverse SDE (Song et al., 2020; Lu et al., 2022; 2023; Zhang & Chen, 2023).

Denoising Diﬀusion Implicit Models (DDIM) (Song et al., 2021) accelerate sampling from Denoising Diﬀu-
sion Probabilistic Models (DDPM) (Ho et al., 2020) by hypothesizing a family of non-Markovian forward
processes, whose reverse process (Markovian) estimators can be trained with the same surrogate objective
as DDPMs, assuming the same parameterization for reverse estimators. It is based on the observation that
a simpliﬁed DDPM training objective
simple,w (Eq. 4) depends on the forward process only through its
marginals at discrete steps t. In other words, one can sample with a pretrained DDPM denoiser by designing
a diﬀerent forward/backward process than the original DDPM given that the forward marginals are the same.

L

1

 
 
 
 
 
 
It has been shown (Xiao et al., 2022; Guo et al., 2023) that the true denoising conditional distributions of
DDPMs are multimodal, especially when the denoising step sizes are large. Although the unimodal Gaussian
kernel in DDIM yields a multimodal denoising conditional distribution, there is potential to improve its ex-
pressiveness with a multimodal kernel. It is not straightforward to use a multimodal kernel while satisfying
the marginal constraints. The work of Watson et al. (2021) shows that it is not necessary to satisfy the
marginal constraints to achieve accelerated sampling. Taking inspiration from these works, we propose using
Gaussian mixtures as the transition kernels of the reverse process in the DDIM framework. This results in
a non-Markovian inference process with Gaussian mixtures as marginals. Further, we constrain the mixture
parameters so that the ﬁrst and second order central moments of the forward marginals match exactly those
of the DDPM forward marginals. For brevity, we refer to central moments as simply moments in the rest of
the paper.

The work on stochastic interpolants (Albergo et al., 2023) uniﬁes diﬀusion and rectiﬁed ﬂow matching
(Liu et al., 2023; Lipman et al., 2023) into a common framework.
It has been shown (Gao et al., 2024)
that the deterministic Euler and DDIM samplers for rectiﬁed ﬂow models are the one and the same. We
build upon this equivalence and derive novel SDE samplers for rectiﬁed ﬂow matching, which allows us to
use the proposed approach within that framework. In summary, our main contributions are as follows:

1. We propose using Gaussian Mixture Models (GMM) as reverse transition operators (kernel) within
the DDIM framework, which results in a non-Markovian inference process with Gaussian mixtures
as marginals.

2. We derive constraints to match the ﬁrst and second order moments of the resulting forward GMM
marginals to those of the DDPM forward marginals. Based on these constraints, we provide three
diﬀerent schemes to compute GMM parameters eﬃciently.

3. We demonstrate experimentally that the proposed method results in further accelerating sampling

from pretrained DDPM models relative to DDIM, especially with fewer sampling steps.

4. We derive novel SDE solvers for rectiﬁed ﬂow matching models and extend them with the proposed

approach demonstrating improvements.

We begin by providing a brief overview of Diﬀusion and the DDIM framework in Section 2. Our approach
for extending DDIMs with Gaussian mixture transition kernels is provided in Section 3. We discuss the
proposed approach in the context of rectiﬁed ﬂow matching in Sec. 4. In Section 5, we conduct experiments
on CelebAHQ (Liu et al., 2015), FFHQ (Karras et al., 2019), ImageNet (Deng et al., 2009), and text-to-
image generation with Stable Diﬀusion (Rombach et al., 2022), and provide quantitative results. We also
report quantitative results of sampling from CIFAR-10 and ImageNet64 using 1-rectiﬁed ﬂow and 2-rectiﬁed
ﬂow models respectively. Prior work on accelerated sampling from diﬀusion models is discussed in Section 6.
Finally we conclude in Section 7.

2 Background

2.1 Denoising Diﬀusion Probabilistic Models

Denoising Diﬀusion Probabilistic Models (Ho et al., 2020) learn a model for the data distribution q(x0) by
designing a forward and backward diﬀusion process.
In the forward process, noise is added to the data
samples following a predetermined schedule, thereby transforming a structured distribution q(x0) at step
(0, I), at step T . This is achieved by setting up a Markov chain at every
t = 0 to Gaussian noise, q(xT ) =
step t with the transition kernel deﬁned by

N

1
(cid:18)
where αt are chosen such that the marginal q(xT ) converges to
forward to obtain the marginal of the latent xt at any step t conditioned on data sample x0 as

αt
αt−1 (cid:19)
(0, I) with large enough T . It is straight-

xt−1) =
|

αt
αt−1

xt−1,

q(xt

(cid:18)r

(1)

N

N

−

(cid:19)

I

,

q(xt

x0) =
|

N

(√αtx0, (1

αt)I) .

−

(2)

2

In the backward process, a parameterized Markovian denoiser pθ(xt−1|
p(xT ) =
N
(ELBO) or minimize

xt), initialized with Gaussian noise,
(0, I), learns to estimate the distribution of xt−1 given xt to maximize the evidence lower bound

ELBO:

L

ELBO = Eq

L

DKL(q(xT

x0)
||
|

pθ(xT ))

DKL(q(xt−1|

pθ(xt−1|
xt, x0)
||

xt))

T

h
+

t=2
X
log pθ(x0|

−

.

x1)
i

(3)

xt, x0), which turns out to
The above is equivalent to training pθ(xt−1|
µθ(xt, t), Σθ(xt, t)), leads
be a Gaussian (Luo, 2022). Assuming a Gaussian form for pθ(xt−1|
to a simpliﬁed loss function for training DDPMs, which is optimized over µθ(xt, t) and Σθ(xt, t). It has
been found that a reparameterized mean estimator µθ(xt, t) as a function of added noise estimator ǫθ(xt, t)
has beneﬁts of better sample quality. The resulting simpliﬁed loss function
simple,w (Ho et al., 2020) is a
weighted version of ELBO with weights wt:

xt) to match the posterior q(xt−1|
(xt−1|

xt) =

N

L

simple,w = Et∼U[1,T ]q(x0)q(xt|x0)ǫ∼N (0,I)
L

wt

ǫ
||

−

ǫθ(xt, t)

2
||

.

(4)

ǫθ(xt, t) is modeled as a U-Net (Ho et al., 2020) or a transformer (Peebles & Xie, 2022). The covariance
estimator Σθ(xt, t) is either learned (Nichol & Dhariwal, 2021; Dhariwal & Nichol, 2021) or ﬁxed (Ho et al.,
2020).

(cid:2)

(cid:3)

2.2 DDIM

Following the same notation as Song et al. (2021), we refer to the forward process as inference and the reverse
process as generative. The key assumption is that the more general non-Markovian inference processes
qσ(x0:T ) have the same marginal distribution qσ(xt
x0) at every t as the DDPM, but not necessarily the
|
same joint distribution over all the latents qσ(x1:T
x0). The form of a Markovian family of generative process
|
in DDIMs (Song et al., 2021) is given by

t=T

(5)

(6)

.

qσ(x1:T

x0) := qσ(xT
|

x0)
|

t=2
Y
(√αT x0, (1

xt, x0),

qσ(xt−1|
αT )I) ,

−

qσ(xT

x0) :=
|

N

where σ

∈

RT

≥0 parameterizes the variances of the reverse transition kernels,

xt, x0) =

qσ(xt−1|
µt(xt, x0) = √αt−1x0 +

N

(cid:0)

µt(xt, x0), σ2
t I

,

(cid:1)
αt−1 −

−

1
q

t > 1,
xt

∀
σ2
t .

−
√1

√αtx0
αt

−
The transition kernel above ensures that the resulting marginals qσ(xt
x0) are identical to the DDPM
|
marginals in Eq. 2. The ODE perspective of DDIM and other related works on accelerated sampling from
diﬀusion models are discussed in Section 6.

3 Approach

We propose using a Gaussian Mixture Model (GMM) within the reverse transition kernels of the DDIM
generative process. Speciﬁcally, the form of transition kernels in Eq. 6 is given by

K

qσ,M(xt−1|

xt, x0) =

t ),

(µk

k=1
X

t , Σk

πk
t N
t = µt(xt, x0) + δk
t ,
t = σ2
t I

∆k
t ,

µk
Σk

−

t > 1,

∀

(7)

3

t , δk
πk

t , ∆k
t

(cid:0)

M

t =

, k = 1 . . . K denote the additional GMM parameters, speciﬁcally the mixture com-
where
ponent priors, mean and covariance oﬀsets relative to the single Gaussian counterparts of Eq. 6, respectively.
Further, we constrain the above kernel so that the ﬁrst and second order moments of the individual latent
x0) are the same as that of an equivalently parameterized DDPM (Eq. 2). This allows us
variables qσ,M(xt
|
to use DDIM sampling on a model trained with the same surrogate objective as the DDPM in Eq. 4 given
that the GMM parameters

t satisfy:

(cid:1)

M

K

πk
t = 1,

t δk
πk

t = 0,

K

k=1
X

t = δk

t (δk

t )T OR ∆k

t =

k=1
X
∆k

1
Kπk
t

K

l=1
X

tδl
πl

t(δl

t)T ,

(8)

where either one of the two constraints on the covariance matrix oﬀset ∆k
t is suﬃcient to yield the correct
moment matching. Please see Appendix A.1 for proof. We also provide an upper bound for the ELBO
loss using the proposed inference process as an augmented version of the
simple,w loss in Appendix A.4.
xt)
The proposed kernel yields a more expressive multimodal denoising conditional distribution qσ,M(xt−1|
compared to DDIM as shown in Section 3.2.

L

3.1 GMM Parameters

Sampling with the DDIM kernel of Eq. 6 requires choosing an appropriate value for the variance σ2
t , which
determines the stochasticity (Song et al., 2021) of the DDIM inference and sampling processes. It is speciﬁed
as a proportion η of the DDPM reverse transition kernel’s variance at the corresponding step t. The proposed
t at every step t during sampling. In what
approach requires choosing the additional GMM parameters
follows, we describe three diﬀerent ways to choose these parameters eﬃciently, without any training, to
satisfy the constraints in Eq. 8 while keeping additional computational requirements relatively low.

M

First we choose the mixture priors πk
t to be uniform or with a suitable random initialization so that they are
non-negative and sum to one. We experiment with choosing the mean oﬀsets δk
t either randomly (DDIM-
GMM-RAND) or followed by orthogonalization (DDIM-GMM-ORTHO) to allow for better exploration of
the latent space (xt, t > 0) as described below.

3.1.1 Method 1: DDIM-GMM-RAND

At every step t we sample random vectors ok
dimensionality equal to that of the latent variables xt
to yield the oﬀsets δt
k.

∈

t , k = 1 . . . K from an isotropic multivariate Gaussian with
RD. These vectors are mean centered and scaled

Ot

∼ N
K

(0, I), Ot

∈

RD×K, K < D,

¯ot =

πk
t Ot[k], Ct[k] = Ot[k]

¯ot,

−

δk
t =

k=1
X

s
Ct[k]

||

||2

Ct[k],

(9)

where Ot[k] denotes the kth column of the matrix Ot,
scale factor that controls the magnitude of the oﬀsets.

3.1.2 Method 2: DDIM-GMM-ORTHO

Ct[k]

||2 denotes the magnitude of Ct[k], and s is a

||

In order to allow for better exploration of the latent space of xt, the set of oﬀsets above is orthonormalized
using an SVD on the matrix Ot with ok
t as columns and choosing the ﬁrst K components of the output Ut

4

factor, i.e. the ﬁrst K eigenvectors of OtOT

t . Speciﬁcally

UtΣtV T
t = SV D(Ot)
K

¯ut =

πk
t Ut[k], Ct[1 : K] = Ut[1 : K]

k=1
X
δk
t = sCt[k],

¯ut,

−

(10)

where Ut[1 : K] are the ﬁrst K columns of Ut. The mean centering above ensures that the oﬀsets δk
the constraint in Eq. 8. The covariance parameters are chosen as

t satisfy

∆k

t =

1
Kπk
t

K

tδl
πl

t(δl

t)T

(11)

l=1
X
to satisfy the covariance constraint of Eq. 8. To sample from the reverse kernel of Eq. 7, we approximate the
covariance matrix σ2
t I
t to be diagonal. A straightforward approximation is to choose only the diagonal
elements of ∆k

∆k
t and subtract from σ2
t .

−

3.1.3 Method 3: DDIM-GMM-ORTHO-VUB

t , we also experiment with an upper bound diagonal approximation of ∆k
Given the random choice of oﬀsets δk
t
by eigen decomposition similar to PCA (Jolliﬀe, 1986), These variance upper bounds (VUB) determine the
maximum allowable variances for the dimensions in ∆k
t keeping the total variance the same. If λi, i = 1 . . . K
are the eigenvalues of ∆k
t , then

s2
Kπk

t  

π1
t −

K

(πl

t)2

λ1 ≤

! ≤

l=1
X
s2
Kπk
t

πi−1
t ≤

λi

≤

s2
Kπk
t

s2
Kπk
t

π1
t ,

πi
t,

i = 2 . . . K.

(12)

It turns out the upper bounds above are independent of the δk

t ’s.
Please see Appendix A.2 for proof.
We use the upper bounds in Eq. 34 to compute the diagonal approximation of σ2
t I
t by oﬀsetting the
ﬁrst K elements of σ2
t I with the eigenvalue upper bounds. The scale s can be chosen such that the upper
bounds are always smaller than σ2
t to ensure positive variances across all dimensions. Note that the above
diagonalization and variance oﬀsetting has to be done only once before sampling, which introduces additional
computation for initialization but not during sampling. We also experiment with sharing GMM parameters
across sampling steps t to save time by avoiding the expensive SVD operation for each step and doing it
only once. We denote this approach as DDIM-GMM-ORTHO-VUB∗. Please see Appendix A.10 for further
discussion on additional computational overhead.

∆k

−

3.2 DDIM-GMM as a Multimodal Denoiser

xt), to be estimated by the denoiser pθ(xt−1|

Recent works (Guo et al., 2023; Xiao et al., 2022) have shown that the target conditional distribution
xt), is multimodal in real-world datasets. Here we
q(xt−1|
show that the proposed DDIM-GMM sampling scheme addresses the unimodal assumption of the single
Gaussian denoisers in pre-trained diﬀusion models better than DDIM. We start by showing that the pro-
xt). Let the true data
posed DDIM-GMM kernel yields a multimodal conditional distribution qσ,M(xt−1|
distribution q(x0) be a Dirac distribution given by

q(x0) =

i
X

wiδ(x0 −

xi

0),

(13)

where xi
can be obtained using Bayes’ rule:

0 are the observed data points. The DDIM-GMM denoiser’s conditional distribution qσ,M(xt−1|

xt)

5

qσ,M(xt−1|

xt) =

∝

=

x0

Z

x0

Z

i
X

qσ,M(xt−1|

xt, x0)qσ,M(x0|

xt) dx0

qσ,M(xt−1|
wiqσ,M(xt−1|

xt, x0)qσ,M(xt

x0)q(x0) dx0
|

xt, xi

0)qσ,M(xt

0),

xi
|

(14)

xi
|

which is a mixture of Gaussians. This follows from the fact that qσ,M(xt−1|
0) is a mixture of Gaussians
given by Eq. 7 and qσ,M(xt
0) is a scalar constant given xt. A similar argument holds even when the
data distribution q(x0) is a mixture of Gaussians. The resulting denoiser is a mixture of Gaussians, whose
form can be obtained by noting that qσ,M(xt
0) is a GMM and accordingly completing squares within
the integrand above (Bishop, 2006). A similar argument as above also enables DDIM sampler to model a
xt, xi
multimodal denoising distribution qσ(xt−1|
0)
in Eq. 14 enables DDIM-GMM to express more complex denoising distributions than DDIM. This has the
xt), especially when the number of sampling
potential to better match the unknown distribution q(xt−1|
steps is small (Guo et al., 2023; Xiao et al., 2022), without any training or ﬁne-tuning with specialized loss
functions.

xt). However the multimodality of the kernel qσ,M(xt−1|

xi
|

xt, xi

4 Rectiﬁed Flow Implicit Models (RFIM)

Flow matching (Lipman et al., 2023; Albergo et al., 2023; Albergo & Vanden-Eijnden, 2023) methods learn
a mapping between a source distribution (x0 ∼
q1) at t = 1 by
formulating a stochastic process (xt
[0, 1]) as a bridge between the two distributions. Speciﬁcally,
rectiﬁed ﬂow matching (Liu et al., 2023) deﬁnes an optimal transport (OT) path between x0 and x1 as

q0) at t = 0 and a target distribution (x1 ∼

pt, t

∼

∈

xt = (1

−

t)x0 + tx1, t

[0, 1].

∈

(15)

In order to transfer samples from source to target distribution or vice versa, a velocity function vθ(xt, t),
conditioned on xt and parameterized by θ, is learnt to match the instantaneous velocity dxt
x0)
in expectation by sampling (x0, x1) from a pre-deﬁned coupling of q0 and q1, usually the independent
q1(x1). This is achieved by minimizing an unconstrained least squared error loss within
coupling q0(x0)
the conditional ﬂow matching framework (Liu et al., 2023):

dt = (x1 −

∗

CF M (θ) = Ex0∼q0,x1∼q1,t∼U(0,1)

L

(x1 −
||

x0)

−

vθ(xt, t)

2
||

,

(16)

where xt is given by Eq. 15. Starting from an initial sample x0 from the source distribution, solving the
following ﬁrst order ODE using the learned velocity vθ(xt, t) yields samples from the target distribution at
t = 1:

(cid:3)

(cid:2)

dxt = vθ(xt, t)dt.

(17)

[0, 1], for
Usually the ODE is run for a ﬁnite number of steps T by discretizing the continuous interval t
e.g. t = i
T , i = 0 . . . T . The Euler update in Eq. 17 has been shown to be equivalent to the deterministic
DDIM step (η = 0) in the context of Diﬀusion models (Gao et al., 2024). Motivated by this equivalence,
we propose a family of implicit models for rectiﬁed ﬂow matching similar to DDIM for diﬀusion. The key
observation is that the simpliﬁed diﬀusion loss of Eq. 4 and the conditional ﬂow matching loss of Eq. 16 have
been shown to be equivalent by a simple reparameterization of the model and an appropriate choice of the
loss weights wt (Lee et al., 2024; Kim et al., 2025). This leads to a family of parameterized implicit models
for sampling, similar to DDIM, whose marginals at every intermediate time step t match the corresponding
marginals from which the training data was sampled.

∈

Assume that q0 is the true data distribution and q1 is the standard Gaussian
to verify that the marginal of the interpolant xt conditioned on x0 is given by

N

(0, I). It is straightforward

q(xt

x0) =
|

N

((1

−

t)x0, t2I).

6

(18)

We deﬁne a parameterized Markov chain to sample the interpolants xti for discrete ti = i

T , i

0 . . . T

as

}

∈ {

qσ(xt1:T |
qσ(xtT |

x0)

x0) := qσ(xtT |
(0, I),
x0) :=

N

i=T

i=2
Y

qσ(xti−1 |

xti, x0),

(19)

where σ
∈
from any t to s, 0

RT

≥0 parameterizes the variances of the Markov chain transition kernels. The transition kernel

≤

s < t

≤
qσ(xs

1, is given by

xt, x0) =
|

µti (xt, x0) =

µt(xt, x0), σ2
t I

σ2
t

xti +

(1

N
s2

(cid:0)
−
t

,

0
∀

≤

(1

s) +

s < t

≤

1,

s2

t)

−
t

(cid:1)
−

−

σ2
t

x0.

(20)

(cid:18)

p
Using the above transition kernel, it can be veriﬁed that the marginal qσ(xs
x0) has the same form as the
|
interpolant marginal of Eq. 18 at t = s. Therefore samples from the proposed Markov chain come from
the same marginal distributions as the interpolant marginals for all t conditioned on x0. In practice, we
do not have access to x0 but use an estimate from a trained model. Speciﬁcally, we use the equivalence
between diﬀusion and rectiﬁed ﬂow matching loss (Lee et al., 2024; Kim et al., 2025) and adapt an x0(xt, t)
prediction model to yield the velocity vt(xt, t). Speciﬁcally, following the Rectiﬁed Flow++ work (Lee et al.,
2024), the velocity vt(xt) can be expressed as

q

(cid:19)

vt(xt) =
where a parametric model for predicting E [x0|
being obtained from the reparameterization of Eq. 21.

:= xt

dxt
dt
−
xt] is trained with the

E [x0|

xt] ,

(21)

CF M loss of Eq. 16 with vθ(xt, t)

L

s2, which leads to a couple of interesting special
Note that in order for Eq. 20 to be a valid kernel, 0
≤
cases at the extremities. If σt = 0, the resulting sampler reduces to a discrete version of the Euler ODE
sampler of Eq. 17. At the other extreme (σt = s), the sampler becomes non-Markovian yielding independent
samples from the interpolant marginal distribution conditioned on x0, i.e., qσ(xs
x0). We
|
[0, 1] similar to DDIM (Song et al., 2021) for
experiment with σt = ηs for a few diﬀerent values of η
diﬀusion models.

xt, x0) = q(xs
|

σ2
t ≤

∈

4.1 Moment Matching Gaussian Mixtures (RFIM-GMM)

We extend our proposed approach to rectiﬁed ﬂow models by using Markov transition kernels that use
Gaussian mixtures in place of unimodal Gaussians. The resulting samplers are denoted with the RFIM-
GMM-* notation similar to DDIM-GMM-*. Speciﬁcally the form of the RFIM-GMM kernels is the same as
Eq. 7 with t replaced by its discrete counterpart ti
(0, 1]. Similarly the constraints of Eq. 8 satisfy the
∈
moment matching condition, i.e., the ﬁrst and second order central moments of the RFIM-GMM sampler
x0) for each
marginals qσ,M(xti |
ti. We experiment with three diﬀerent choices for choosing the mean oﬀsets of the Gaussian mixture kernels
described in Sec. 3.1 and refer to them as RFIM-GMM-RAND, RFIM-GMM-ORTHO and RFIM-GMM-
ORTHO-VUB respectively.

x0) match the correpsonding moments of the interpolant marginals q(xti |

We show that the RFIM-GMM kernels provide more expressivity relative to their unimodal counterparts
(RFIM) in modeling the true denoising distributions q(xti−1 |
within the rectiﬁed ﬂow match-
ing framework. This can be seen by examining the form of the denoising distributions between two successive
discretized interpolant time points (ti−1, ti) as

1 . . . t
}

xti ), i

∈ {

q(xti−1 |

xti) =

x0

Z

∝

x0

Z

∝

x0

Z

q(xti−1|

xti , x0)q(x0|

xti) dx0

δ(xti−1 −

xti −

(xti −

x0)(ti−1 −

ti))q(xti |

x0)q(x0) dx0

δ(xti−1 −

xti −

(xti −

x0)(ti−1 −

ti)) exp

xti −
||

(1

−
t2
i

2

ti)x0||

(cid:19)

−

(cid:18)

q(x0) dx0,

(22)

7

30

25

D
I
F

20

15

25

D
I
F

20

15

η

= 0.0

η

= 0.2

η

= 0.5

η

= 1.0

DDIM

DDIM

DDIM

DDIM

40

70

DDIM-GMM-RAND

DDIM-GMM-RAND

DDIM-GMM-RAND

DDIM-GMM-RAND

30

35

60

DDIM-GMM-ORTHO

DDIM-GMM-ORTHO

DDIM-GMM-ORTHO

DDIM-GMM-ORTHO

DDIM-GMM-ORTHO-VUB

DDIM-GMM-ORTHO-VUB

DDIM-GMM-ORTHO-VUB

DDIM-GMM-ORTHO-VUB

DDPM

25

DDPM

D
I
F

20

15

DDPM

30

D
I
F

25

20

15

DDPM

50

D
I
F

40

30

20

10

20

40

60

80

100

20

40

60

80

100

20

40

60

80

100

20

40

60

80

100

#Steps

#Steps

#Steps

#Steps

η

= 0.0

η

= 0.2

η

= 0.5

η

= 1.0

DDIM

DDIM

35

DDIM

80

DDIM

30

DDIM-GMM-RAND

DDIM-GMM-RAND

DDIM-GMM-RAND

DDIM-GMM-RAND

70

DDIM-GMM-ORTHO

DDIM-GMM-ORTHO

30

DDIM-GMM-ORTHO

DDIM-GMM-ORTHO

25

DDIM-GMM-ORTHO-VUB

DDIM-GMM-ORTHO-VUB

DDIM-GMM-ORTHO-VUB

60

DDIM-GMM-ORTHO-VUB

DDPM

DDPM

DDPM

DDPM

D
I
F

20

15

25

D
I
F

20

15

10

50

D
I
F

40

30

20

10

10

10

20

40

60

80

100

20

40

60

80

100

20

40

60

80

100

20

40

60

80

100

#Steps

#Steps

#Steps

#Steps

Figure 1: CelebAHQ (top) and FFHQ (bottom). FID (
↓
for 1000 steps.

). The horizontal line is the DDPM baseline run

In general, the true data distribution q(x0) is
where we have made use of Eq. 18 to arrive at Eq. 22.
multimodal (e.g. Eq. 13) implying a complex multimodal form of the true denoising distribution. Using the
xti ) relative to
same arguments as Sec. 3.2, RFIM-GMM kernels enable more ﬂexibility in modeling q(xti−1 |
RFIM kernels.

5 Experiments

In this section we compare the quality of samples generated using the proposed approach with those gen-
erated by the original DDIM sampling. Using diﬀusion models, we conduct experiments on CelebAHQ
(Liu et al., 2015) and FFHQ (Karras et al., 2019), which are high resolution face datasets used as standard
benchmarks for evaluating generative models. We also evaluate the eﬀectiveness of the proposed approach
on sampling from class-conditional distributions by training on the ImageNet dataset with conditioning
on class labels (Rombach et al., 2022). The sample quality is measured using Frechét Inception Distance
(FID) (Heusel et al., 2017) and Inception score (IS) (Salimans et al., 2016) for class-conditional generation.
We train diﬀusion models using the unweighted DDPM objective (Ho et al., 2020) in the latent space of a
VQVAE (Rombach et al., 2022). More experimental details can be found in the Appendix A.5. For each
dataset, we generate as many samples as in the standard validation split of the dataset to compute the FID
and IS metrics, i.e., 5000 for CelebAHQ, 10000 for FFHQ and 50000 for ImageNet respectively. Experimental
results on text-to-image generation using Stable Diﬀusion v2.1 on the COYO700M dataset (Byeon et al.,
2022) are reported in Sec. 5.3. We also report experimental results with rectiﬁed ﬂow matching models
in Sec. 5.4. Speciﬁcally, we compare RFIM vs. RFIM-GMM samplers on the FID metric computed using
50000 samples generated from 1-rectiﬁed ﬂow and 2-rectiﬁed ﬂow models on CIFAR10 (Krizhevsky, 2009)
and ImageNet64 (Deng et al., 2009) datasets respectively.

5.1 Unconditional Models on CelebAHQ and FFHQ

The FID scores of unconditional generation models on CelebAHQ and FFHQ datasets are reported in Fig. 1.
We run both DDIM and the proposed variants of DDIM-GMM samplers for diﬀerent numbers of steps (10,
20, 50, 100) using diﬀerent values of the stochasticity parameter η (Song et al., 2021). For each DDIM-
GMM variant, we choose a GMM with 8 mixture components with uniform priors (πk
t =0.125) for all steps
t. We also search for the best value of scaling s among
and report the best result with
the chosen value for s. For all the experiments here, we set the value of s to be the same for all steps
t.
It is possible to further tune these parameters. For instance one could search for an optimal set of

0.01, 0.1, 1.0, 10.0

}

{

8

16

14

12

10

D
I
F

8

6

4

2

0

350

300

250

200

S

I

150

100

50

0

η

= 0.0

η

= 0.2

η

= 0.5

η

= 1.0

DDIM

DDIM-GMM-ORTHO

DDIM

DDIM-GMM-ORTHO

DDIM

DDIM-GMM-ORTHO

DDIM

DDIM-GMM-ORTHO

16

16

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

20

14

14

DDPM

DDPM

DDPM

DDPM

DDPM

DDPM

12

10

D
I
F

8

6

4

2

0

0
1

0
0
1

12

10

D
I
F

8

6

4

2

0

0
1

0
0
1

15

D
I
F

10

5

0

DDPM

DDPM

0
1

0
0
1

#Steps

#Steps

#Steps

0
1

0
0
1

#Steps

η

= 0.0

η

= 0.2

η

= 0.5

η

= 1.0

DDIM

DDIM-GMM-ORTHO

DDIM

DDIM-GMM-ORTHO

DDIM

DDIM-GMM-ORTHO

DDIM

DDIM-GMM-ORTHO

350

350

350

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

300

300

300

DDPM

DDPM

DDPM

DDPM

DDPM

DDPM

DDPM

DDPM

250

250

250

200

S

I

200

S

I

200

S

I

150

150

150

100

100

100

0
1

0
0
1

#Steps

50

0

0
1

0
0
1

50

0

0
1

0
0
1

50

0

0
1

0
0
1

#Steps

#Steps

#Steps

Figure 2: Class-conditional ImageNet with Classiﬁer-free Guidance. FID (
)
) (top) and IS (
↑
↓
(bottom) on the 50k ImageNet validation set. Classiﬁer-free guidance with a scale of 2.5 is used during
inference. The horizontal line is the DDPM baseline run for 1000 steps.

L

parameters using a suitable objective (Watson et al., 2021; Mathiasen & Hvilshøj, 2021) on the training set.
We also run full DDPM sampling for 1000 steps as a baseline since all the models were trained for 1000
simple,w DDPM objective (Eq. 4) with uniform weights (w = 1). We
steps in the forward process using the
observe that sampling with a GMM transition kernel (DDIM-GMM-*) shows signiﬁcant improvements in
sample quality over the Gaussian kernel (DDIM) at lower values of sampling steps and higher values of η for
unconditional generation on both CelebAHQ and FFHQ (see also Tables 7 and 8, Appendix A.14). Among
the diﬀerent choices for computing GMM oﬀset parameters, DDIM-GMM-RAND and DDIM-GMM-ORTHO
produce similar quality results. We observe signiﬁcant improvements with upper bounding variances with
the DDIM-GMM-ORTHO-VUB variant. Our hypothesis is that the GMM kernel allows exploring the latent
space better than the Gaussian kernel under those settings. Variance upper bounding further encourages
this by lumping variances into fewer dimensions of ∆k
t . This is favorable since sampling time is a signiﬁcant
bottleneck for the use of DDPM in real-time applications.

5.2 Class-conditional ImageNet

We train class-conditional models on ImageNet and experiment with guided sampling using classiﬁer-free
guidance (Ho & Salimans, 2021). Speciﬁcally, we jointly train a class-conditional and unconditional model
with parameter sharing (Ho & Salimans, 2021) by setting the unconditional training probability to 0.1. We
then sample from this model using a guidance scale of 2.5 for 10 and 100 sampling steps. Note that each
sampling step involves two Neural Function Evaluations (NFE) using classiﬁer-free guidance in order to
compute conditional and unconditional scores with the same denoising network. The FID and IS results are
shown in Fig. 2. Using fewer sampling steps (10), the FID and IS scores of the samples improve signiﬁcantly
when any DDIM-GMM-* sampler is used, relative to DDIM (see Tables 11 and 12, Appendix A.14.3). No-
tably, the deterministic (η = 0) DDIM-GMM-ORTHO-VUB∗ sampler and the DDIM-GMM-ORTHO-VUB
sampler at η = 0.5 yield the best FID (6.72) and IS (211.93) respectively. Similar to previous results, among
DDIM-GMM-*, using variance bounding leads to more signiﬁcant improvements at higher η relative to other
variants.

Using 100 steps, samples from DDIM and DDIM-GMM-* variants have similar metrics in most settings with
some exceptions. DDIM-GMM-RAND and DDDIM-GMM-ORTHO yield signiﬁcantly better FID values
than DDIM under the η = 1 setting. DDIM-GMM-ORTHO-VUB samples have consistently higher IS values
under all settings and yield the best FID (9.13) at η = 0. We posit that the similar performance of DDIM
and DDIM-GMM-* samplers using larger number of steps is due to the possibility that the multimodality

9

η

= 0.0

η

= 0.2

η

= 0.5

η

= 1.0

25

25

25

DDIM

DDIM-GMM-ORTHO

DDIM

DDIM-GMM-ORTHO

DDIM

DDIM-GMM-ORTHO

DDIM

DDIM-GMM-ORTHO

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

30

20

15

D
I
F

10

5

0

5

0
1

#Steps

20

15

D
I
F

10

5

0

5

0
1

20

15

D
I
F

10

5

0

5

0
1

25

20

D
I
F

15

10

5

0

5

0
1

#Steps

#Steps

#Steps

Figure 3: Text-to-Image Generation. FID (
↓
v2.1 model. Classiﬁer-free guidance with a scale of 7.5 is used during inference.

) on a 30k subset of COYO-700M using the Stable Diﬀusion

η

= 0.0

η

= 0.2

η

= 0.5

η

= 1.0

35

DDIM

DDIM-GMM-ORTHO

DDIM

DDIM-GMM-ORTHO

DDIM

DDIM-GMM-ORTHO

30

DDIM

DDIM-GMM-ORTHO

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

30

25

20

S

I

15

10

5

0

5

0
1

#Steps

30

25

20

S

I

15

10

5

0

5

0
1

30

25

20

S

I

15

10

5

0

5

0
1

25

20

S

I

15

10

5

0

5

0
1

#Steps

#Steps

#Steps

Figure 4: Text-to-Image Generation. IS (
↑
v2.1 model. Classiﬁer-free guidance with a scale of 7.5 is used during inference.

) on a 30k subset of COYO-700M using the Stable Diﬀusion

xt)) is modeled equally well by both the samplers
of the true denoiser conditional distribution (q(xt−1|
(Guo et al., 2023; Xiao et al., 2022). See Appendix A.6 for additional results using classiﬁer-guidance. Also,
Appendix A.13 shows some qualitative results of sampling with the proposed approach compared to DDIM.

5.3 Text-to-Image Generation

In this section, we experiment with a pretrained text-to-image diﬀusion model. Speciﬁcally, we use the
publicly available Stable Diﬀusion v2.1 (Rombach et al., 2022) on a subset of 30,000 text and image pairs
from the large scale COYO-700M image-text pair dataset (Byeon et al., 2022). Stable Diﬀusion v2.1 is
a text-to-image diﬀusion model conditioned on text captions. It is trained on a subset of the large-scale
LAION-5B image-text pair dataset (Schuhmann et al., 2022). We use DDIM and the variants of DDIM-
GMM samplers, each with 5 and 10 sampling steps, to generate images at 256x256 resolution conditioned
on captions from the COYO-700M data subset. Fig. 3 and 4 show the FID and IS metrics respectively.
The FID metric improves consistently with the DDIM-GMM-* samplers relative to DDIM for all settings
of η using 10 sampling steps. The relative improvements with DDIM-GMM-ORTHO-VUB over DDIM are
more signiﬁcant with increasing η compared to DDIM-GMM-RAND and DDIM-GMM-ORTHO suggesting
better exploration of latent space, similar to results with unconditional models. Using 5 sampling steps, the
DDIM-GMM-* samplers show most improvements with η = 1. On the IS metric, all variants of DDIM-GMM
samplers show signiﬁcant improvements over DDIM under diﬀerent settings of η and sampling steps. Fig. 11
in Appendix A.13 shows some sample generated images.

5.4 Sampling from Rectiﬁed Flow Matching models

We report quantitative results by sampling from rectiﬁed ﬂow matching models using the proposed RFIM
and RFIM-GMM kernels. We experiment with both 1-rectiﬁed ﬂow and 2-rectiﬁed ﬂow matching models and
compute the FID (Heusel et al., 2017) metric to quantify sample quality. Speciﬁcally, we train a 1-rectiﬁed
ﬂow model on the CIFAR-10 dataset using the UNet architecture and OT-CFM loss (Tong et al., 2024). For
2-rectiﬁed ﬂow, we use the pre-trained ImageNet64 model from Lee et al. (2024) and sample from it. For
as we found that higher values of η led to worse sample
all our experiments, we choose η

0.0, 0.2, 0.5

∈ {

}

10

quality, likely due to the sampling chain becoming progressively non-Markovian and completely independent
at η = 1 (see Sec. 4). Table 1 shows the results of sampling from the 1-rectiﬁed ﬂow CIFAR-10 model for
up to 50 sampling steps. The RFIM-GMM kernels consistently lead to better sample quality (lower FID)
using diﬀerent values of η. We also see small improvements with the RFIM-GMM kernels on the pre-trained
2-rectiﬁed ﬂow ImageNet64 model when using only 1 or 2 sampling steps. In this case, the relative improve-
ments are the same for all the diﬀerent RFIM-GMM-* kernels. Interestingly, the additional stochasticity
from the oﬀsets of the RFIM-GMM kernels seems to improve the FID relative to the deterministic Euler
ODE (RFIM, η = 0) sampler.

Table 2: 2-Rectiﬁed Flow, Ima-
geNet64. FID (
)
↓
Steps

1

2

RFIM
RFIM-GMM-*
RFIM

η
0
0
0.2
0.2 RFIM-GMM-*
0.5
0.5 RFIM-GMM-*

RFIM

4.42
4.38
4.42
4.38
4.42
4.38

3.97
3.94
3.99
3.95
4.25
4.20

Table 1: 1-Rectiﬁed Flow, CIFAR10.
FID (
↓

)

Steps

2

5

10

50

RFIM
RFIM-GMM-RAND
RFIM-GMM-ORTHO

η
89.08
0
88.10
0
88.11
0
RFIM-GMM-ORTHO-VUB 88.04
0
89.66
RFIM
0.2
88.65
RFIM-GMM-RAND
0.2
0.2
88.66
RFIM-GMM-ORTHO
0.2 RFIM-GMM-ORTHO-VUB 88.65
93.15
RFIM
0.5
92.01
RFIM-GMM-RAND
0.5
0.5
92.00
RFIM-GMM-ORTHO
0.5 RFIM-GMM-ORTHO-VUB 92.00

25.20
24.61
24.62
24.62
25.87
25.17
25.18
25.17
30.02
29.03
29.02
29.02

14.13
13.69
13.69
13.69
14.41
14.20
14.19
14.20
17.90
17.59
17.59
17.59

6.37
6.16
6.13
6.13
6.52
6.27
6.26
6.27
17.67
17.32
17.34
17.35

6 Related Work

Prior and concurrent work on accelerated sampling for pretrained diﬀusion models can be broadly
categorized into implicit modeling (Song et al., 2021; Zhang et al., 2022; Watson et al., 2021), distilla-
tion (Luhman & Luhman, 2021; Salimans & Ho, 2022; Meng et al., 2023; Yin et al., 2024; Sauer et al.,
2024), consistency models (Song et al., 2023; Kim et al., 2023) and ODE solver (Song et al., 2020;
Jolicoeur-Martineau et al., 2021; Zhang & Chen, 2023; Karras et al., 2022; Lu et al., 2023; Liu et al., 2022)
based approaches. Watson et al. (2021) extend DDIMs by introducing a more general family of implicit dis-
tributions with learnable parameters trained with backpropagation using perceptual loss. While the proposed
approach also introduces learnable parameters, our marginals are Gaussian mixtures and we ensure that the
moments are matched exactly with those of the DDPM marginals. Zhang et al. (2022) analyze the workings
of DDIM using a limiting case of Dirac distribution in the data space and generalize it to non-isotropic diﬀu-
sion models. Other approaches propose accelerated sampling by modeling DDPMs with non-Gaussian noise
(Nachmani et al., 2021) or learning noise levels of the reverse process separately (San-Roman et al., 2021).
Diﬀerent from these, the proposed approach introduces a diﬀerent sampling kernel in the reverse process of
the DDIM framework (Song et al., 2021). Our work is also complementary to distillation based approaches,
which might further beneﬁt from an improved DDIM teacher (Salimans & Ho, 2022; Meng et al., 2023).

By treating sampling as solving reverse direction diﬀusion ODEs (Song et al., 2020), acceleration is achieved
by discretization with linear (Song et al., 2021) or higher order approximations (Jolicoeur-Martineau et al.,
2021; Lu et al., 2022; Zhang & Chen, 2023). Being a moment matching version of DDIM, the proposed
approach can be thought of as a linear solver. It is observed that higher-order solvers are inherently unstable
in the guided sampling regime, especially if the guidance weight is high (Lu et al., 2023). Our empirical
results suggest that, even with a high guidance weight, moment matching based DDIM-GMM is beneﬁcial
for guided sampling with few sampling steps. Rectiﬁed ﬂow matching (Liu et al., 2023; Lipman et al., 2023;
Albergo et al., 2023) framework inherently learns to produce straighter sampling paths compared to diﬀusion
models leading to relatively faster sampling using ODE solvers such as the ﬁrst order Euler or the second
order Heun (Lee et al., 2024) solvers. Multiple rectiﬁcation iterations (Liu et al., 2023; Lee et al., 2024;

11

Kim et al., 2025) further help straighten the paths improving sampling eﬃciency. Our work proposes novel
SDE solvers that includes Euler ODE as a special case.

7 Conclusions

We propose improved DDIM sampling by using Gaussian mixture transition kernels whose marginal ﬁrst and
second order moments match the corresponding moments of the DDPM forward marginals. Our experiments
suggest that moment matching is suﬃcient to produce samples of the same or better quality than the original
DDIM sampler. This is especially true if the number of sampling steps is small (e.g. 10) using unconditional
models trained on CelebAHQ and FFHQ. For guided ImageNet class-conditional models, the GMM kernel
based samplers lead to improvements in both FID and IS metrics under almost all settings of η and number
of sampling steps (10 and 100). This seems to suggest that the GMM kernel allows for a better exploration
of the latent space with a small number of sampling steps. We also demonstrate that DDIM-GMM shows
improvements over DDIM for few step sampling from text-to-image models. Using the equivalence between
diﬀusion and rectiﬁed ﬂow matching, we derive novel SDE samplers for rectiﬁed ﬂow models and demonstrate
improvements with the proposed approach.

The gap between DDIM and the proposed DDIM-GMM becomes smaller with larger number of steps using
training-free GMM parameter selection. An interesting future direction would be to optimize the GMM pa-
rameters
t to maximize a suitable metric such as KID (Watson et al., 2021) or FID (Mathiasen & Hvilshøj,
2021) on the training dataset.

M

Broader Impact

Our work aims to accelerate sampling from pre-trained diﬀusion and rectiﬁed ﬂow matching models by
building upon the widely used DDIM framework. On one hand, fast sampling from these models has the
potential for positive societal beneﬁts by reducing the computational burden and thereby the carbon footprint
resulting from large-scale deployment of diﬀusion models. On the other hand, generative models are known
to pose risks to society such as aiding misinformation, privacy invasion and phishing. All these have been
widely discussed and we wish not to list them in detail here.

Acknowledgments

We thank Miguel Angel Bautista Martin, Navdeep Jaitly, Ian Fasel and Barry Theobald for their invaluable
help in the review and publishing process of this work.

References

Michael S. Albergo, Nicholas M. Boﬃ, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying frame-

work for ﬂows and diﬀusions, 2023. URL https://arxiv.org/abs/2303.08797.

Michael Samuel Albergo and Eric Vanden-Eijnden. Building normalizing ﬂows with stochastic in-
In The Eleventh International Conference on Learning Representations, 2023. URL

terpolants.
https://arxiv.org/abs/2209.15571.

Christopher M. Bishop. Pattern Recognition and Machine Learning (Information Science and Statistics).

Springer-Verlag, Berlin, Heidelberg, 2006. ISBN 0387310738.

Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and
Karsten Kreis. Align your latents: High-resolution video synthesis with latent diﬀusion models. In IEEE
Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

Andrew Brock, Jeﬀ Donahue, and Karen Simonyan. Large scale GAN training for high ﬁdelity natural image

synthesis. In International Conference on Learning Representations, 2019.

12

Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-

700m: Image-text pair dataset. https://github.com/kakaobrain/coyo-dataset, 2022.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei.

Imagenet: A large-scale hierar-
chical image database. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pp. 248–255, 2009.

Prafulla Dhariwal and Alexander Quinn Nichol. Diﬀusion models beat GANs on image synthesis.

In
A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan (eds.), Advances in Neural Informa-
tion Processing Systems, 2021.

Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real NVP. In 5th In-
ternational Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017,
Conference Track Proceedings, 2017.

Minh Do. Fast approximation of kullback-leibler distance for dependence trees and hidden markov models.

Signal Processing Letters, IEEE, 10:115 – 118, 05 2003.

Ruiqi Gao, Emiel Hoogeboom, Jonathan Heek, Valentin De Bortoli, Kevin P. Murphy, and Tim Salimans. Dif-
fusion meets ﬂow matching: Two sides of the same coin, 2024. URL https://diffusionflow.github.io/.

Gene H. Golub. Some modiﬁed matrix eigenvalue problems. SIAM Review, 15(2):318–334, 1973.

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron
Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing
Systems, volume 27, 2014.

Hanzhong Allan Guo, Cheng Lu, Fan Bao, Tianyu Pang, Shuicheng YAN, Chao Du, and Chongxuan Li.
Gaussian mixture solvers for diﬀusion models. In Thirty-seventh Conference on Neural Information Pro-
cessing Systems, 2023.

John Hershey and Peder Olsen. Approximating the kullback leibler divergence between gaussian mixture

models. volume 4, pp. IV–317, 05 2007.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans
trained by a two time-scale update rule converge to a local nash equilibrium.
In I. Guyon, U. Von
Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in Neural
Information Processing Systems, volume 30, 2017.

Jonathan Ho and Tim Salimans. Classiﬁer-free diﬀusion guidance. In NeurIPS 2021 Workshop on Deep

Generative Models and Downstream Applications, 2021.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diﬀusion probabilistic models. In Advances in Neural
Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020,
NeurIPS 2020, December 6-12, 2020, virtual, 2020.

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet.

Video diﬀusion models. arXiv:2204.03458, 2022.

Alexia Jolicoeur-Martineau, Ke Li, Rémi Piché-Taillefer, Tal Kachman, and Ioannis Mitliagkas. Gotta go

fast when generating data with score-based models. arXiv preprint arXiv:2105.14080, 2021.

I.T. Jolliﬀe. Principal Component Analysis. Springer Verlag, 1986.

Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial

networks. CoRR, 2018.

Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial
networks. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach,
CA, USA, June 16-20, 2019, pp. 4401–4410. Computer Vision Foundation / IEEE, 2019.

13

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diﬀusion-based

generative models. In Proc. NeurIPS, 2022.

Beomsu Kim, Yu-Guan Hsieh, Michal Klein, marco cuturi, Jong Chul Ye, Bahjat Kawar, and James Thorn-
ton. Simple reﬂow: Improved techniques for fast ﬂow models. In The Thirteenth International Conference
on Learning Representations, 2025.

Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong
He, Yuki Mitsufuji, and Stefano Ermon. Consistency trajectory models: Learning probability ﬂow ode
trajectory of diﬀusion. arXiv preprint arXiv:2310.02279, 2023.

Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. In Yoshua Bengio and Yann LeCun
(eds.), 2nd International Conference on Learning Representations, ICLR 2014, Banﬀ, AB, Canada, April
14-16, 2014, Conference Track Proceedings, 2014.

Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. Diﬀwave: A versatile diﬀusion
In 9th International Conference on Learning Representations, ICLR 2021,

model for audio synthesis.
Virtual Event, Austria, May 3-7, 2021, 2021.

Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, 2009.

Sangyun Lee, Zinan Lin, and Giulia Fanti. Improving the training of rectiﬁed ﬂows. In Advances in Neural

Information Processing Systems, volume 37, pp. 63082–63109. Curran Associates, Inc., 2024.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matthew Le, Brian Karrer, David
Lopez-Paz, and Itai Gat. Flow matching for generative modeling. In International Conference on Learning
Representations (ICLR), 2023.

Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diﬀusion models on manifolds.

In International Conference on Learning Representations, 2022.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer

data with rectiﬁed ﬂow. In ICLR, 2023.

Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild.

In

Proceedings of International Conference on Computer Vision (ICCV), December 2015.

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver
for diﬀusion probabilistic model sampling in around 10 steps. arXiv preprint arXiv:2206.00927, 2022.

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver

for guided sampling of diﬀusion probabilistic models. arXiv preprint arXiv:2211.01095, 2023.

Eric Luhman and Troy Luhman. Knowledge distillation in iterative generative models for improved sampling

speed. 2021.

Calvin Luo. Understanding diﬀusion models: A uniﬁed perspective. ArXiv, abs/2208.11970, 2022.

Alexander Mathiasen and Frederik Hvilshøj. Backpropagating through fréchet inception distance. arXiv

preprint arXiv:2009.14075, 2021.

Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim
Salimans. On distillation of guided diﬀusion models. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), pp. 14297–14306, June 2023.

Eliya Nachmani, Robin San Roman, and Lior Wolf. Non gaussian denoising diﬀusion models. arXiv preprint

arXiv:2106.07582, 2021.

Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diﬀusion probabilistic models. In Marina
Meila and Tong Zhang (eds.), Proceedings of the 38th International Conference on Machine Learning,
volume 139 of Proceedings of Machine Learning Research, pp. 8162–8171. PMLR, 18–24 Jul 2021.

14

William Peebles and Saining Xie.

Scalable diﬀusion models with transformers.

arXiv preprint

arXiv:2212.09748, 2022.

Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diﬀusion.

arXiv, 2022.

Danilo Rezende and Shakir Mohamed. Variational inference with normalizing ﬂows.

In Proceedings of
the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning
Research, pp. 1530–1538. PMLR, 2015.

Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation and approximate
In Proceedings of the 31st International Conference on Machine

inference in deep generative models.
Learning, volume 32 of Proceedings of Machine Learning Research, pp. 1278–1286. PMLR, 2014.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution
image synthesis with latent diﬀusion models. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pp. 10684–10695, 2022.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar
Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, Jonathan Ho, David J Fleet,
and Mohammad Norouzi. Photorealistic text-to-image diﬀusion models with deep language understanding.
In Advances in Neural Information Processing Systems, volume 35, pp. 36479–36494, 2022.

Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diﬀusion models. In The Tenth
International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022,
2022.

Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen, and Xi Chen.
Improved techniques for training gans. In Advances in Neural Information Processing Systems, volume 29,
2016.

Robin San-Roman, Eliya Nachmani, and Lior Wolf. Noise estimation for generative diﬀusion models. arXiv

preprint arXiv:2104.02600, 2021.

Andreas Sauer, Klaus Greﬀ, Michael Geiger, Nils Schindler, and Stefan Wermter. Adversarial diﬀusion

distillation. In European Conference on Computer Vision (ECCV), 2024.

Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade W Gordon, Ross Wightman, Mehdi Cherti,
Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa R
Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5b: An
open large-scale dataset for training next generation image-text models. In Thirty-sixth Conference on
Neural Information Processing Systems Datasets and Benchmarks Track, 2022.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning
using nonequilibrium thermodynamics. In Francis Bach and David Blei (eds.), Proceedings of the 32nd
International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research,
pp. 2256–2265. PMLR, 07–09 Jul 2015.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diﬀusion implicit models. In International
Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=St1giarCHLP.

Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In

Advances in Neural Information Processing Systems, volume 32, 2019.

Yang Song, Jascha Narain Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben
Poole. Score-based generative modeling through stochastic diﬀerential equations. ArXiv, abs/2011.13456,
2020.

Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models.

arXiv preprint

arXiv:2303.01469, 2023.

15

Alexander Tong, Kilian FATRAS, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks,
Improving and generalizing ﬂow-based generative models with mini-
ISSN 2835-8856. URL

Guy Wolf, and Yoshua Bengio.
batch optimal transport. Transactions on Machine Learning Research, 2024.
https://openreview.net/forum?id=CD9Snc73AW. Expert Certiﬁcation.

Aäron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal
Kalchbrenner, Andrew Senior, and Koray Kavukcuoglu. Wavenet: A generative model for raw audio. In
9th ISCA Speech Synthesis Workshop, 2016a.

Aäron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks. In ICML,

volume 48 of JMLR Workshop and Conference Proceedings, pp. 1747–1756, 2016b.

Daniel Watson, William Chan, Jonathan Ho, and Mohammad Norouzi. Learning fast samplers for diﬀusion
models by diﬀerentiating through sample quality. In International Conference on Learning Representa-
tions, 2021.

Daniel Watson, William Chan, Ricardo Martin-Brualla, Jonathan Ho, Andrea Tagliasacchi, and Mohammad

Norouzi. Novel view synthesis with diﬀusion models. ArXiv, abs/2210.04628, 2022.

Zhisheng Xiao, Karsten Kreis, and Arash Vahdat. Tackling the generative learning trilemma with denoising

diﬀusion GANs. In International Conference on Learning Representations, 2022.

Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T Freeman, and

Taesung Park. One-step diﬀusion with distribution matching distillation. In CVPR, 2024.

Fisher Yu, Yinda Zhang, Shuran Song, Ari Seﬀ, and Jianxiong Xiao. Lsun: Construction of a large-scale

image dataset using deep learning with humans in the loop. arXiv preprint arXiv:1506.03365, 2015.

Qinsheng Zhang and Yongxin Chen. Fast sampling of diﬀusion models with exponential integrator.

In

International Conference on Learning Representations, 2023.

Qinsheng Zhang, Molei Tao, and Yongxin Chen. gddim: Generalized denoising diﬀusion implicit models,

2022.

A Appendix

A.1 Proof of Constraints on GMM Parameters

Our proof for the constraints in Eq. 8 follows by induction (Song et al., 2021). The marginal of xT is already
equal to the DDPM marginal at step T by deﬁnition (Eq. 19). We show below that the marginals of all the
random variables xt, t < T are Gaussian mixtures with their ﬁrst and second order moments equal to the
desired values, given the constraints in Eq. 8. We derive the forms of the marginals for T
2 and

1 and T

−

−

16

the proof follows inductively for all t < T

−

2. Using Bayes’ rule, the marginal at xT −1 is given by

qσ,M(xT −1|

x0) =

xT

Z

qσ,M(xT −1|

xT , x0)qσ,M(xT

x0) dxT
|

=

ZxT

K

k=1
X

πk
T N

√αT −1x0 +

1

(cid:18)

q

αT −1 −

−

σ2
T .

xT

−
√1

√αT x0
αT

−

qσ,M(xT

x0) dxT
|

K

=

πk
T

k=1
X

ZxT N

√αT −1x0 +

1

(cid:18)

q

αT −1 −

−

σ2
T .

xT

−
√1

√αT x0
αT

−

(√αT x0, (1

−

N

αT )I) dxT ,

K

=

k=1
X

πk
T N

√αT −1x0 + δk

T , (1

αT −1)I

−

∆k
T

,

−

(cid:0)

(cid:1)

+ δk

T , σ2

T I

+ δk

T , σ2

T I

∆k
T

−

(cid:19)

∆k
T

−

(cid:19)

(23)

which is also a GMM with the same mixing weights πk
T . This is due to the fact that each of the above
integrals is a Gaussian, whose parameters can be determined by using Gaussian marginalization identities
(Bishop, 2006)(2.115). The mean µGMM
parameters of the above GMM are
given by

and the covariance ΣGMM

T −1

T −1

K

µGMM

T −1 =

πk
T

√αT −1x0 + δk
T

k=1
X

(cid:0)

= √αT −1x0 +

T δk
πk
T

K

k=1
X

(cid:1)

K

∆k
T

+

ΣGMM

T −1 =

K

k=1
X

(cid:0)

πk
T

(1

−

αT −1)I

−

= (1

−

αT −1)I +

(cid:1)

(δk

T −

k=1
X
¯δT )(δk

T −

K

πk
T

k=1
X

(cid:16)

πk
T (δk

¯δT )(δk

T −

¯δT )T

T −

T
¯δT )

−

∆k
T

,

(cid:17)

(24)

K
k=1 πk

T δk
T .

where ¯δT =
It is straightforward to verify that these are equal to the desired means and
covariance parameters of the equivalent DDPM forward marginal if the constraints in Eq. 8 are satisﬁed.
Speciﬁcally the constraint ¯δT = 0 and either one of the constraints on ∆k
T in Eq. 8 lead to the following
expressions for the ﬁrst and second order moments:

P

µGMM
ΣGMM

T −1 = √αT −1x0
T −1 = (1

αT −1)I,

−

(25)

as desired.

17

∆l

T −1

−

(cid:19)

(26)

K

L

k=1
X

l=1
X

K

L

=

=

k=1
X

l=1
X

The marginal of xT −2 can be derived similarly by invoking Bayes’ rule and using the form of the GMM for
xT −1. Speciﬁcally

qσ,M(xT −2|

x0) =

xT −1

Z

qσ,M(xT −2|

xT −1, x0)qσ,M(xT −1|

x0) dxT −1

=

ZxT −1

L

l=1
X

πl
T −1N

(cid:18)

√αT −2x0 +

1

q

αT −2 −

−

σ2
T −1.

xT −1 −
√1
−

√αT −1x0
αT −1

+ δl

T −1, σ2

T −1I

∆l

T −1

−

(cid:19)

=

ZxT −1 (

L

l=1
X

πl
T −1N

qσ,M(xT −1|

x0) dxT −1

√αT −2x0 +

1

(cid:18)

q

αT −2 −

−

σ2
T −1.

xT −1 −
√1
−

√αT −1x0
αT −1

+ δl

T −1, σ2

T −1I

∆l

T −1

−

(cid:19))

K

(

k=1
X

πk
T N

√αT −1x0 + δk

T , (1

αT −1)I

−

∆k
T

−

(cid:0)

dxT −1

)
(cid:1)

T πl
πk

T −1

ZxT −1 N

(cid:18)

√αT −2x0 +

1

q

αT −2 −

−

σ2
T −1.

xT −1 −
√1
−

√αT −1x0
αT −1

+ δl

T −1, σ2

T −1I

√αT −1x0 + δk

T , (1

αT −1)I

−

∆k
T

−

dxT −1

N

(cid:0)
µT,T −1
k,l

, ΣT,T −1
k,l

,

(cid:1)

T πl
πk

T −1N

(cid:16)

(cid:17)

where we assume that the transition kernel from xT −1 to xT −2 is a GMM with L components and pa-
, l = 1 . . . L. Each of the integrals above is a Gaussian whose mean µT,T −1
T −1, δl
πl
rameters
and covariance ΣT,T −1
parameters can be deduced by invoking Gaussian marginalization identities (Bishop,
(cid:1)
(cid:0)
2006)(2.115) and are given by:

T −1, ∆l

T −1

k,l

k,l

µT,T −1
k,l

= √αT −2x0 +

ΣT,T −1
k,l

= (1

αT −2)I

−

q

−

σ2

T −1

σ2

T −1

1

−
√1

1

−

αT −2 −
αT −1
−
αT −2 −
αT −1
1
−

δk
T + δl

T −1

∆k

T −

∆l

T −1.

(27)

Using the above expressions for the mean and covariance parameters of individual Gaussian components,
the corresponding parameters µGMM

for the GMM marginal of xT −2 are given by:

and ΣGMM

T −2

T −2

K

L

µGMM

T −2 =

T πl
πk

T −1µT,T −1

k,l

k=1
X

l=1
X

= √αT −2x0

K

L

ΣGMM

T −2 =

T πl
πk

T −1ΣT,T −1

k,l

k=1
X
−

= (1

l=1
X
αT −2)I,

K

L

+

k=1
X

l=1
X

T πl
πk

T −1[AδT

k + δT −1
l

where

1

A =

q

σ2

T −1

−
√1

αT −2 −
αT −1
−

.

18

][AδT

k + δT −1
l

]T

(28)
(29)

(30)

(31)

k , δT −1
We have made use of the constraints in Eq. 8, for the parameters (πT
), to arrive
at the above expressions and noting that A is independent of the GMM parameters
T −1. The
ﬁrst and second order moments in Eq. 29 and Eq. 30 correspond to the DDPM forward marginal moments
of xT −2. The proof for all latents xt, t < T
2 follows from a similar argument as above noting that the
form of the marginal in Eq. 26 is a GMM with M = KL components. Further each component’s mean and
1 and T ) step transition kernels’ oﬀset parameters (δ’s and ∆’s) as
covariances in Eq. 27 carry future (T
linear additive factors with coeﬃcients (A, A2 and 1) that are independent of those parameters. This makes
it easier to see why proof by induction should work for t < T

k , ∆T −1
l
T and

k , πT −1

, ∆T

, δT

M

M

2.

−

−

l

l

−

A.2 Variance Upper Bounds

The upper bounds of the eigenvalues of the matrix ∆k
of outer-products of mean centered orthonormal vectors. Speciﬁcally ∆k

t are tractable because the matrix is a weighted sum
t can be written as

∆k

t =

s2
Kπk

t  

K

l=1
X

t (ul
πk

t)(ul

t)T

¯ut ¯uT
t

−

,

!

(32)

where uk
matrix Ut[1 : K] leads to a matrix M k

t = Ut[k] (Section 3.1.2). Diagonalizing the ﬁrst term in Eq. 32 by pre and post multiplying by the

t , which is a sum of a diagonal matrix and a rank one matrix,
M k

t = Ut[1 : K]T ∆k

t Ut[1 : K]

=

s2
Kπk
t

(Dk

t −

πtπT

t ),

(33)

t is a diagonal matrix with πk

where Dk
proportions πk
one matrix, if λi, i = 1 . . . K are the eigenvalues of M k

t , k = 1 . . . K, along its diagonal and πt is a column vector of mixture
t . Using a bound (Golub, 1973) on the eigenvalues of a diagonal matrix modiﬁed by a rank

t , then

s2
Kπk

t  

π1
t −

K

(πl

t)2

λ1 ≤

! ≤

l=1
X
s2
Kπk
t

πi−1
t ≤

λi

≤

s2
Kπk
t

s2
Kπk
t

π1
t

πi
t,

i = 2 . . . K.

(34)

A.3 Forward Process

We can derive the forward process using Bayes’ rule. Speciﬁcally

qσ,M(xt

xt−1, x0) =
|

qσ,M(xt−1|

xt, x0)qσ,M(xt

x0)
|

=

qσ,M(xt−1|

x0)

GM M (t, t

−

GM M (t

1)GM M (t)
1)

−

where GM M (t, t
marginal GMMs at steps t
model is non-Gaussian and non-Markovian in general and diﬀerent from Gaussian diﬀusion.

1) and GM M (t) are the
1 and t respectively. Note that the inference process of the proposed implicit

1) is the transition GMM from step t to t

1 and GM M (t

−

−

−

−

A.4 Upper Bound of ELBO using the DDIM-GMM Inference Process

In this section we provide an upper bound for the ELBO loss using the proposed DDIM-GMM inference
simple,w loss, w.r.t. the denoiser parameters θ.
process in terms of an augmented version of the DDPM
The ELBO loss

ELBO,qσ,M(θ) using the proposed DDIM-GMM inference process is given by

L

L

ELBO,qσ,M(θ) = Eqσ,M

L

DKL(qσ,M(xT

"

x0)
||
|

pθ(xT )) +

Eqσ,M [log pθ(x0|

x1)]

−

T

T

t=2
X

DKL(qσ,M(xt−1|

pθ(xt−1|
xt, x0)
||

xt))

#

= Eqσ,M

"

t=2
X

DKL(qσ,M(xt−1|

pθ(xt−1|
xt, x0)
||

xt))

−

19

log pθ(x0|

x1)

#

+ const.,

(35)

where const. is a term independent of θ because pθ(xT ) =
a normal likelihood function for the observation x0 given x1, i.e.,

N

(0, I). We ignore the constant term and assume

where fθ(xt, t) is the denoiser estimate of x0, given by:

pθ(x0|

x1) =

N

(fθ(x1, 1), σ2

1I),

fθ(xt, t) =

xt

−

√1

αtǫθ(xt, t)

−
√αt

.

The ELBO loss in Eq. 35 reduces to

T

ELBO,qσ,M(θ) =

L

Eqσ,Mt [DKL(qσ,Mt (xt−1|
log pθ(x0|

t=2
X
+ Eqσ,M(x0,x1) [

−

x1)]

qσ,Mt (xt−1|
xt, x0)
||

xt, fθ(xt, t))))]

:=K1 + K2

T

=

K1,t + K2,

(36)

(37)

(38)

t=2
X
xt) = qσ,M(xt−1|

xt, fθ(xt, t)) as in DDIM (Song et al., 2021). The ﬁrst term K1
where we use pθ(xt−1|
involves KL-Divergences between mixtures of Gaussians, which is analytically intractable. However, we can
use a suitable upper bound (Hershey & Olsen, 2007) as a surrogate for optimization. Assuming that there
is a one-to-one correspondence between the mixture components of the GMMs using the true and estimated
value of x0 above, we can use the matched bound (Hershey & Olsen, 2007; Do, 2003) as the upper bound
to each of the KLD terms K1,t at step t, i.e. for any t > 1

(xt−1|

xt, x0)
qσ,Mk
||

t

(xt−1|

xt, fθ(xt, t)))))

#

πk
t
νk
t !

(1

αt)

−
2αt

ǫ
||

−

ǫθ(xt, t)

2
||

#

Eqσ,M

"

k
X

πk
t DKL((qσ,Mk

t

Eq(x0)qσ,M(xt|x0)ǫ∼N (0,I)

" 

k
X

K1,t

≤

≤

=

ξGMM,l
t

l
X

E

q(x0)ql

σ,M(xt|x0)ǫ∼N (0,I)

" 

= E

l∼ξGMM
t

q(x0)ql

σ,M(xt|x0)ǫ∼N (0,I)

" 

k
X

k
X
πk
t
νk
t !

πk
t
νk
t !

(1

αt)

−
2αt

ǫ
||

−

ǫθ(xt, t)

2
||

#

(1

αt)

−
2αt

ǫ
||

−

ǫθ(xt, t)

2
||

#

(39)

where νk
t is the minimum variance within a diagonal approximation of the covariance matrix σ2
t I
t , i.e.,
νk
t = min diag((σ2
xt, x0) is used to refer to
t I
the kth mixture component’s density function of the GMM transition kernel. Similarly, ql
x0) refers
σ,M(xt
|
to the lth component of the DDIM-GMM’s forward marginal GMM at step t. The upper bound of Eq. 39
can be interpreted as an augmented form of

t ))). Note that in the above, qσ,Mk

diag_approx(∆k

simple,w with weights

(xt−1|

∆k

−

−

t

L

wt =

πk
t
νk
t !

(1

αt)

−
2αt

k
X

(40)

and the DDPM marginals’ mean and covariance randomly modiﬁed with shifts from one of the DDIM-GMM
forward marginal’s components at every step t (e.g., Eq. 27 for t = T
2). The choice of the shifts is
according to a discrete distribution with proportions given by the DDIM-GMM marginal’s mixture priors
ξGMM
t

(e.g. Eq. 26 for t = T

2).

−

−

20

 
For t = 1, the loss term K2 is given by

K2 = Eqσ,M(x0,x1) [

log pθ(x0|
= Eq(x0)qσ,M(x1|x0)ǫ∼N (0,I)

−

= E

l∼ξGMM
1

q(x0)ql

σ,M(x1|x0)ǫ∼N (0,I)

x1)]
α1)
(1
ǫ
−
2σ2
1α1 ||
−
α1)
(1
ǫ
−
2σ2
1α1 ||

(cid:20)

(cid:20)

2
ǫθ(x1, 1)
||

+ const.

(cid:21)
2
ǫθ(x1, 1)
||

−

,

(cid:21)

(41)

where we have ignored the const. term independent of θ. Combining Eqs. 39 and 41, the
be interpreted as upper bounded by an augmented version of
41.

ELBO,qσ,M(θ) can
simple,w with weights wt given in Eqs. 40 and

L

L

A.5 Experimental Details

We provide additional details on the experiments reported in Section 5, speciﬁcally for the CelebAHQ,
FFHQ and ImageNet experiments. All our diﬀusion models are trained in the latent space of a VQVAE
(Rombach et al., 2022). The input images to the VQVAE are at a resolution of 256x256 pixels. Each of
the VQVAEs are trained on a large scale dataset. Speciﬁcally the VQVAEs for unconditional generation on
CelebAHQ and class-conditional generation on ImageNet are trained on OpenImages. We use the publicly
available f 4 VQVAE (Table 8, Section D.2. of Rombach et al. (2022)) for training CelebAHQ models and
f 8 VQVAE for class-conditional ImageNet models respectively. The f4 VQVAE (#embeddings=8192) does
not use attention layers at any resolution within the model architecture, whereas the f 8 VQVAE (#em-
beddings=16384) uses attention at resolution 32. We train a f 4 VQVAE (#embeddings=8192), with no
attention layers, on ImageNet for 712k steps and use its latent space to train the diﬀusion models on FFHQ.

αt
All our diﬀusion models are trained with 1000 forward steps using a linear noise (βt = 1
αt−1 ) schedule
of [β0 = 0.0015, β1000 = 0.0195]. We use the U-Net architecture (Ho et al., 2020; Rombach et al., 2022) for
the denoiser. Speciﬁcally, the unconditional U-Net encoders operating on the f 4 VQVAE latent space have
four 2x downsampling levels with channel multiplication factors of [1, 2, 3, 4] starting from a base set of 224
channels. Each level uses two residual blocks. Attention blocks are used within levels at downsampling
factors [2, 4, 8]. Similarly, the class-conditional U-Net encoders operating on f 8 VQVAE latent space have
three 2x downsampling levels with channel multiplication factors of [1, 2, 4] starting from a base set of 256
channels. Other architectural details remain the same as before except that the attention blocks are at
downsampling levels [1, 2, 4].

−

For each DDIM-GMM variant, we choose a GMM with 8 mixture components with uniform priors (πk
for all steps t. We also search for the best value of scaling s among
result with the chosen value for s. It is possible that for some choices of s, the diagonal elements of ∆t
the corresponding upper bounds in the DDIM-GMM-VUB sampler could be larger than σ2
we clip the negative elements of (σ2
t I
variances in those dimensions in the latent space.

t =0.125)
and report the best
k or
t . In such cases
t )) to zero, which amounts to sampling with zero

diag_approx(∆k

0.01, 0.1, 1.0, 10.0

−

{

}

A.6 Class-conditional ImageNet with Classiﬁer Guidance

For classiﬁer guidance, we train a separate classiﬁer at diﬀerent levels of noise and use it with two guidance
scales (1, 10) for sampling with 10 and 100 steps. The FID and IS results are in Figs. 5 and 6 respectively.
Using smaller guiance scale (1), DDIM-GMM-* samplers show improvements over DDIM only under the
highest η(= 1) setting. FID improves using the fewest sampling steps (10) and IS improves using both
10 and 100 sampling steps. This can be attributed to a similar argument as for unconditional sampling,
especially for the least number of sampling steps. With a higher guidance scale (10), all variants of DDIM-
GMM-* samplers yield signiﬁcantly lower FIDs than the DDIM sampler when the number of sampling steps
is small (10) (see Table 9, Appendix A.14.3). The FIDs with variance upper bounding, relative to without,
improve signiﬁcantly with higher values of η possibly due to greater exploration of the latent space under
those settings. The diﬀerences between DDIM and DDIM-GMM-* are marginal using 100 sampling steps
with the exception of DDIM-GMM-RAND and DDIM-GMM-ORTHO for the highest η setting. With a

21

higher guidance scale (10), the IS scores of samples from DDIM-GMM-* samplers are almost always higher
than from the DDIM sampler (see Table 10, Appendix A.14.3). The only exception is the DDIM-GMM-
ORTHO sampler run for 100 steps using η = 1, which is only marginally worse. Similar to FID results,
the diﬀerences between samplers with and without variance upper bounding are ampliﬁed by η. This is an
interesting result indicating that better exploration of latent spaces with a multimodal reverse kernel not
only helps with coverage (FID) but also sample sharpness (IS) since the guidance scale is known to trade-oﬀ
one versus the other (Dhariwal & Nichol, 2021) and poses challenges for higher order ODE solvers (Lu et al.,
2023).

η

= 0.0

η

= 0.2

η

= 0.5

η

= 1.0

DDIM

DDIM-GMM-ORTHO

DDIM

DDIM-GMM-ORTHO

DDIM

DDIM-GMM-ORTHO

DDIM

DDIM-GMM-ORTHO

60

30

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

25

20

D
I
F

15

10

5

0

DDPM

DDPM

0
1

0
0
1

#Steps

25

20

D
I
F

15

10

5

0

25

20

D
I
F

15

10

5

0

DDPM

DDPM

0
1

0
0
1

50

40

D
I
F

30

20

10

0

DDPM

DDPM

0
1

0
0
1

DDPM

DDPM

0
1

0
0
1

#Steps

#Steps

#Steps

20.0

20.0

DDIM

DDIM-GMM-ORTHO

DDIM

DDIM-GMM-ORTHO

DDIM

DDIM-GMM-ORTHO

DDIM

DDIM-GMM-ORTHO

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

20

η

= 0.0

η

= 0.2

η

= 0.5

η

= 1.0

35

17.5

15.0

12.5

D
I
F

10.0

7.5

5.0

2.5

0.0

17.5

15.0

12.5

D
I
F

10.0

7.5

5.0

2.5

0.0

DDPM

DDPM

0
1

0
0
1

#Steps

DDPM

DDPM

0
1

0
0
1

15

D
I
F

10

5

0

DDPM

DDPM

0
1

0
0
1

30

25

20

D
I
F

15

10

5

0

DDPM

DDPM

0
1

0
0
1

#Steps

#Steps

#Steps

Figure 5: Class-conditional ImageNet with Classiﬁer Guidance. FID (
↓
(top) and 10.0 (bottom) respectively. The horizontal line is the DDPM baseline run for 1000 steps.

) with guidance scale of 1.0

140

120

100

S

I

80

60

40

20

0

η

= 0.0

η

= 0.2

η

= 0.5

η

= 1.0

DDIM

DDIM-GMM-ORTHO

DDIM

DDIM-GMM-ORTHO

DDIM

DDIM-GMM-ORTHO

DDIM

DDIM-GMM-ORTHO

140

140

140

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

DDPM

DDPM

DDPM

DDPM

DDPM

DDPM

DDPM

DDPM

120

120

120

100

S

I

80

60

40

20

0

0
1

0
0
1

100

S

I

80

60

40

20

0

0
1

0
0
1

100

S

I

80

60

40

20

0

0
1

0
0
1

#Steps

#Steps

#Steps

0
1

0
0
1

#Steps

η

= 0.0

η

= 0.2

η

= 0.5

η

= 1.0

DDIM

DDIM-GMM-ORTHO

DDIM

DDIM-GMM-ORTHO

DDIM

DDIM-GMM-ORTHO

DDIM

DDIM-GMM-ORTHO

250

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

250

250

250

DDPM

DDPM

DDPM

DDPM

DDPM

DDPM

DDPM

DDPM

200

200

200

200

S

I

150

100

50

0

S

I

150

S

I

150

S

I

150

100

100

100

50

50

50

0
1

0
0
1

#Steps

0

0
1

0
0
1

0

0
1

0
0
1

0

0
1

0
0
1

#Steps

#Steps

#Steps

Figure 6: Class-conditional ImageNet with Classiﬁer Guidance. IS (
↑
(top) and 10.0 (bottom) respectively. The horizontal line is the DDPM baseline run for 1000 steps.

) with guidance scale of 1.0

A.7 Class-conditional ImageNet with Classiﬁer-free Guidance

In addition to the results reported in Section 5.2, we conduct experiments using a higher classiﬁer-free
guidance scale of 5.0. The FID and IS metrics are shown in Fig. 7. As expected, higher guidance leads to

22

better IS values at the expense of FID compared to lower guidance. Using 10 sampling steps, the best IS of
320.66 (Table 12) is obtained with a deterministic (η = 0) DDIM-GMM-ORTHO sampler. Similarly, using
100 sampling steps, DDIM-GMM-ORTHO and DDIM-GMM-RAND yield the best IS, around 355, at η = 1.
The FIDs are generally worse relative to sampling with lower guidance scale with the best value of 15.73
achieved using both DDIM-GMM-RAND and DDIM-GMM-ORTHO variants at η = 0.5.

30

25

20

D
I
F

15

10

5

0

0
1

0
0
1

#Steps

η

= 0.0

η

= 0.2

η

= 0.5

η

= 1.0

DDIM

DDIM-GMM-ORTHO

DDIM

DDIM-GMM-ORTHO

DDIM

DDIM-GMM-ORTHO

DDIM

DDIM-GMM-ORTHO

30

30

30

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

DDPM

DDPM

DDPM

DDPM

DDPM

DDPM

25

25

20

D
I
F

15

20

D
I
F

15

10

10

5

0

0
1

0
0
1

5

0

0
1

0
0
1

25

20

D
I
F

15

10

5

0

DDPM

DDPM

0
1

0
0
1

#Steps

#Steps

#Steps

η

= 0.0

η

= 0.2

η

= 0.5

η

= 1.0

DDIM

DDIM-GMM-ORTHO

DDIM

DDIM-GMM-ORTHO

DDIM

DDIM-GMM-ORTHO

DDIM

DDIM-GMM-ORTHO

400

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

400

DDIM-GMM-RAND

DDIM-GMM-ORTHO-VUB

400

400

DDPM

DDPM

DDPM

DDPM

DDPM

DDPM

DDPM

DDPM

300

S

I

200

100

0

0
1

0
0
1

#Steps

300

300

S

I

S

I

200

200

100

100

0

0
1

0
0
1

0

0
1

0
0
1

300

S

I

200

100

0

0
1

0
0
1

#Steps

#Steps

#Steps

Figure 7: Class-conditional ImageNet with Classiﬁer-free Guidance. FID (
)
) (top) and IS (
↑
↓
(bottom) respectively with a guidance scale of 5.0. The horizontal line is the DDPM baseline run for 1000
steps.

A.8 Ablations

In this section, we perform an ablative study on the number of mixture components and oﬀset scaling factor
s of the GMM parameters using the unconditional model trained on the CelebAHQ dataset as described in
Section 5. We use the GMM-ORTHO-VUB sampler and ﬁx the mixture weights of the components to be
uniform in all these experiments.

A.8.1 Number of mixture components

We compute the FID on the validation set by choosing one of 8, 256, or 1024 components (n) at each step
during sampling, using diﬀerent values of η. For each choice of n, we select the scale s among (0.01, 0.1, 1.0,
10.0) that leads to the lowest FID. The results are plotted in Fig. 8 (top). For lower values of η (0, 0.2),
the performance is the same across all the choices, since the oﬀsets perturb only the means as the oﬀset
variances (diag_approx(σ2
t I
t )) are small. At higher η values (0.5, 1.0), the oﬀsets aﬀect variances in
as many dimensions and inﬂuence the exploration of latent spaces xt. The choices 8 and 256 lead to better
samples than 1024 because the latter restricts the variances in those many dimensions impacting exploration.
8 performs slightly better than 256 at η = 0.5 and vice-versa at η = 1.0. As the number of steps increases,
all the choices lead to similar results, likely due to small s (See Table 7, Appendix A.14.1).

∆k

−

A.8.2 Oﬀset scaling s

In order to study the eﬀect of s, we ﬁx the number of mixture components to 8 and choose a value for s
within four choices: (0.01, 0.1, 1.0, 10.0). The results are shown in Fig. 8 (bottom). The sample quality
is almost the same with smaller values of s (0.01-1.0). The highest value s = 10 gives the best results for
the least number of sampling steps (10) (See Table 7). As the number of steps increases, this leads to poor
quality samples and smaller oﬀsets are preferable, with one exception: at η = 1 it is still the best choice
for up to 50 steps. This can be explained using the hypothesis (Guo et al., 2023; Xiao et al., 2022) that

23

η

= 0

η

= 0.2

η

= 0.5

η

= 1

27.5

8

256

27.5

8

256

35

8

256

60

8

256

25.0

1024

1024

1024

1024

22.5

D
I
F

20.0

17.5

25.0

22.5

D
I
F

20.0

17.5

15.0

15.0

12.5

12.5

30

D
I
F

25

20

15

50

D
I
F

40

30

20

20

40

60

80

100

20

40

60

80

100

20

40

60

80

100

20

40

60

80

100

#Steps

#Steps

#Steps

#Steps

η

= 0

η

= 0.2

η

= 0.5

η

= 1

300

250

0.01

0.01

0.01

175

0.1

1.0

250

0.1

1.0

150

0.1

1.0

200

10.0

200

10.0

125

10.0

D
I
F

150

100

50

0

D
I
F

150

100

50

0

D
I
F

100

75

50

25

0.01

0.1

1.0

10.0

70

60

50

D
I
F

40

30

20

20

40

60

80

100

20

40

60

80

100

20

40

60

80

100

20

40

60

80

100

#Steps

#Steps

#Steps

#Steps

Figure 8: CelebAHQ. FID (
↓
factor s (bottom).

). Ablations on the number of mixture components (top) and oﬀset scaling

true denoising distributions are multimodal at fewer sampling steps and larger exploration (higher s) with
a multimodal kernel is favorable. This advantage vanishes as the number of sampling steps increase. At the
highest η(= 1), we hypothesize that s = 10 reduces the oﬀset variances (diag_approx(σ2
t I
t )) more than
the other choices, at least up to 50 steps, keeping sampling quality high.

∆k

−

A.9 Sharing GMM Parameters across Sampling Steps

t across sampling steps t
As discussed in Section 3.1.3, we also experiment with sharing GMM parameters
by choosing the oﬀsets only once followed by orthogonalization (SVD), scaling and variance upper bounding.
This saves some compute time during initialization. Table 3 compares the FIDs between DDIM-GMM-
ORTHO-VUB samplers that share GMM parameters (ORTHO-VUB∗) with the corresponding ones that
set them independently (ORTHO-VUB) across sampling steps t. We observe that there is no signiﬁcant
(more than 1 FID point) impact on sample FIDs across all datasets, except with FFHQ with 10 sampling
steps using η = 1, where the shared parameter sampler yields slightly better results. Here we show results
with unconditional models on CelebAHQ and FFHQ. More results can be found in Tables 7 to 12 in the
Appendix.

M

Table 3: Sharing GMM Parameters across Steps. FID (
↓

)

Dataset
Steps

CelebAHQ
100
10

FFHQ

10

100

η
ORTHO-VUB
0
ORTHO-VUB∗
0
1.0 ORTHO-VUB
1.0 ORTHO-VUB∗

27.94
27.84
60.65
61.15

11.44
11.42
15.60
15.94

26.46
27.18
69.25
67.72

11.12
11.24
11.44
11.38

A.10 Computational Overhead

As discussed brieﬂy in Section 3.1, the proposed approach introduces additional computational overhead in
an initialization phase prior to sampling. All the GMM mean and variance oﬀsets are precomputed and saved
in memory before sampling. We can choose to precompute a single set of GMM parameters per batch or the
entire sample set. We experimented with both the options and did not see a signiﬁcant diﬀerence in metrics.

24

So it is computationally more eﬃcient to precompute oﬀsets once and ﬁx them. We also experimented with
choosing diﬀerent GMM parameters for diﬀerent sampling steps t and found no signiﬁcant diﬀerence with
setting them the same across all t in our experiments. Due to storing the additional GMM parameters,
there is some memory overhead relative to DDIM but it is negligible, especially in the scenario of choosing a
single set of oﬀsets across all samples and time steps. In the scenario of using diﬀerent oﬀsets across subsets
(batches) of samples or sampling steps, the overhead scales linearly along the sample subset size and number
of step dimensions. The dimensionality of the latent spaces xt also inﬂuence the computational and memory
requirements of the GMM oﬀset parameters. For instance, it might be infeasible to compute the outer
products of centered oﬀsets (Eq. 11) if the dimensionality of the latent spaces is high, e.g. high-resolution
image space diﬀusion models. In such cases, the DDIM-GMM-ORTHO-VUB sampler is more feasible as it
provides an upper bound for the variance oﬀsets without explicitly computing them.

A.11 Comparison with DPM-Solver

In this section, we compare DDIM-GMM-ORTHO-VUB and DPM-Solver (Lu et al., 2022) samplers on the
class-conditional model trained on ImageNet. During inference, we use classiﬁer-free guidance with weights
(2.5, 5) and run each sampler for 10 and 100 steps. For the DDIM-GMM-ORTHO-VUB sampler, η is set to
0. The results listed in Table 6 suggest DDIM-GMM-ORTHO-VUB is superior to DPM-Solver in all cases
on the FID metric. This is also true with the IS metric with the exception of lower guidance scale (2.5) using
10 sampling steps.

Steps
Guidance Scale
DPM-Solver

2.5
9.75/225.68
DDIM-GMM-ORTHO-VUB 6.94/207.85

10

100

5
19.22/309.14
16.61/319.13

2.5
9.30/239.11
9.13/240.88

5
19.82/323.87
19.66/323.84

Table 4: Comparison with DPM-Solver. Class-conditional ImageNet with classiﬁer free guidance (FID
↓
IS
↑

).

/

A.12 Additional Experiments

In this section, we report results on additional experiments on the LSUN benchmarks (Yu et al., 2015) using
the same settings as the DDIM (Song et al., 2021) work. Speciﬁcally, we use the pretrained DDPM models
(Ho et al., 2020) on LSUN Bedroom and Church datasets to compare DDIM vs. DDIM-GMM-ORTHO-VUB
samplers for diﬀerent number of sampling steps (10, 20, 50 and 100).

Steps
DDIM

10
16.93
DDIM-GMM 16.86

20
8.77
8.76

50
6.68
6.62

100
6.76
6.67

Table 5: LSUN Bedroom. Comparison between DDIM and DDIM-GMM on the FID(
↓
both samplers.

) metric. η = 0 for

Steps
DDIM

10
19.39
DDIM-GMM 19.33

20
12.33
12.37

50
11.04
10.85

100
10.85
10.81

Table 6: LSUN Church. Comparison between DDIM and DDIM-GMM on the FID(
↓
both samplers.

) metric. η = 0 for

25

Figure 9: Class-conditional ImageNet with classiﬁer guidance, 10 sampling steps. Random samples
from the class-conditional ImageNet model using DDIM (left) and DDIM-GMM (right) sampler conditioned
on the class labels pelican (top) and cairn terrier (bottom) respectively. 10 sampling steps are used for each
sampler with a classiﬁer guidance weight of 10 (η = 1).

A.13 Qualitative Results

In this section we show some qualitative results of sampling with the proposed approach compared to original
DDIM. Speciﬁcally, we use the DDIM-GMM-ORTHO-VUB∗ (Section 3.1.3) method to obtain the samples
and refer to them with the label DDIM-GMM for brevity. In Fig. 9 we plot samples from the classiﬁer-guided
class conditional model trained on ImageNet with 10 sampling steps for both DDIM (left) and DDIM-GMM
(right). The top and bottom group of images correspond to input class labels “pelican" and “cairn terrier"
respectively. See Fig. 10 for comparisons using classiﬁer-free guidance. Also Fig. 11 shows images generated
by conditioning on text prompts using the Stable Diﬀusion v2.1 model.

26

Figure 10: Class-conditional ImageNet with classiﬁer-free guidance, 10 sampling steps. Random
samples from the class-conditional ImageNet model using DDIM (left) and DDIM-GMM (right) sampler
conditioned on the class labels pelican (top) and cairn terrier (bottom) respectively. 10 sampling steps are
used for each sampler with a classiﬁer free guidance weight of 5 (η = 0).

27

Figure 11: Text-to-image-generation using Stable Diﬀusion v2.1, 10 sampling steps. Random
samples from the Stable Diﬀusion v2.1 model using DDIM (left) and DDIM-GMM (right) sampler conditioned
on text prompts, displayed below each image. 10 sampling steps are used for each sampler with a classiﬁer
free guidance weight of 7.5 (η = 0).

28

A.14 FID and IS Metrics

In this section we list the metrics plotted in Section 5 in a tabular format. The numbers in bold emphasize
improvement of the corresponding sampling method’s metric if the diﬀerence in metric (FID or IS) is at least
1 unit from the worst result within the same group (same η and number of sampling steps). The number in
parentheses denotes the scale parameter s that resulted in the best metric for the particular DDIM-GMM-*
sampling method under a given setting. We omit the best s for ImageNet results.

A.14.1 CelebAHQ

Steps

10

20

50

100

1000

DDIM
DDIM-GMM-RAND
DDIM-GMM-ORTHO
DDIM-GMM-ORTHO-VUB
DDIM-GMM-ORTHO-VUB∗
DDIM
DDIM-GMM-RAND
DDIM-GMM-ORTHO

η
0
0
0
0
0
0.2
0.2
0.2
0.2 DDIM-GMM-ORTHO-VUB
0.2 DDIM-GMM-ORTHO-VUB∗
0.5
0.5
0.5
0.5 DDIM-GMM-ORTHO-VUB
0.5 DDIM-GMM-ORTHO-VUB∗
1.0
1.0
1.0
1.0 DDIM-GMM-ORTHO-VUB
1.0 DDIM-GMM-ORTHO-VUB∗
1.0

DDIM
DDIM-GMM-RAND
DDIM-GMM-ORTHO

DDIM
DDIM-GMM-RAND
DDIM-GMM-ORTHO

DDPM

32.95
28.01 (10)
27.97 (10)
27.94 (10)
27.84 (10)
33.74
32.32 (10)
32.42 (10)
28.32 (10)
27.68 (10)
39.04
37.15 (10)
37.27 (10)
31.00 (10)
31.42 (10)
68.67
66.94 (10)
67.15 (10)
60.65 (10)
61.15 (10)

18.58
18.74 (1)
18.71 (1)
18.71 (1)
18.53 (1)
19.48
17.26 (10)
17.33 (10)
19.18 (1)
19.77 (0.01)
22.01
20.66 (10)
20.78 (10)
20.65 (10)
20.89 (10)
39.20
37.63 (10)
37.79(1)
28.03 (10)
27.41 (10)

12.65
12.33 (1)
12.41 (1)
12.41 (1)
12.62 (1)
12.79
12.80 (0.01)
12.79 (1)
12.58 (1)
12.81 (1)
13.99
12.95 (10)
12.98 (10)
13.87 (1)
14.09 (1)
21.53
19.33 (10)
19.36 (10)
15.97 (10)
16.48 (10)

11.42
11.35 (1)
11.44 (1)
11.44 (1)
11.42 (0.1)
11.41
11.36 (0.1)
11.37 (0.01)
11.29 (1)
11.34 (1)
12.05
12.26 (1)
12.25 (0.1)
12.00 (1)
11.91 (1)
16.09
14.14 (10)
14.37 (10)
15.60 (1)
15.94 (1)

Table 7: CelebAHQ (FID
↓

)

11.59

29

A.14.2 FFHQ

Steps

10

20

50

100

1000

DDIM
DDIM-GMM-RAND
DDIM-GMM-ORTHO
DDIM-GMM-ORTHO-VUB
DDIM-GMM-ORTHO-VUB∗
DDIM
DDIM-GMM-RAND
DDIM-GMM-ORTHO

η
0
0
0
0
0
0.2
0.2
0.2
0.2 DDIM-GMM-ORTHO-VUB
0.2 DDIM-GMM-ORTHO-VUB∗
0.5
0.5
0.5
0.5 DDIM-GMM-ORTHO-VUB
0.5 DDIM-GMM-ORTHO-VUB∗
1.0
1.0
1.0
1.0 DDIM-GMM-ORTHO-VUB
1.0 DDIM-GMM-ORTHO-VUB∗
1.0

DDIM
DDIM-GMM-RAND
DDIM-GMM-ORTHO

DDIM
DDIM-GMM-RAND
DDIM-GMM-ORTHO

DDPM

28.73
26.55 (10)
26.46 (10)
26.46 (10)
27.18 (10)
29.33
29.27 (0.1)
29.09 (10)
26.90 (10)
27.35 (10)
35.53
35.16 (10)
35.00 (10)
29.03 (10)
28.73 (10)
81.88
79.93
80.18 (10)
69.25 (10)
67.72 (10)

15.68
15.64 (1)
15.67 (1)
15.67 (1)
15.43 (1)
15.83
16.03 (0.1)
16.01 (1)
15.96 (1)
15.85 (0.01)
17.83
18.01 (10)
17.92 (10)
18.16 (1)
17.89 (1)
37.09
35.85 (10)
35.93 (10)
23.88 (1)
23.76 (10)

11.67
11.83 (0.01)
11.83 (0.01)
11.83 (0.01)
11.77 (0.1)
11.66
11.87 (0.01)
11.87 (0.01)
11.87 (0.01)
11.62 (0.01)
11.89
11.85 (1)
11.87 (0.1)
11.78 (1)
11.92 (1)
15.45
15.17 (10)
15.03 (10)
15.44 (0.1)
15.56 (1)

11.17
11.12 (0.01)
11.12 (0.01)
11.12 (0.01)
11.24 (0.1)
11.07
10.92 (0.01)
10.92 (0.01)
10.91 (0.1)
11.11 (0.01)
10.53
10.49 (0.01)
10.47 (0.1)
10.45 (0.1)
10.81 (1)
11.33
11.50 (1)
11.51 (1)
11.44 (1)
11.38 (0.1)

Table 8: FFHQ (FID
↓

)

9.69

30

A.14.3

ImageNet

The FID and IS results with classiﬁer and classiﬁer-free guidance are in Tables 9-10 and Tables 11-12
respectively.

Steps
Guidance Scale

10

100

1000

1

10

1

10

1

10

DDIM
DDIM-GMM-RAND
DDIM-GMM-ORTHO
DDIM-GMM-ORTHO-VUB
DDIM-GMM-ORTHO-VUB∗
DDIM
DDIM-GMM-RAND
DDIM-GMM-ORTHO

η
0
0
0
0
0
0.2
0.2
0.2
0.2 DDIM-GMM-ORTHO-VUB
0.2 DDIM-GMM-ORTHO-VUB∗
0.5
0.5
0.5
0.5 DDIM-GMM-ORTHO-VUB
0.5 DDIM-GMM-ORTHO-VUB∗
1.0
1.0
1.0
1.0 DDIM-GMM-ORTHO-VUB
1.0 DDIM-GMM-ORTHO-VUB∗
1.0

DDIM
DDIM-GMM-RAND
DDIM-GMM-ORTHO

DDIM
DDIM-GMM-RAND
DDIM-GMM-ORTHO

DDPM

21.94
21.94
22.00
22.00
21.95
22.39
22.25
22.11
22.43
22.39
25.31
25.39
25.35
24.72
24.96
47.61
46.55
46.45
37.97
38.08

15.65
11.32
11.26
11.26
11.54
15.83
13.10
12.95
11.29
11.50
16.72
15.36
15.27
11.79
11.93
26.09
23.71
23.57
18.60
18.68

10.78
11.14
11.22
11.24
10.81
10.59
10.50
10.50
10.79
10.50
10.04
9.93
9.88
9.94
9.91
8.97
8.94
8.91
8.82
8.80

11.66
11.28
11.28
11.28
11.31
11.55
11.58
11.57
11.20
11.24
11.39
11.25
11.25
11.01
10.96
11.48
10.31
10.28
11.38
11.40

8.50

10.75

Table 9: Class-conditional ImageNet with classiﬁer guidance(FID
↓

)

31

Steps
Guidance Scale

10

100

1000

1

10

1

10

1

10

DDIM
DDIM-GMM-RAND
DDIM-GMM-ORTHO
DDIM-GMM-ORTHO-VUB
DDIM-GMM-ORTHO-VUB∗
DDIM
DDIM-GMM-RAND
DDIM-GMM-ORTHO

η
0
0
0
0
0
0.2
0.2
0.2
0.2 DDIM-GMM-ORTHO-VUB
0.2 DDIM-GMM-ORTHO-VUB∗
0.5
0.5
0.5
0.5 DDIM-GMM-ORTHO-VUB
0.5 DDIM-GMM-ORTHO-VUB∗
1.0
1.0
1.0
1.0 DDIM-GMM-ORTHO-VUB
1.0 DDIM-GMM-ORTHO-VUB∗
1.0

DDIM
DDIM-GMM-RAND
DDIM-GMM-ORTHO

DDIM
DDIM-GMM-RAND
DDIM-GMM-ORTHO

DDPM

66.15
66.26
66.16
66.17
66.48
66.17
69.51
69.48
65.35
66.50
62.42
62.25
62.53
67.45
65.35
35.19
36.78
37.02
48.74
35.03

136.76
161.19
161.47
161.45
154.64
135.01
152.48
152.10
162.13
156.31
131.49
138.43
139.64
161.57
156.91
86.76
95.47
96.90
120.63
118.57

98.71
98.34
97.25
97.14
98.75
99.4
99.60
100.12
99.00
99.64
105.02
106.46
106.18
106.74
106.46
116.59
116.54
117.09
117.19
117.52

179.62
181.43
181.33
181.33
180.60
179.86
182.41
182.25
183.91
182.60
190.45
191.35
190.87
192.72
192.17
207.78
207.52
208.34
209.36
208.74

118.28

210.53

Table 10: Class-conditional ImageNet with classiﬁer guidance(IS
↑

)

Steps
Guidance Scale

10

100

1000

2.5

5

2.5

5

2.5

5

DDIM
DDIM-GMM-RAND
DDIM-GMM-ORTHO
DDIM-GMM-ORTHO-VUB
DDIM-GMM-ORTHO-VUB∗
DDIM
DDIM-GMM-RAND
DDIM-GMM-ORTHO

η
0
0
0
0
0
0.2
0.2
0.2
0.2 DDIM-GMM-ORTHO-VUB
0.2 DDIM-GMM-ORTHO-VUB∗
0.5
0.5
0.5
0.5 DDIM-GMM-ORTHO-VUB
0.5 DDIM-GMM-ORTHO-VUB∗
1.0
1.0
1.0
1.0 DDIM-GMM-ORTHO-VUB
1.0 DDIM-GMM-ORTHO-VUB∗
1.0

DDIM
DDIM-GMM-RAND
DDIM-GMM-ORTHO

DDIM
DDIM-GMM-RAND
DDIM-GMM-ORTHO

DDPM

10.15
6.90
6.90
6.94
6.72
10.35
8.52
8.54
7.06
6.75
11.15
10.28
10.29
7.59
7.44
17.50
15.95
15.94
12.38
12.34

18.56
16.77
16.64
16.61
16.14
18.60
17.90
17.85
16.78
16.49
19.13
18.61
18.54
17.91
17.51
20.42
19.91
19.92
19.70
19.64

9.57
9.23
9.20
9.13
9.22
9.67
9.70
9.73
9.27
9.38
10.70
10.60
10.56
10.27
10.38
12.97
10.84
10.86
12.80
12.80

19.94
19.83
19.84
19.66
19.77
20.29
20.17
20.18
20.04
20.05
21.41
15.73
15.73
21.10
21.17
23.56
22.38
22.26
22.94
23.59

12.61

23.38

Table 11: Class-conditional ImageNet with classiﬁer free guidance(FID
↓

)

32

Steps
Guidance Scale

10

100

1000

2.5

5

2.5

5

2.5

5

DDIM
DDIM-GMM-RAND
DDIM-GMM-ORTHO
DDIM-GMM-ORTHO-VUB
DDIM-GMM-ORTHO-VUB∗
DDIM
DDIM-GMM-RAND
DDIM-GMM-ORTHO

η
0
0
0
0
0
0.2
0.2
0.2
0.2 DDIM-GMM-ORTHO-VUB
0.2 DDIM-GMM-ORTHO-VUB∗
0.5
0.5
0.5
0.5 DDIM-GMM-ORTHO-VUB
0.5 DDIM-GMM-ORTHO-VUB∗
1.0
1.0
1.0
1.0 DDIM-GMM-ORTHO-VUB
1.0 DDIM-GMM-ORTHO-VUB∗
1.0

DDIM
DDIM-GMM-RAND
DDIM-GMM-ORTHO

DDIM
DDIM-GMM-RAND
DDIM-GMM-ORTHO

DDPM

196.73
209.90
207.65
207.85
200.89
196.15
209.22
209.02
208.91
204.41
193.44
198.14
197.84
211.93
210.79
148.96
157.54
158.41
187.49
185.85

296.89
318.73
320.66
319.13
316.90
298.19
311.00
313.10
318.41
318.19
297.65
306.79
305.75
319.87
320.60
281.86
294.47
294.03
309.10
310.05

237.98
238.63
238.97
240.88
238.97
241.48
239.52
238.85
243.14
241.09
250.97
251.02
250.27
251.72
251.74
272.39
270.73
270.28
273.73
271.89

321.59
321.68
322.90
323.84
321.40
323.87
323.23
322.96
326.02
324.86
330.86
331.14
331.16
333.34
333.29
345.19
354.69
355.00
349.01
346.74

277.40

349.48

Table 12: Class-conditional ImageNet with classiﬁer free guidance(IS
↑

)

33

