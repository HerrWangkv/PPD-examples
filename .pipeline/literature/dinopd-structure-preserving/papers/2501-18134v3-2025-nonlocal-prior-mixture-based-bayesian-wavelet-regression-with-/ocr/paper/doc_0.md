Nonlocal Prior Mixture-Based Bayesian
Wavelet Regression with Application to Noisy
Imaging and Audio Data

Nilotpal Sanyal

Department of Mathematical Sciences, University of Texas at El Paso,
nsanyal@utep.edu

Abstract

We propose a novel Bayesian wavelet regression approach using a three-
component spike-and-slab prior for wavelet coefficients, combining a point
mass at zero, a moment (MOM) prior, and an inverse moment (IMOM) prior.
This flexible prior supports small and large coefficients differently, offering
advantages for highly dispersed data where wavelet coefficients span multi-
ple scales. The IMOM prior’s heavy tails capture large coefficients, while
the MOM prior is better suited for smaller non-zero coefficients. Further,
our method introduces innovative hyperparameter specifications for mixture
probabilities and scale parameters, including generalized logit, hyperbolic se-
cant, and generalized normal decay for probabilities, and double exponential
decay for scaling. Hyperparameters are estimated via an empirical Bayes ap-
proach, enabling posterior inference tailored to the data. Extensive simula-
tions demonstrate significant performance gains over two-component wavelet
methods. Applications to electroencephalography and noisy audio data il-
lustrate the method’s utility in capturing complex signal characteristics. We
implement our method in an R package, NLPwavelet (≥ 1.1).

Keywords: Three-component spike-and-slab prior; wavelet analysis; nonlocal
prior; generalized logit decay; hyperbolic secant decay; generalized normal
decay.

5
2
0
2

g
u
A
3
2

]
E
M

.
t
a
t
s
[

3
v
4
3
1
8
1
.
1
0
5
2
:
v
i
X
r
a

 
 
 
 
 
 
1

Introduction

We focus on spike-and-slab mixture models for wavelet-based Bayesian
nonparametric regression. There are several existing approaches that con-
sider for the wavelet coefficients various spike-and-slab mixture models, such
as mixtures of two Gaussian distributions with different standard devia-
tions (Chipman et al., 1997), mixtures of a Gaussian and a point mass at
zero (Clyde et al., 1998; Abramovich et al., 1998; Sanyal and Ferreira, 2012),
mixtures of a heavy-tailed distribution and a point mass at zero (Clyde and
George, 2000; Johnstone and Silverman, 2005), mixtures of a logistic distri-
bution and a point mass at zero (dos Santos Sousa, 2024), and mixtures of a
nonlocal prior and a point mass at zero (Sanyal and Ferreira, 2017). Nonlo-
cal priors (Johnson and Rossell, 2010) are a class of priors that assign zero
probability density in a neighborhood of the null value (often zero) of the pa-
rameter, unlike local priors, which are positive everywhere. In contrast to the
other works mentioned above that used traditionally used local priors for the
wavelet coefficients, ref. Sanyal and Ferreira (2017) pioneered the use of non-
local priors for wavelet regression. Specifically, ref. Sanyal and Ferreira (2017)
used two different nonlocal priors, namely the moment prior (MOM) and the
inverse moment (IMOM) prior, in their mixture model.
In this work, we
flexibly extend all the previous approaches by proposing a three-component
spike-and-slab mixture model for the wavelet coefficients, where, along with
a point mass for the spike part, a mixture of nonlocal priors is used to model
the slab component. In addition, we introduce novel hyperparameter specifi-
cations that are shown to provide improved estimates in extensive simulation
experiments with highly dispersed data.

In the Bayesian paradigm, nonlocal priors have been shown to encourage
model selection parsimony and selective shrinkage (unlike local priors) for
spurious coefficients (Johnson and Rossell, 2012; Rossell and Telesca, 2017).
They create a gap around zero, yielding harder exclusion and lower bias for
sizable effects. In contrast, local shrinkage priors provide continuous shrink-
age that is particularly suitable for estimation but less effective for variable
selection unless combined with explicit selection rules. Previous work (Sanyal
et al., 2019) has observed, in the context of high-dimensional genomics data,
that different nonlocal priors provide better support for large and small re-
gression coefficients. In wavelet regression, the wavelet coefficients at multiple
resolution levels capture both location and scale characteristics of the under-
lying function (Mallat, 2008). If the underlying function is highly dispersed,
its energy is spread across a wide range of location or scale components. For
such a function, while most of the wavelet coefficients will likely be small or

2

near zero, non-zero wavelet coefficients will span across multiple scales. In
other words, no single scale will dominate the wavelet coefficients, as different
scales will capture different portions of the signal’s energy. This will lead to
significant coefficients at both coarse and fine scales. We conjecture that, if
the underlying function is highly dispersed, a mixture of nonlocal priors will
provide better support to the distribution of the wavelet coefficients compared
to individual nonlocal priors. Specifically, in this work, we consider for the
wavelet coefficients a prior that is a mixture of a point mass at zero, a MOM
prior, and an IMOM prior.

Our motivation for combining MOM and IMOM priors stems from the fact
that they offer complementary strengths in sparse high-dimensional regres-
sion. The MOM prior places more mass around moderate non-zero values,
thereby offering greater sensitivity to small but meaningful signals. On the
other hand, the IMOM prior has heavier tails, allowing it to support large
coefficients better. In the context of wavelet regression of highly dispersed
data, a single prior, MOM or IMOM, may over-shrink large coefficients or
over-allow small noise fluctuations. A mixture of MOM and IMOM provides
adaptive flexibility, simultaneously supporting both ends of the coefficient
spectrum. This idea echoes adaptive shrinkage principles in sparse Bayesian
learning. From a signal processing perspective, the heterogeneous scaling
behavior of wavelet coefficients for highly dispersed signals motivates a prior
that can adaptively vary shrinkage across scales. This hybrid slab also enables
the prior predictive distribution to cover a broader range of plausible signals,
reducing prior-data conflict. While our empirical results support this mix-
ture’s performance, the theoretical justification lies in its capacity to model
heterogeneity in signal strengths, a feature not adequately captured by any
one nonlocal prior alone.

In Bayesian wavelet regression, the probability weights associated with dif-
ferent component distributions of a spike-and-slab mixture prior, henceforth
called mixture probabilities, and the scaling parameters of the component dis-
tributions are often governed by multiple hyperparameters (Abramovich et al.,
1998; Clyde et al., 1998; Sanyal and Ferreira, 2017). For the mixture probabil-
ities at different resolution levels, previous work considered exponential decay
specification (Abramovich et al., 1998), Bernoulli distribution (Clyde et al.,
1998), and logit specification (Sanyal and Ferreira, 2017). Further, for the
scaling parameters, exponential decay (Abramovich et al., 1998) and polyno-
mial decay specifications (Sanyal and Ferreira, 2017) have been considered.
In this work, we propose several novel specifications that flexibly model the
variations in the mixture probabilities and scaling components. For the mix-

3

ture probabilities, we consider a generalized logit decay, a hyperbolic secant
decay, and a generalized normal decay, whereas for the scaling parameter, we
consider a double exponential decay. Each of these specifications is controlled
by a few hyperparameters. Following an empirical Bayes approach, we esti-
mate the hyperparameters from the data and develop a posterior inference
conditional on the hyperparameter estimates.

Through extensive simulation studies, we assess the performance gains of
our approach under the different hyperparameter specifications, comparing
it to two-component spike-and-slab prior-based wavelet methods. Finally,
applications to real-world data, including electroencephalography (EEG) data
from a meditation study and audio data from a noisy musical recording,
illustrate the practical utility of the proposed method.

Although this work focuses on Bayesian spike-and-slab priors for wavelet
analysis, it is still relevant to acknowledge other types of priors explored in
the wavelet literature. These include scale mixtures of Gaussians (Vidakovic,
1998; Vidakovic and Ruggeri, 2001; Portilla et al., 2003; Cutillo et al., 2008),
hidden Markov model-based priors (Crouse et al., 1998), the generalized Gaus-
sian distribution prior (Chang et al., 2000), Jeffreys’ prior (Figueiredo and
Nowak, 2001), the Bessel K Form prior (Boubchir and Boashash, 2013), the
double Weibull prior (Rem´enyi and Vidakovic, 2015), Bayes factor threshold-
ing based on mixtures of conjugate priors (Afshari et al., 2017), the beta prior
(Sousa et al., 2021), and the logistic prior (dos Santos Sousa, 2022). For com-
parative reviews and summaries of wavelet-based nonparametric regression
methods, see Vidakovic (1999); Antoniadis et al. (2001).

In what follows, Section 2 describes the proposed Bayesian hierarchical
model along with the hyperparameter specifications, and Section 3 describes
the empirical Bayes inference procedure. Subsequently, Section 4 describes
the simulation experiment and results. The real data applications appear
in Section 5 and Section 6. Finally, Section 7 concludes with an overall
discussion of our work and relevant remarks. Theoretical proofs are included
in Appendix 7.

2 Bayesian Hierarchical Model

2.1 Observation Model

Suppose y1, . . . , yn represent n noisy observations from an unknown func-
tion f (t), which we aim to estimate. We consider the observation model
i = 1, . . . , n, where ti = i/n represents the equispaced sam-
yi = f (ti) + ϵi,

4

pling points, and the errors ϵi, . . . , ϵn are assumed to be independent and iden-
tically distributed (i.i.d.) normal random variables, ϵi ∼ N (0, σ2), with un-
known variance σ2. Let y = (y1, . . . , yn) denote the vector of observations,
f = (f (t1), . . . , f (tn)) the vector of true functional values, and ϵ = (ϵ1, . . . , ϵn)
the vector of errors. The observation model can be expressed in matrix form
as

y = f + ϵ,

ϵ ∼ N (0, σ2In),

(1)

where In is the n-dimensional identity matrix. In wavelet regression, the goal
is to estimate the function f (t) from the observations y by decomposing f (t)
into wavelet basis functions. This decomposition allows us to exploit the
multiscale nature of wavelets to capture both local and global features of
the function. A chief feature of wavelet regression is its ability to adapt to
different levels of smoothness and to handle noisy data efficiently, making it
especially useful in nonparametric function estimation problems.

2.2 Wavelet Coefficient Model

We represent f using an orthogonal wavelet basis matrix as f = Wd
(Donoho and Johnstone, 1995), where W is the orthogonal basis matrix and
d is a vector whose elements include the scaling coefficient at the coarsest
resolution level along with the wavelet coefficients at all resolution levels. Let
bd = WT y denote the vector of empirical wavelet coefficients. Since W is
orthogonal, we can express bd as bd = d + ϵ∗, where ϵ∗ = WT ϵ represents the
transformed error vector with ϵ∗ ∼ N (0, σ2In).

Suppose dlj denotes the wavelet coefficient at position j and resolution
level l, with bdlj defined similarly for the empirical wavelet coefficients. Then,
the model in terms of individual coefficients is given by

bdlj = dlj + ϵ∗
lj,

lj ∼ N (0, σ2)
ϵ∗

(2)

2.3 Mixture Prior for the Wavelet Coefficients

For the wavelet coefficient dlj, we consider a spike-and-slab mixture prior
that is a mixture of three components—a MOM prior, an IMOM prior, and a

5

point mass at zero—given by
dlj|γ(1)
l

, γ(2)
l

, τ (2)
, τ (1)
l
l
(cid:16)
1 − γ(1)

, σ2, r, ν ∼ γ(1)
l M OM
(cid:16)
(cid:17)
τ (2)
l

γ(2)
l IM OM

, ν, σ2(cid:17)

l

+

(cid:16)

τ (1)
l

+

, r, σ2(cid:17)
(cid:16)
1 − γ(1)
l
0 < γ(1)

l

(cid:17) (cid:16)

(cid:17)

δ0(·),

l

1 − γ(2)
, γ(2)

l < 1,

(3)

1 and γ(2)
where γ(l)
and variance component τ (1)
following density function:

l

are mixture probabilities. The M OM prior with order r
l σ2, involving the scale parameter τ (1)
, has the

l

mom(dlj|τ (1)

l

, r, σ2) = fMr

(cid:16)

l σ2(cid:17)−r−1/2
τ (1)

d2r
lj exp


−


 ,

d2
lj
2τ (1)
l σ2

r > 1, τ (1)

l > 0,

where fMr = (2π)−1/2/(2r − 1)!! and (2r − 1)!! = 1 × 3 × . . . × (2r − 1).
The IM OM prior with shape parameter ν and variance component τ (2)
l σ2,
involving the scale parameter τ (2)
l σ2(cid:17)ν/2
τ (2)
Γ(ν/2)

, has the following density function:

imom(dlj|τ (2)

|dlj|−ν−1 exp

ν > 1, τ (2)

, ν, σ2) =


−

l > 0.


 ,

(cid:16)

l

l

τ (2)
l σ2
d2
lj

The conditional density of the wavelet coefficient dlj given that dlj ̸= 0 can

be expressed as

π(dlj|dlj ̸= 0) =

γ(1)
l
1 − γ(1)

l

(cid:16)

(cid:17)

γ(2)
l

γ(1)
l +

fMr

(cid:16)

l σ2(cid:17)−r−1/2
τ (1)

d2r
lj exp

(cid:16)

(cid:17)

1 − γ(1)
(cid:16)

γ(2)
l
l
(cid:17)
1 − γ(1)

l

γ(1)
l +

γ(2)
l

(cid:16)

l σ2(cid:17)ν/2
τ (2)
Γ(ν/2)

|dlj|−ν−1 exp


−





d2
lj
2τ (1)
l σ2


−


 .

τ (2)
l σ2
d2
lj

Figure 1 shows plots of our proposed three-component spike-and-slab mix-
ture model (solid line) for dlj with γ1l = γ2l = 0.25, τ1l = τ2l = 0.2, r = ν = 1,
and σ = 1 along with two-component spike-and-slab mixture models based
on MOM (dashed line) and IMOM (dotted line) priors with γl = 0.5, τl = 1,
and σ = 1.

2.4 Hyperparameter Specifications

The mixture prior given in (3) depends on the mixture probabilities γ(l)
1
, order r, and shape parameter ν. This
. Previous

and γ(2)
section considers different specifications for γ(l)

, scale parameters τ (1)

, and τ (1)

and τ (1)

l

l

l

1 , γ(2)

l

, τ (1)
l

l

6

work in wavelet regression considered two-component spike-and-slab mixture
priors, where the mixture probabilities were specified using exponential decay
specification (Abramovich et al., 1998), Bernoulli distribution (Clyde et al.,
1998), and logit specification (Sanyal and Ferreira, 2017). In this work, we
examine three specifications for the mixture probabilities that flexibly model
the variations in the mixture probabilities and, to the best of our knowledge,
are hitherto unused in wavelet regression. These novel specifications were
motivated by an exploratory data analysis where, for multiple highly dispersed
data, we observed how the wavelet coefficients changed with resolution level.
We compare these novel specifications with logit specification for γ(1)
and γ(2)
,
given by γ(1)
l = exp(θγ
1 − θγ
2 l)/{1 + exp(θγ
3 − θ4l) {1 +
2 , θγ
3 ∈ R, θγ
3 − θγ
exp(θγ
4 > 0 (Sanyal and Ferreira, 2017).
Figure 2 shows, in the left panel, the plots of the different specifications for
the mixture probabilities against resolution level, with specific values of the
hyperparameters. The novel specifications are described as follows:

l = exp(θγ
4 l)}, where θγ

1 − θγ
1 , θγ

2 l)}, γ(2)

l

l

(a) Generalized logit or Richards decay specifications, given by

γ(1)
l =

γ(2)
l =

1
[1 + exp{−(θγ
1
[1 + exp{−(θγ

1 − θγ

2 l)}]θγ

3

,

1 ∈ R, θγ
θγ

2 , θγ

3 > 0

5 l)}]θγ
This form corresponds to a flexible S-shaped decay curve that reduces
3 (or θγ
to standard logistic decay when θγ
6 ) and controls the steepness of
the curve equal to one.

4 − θγ

6

,

4 ∈ R, θγ
θγ

5 , θγ

6 > 0.

(b) Hyperbolic secant decay specifications, given by

γ(1)
l =

γ(2)
l =

2
π
2
π

arctan

arctan

(cid:20)

(cid:20)

exp

exp

(cid:18)π
2
(cid:18)π
2

(cid:19)(cid:21)

(θγ

1 − θγ

2 l)

(cid:19)(cid:21)

(θγ

3 − θγ

4 l)

,

,

1 ∈ R, θγ
θγ

2 > 0

3 ∈ R, θγ
θγ

4 > 0.

This form, although less intuitive, is also sigmoid-like but with heavier
tails and slower decay than logistic, indicating that the mixture param-
2 (or θγ
eters approach one more slowly with a smoother transition. θγ
4 )
controls steepness and θγ

3 ) controls shift.

1 (or θγ

7

(c) Generalized normal decay specifications, given by

γ(1)
l =

γ(2)
l =

1
2

1
2

+ sign(θγ

1 − l)

+ sign(θγ

4 − l)

1
2Γ(1/θγ
2 )

1
2Γ(1/θγ
5 )


1/θγ
2 ,

γ


1/θγ
5 ,

γ

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

θγ
1 − l
θγ
3

θγ
4 − l
θγ
6

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

θγ
2


 ,

θγ
5


 ,

1 ∈ R, θγ
θγ

2 , θγ

3 > 0

4 ∈ R, θγ
θγ

5 , θγ

6 > 0.

This form is most flexible and can mimic normal CDF, Laplace CDF,
and many other distributions, but also most complex. Whereas θγ
2 (or
θγ
2 ) controls tail behavior with larger values indicating lighter tails, others
control scale and shift.

For scale parameters of spike-and-slab mixture priors, previously consid-
ered specifications include exponential decay (Abramovich et al., 1998) and
polynomial decay (Sanyal and Ferreira, 2017). Here, for τ (1)
, we con-
l
sider the polynomial decay specification given by τ (1)
3 l−θτ
l = θτ
with θτ
4 > 0. In addition, for modeling the variations in the scale
parameters more flexibly, we novelly propose the following:

and τ (2)
l
2 , τ (2)
l = θτ

1 l−θτ

3 , θτ

2 , θτ

1 , θτ

4

(d) Double exponential decay specifications, given by

τ (1)
l = θτ
τ (2)
l = θτ

1 exp(−θτ

2 l) + θτ

3 exp(−θτ

4 l),

5 exp(−θτ

6 l) + θτ

7 exp(−θτ

8 l),

θτ
1 , θτ

2 , θτ

3 , θτ

4 > 0

θτ
5 , θτ

6 , θτ

7 , θτ

8 > 0

Figure 2 shows, in the right panel, the plots of the different specifications
for the scale parameters against resolution level, with specific values of the
hyperparameters.

Each of the proposed specifications admits interpretable controls over rate
and shape of decay across scales, crucial for adaptivity in multi-resolution
modeling, and is governed by a small number of hyperparameters. Note that
the hyperparameters for the mixture probabilities are superscripted with γ
and those for the scale parameters are superscripted with τ . In our simula-
tion studies described in Section 4, we analyze each simulated dataset using
4 × 2 = 8 configurations arising out of the combinations of the above spec-
ifications for the mixture probabilities and the scale parameters. While no
formal optimality results are available for these specific forms, their empirical
performance, as shown in Section 4, supports their practical relevance. Let θ
generically denote the set of all hyperparameters for any given configuration.

8

Figure 1: Plots of our proposed three-component spike-and-slab mixture
model (solid line) with γ1l = γ2l = 0.25, τ1l = τ2l = 0.2, r = ν = 1, and σ = 1
along with two-component spike-and-slab mixture models based on MOM
(dashed line) and IMOM (dotted line) priors with γl = 0.5, τl = 1, and
σ = 1.

9

−4−20240.000.050.100.150.200.250.30Wavelet coefficient (d)Density2−component MOM2−component IMOMProposedl

and γ(2)

Figure 2: Plots of the different specifications considered for the mixture prob-
abilities γ(1)
(logit, generalized logit, hyperbolic secant, and general-
l
ized normal) and scale parameters τ (1)
(polynomial decay and double
exponential decay) against resolution level, with specified values of the hy-
perparameters.

and τ (2)

l

l

3

Inference

For inference in our proposed Bayesian hierarchical wavelet regression
model based on three-component spike-and-slab mixture priors, we adopt
the empirical Bayes approach (Clyde and George, 2000; Sanyal and Ferreira,
2017). This methodology estimates the hyperparameters from the data and
performs posterior inference conditioned on these estimated hyperparameters.
For simplicity, in our simulation analysis, we set r = 1 and ν = 1. Further, we
estimate the error variance σ2 using the median absolute deviation estimator
(Donoho et al., 1995) ˆσ = 0.6745−1 medianj(| bdLj − medianj( bdLj)|), which
is a well-established practice in wavelet regression (Abramovich et al., 1998;
Clyde et al., 1998; Clyde and George, 2000; Johnstone and Silverman, 2005).

3.1 Hyperparameter Estimation

Let δ(x) denote the value of the point mass function δ(0) at x. The fol-

lowing result is used to obtain the hyperparameter estimates.

Result 1 Integrating out dlj from the wavelet coefficient model (2) using the
mixture prior of the wavelet coefficients in (3) and using the Laplace approx-
imation for the IMOM prior component, we get the marginal distribution of

10

the empirical wavelet coefficients, bdlj, as

π( bdlj|σ2, θ, r, ν) ≈ γ(1)

l

(cid:16)

1 + τ (1)

l

(cid:17)−r

M ∗

, σ2) ϕ

(cid:16)

bdlj; 0, σ2 (cid:16)

1 + τ (1)

l

(cid:17)(cid:17)

(cid:16)

+

(cid:16)

1 − γ(1)

l

(cid:17)

γ(2)
l

(cid:16)

bdlj; 0, σ2(cid:17) √

2πσ∗h(d∗

lj( bdlj))

l

r ( bdlj, τ (1)
l σ2(cid:17) ν
τ (2)
Γ(ν/2)

2

ϕ

+

(cid:16)

1 − γ(1)

l

(cid:17) (cid:16)

1 − γ(2)

l

(cid:17)

(cid:16)

bdlj; 0, σ2(cid:17)

,

ϕ

(4)

where, in the offshoot of the MOM component,

M ⋆
r

(cid:16)
bdlj, τ (1)
l

, σ2(cid:17)

=

1
(2r − 1)!!

r
X

i=0

(2r)!
(2i)!(r − i)!2r−i

and in the offshoot of the IMOM component,

v
u
u
u
t






τ (1)
l
1 + τ (1)

l

bdlj
σ

2i






,

h(dlj) = |dlj|−(ν+1) exp






−

1
2σ2

(cid:16)

d2
lj − 2dlj bdlj

(cid:17)

−

τ (2)
l σ2
d2
lj






,

d∗
lj( bdlj) is the global maxima of h(dlj), and σ2
log(h(dlj)).

∗ = −1/L′′

h(d∗

lj( bdlj)), with Lh(dlj) =

The proof of Result 1 is given in Appendix 7. Using (4), the marginal likeli-
hood function is approximately given by

(cid:20)
γ(1)
l

(cid:16)

1 + τ (1)

l

(cid:17)−r

Y

Y

l

j

M ∗

r ( bdlj, τ (1)

l

, σ2) ϕ

bdlj; 0, σ2 (cid:16)
(cid:16)

1 + τ (1)

l

(cid:17)(cid:17)

(cid:16)

+

1 − γ(1)

l

(cid:17)

γ(2)
l

(cid:16)

2

l σ2(cid:17) ν
τ (2)
Γ(ν/2)

(cid:16)

bdlj; 0, σ2(cid:17) √

ϕ

2πσ∗h(d∗

lj( bdlj))

+

(cid:16)
1 − γ(1)

l

(cid:17) (cid:16)

1 − γ(2)

l

(cid:17)

(cid:16)

bdlj; 0, σ2(cid:17)i

.

ϕ

This is a function only of the data and the hyperparameters. Hence, we
maximize this function with respect to the hyperparameters to obtain their
estimates, bθ. With θ = bθ, the prior distribution in (3) is fully known.

3.2 Posterior Distribution

The following two results provide the posterior estimates of the wavelet

coefficients.

Result 2 The conditional posterior density of the wavelet coefficient dlj, given

11

that dlj ̸= 0 and the hyperparameter estimates ˆθ, by using the Laplace approx-
imation for the IMOM prior component, can be expressed as

π(dlj|dlj ̸= 0, σ2, θ, r, ν, y) =

p(1)
lj
lj + p(2)
p(1)

lj

(cid:16)

M ∗
r

exp






−

2σ2

1
τ (1)
l(cid:16)
1+τ (1)
l

(cid:17)


dlj −

τ (1)
l
1 + τ (1)

l

2


bdlj



fMr
bdlj, τ (1)
l




+

, σ2(cid:17) d2r

lj

p(2)
lj
lj + p(2)
p(1)

lj

(cid:16)

ϕ

dlj|d∗

lj( bdlj), σ2
∗

(cid:17)

,

where

O(1)

lj =

(cid:16)

and

p(1)
lj =

,

p(2)
lj =

O(1)
lj
lj + O(2)
1 + O(1)
γ(1)
l
(cid:17) (cid:16)

(cid:17)

lj

1 − γ(2)

l

1 − γ(1)

l

(cid:16)
1 + τ (1)

l

(cid:17)−r−1/2

,

O(2)
lj
lj + O(2)
1 + O(1)

1

2σ2

r exp

M ∗

lj



τ (1)
l
1 + τ (1)

l






,

bd2
lj

O(2)

lj =

γ(2)
l
1 − γ(2)

l

(cid:16)

2

l σ2(cid:17) ν
τ (2)
Γ(ν/2)

√

2πσ∗h(d∗

lj( bdlj)).

The proof is given in Appendix 7.

Result 3 The posterior expectation of the wavelet coefficients dlj is

¯dlj = E(dlj|y) = p(1)
lj

v
u
u
u
t

M ∗∗
r
M ∗
r

τ (1)
l
1 + τ (1)

l

σ + p(2)

lj d∗

lj( bdlj),

where

M ⋆⋆
r

(cid:16)

bdlj, τl, σ2(cid:17)

=

1
(2r − 1)!!

r+1
X

i=1

(2r + 1)!
(2i − 1)!(r + 1 − i)!2r+1−i

v
u
u
u
t






τ (1)
l
1 + τ (1)

l

bdlj
σ

2i−1






.

The proof of Result 3 is immediate from the proof of Result 2 and hence
is omitted. Let ¯d = {dlj : l = 1, . . . , L, j = 1 . . . , J} denote the vector of
posterior means of the wavelet coefficients. We use ¯d in the inverse discrete
wavelet transform to obtain the posterior mean of the unknown function f ,
given by

¯f = E(f |y) = W E(d|y) = W¯d.

12

4 Simulation study

In this section, we conduct an extensive simulation analysis to compara-
tively evaluate the performance of the proposed method with different hy-
perparameter configurations. We consider three well-known test functions
proposed by Donoho and Johnstone (1994)—blocks, bumps, and doppler —
that are used as standard test functions in the wavelet literature. However,
we modify the coefficients used by Donoho and Johnstone (1994) to obtain
more highly dispersed signals. We define our modified test functions as

fblocks(t) = X hjK(t − tj), where K(t) = {1 + sign(t)}/2,

(tj) = (0.1, 0.13, 0.15, 0.23, 0.25, 0.40, 0.44, 0.65, 0.76, 0.78, 0.81),
(hj) = (4, −8, 3, −4, 8, −4.2, 2.1, 4.3, −6.1, 2.1, −4.7);

fbumps(t) = X hjK((t − tj)/wj), where K(t) = (1 + |t|)−4,

(tj) = (0.1, 0.13, 0.15, 0.23, 0.25, 0.40, 0.44, 0.65, 0.76, 0.78, 0.81),
(hj) = (2, 10, 1, 4, 8, 4.2, 2.1, 4.3, 1.1, 3.1, 8.2),
(wj) = (0.005, 0.005, 0.006, 0.01, 0.01, 0.03, 0.01, 0.01, 0.005, 0.008, 0.005);

fdoppler(t) = {t(1 − t)}1/2 sin{2π(1 + ϵ)/(t + ϵ)}, ϵ = 0.01.

As these modifications involved adjustments to specific numerical values only,
the overall functional forms remained consistent with the original Donoho–
Johnstone test functions. In addition, we consider three different linear com-
binations of these functions—lcomb1, lcomb2, and lcomb3 —that represent
curves that combine various features—such as blockiness, bumpiness, and
changing frequency—in different proportions, given by

flcomb1(t) = 0.4fblocks(t) + 0.4fbumps(t) + 0.2fdoppler(t)
flcomb2(t) = 0.4fblocks(t) + 0.2fbumps(t) + 0.4fdoppler(t)
flcomb3(t) = 0.2fblocks(t) + 0.4fbumps(t) + 0.4fdoppler(t).

Figure 3 shows the plots of the test functions blocks, bumps, and doppler
using Donoho–Johnstone specifications (dashed line) and our specifications
(solid line), and lcomb1, lcomb2, and lcomb3 based on our specifications, all
evaluated at 1024 equally spaced in (0,1). We evaluate each considered test
function at n = 512, 1024, 2048, and 4096 equidistant points in the interval
(0,1) and add random Gaussian noise with mean 0 to generate data with
signal-to-noise ratio, SNR = 3, 5, and 7. For each combination of n and SNR,
we consider 100 replications. The simulated datasets are analyzed using the
proposed methodology with eight different hyperparameter configurations de-

13

scribed in Section 2. For comparison, we also analyzed the datasets using indi-
vidual MOM and IMOM prior-based two-component mixture models (Sanyal
and Ferreira, 2017). Thus, a total of 24 analysis methods were applied to
each dataset. Note that the prior literature (Sanyal and Ferreira, 2017) has
already shown that nonlocal prior-based wavelet analysis generally performs
better than other existing wavelet-based methods such as sure (Donoho and
Johnstone, 1995), BayesThresh (Abramovich et al., 1998), cv (Nason, 1996),
fdr (Abramovich and Benjamini, 1996), and Ebayesthresh (Johnstone and Sil-
verman, 2005). So, for brevity, we do not compare with these methods in this
work. For wavelet transformation, we consider the Daubechies least asym-
metric wavelet with six vanishing moments and periodic boundary conditions.
Wavelet computations are implemented using the R package wavethresh (Na-
son, 2024).

Figure 3: Plots of the test functions blocks, bumps, and doppler using the
Donoho–Johnstone (DJ) specifications (dashed line) and our specifications
(solid line), and three linear combinations of them—lcomb1, lcomb2, and
lcomb3 —based on our specifications, all evaluated at 1024 equally spaced
in (0,1).

Table 1 presents, for each test function and analysis method, the number of
(n, SNR) combinations where the method achieved the lowest mean squared

14

error (MSE). For each test function, the method with the highest frequency
of best performance (i.e., the lowest MSE) is highlighted in bold. We observe
the following:

(a) For every function, the highest frequency of best performance was shown
by a three-component mixture method (with one tie with a two-component
mixture method for the blocks function), which is proposed in the current
work.

(b) Out of all eight three-component mixture methods, the one with general-
ized normal specification for the mixture probabilities and double expo-
nential decay specification for the scale parameters (mixture-gennormal-
doubleexp) showed the maximum number of best performances in total
(12 times) for all the test functions. This was followed by the method us-
ing logit specification for the mixture probabilities and polynomial decay
specification for the scale parameters (mixture-logit-polynom) (11 times)
and the method using generalized normal specification for the mixture
probabilities and polynomial decay specification for the scale parameters
(mixture-logit-polynom) (9 times).

(c) Considering only the test functions lcomb1, lcomb2, and lcomb3 that
represent signals with mixed characteristics in various proportions, the
mixture-gennormal-doubleexp method showed the maximum number of
best performances (10 times).

For real data, SNR and the original function will not be known. So, next,
in Figure 4, we show, for each of the eight three-component mixture methods,
the MSE for the four different sample sizes (n), averaged over the three SNRs
and six test functions considered in our study. Overall, methods with a double
exponential decay specification for the scale parameters showed a better per-
formance. Specifically, for the two largest sample sizes (2048 and 4096), the
method with hyperbolic secant and double exponential decay specifications
(mixture-hypsec-doubleexp) and the method with generalized logit and dou-
ble exponential decay specifications (mixture-genlogit-doubleexp) provided top
performances. Notably, the method with generalized normal and polynomial
decay specifications showed a worse performance in Figure 4, even though the
methods with generalized normal specifications showed the best performance
in Table 1. This implies that the methods with generalized normal spec-
ifications, although most often provide the best denoising, may sometimes
produce large bias that negatively affects their overall average performance.
So, the results obtained by their usage should be validated by field knowledge
or external means (such as auditory assessment for sound signals).

15

Table 1: Method comparison: Data were simulated for each test function
across 12 combinations of sample size (n) and SNR, with 100 replications
per combination. A total of 24 analysis methods were applied to each
dataset—8 methods with MOM-based two-component mixture prior, 8
methods with IMOM-based two-component mixture prior, and 8 methods
with the proposed three-component mixture prior. The average MSE was
computed for each method across the replications. The table presents, for
each test function and analysis method, the number of (n, SNR) combinations
where the method achieved the lowest MSE. For each test function, the
method with the highest frequency of best performance is highlighted in bold.

Method
mom-logit-polynom
mom-logit-doubleexp
mom-genlogit-polynom
mom-genlogit-doubleexp
mom-hypsec-polynom
mom-hypsec-doubleexp
mom-gennormal-polynom
mom-gennormal-doubleexp

imom-logit-polynom
imom-logit-doubleexp
imom-genlogit-polynom
imom-genlogit-doubleexp
imom-hypsec-polynom
imom-hypsec-doubleexp
imom-gennormal-polynom
imom-gennormal-doubleexp

mixture-logit-polynom
mixture-logit-doubleexp
mixture-genlogit-polynom
mixture-genlogit-doubleexp
mixture-hypsec-polynom
mixture-hypsec-doubleexp
mixture-gennormal-polynom
mixture-gennormal-doubleexp
Total

blocks bumps doppler lcomb1 lcomb2 lcomb3 Total
1
1
0
0
1
3
0
0

1
1
0
0
0
3
0
0

0
0
0
0
1
0
0
0

0
0
0
0
0
0
0
0

0
0
0
0
0
0
0
0

0
0
0
0
0
0
0
0

0
0
0
0
0
0
0
0

0
0
0
0
0
0
0
0

0
2
2
0
4
0
2
2
12

0
0
0
0
0
0
0
0

1
1
0
0
1
4
1
4
12

0
0
0
0
0
0
0
0

3
2
1
1
1
1
1
1
12

0
0
0
0
0
0
0
0

0
1
2
0
0
2
2
5
12

0
0
0
0
0
0
0
0

11
7
8
4
7
8
9
12
72

0
0
0
0
0
0
0
0

1
0
2
3
1
0
0
0
12

0
0
0
0
0
0
0
0

6
1
1
0
0
1
3
0
12

16

Figure 4: For the proposed three-component mixture-based methods, MSE
for different sample sizes (n), averaged over 3 SNRs and 6 test functions.

5 EEG Meditation Study

We demonstrate the utility and flexibility of the proposed methodology
through the analysis of EEG data from a meditation study, investigating
the relation between mind wandering and meditation practice. Detailed in-
formation about the study is provided in Brandmeyer and Delorme (2018),
and the dataset is available on the OpenNeuro platform (see the Supplemen-
tary Materials Section S1 for the link). The meditation experiment involved
24 subjects—12 experienced meditators (10 males, 2 females) and 12 novices
(2 males, 10 females). Participants meditated while being interrupted ap-
proximately every two minutes to report their level of concentration and mind
wandering via three probing questions. Each participant completed two to
three sessions lasting 45 to 90 minutes, with a minimum of 30 probes per
participant. EEG data were collected using a 64-channel Biosemi system
(channels A1–A32 and B1–B32) with a Biosemi 10–20 head cap montage at
a sampling rate of 2048 Hz, providing spatial information about brain activ-
ity across different scalp regions. The dataset available on OpenNeuro has
already been downsampled to 256 Hz.

For our analysis, we focused on the data from session 1 of four partici-
pants: two expert meditators (one male and one female, identified as subjects

17

0.000.020.040.060.080.100.120.14mixture−genlogit−doubleexpmixture−genlogit−polynommixture−gennormal−doubleexpmixture−gennormal−polynommixture−hypsec−doubleexpmixture−hypsec−polynommixture−logit−doubleexpmixture−logit−polynomn51210242048409610 and 5 in the original dataset, respectively) and two novices (one male
and one female, identified as subjects 23 and 14, respectively). The data
were average-referenced to mitigate common noise and artifacts by calculat-
ing the mean signal across all electrodes at each time point and subtracting
it from the signal at each electrode. Following this, a 2 Hz high-pass filter
was applied using an infinite impulse response (IIR) filter with a transition
bandwidth of 0.7 Hz and an order of six. Denoting the time of probe 1
by t, we analyzed the EEG signal within the 16 s interval (t − 8, t + 8).
With a sampling rate of 256 Hz, this interval comprised 256×16 = 4096
data points, representing noisy observations of the underlying signal. We
analyzed the EEG data using the mixture-hypsec-doubleexp, mixture-genlogit-
doubleexp, and mixture-gennormal-polynom methods (see Section 4), employ-
ing the Daubechies least asymmetric wavelet transform with six vanishing
moments and periodic boundary conditions.

Figure 5 presents the plots of the posterior means of the EEG signal (in
slategray) based on the mixture-genlogit-doubleexp (left column), mixture-
hypsec-doubleexp (middle column), and mixture-gennormal-doubleexp (right
column) methods, superimposed on the observed data (in black), obtained
during the 16 s interval from the A10 channel of the four considered partic-
ipants. Similar plots for some other channels are presented in the Supple-
mentary Materials Section S3. The plots clearly indicate that our method,
with all the considered hyperparameter configurations, yielded significantly
denoised estimates.

6 Musical Sound Study

To further demonstrate the utility of the proposed method, we consider
analyzing musical sounds that often exhibit sudden frequency changes over
short periods, resulting in highly dispersed signals. For this study, we focus on
a vocal music recording from 1934, originally published on a 78 RPM record of
Hindustani classical music (one of India’s two classical music traditions) and
performed by eminent Ustad Amir Khan. A noisy copy of this recording was
sourced from YouTube (link provided in the Supplementary Materials Section
S1). From the recording, we extracted a 15-second segment representing a
wide frequency range and saved it as a 16-bit WAV audio file, which represents
noisy data.

For our analysis, we divided each of the two audio file channels (left
and right) into sections of 4096 data points to enhance computational effi-
ciency. Each section was independently analyzed using the mixture-genlogit-

18

Figure 5: Plots of the posterior means of the EEG signal (in slate gray) based
on the imom-logit-polynom (left), mixture-genlogit-doubleexp (middle),
and mixture-gennorm-polynom (right) methods, superimposed on the ob-
served data (in black), obtained during the 16 s interval (t − 8, t + 8), t being
probe 1 onset time, from the A10 channel of the 4 considered participants.

19

−2001020timesignal02468101316−2001020timesignal02468101316−2001020timesignal02468101316−20−1001020timesignal02468101316−20−1001020timesignal02468101316−20−1001020timesignal02468101316−15−50510timesignal02468101316−15−50510timesignal02468101316−15−50510timesignal02468101316−505timesignal02468101316−505timesignal02468101316−505timesignal02468101316A10:mixture−genlogit−doubleexpA10:mixture−hypsec−doubleexpA10:mixture−gennormal−doubleexpSubject14Subject 23Subject 5Subject 10doubleexp and mixture-hypsec-doubleexp methods (see Section 4) employing
the Daubechies coiflets wavelet transform with five vanishing moments and
periodic boundary conditions. The posterior estimates of the individual sec-
tions were then combined to reconstruct the posterior estimate of the entire
audio segment.

In Figure 6, we present the posterior mean of the right-channel signal of
the selected audio segment (in slate gray), superimposed on the noisy data
(in black). The audio files corresponding to these posterior estimates, along
with the original data and posterior estimates using the hard thresholding
rule, are provided in the Supplementary Materials for auditory comparison
and assessment. The posterior estimates show significant denoising and pre-
cise recovery of the signal. There is a slight systematic noise present in the
posterior audio files, which is due to analyzing the whole segment in disjoint
sections for computational convenience. That can be mitigated by analyz-
ing larger sections or analyzing partially overlapping sections with weighted
averaging (e.g., cross-fading or Hann windowing) to smoothly combine over-
lapping regions and reduce boundary artifacts.

Figure 6: Plots of the posterior means (in slate gray) of the right-channel
signal of the chosen audio segment of the vocal music recording, superimposed
on the noisy data (in red).

20

indexsignal−15,000−5,0005,00015,000mixture−hypsec−doubleexpindexsignal−15,000−5,0005,00015,000mixture−genlogit−doubleexpOriginalPosterior7 Discussion

In this work, we proposed the use of nonlocal prior mixtures for wavelet-
based nonparametric function estimation. The main innovations of our method-
ology are as follows:

(a) We introduce a three-component spike-and-slab prior for the wavelet
coefficients. This structure is particularly suited for modeling highly
dispersed signals. The slab component is a mixture of two nonlocal
priors—the MOM and IMOM priors, which offer enhanced adaptability
to signal characteristics.

(b) We propose flexible and previously unexplored hyperparameter specifi-
cations. These include generalized logit (or Richards), hyperbolic secant,
and generalized normal decay specifications for the mixture probabilities,
as well as a double exponential decay structure for the scale parameter.
These enhancements provide improved flexibility and accuracy in mod-
eling complex signal patterns, as demonstrated in our simulation study.

(c) We implement our methodology within the R programming language (R
Core Team, 2024) as a package named NLPwavelet (Sanyal, 2025), which
performs nonlocal prior (NLP)-based wavelet analysis.

In the simulation study, using more dispersed versions of the Donoho–
Johnstone test functions and various linear combinations of them, we com-
pared the performance of the proposed approach with the existing two-component
spike-and-slab mixture prior and demonstrated the superior flexibility of the
proposed approach. Further, analysis using several novel hyperparameter con-
figurations provided valuable insights into the relative advantages. Although
no formal optimality results are available for these specific forms, their flex-
ibility is supported by established theoretical principles for scale-dependent
shrinkage, and their empirical performance, demonstrated in the simulation
study of Section 4, provides strong evidence of their practical relevance.

Note that, in our empirical Bayes implementation, we did not encounter
convergence failures. However, in smaller samples or high-noise scenarios,
certain decay specifications—particularly those involving generalized normal
parameters—showed greater variability in estimated hyperparameters. To re-
duce potential overfitting and identifiability issues, we recommend sensitivity
checks across multiple decay specifications and inspection of hyperparameter
estimates for plausibility. Such practices can help ensure the robustness of
conclusions in applied settings.

A necessary limitation of the proposed approach is its higher computa-

21

tional cost relative to two-component mixture priors. Supplementary Materi-
als Section S2 reports the average runtime (in seconds) for 24 analysis meth-
ods across the sample sizes n considered in our simulation study. Within
each method, the specification of the mixture probability has only a mod-
est effect on runtime, while for the scaling parameter, doubleexp options are
generally slightly slower than polynom options. The runtime growth of the
mixture methods suggests approximately linear scaling with sample size. For
instance, for mixture-logit-polynom, the runtime for n = 1024 is about 1.96
times that for n = 512; for n = 2048, it is about 1.96 times that for n = 1024;
and for n = 4096, about 1.97 times that for n = 1024. That near doubling of
runtime when n doubles is consistent across the other mixture methods too.
This near doubling with each doubling of n is consistent across other mixture
methods, indicating a computational complexity of roughly O(n), which is
generally considered good and efficient for large-scale problems. Nonetheless,
the larger constant factor of the proposed three-component methods, relative
to two-component methods, results in longer absolute runtimes, which is an
anticipated trade-off for their enhanced modeling flexibility. While our simu-
lations focus on one-dimensional signals, the linear scaling behavior suggests
the approach can extend to higher-dimensional settings (e.g., 2D/3D images),
subject to the corresponding increase in the number of coefficients.

Further, while the EEG and audio denoising examples illustrate the prac-
tical use of the proposed method, their evaluation is primarily qualitative
due to the lack of ground truth reference signals. For EEG data, objective
measures such as SNR improvement or reconstruction error cannot be com-
puted reliably without a known clean signal. Similarly, in the audio example,
perceptual quality metrics such as PESQ (Recommendation, 2001) or STOI
(Taal et al., 2010) require access to a clean reference, which was not available
for the real-world recording used here. Future work, incorporating controlled
experiments in which synthetic noise is added to high-quality EEG or audio
recordings, would allow the computation of such quantitative metrics and
hence direct, reproducible comparisons with existing denoising techniques,
complementing the qualitative assessments presented in this study.

An obvious extension of the proposed approach is to adapt it to multi-
dimensional wavelet regression, such as 2D or 3D image processing tasks.
Another possible avenue is to develop fully Bayesian hierarchical models by
specifying prior distributions for the hyperparameters and employing Markov
chain Monte Carlo tools or variational methods for posterior inference.
In
addition, one can explore combining wavelet decompositions with local poly-
nomial regression to mitigate the boundary bias issues. Further, adapting our

22

methodology to non-Gaussian or skewed data can be an interesting enterprise.
We leave all these to future research.

Acknowledgements: The authors thank the JAKAR High-Performance
Cluster at the University of Texas at El Paso for providing computational
resources free of charge.

Abbreviations: The following abbreviations are used in this manuscript:

MOM prior Moment prior
IMOM prior
NLP
SNR
MSE
EEG

Inverse moment prior
Nonlocal prior
Signal-to-noise ratio
Mean squared error
Electroencephalogram

Appendix A

Appendix A.1. Proof of Result 1

From the wavelet coefficient model (2) and the mixture prior of the wavelet
coefficients in (3), using the Laplace approximation for the IMOM prior com-
ponent, we get the marginal distribution of the empirical wavelet coefficients,
bdlj, as

23

( bdlj − dlj)2
2σ2
l σ2(cid:17) ν
τ (2)
Γ(ν/2)

2

(cid:16)

π( bdlj|σ2, θ, r, ν)

Z

Z

=

=

π( bdlj|dlj, σ2, θ, r, ν)π(dlj|σ2, θ, r, ν)ddlj

(2πσ2)− 1

2 exp






−


γ(1)

l fMr

(cid:16)






l σ2(cid:17)−r− 1
τ (1)

2 d2r

lj exp


−





d2
lj
2τ (1)
l σ2



+

(cid:16)
1 − γ(1)

l

(cid:17)

γ(2)
l

|dlj|−ν−1 exp


−

τ (2)
l σ2
d2
lj


 +

(cid:16)

1 − γ(1)

l

(cid:17) (cid:16)

1 − γ(2)

l

(cid:17)


δ(dlj)



ddlj

= (2πσ2)− 1

2


γ(1)

l fMr

(cid:16)

+

1 − γ(1)

l

(cid:17)

γ(2)
l

(cid:16)

l σ2(cid:17)−r− 1
2 Z
τ (1)
l σ2(cid:17) ν
(cid:16)
τ (2)
Γ(ν/2)

Z

2

d2r
lj exp






−

1
2σ2 (( bdlj − dlj)2 +



)


ddlj

|dlj|−ν−1 exp






−

( bdlj − dlj)2
2σ2

−

τ (2)
l σ2
d2
lj

ddlj

d2
lj
τ (1)
l





(cid:16)

+

1 − γ(1)

l

(cid:17) (cid:16)

1 − γ(2)

l

(cid:17)

exp


−









bd2
lj
2σ2

= (2πσ2)− 1

2


γ(1)


l fMr

(cid:16)

l σ2(cid:17)−r− 1
τ (1)

2 exp






−

2σ2 (cid:16)

bd2
lj
1 + τ (1)

l

(cid:17)







2πσ2

τ (1)
l
1 + τ (1)

l



1/2 



σ2



r



τ (1)
l
1 + τ (1)

l

r
X

i=0

(2r)!
(2i)!(r − i)!2r−1









(cid:16)

+

1 − γ(1)

l

(cid:17) (cid:16)

1 − γ(2)

l



2i







bdlj

τ (1)
l
1+τ (1)
l
s
τ (1)
l
1+τ (1)
l

−

exp

σ

(cid:17)

+

(cid:16)

1 − γ(1)

l

(cid:17)

γ(2)
l

(cid:16)

2

l σ2(cid:17) ν
τ (2)
Γ(ν/2)

exp


−





bd2
lj
2σ2

√

2πσ∗h(d∗

lj( bdlj))









bd2
lj
2σ2

(using Laplace approximation)

1 + τ (1)

l

(cid:17)(cid:17)

+

(cid:16)

1 − γ(1)

l

(cid:17)

γ(2)
l

(cid:16)

2

l σ2(cid:17) ν
τ (2)
Γ(ν/2)

(cid:16)

(cid:17)−r

l

1 + τ (1)
bdlj; 0, σ2(cid:17) √

= γ(1)
l

(cid:16)

ϕ

where

M ∗
r

(cid:16)

bdlj, τ (1)
l

, σ2(cid:17)

(cid:16)

ϕ

bdlj; 0, σ2 (cid:16)
(cid:17) (cid:16)

2πσ∗h(d∗

lj( bdlj)) +

(cid:16)

1 − γ(1)

l

1 − γ(2)

l

M ⋆
r

(cid:16)

bdlj, τl, σ2(cid:17)

=

1
(2r − 1)!!

r
X

i=0

(2r)!
(2i)!(r − i)!2r−i

v
u
u
u
t






(cid:17)

(cid:16)

bdlj; 0, σ2(cid:17)

.

ϕ

τ (1)
l
1 + τ (1)

l

bdlj
σ

2i






,

h(dlj) = |dlj|−(ν+1) exp






−

1
2σ2

(cid:16)

d2
lj − 2dlj bdlj

(cid:17)

−

τ (1)
l σ2
d2
lj






,

d∗
lj( bdlj) is the global maxima of h(dlj), and σ2
log(h(dlj)).

∗ = −1/L′′

h(d∗

lj( bdlj)), with Lh(dlj) =

24

Appendix A.2. Proof of Result 2

(cid:16)

π

dlj|σ2, θ, r, ν, y

(cid:17)

(cid:16)

∝ π

y|dlj, σ2, θ, r, ν

(cid:17)

(cid:16)

π

dlj|σ2, θ, r, ν

(cid:17)

.

The proportionality constant is

C =

Z

(cid:16)

π

y|dlj, σ2, θ, r, ν

(cid:16)

(cid:17)

π

dlj|σ2, θ, r, ν

(cid:17)

ddlj

= (2πσ2)− 1

2


γ(1)


l

(cid:16)

1 + τ (1)

l

(cid:17)−r− 1

2 M ∗
r

(cid:16)

bdlj, τ (1)
l

, σ2(cid:17)

exp






−

bd2
lj
1 + τ (1)

l

(cid:17)






2σ2 (cid:16)

(cid:16)

+

1 − γ(1)

l

(cid:17)

γ(2)
l

(cid:16)

2

l σ2(cid:17) ν
τ (2)
Γ(ν/2)

(cid:16)

+

1 − γ(1)

l

(cid:17) (cid:16)

1 − γ(2)

l

(cid:17)

exp


−

exp


−

bd2
lj
2σ2

bd2
lj
2σ2







√





2πσ∗h(d∗

lj( bdlj))

= (2πσ2)− 1

2

(cid:16)

1 − γ(1)

l

(cid:17) (cid:16)

1 − γ(2)

l

(cid:17)

exp


−





h

bd2
lj
2σ2

Thus,

O(1)

lj + O(2)

lj + 1

i

,

(cid:16)

π
1
C

=

dlj|σ2, θ, r, ν, y

(cid:17)

(cid:16)

π

y|dlj, σ2, θ, r, ν

(cid:17)

(cid:16)

π

dlj|σ2, θ, r, ν

(cid:17)

1

=

=

(2πσ2)− 1

2

(cid:16)

1 − γ(1)

l

(cid:17) (cid:16)

1 − γ(2)

l

(cid:17)

exp

!

h

− bd2
lj
2σ2



(cid:16)


γ(1)

l fMr



l σ2(cid:17)−r− 1
τ (1)

2 exp




bd2
lj
1 + τ (1)

l

(cid:17)






−

2σ2 (cid:16)

O(1)

lj + O(2)
lj + 1


−


d2r
lj exp

1
2σ2 τ (1)
l
1+τ (1)
l

(2πσ2)− 1

2

i


dlj −



2

bdlj



τ (1)
l
1 + τ (1)

l





(cid:16)

(cid:16)

+

1 − γ(1)

l

(cid:17)

γ(2)
l

exp


−





bd2
lj
2σ2

√

2πσ∗h(d∗

lj( bdlj))ϕ

(cid:16)

dlj|d∗

lj( bdlj), σ2
∗

(cid:17)

2


l σ2(cid:17) ν
τ (2)
Γ(ν/2)



exp






bdljdlj
σ2 −

d2
lj
2σ2




δ(dlj)


O(1)
lj
lj + O(2)

O(1)

lj + 1

fMr
bdlj, τ (1)
l

(cid:16)

M ∗
r

, σ2(cid:17) d2r

lj exp






−

2σ2

1
τ (1)
l(cid:16)
1+τ (1)
l

(cid:17)


dlj −

2


bdlj



τ (1)
l
1 + τ (1)

l






25

 
+

O(2)
lj
lj + O(2)

O(1)

lj + 1

(cid:16)

ϕ

dlj|d∗

lj( bdlj), σ2
∗

(cid:17)

+

1
lj + O(2)

O(1)

lj + 1

exp






bdljdlj
σ2 −

d2
lj
2σ2






δ(dlj)

= p(1)
lj

fMr
bdlj, τ (1)
l

(cid:16)

M ∗
r

, σ2(cid:17) d2r

lj exp






−

2σ2

1
τ (1)
l(cid:16)
1+τ (1)
l

(cid:17)


dlj −

2


bdlj



τ (1)
l
1 + τ (1)

l






+ p(2)
lj ϕ

(cid:16)

dlj|d∗

lj( bdlj), σ2
∗

(cid:17)

(cid:16)

+

1 − p(1)

lj − p(2)

lj

(cid:17)

exp






bdljdlj
σ2 −

d2
lj
2σ2






δ(dlj),

from which Result 2 follows.

References

Abramovich, F. and Benjamini, Y. (1996). “Adaptive thresholding of wavelet
coefficients.” Computational Statistics & Data Analysis, 22, 4, 351–361.

Abramovich, F., Sapatinas, T., and Silverman, B. (1998). “Wavelet thresh-
olding via a Bayesian approach.” Journal of the Royal Statistical Society –
Series B , 60, 4, 725–749.

Afshari, M., Lak, F., and Gholizadeh, B. (2017). “A new Bayesian wavelet
thresholding estimator of nonparametric regression.” Journal of Applied
Statistics, 44, 4, 649–666.

Antoniadis, A., Bigot, J., and Sapatinas, T. (2001). “Wavelet Estimators in
Nonparametric Regression: A Comparative Simulation Study.” Journal of
Statistical Software, vol. 6, pp. 1–83.

Boubchir, L. and Boashash, B. (2013). “Wavelet denoising based on the
MAP estimation using the BKF Prior with application to images and EEG
signals.” IEEE Transactions on Signal Processing, 61, 8, 1880–1894.

Brandmeyer, T. and Delorme, A. (2018). “Reduced mind wandering in ex-
perienced meditators and associated EEG correlates.” Experimental brain
research, 236, 2519–2528.

Chang, S., Yu, B., and Vetterli, M. (2000). “Adaptive wavelet threshold-
ing for image denoising and compression.” IEEE Transactions on Image
Processing, 9, 9, 1532–1546.

Chipman, H. A., Kolaczyk, E. D., and McCulloch, R. E. (1997). “Adaptive
Bayesian Wavelet Shrinkage.” Journal of the American Statistical Associ-
ation, 92, 440, 1413–1421.

26

Clyde, M. and George, E. I. (2000). “Flexible empirical Bayes estimation
for wavelets.” Journal of the Royal Statistical Society – Series B , 62, 4,
681–698.

Clyde, M., Parmigiani, G., and Vidakovic, B. (1998). “Multiple shrinkage

and subset selection in wavelets.” Biometrika, 85, 2, 391–401.

Crouse, M., Nowak, R., and Baraniuk, R. (1998). “Wavelet-based statistical
signal processing using hidden Markov models.” IEEE Transactions on
Signal Processing, 46, 4, 886–902.

Cutillo, L., Jung, Y. Y., Ruggeri, F., and Vidakovic, B. (2008). “Larger pos-
terior mode wavelet thresholding and applications.” Journal of Statistical
Planning and Inference, 138, 12, 3758 – 3773.

Donoho, D. and Johnstone, J. (1994). “Ideal spatial adaptation by wavelet

shrinkage.” Biometrika, 81, 3, 425–455.

Donoho, D. L. and Johnstone, I. M. (1995). “Adapting to Unknown Smooth-
ness via Wavelet Shrinkage.” Journal of the American Statistical Associa-
tion, 90, 432, 1200–1224.

Donoho, D. L., Johnstone, I. M., Kerkyacharian, G., and Picard, D. (1995).
“Wavelet Shrinkage: Asymptopia?” Journal of the Royal Statistical Society
– Series B , 57, 2, pp. 301–369.

dos Santos Sousa, A. R. (2022). “Bayesian wavelet shrinkage with logistic
prior.” Communications in Statistics-Simulation and Computation, 51, 8,
4700–4714.

— (2024). “A Bayesian wavelet shrinkage rule under LINEX loss function.”

Research in Statistics, 2, 1, 2362926.

Figueiredo, M. and Nowak, R. (2001). “Wavelet-based image estimation:
an empirical Bayes approach using Jeffrey’s noninformative prior.” IEEE
Transactions on Image Processing, 10, 9, 1322–1331.

Johnson, V. and Rossell, D. (2010). “On the use of non-local prior densities
in Bayesian hypothesis tests.” Journal of the Royal Statistical Society –
Series B , 72, 2, 143–170.

— (2012). “Bayesian model selection in high-dimensional settings.” Journal

of the American Statistical Association, 107, 498, 649–660.

Johnstone, I. M. and Silverman, B. W. (2005). “Empirical Bayes Selection of

Wavelet Thresholds.” The Annals of Statistics, 33, 4, pp. 1700–1752.

27

Mallat, S. (2008). A Wavelet Tour of Signal Processing, Third Edition: The

Sparse Way. 3rd ed. USA: Academic Press, Inc.

Nason, G. (2024). wavethresh: Wavelets Statistics and Transforms. R package

version 4.7.3.

Nason, G. P. (1996). “Wavelet Shrinkage Using Cross-Validation.” Journal

of the Royal Statistical Society – Series B , 58, 2, pp. 463–479.

Portilla, J., Strela, V., Wainwright, M., and Simoncelli, E. (2003). “Image
denoising using scale mixtures of Gaussians in the wavelet domain.” IEEE
Transactions on Image Processing, 12, 11, 1338–1351.

R Core Team (2024). R: A Language and Environment for Statistical Com-

puting. R Foundation for Statistical Computing, Vienna, Austria.

Recommendation, I.-T. (2001).

“Perceptual evaluation of speech quality
(PESQ): An objective method for end-to-end speech quality assessment
of narrow-band telephone networks and speech codecs.” Rec. ITU-T P.
862 .

Rem´enyi, N. and Vidakovic, B. (2015). “Wavelet shrinkage with double
Weibull prior.” Communications in Statistics: Simulation and Compu-
tation, 44, 1, 88–104.

Rossell, D. and Telesca, D. (2017). “Nonlocal priors for high-dimensional
estimation.” Journal of the American Statistical Association, 112, 517,
254–265.

Sanyal, N. (2025). NLPwavelet: Bayesian Wavelet Analysis Using Non-Local

Priors. R package version 1.1.

Sanyal, N. and Ferreira, M. A. (2012). “Bayesian hierarchical multi-subject
multiscale analysis of functional {MRI} data.” NeuroImage, 63, 3, 1519 –
1531.

— (2017). “Bayesian wavelet analysis using nonlocal priors with an applica-

tion to FMRI analysis.” Sankhya B , 79, 2, 361–388.

Sanyal, N., Lo, M.-T., Kauppi, K., Djurovic, S., Andreassen, O. A., Johnson,
V. E., and Chen, C.-H. (2019). “GWASinlps: non-local prior based iterative
SNP selection tool for genome-wide association studies.” Bioinformatics,
35, 1, 1–11.

Sousa, A. R. d. S., Garcia, N. L., and Vidakovic, B. (2021). “Bayesian wavelet
shrinkage with beta priors.” Computational Statistics, 36, 2, 1341–1363.

28

Taal, C. H., Hendriks, R. C., Heusdens, R., and Jensen, J. (2010). “A short-
time objective intelligibility measure for time-frequency weighted noisy
speech.” In 2010 IEEE international conference on acoustics, speech and
signal processing, 4214–4217. IEEE.

Vidakovic, B. (1998). “Nonlinear Wavelet Shrinkage with Bayes Rules and
Bayes Factors.” Journal of the American Statistical Association, 93, 441,
pp. 173–179.

— (1999). Statistical Modeling by Wavelets. Wiley Series in Probability and

Statistics. Wiley.

Vidakovic, B. and Ruggeri, F. (2001). “BAMS Method: Theory and Simu-
lations.” Sankhy¯a: The Indian Journal of Statistics – Series B , 63, 2, pp.
234–249.

29

