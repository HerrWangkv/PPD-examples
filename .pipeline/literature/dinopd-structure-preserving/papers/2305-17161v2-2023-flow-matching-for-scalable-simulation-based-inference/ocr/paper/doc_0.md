3
2
0
2

t
c
O
7
2

]

G
L
.
s
c
[

2
v
1
6
1
7
1
.
5
0
3
2
:
v
i
X
r
a

Flow Matching for Scalable Simulation-Based
Inference

Maximilian Dax∗
Max Planck Institute for Intelligent Systems
Tübingen, Germany
maximilian.dax@tuebingen.mpg.de

Jonas Wildberger∗
Max Planck Institute for Intelligent Systems
Tübingen, Germany
wildberger.jonas@tuebingen.mpg.de

Simon Buchholz∗
Max Planck Institute for Intelligent Systems
Tübingen, Germany
sbuchholz@tue.mpg.de

Stephen R. Green
University of Nottingham
Nottingham, United Kingdom

Jakob H. Macke
Max Planck Institute for Intelligent Systems &
Machine Learning in Science,
University of Tübingen
Tübingen, Germany

Bernhard Schölkopf
Max Planck Institute for Intelligent Systems
Tübingen, Germany

Abstract

Neural posterior estimation methods based on discrete normalizing flows have
become established tools for simulation-based inference (SBI), but scaling them
to high-dimensional problems can be challenging. Building on recent advances in
generative modeling, we here present flow matching posterior estimation (FMPE),
a technique for SBI using continuous normalizing flows. Like diffusion models,
and in contrast to discrete flows, flow matching allows for unconstrained architec-
tures, providing enhanced flexibility for complex data modalities. Flow matching,
therefore, enables exact density evaluation, fast training, and seamless scalability
to large architectures—making it ideal for SBI. We show that FMPE achieves com-
petitive performance on an established SBI benchmark, and then demonstrate its
improved scalability on a challenging scientific problem: for gravitational-wave in-
ference, FMPE outperforms methods based on comparable discrete flows, reducing
training time by 30% with substantially improved accuracy. Our work underscores
the potential of FMPE to enhance performance in challenging inference scenarios,
thereby paving the way for more advanced applications to scientific problems.

1

Introduction

The ability to readily represent Bayesian posteriors of arbitrary complexity using neural networks
would herald a revolution in scientific data analysis. Such networks could be trained using simulated
data and used for amortized inference across observations—bringing tractable inference and speed to
a myriad of scientific models. Thanks to innovative architectures such as normalizing flows [1, 2],
approaches to neural simulation-based inference (SBI) [3] have seen remarkable progress in recent
years. Here, we show that modern approaches to deep generative modeling (particularly flow
matching) deliver substantial improvements in simplicity, flexibility and scaling when adapted to SBI.

∗Equal contribution

37th Conference on Neural Information Processing Systems (NeurIPS 2023).

 
 
 
 
 
 
Figure 1: Comparison of network architectures (left) and flow trajectories (right). Discrete flows
(NPE, top) require a specialized architecture for the density estimator. Continuous flows (FMPE,
bottom) are based on a vector field parametrized with an unconstrained architecture. FMPE uses
this additional flexibility to put an enhanced emphasis on the conditioning data x, which in the SBI
context is typically high dimensional and in a complex domain. Further, the optimal transport path
produces simple flow trajectories from the base distribution (inset) to the target.

The Bayesian approach to data analysis is to compare observations to models via the posterior
x). This gives our degree of belief that model parameters θ gave rise to an observation
distribution p(θ
|
x, and is proportional to the model likelihood p(x
θ) times the prior p(θ). One is typically interested
|
in representing the posterior in terms of a collection of samples, however obtaining these through
standard likelihood-based algorithms can be challenging for intractable or expensive likelihoods. In
such cases, SBI offers an alternative based instead on data simulations x
θ). Combined with
deep generative modeling, SBI becomes a powerful paradigm for scientific inference [3]. Neural
posterior estimation (NPE) [4–6], for instance, trains a conditional density estimator q(θ
x) to
|
approximate the posterior, allowing for rapid sampling and density estimation for any x consistent
with the training distribution.

p(x

∼

|

|

The NPE density estimator q(θ
x) is commonly taken to be a (discrete) normalizing flow [1, 2], an
approach that has brought state-of-the-art performance in challenging problems such as gravitational-
wave inference [7]. Naturally, performance hinges on the expressiveness of q(θ
x). Normalizing
|
flows transform noise to samples through a discrete sequence of basic transforms. These have been
carefully engineered to be invertible with simple Jacobian determinant, enabling efficient maximum
likelihood training, while at the same time producing expressive q(θ
x). Although many such discrete
|
flows are universal density approximators [2], in practice, they can be challenging to scale to very
large networks, which are needed for big-data experiments.

Recent studies [8, 9] propose neural posterior score estimation (NPSE), a rather different approach
that models the posterior distribution with score-matching (or diffusion) networks. These techniques
were originally developed for generative modeling [10–12], achieving state-of-the-art results in many
domains, including image generation [13, 14]. Like discrete normalizing flows, diffusion models
transform noise into samples, but with trajectories parametrized by a continuous “time” parameter
t. The trajectories solve a stochastic differential equation [15] (SDE) defined in terms of a vector
field vt, which is trained to match the score of the intermediate distributions pt. NPSE has several
advantages compared to NPE, including the ability to combine multiple observations at inference
time [9] and, importantly, the freedom to use unconstrained network architectures.

We here propose to use flow matching, another recent technique for generative modeling, for Bayesian
inference, an approach we refer to as flow-matching posterior estimation (FMPE). Flow matching is
also based on a vector field vt and thereby also admits flexible network architectures (Fig. 1). For flow

2

<latexit sha1_base64="8BuNO7isAM/AKew+VVR/qqMdSIc=">AAACAXicbVDJSgNBEO2JW4xb1IvgpTEI8RJmxO0kAS8eI5gFkhB6OpWkSc9id40kjOPFX/HiQRGv/oU3/8bOctDEBwWP96qoqueGUmi07W8rtbC4tLySXs2srW9sbmW3dyo6iBSHMg9koGou0yCFD2UUKKEWKmCeK6Hq9q9GfvUelBaBf4vDEJoe6/qiIzhDI7Wye3f5BvYAGX2gg1bcQBhgHCTJUSubswv2GHSeOFOSI1OUWtmvRjvgkQc+csm0rjt2iM2YKRRcQpJpRBpCxvusC3VDfeaBbsbjDxJ6aJQ27QTKlI90rP6eiJmn9dBzTafHsKdnvZH4n1ePsHPRjIUfRgg+nyzqRJJiQEdx0LZQwFEODWFcCXMr5T2mGEcTWsaE4My+PE8qxwXnrHB6c5IrXk7jSJN9ckDyxCHnpEiuSYmUCSeP5Jm8kjfryXqx3q2PSWvKms7skj+wPn8AdUiW6A==</latexit>q(✓|xo)<latexit sha1_base64="DLwt0BlAmcsm/EAanbxCyQKHBcU=">AAAB+nicbVDLSsNAFJ34rPWV6tLNYBEqSEnE10oKblxWsA9oYplMJ+3QySTM3Kgl9lPcuFDErV/izr9x+lho64ELh3Pu5d57gkRwDY7zbS0sLi2vrObW8usbm1vbdmGnruNUUVajsYhVMyCaCS5ZDTgI1kwUI1EgWCPoX438xj1TmsfyFgYJ8yPSlTzklICR2nbBSzS/4yUPegzIEX48bNtFp+yMgeeJOyVFNEW1bX95nZimEZNABdG65ToJ+BlRwKlgw7yXapYQ2idd1jJUkohpPxufPsQHRungMFamJOCx+nsiI5HWgygwnRGBnp71RuJ/XiuF8MLPuExSYJJOFoWpwBDjUQ64wxWjIAaGEKq4uRXTHlGEgkkrb0JwZ1+eJ/XjsntWPr05KVYup3Hk0B7aRyXkonNUQdeoimqIogf0jF7Rm/VkvVjv1sekdcGazuyiP7A+fwA2lJNR</latexit> i(✓,x)<latexit sha1_base64="hZNTGLrhsrF4oeJ6l5cdp3RQYJM=">AAAB+nicbVDLSsNAFJ34rPWV6tLNYBEqSEnE10oKblxWsA9oYplMJ+3QySTM3Kgl9lPcuFDErV/izr9x+lho64ELh3Pu5d57gkRwDY7zbS0sLi2vrObW8usbm1vbdmGnruNUUVajsYhVMyCaCS5ZDTgI1kwUI1EgWCPoX438xj1TmsfyFgYJ8yPSlTzklICR2nbBSzS/kyUPegzIEX48bNtFp+yMgeeJOyVFNEW1bX95nZimEZNABdG65ToJ+BlRwKlgw7yXapYQ2idd1jJUkohpPxufPsQHRungMFamJOCx+nsiI5HWgygwnRGBnp71RuJ/XiuF8MLPuExSYJJOFoWpwBDjUQ64wxWjIAaGEKq4uRXTHlGEgkkrb0JwZ1+eJ/XjsntWPr05KVYup3Hk0B7aRyXkonNUQdeoimqIogf0jF7Rm/VkvVjv1sekdcGazuyiP7A+fwA+X5NW</latexit> n(✓,x)<latexit sha1_base64="Dpdu/hs0ZHlPMJZnMpQHnoZv6GQ=">AAAB+nicbVDLSsNAFJ34rPWV6tLNYBEqSEnE10oKblxWsA9oYplMJ+3QySTM3Kgl9lPcuFDErV/izr9x+lho64ELh3Pu5d57gkRwDY7zbS0sLi2vrObW8usbm1vbdmGnruNUUVajsYhVMyCaCS5ZDTgI1kwUI1EgWCPoX438xj1TmsfyFgYJ8yPSlTzklICR2nbBSzS/c0se9BiQI/x42LaLTtkZA88Td0qKaIpq2/7yOjFNIyaBCqJ1y3US8DOigFPBhnkv1SwhtE+6rGWoJBHTfjY+fYgPjNLBYaxMScBj9fdERiKtB1FgOiMCPT3rjcT/vFYK4YWfcZmkwCSdLApTgSHGoxxwhytGQQwMIVRxcyumPaIIBZNW3oTgzr48T+rHZfesfHpzUqxcTuPIoT20j0rIReeogq5RFdUQRQ/oGb2iN+vJerHerY9J64I1ndlFf2B9/gDfPZMZ</latexit> 1(✓,x)<latexit sha1_base64="ieG5VY1tQ2vm/Ks9uOyONqK4v6I=">AAACD3icbVDLSsNAFJ3UV62vqEs3g0WpICURXyspuNGNVLAPaEKYTKft0MmDmRuhhPyBG3/FjQtF3Lp15984abvQ6oELh3Pu5d57/FhwBZb1ZRTm5hcWl4rLpZXVtfUNc3OrqaJEUtagkYhk2yeKCR6yBnAQrB1LRgJfsJY/vMz91j2TikfhHYxi5gakH/IepwS05Jn7DgwYEOwoHmAnIDCgRKQ3mZdah/g6q0zsA88sW1VrDPyX2FNSRlPUPfPT6UY0CVgIVBClOrYVg5sSCZwKlpWcRLGY0CHps46mIQmYctPxPxne00oX9yKpKwQ8Vn9OpCRQahT4ujO/WM16ufif10mgd+6mPIwTYCGdLOolAkOE83Bwl0tGQYw0IVRyfSumAyIJBR1hSYdgz778lzSPqvZp9eT2uFy7mMZRRDtoF1WQjc5QDV2hOmogih7QE3pBr8aj8Wy8Ge+T1oIxndlGv2B8fAPO3pvf</latexit>✓⇠N0,I(✓)<latexit sha1_base64="dr4YELvgPteXpcKmc1H8erO9Ftg=">AAAB83icbVDJSgNBEO2JW4xb1KOXxiDES5gRt5MEvHiMYBbIhNDTqUma9Cx214hhzG948aCIV3/Gm39jJ5mDJj4oeLxXRVU9L5ZCo21/W7ml5ZXVtfx6YWNza3unuLvX0FGiONR5JCPV8pgGKUKoo0AJrVgBCzwJTW94PfGbD6C0iMI7HMXQCVg/FL7gDI3k3pddHAAy+vR43C2W7Io9BV0kTkZKJEOtW/xyexFPAgiRS6Z127Fj7KRMoeASxgU30RAzPmR9aBsasgB0J53ePKZHRulRP1KmQqRT9fdEygKtR4FnOgOGAz3vTcT/vHaC/mUnFWGcIIR8tshPJMWITgKgPaGAoxwZwrgS5lbKB0wxjiamggnBmX95kTROKs555ez2tFS9yuLIkwNySMrEIRekSm5IjdQJJzF5Jq/kzUqsF+vd+pi15qxsZp/8gfX5A15MkUA=</latexit>q(✓|x)<latexit sha1_base64="7iRIRqjbhZvJmXxix4/SldHgJ5s=">AAAB6HicbVDLSgNBEOyNrxhfUY9eBoPgKeyKr5MEvHhMwDwgWcLspJOMmZ1dZmbFsOQLvHhQxKuf5M2/cZLsQRMLGoqqbrq7glhwbVz328mtrK6tb+Q3C1vbO7t7xf2Dho4SxbDOIhGpVkA1Ci6xbrgR2IoV0jAQ2AxGt1O/+YhK80jem3GMfkgHkvc5o8ZKtaduseSW3RnIMvEyUoIM1W7xq9OLWBKiNExQrdueGxs/pcpwJnBS6CQaY8pGdIBtSyUNUfvp7NAJObFKj/QjZUsaMlN/T6Q01HocBrYzpGaoF72p+J/XTkz/2k+5jBODks0X9RNBTESmX5MeV8iMGFtCmeL2VsKGVFFmbDYFG4K3+PIyaZyVvcvyRe28VLnJ4sjDERzDKXhwBRW4gyrUgQHCM7zCm/PgvDjvzse8NedkM4fwB87nD+hFjQI=</latexit>x<latexit sha1_base64="8BuNO7isAM/AKew+VVR/qqMdSIc=">AAACAXicbVDJSgNBEO2JW4xb1IvgpTEI8RJmxO0kAS8eI5gFkhB6OpWkSc9id40kjOPFX/HiQRGv/oU3/8bOctDEBwWP96qoqueGUmi07W8rtbC4tLySXs2srW9sbmW3dyo6iBSHMg9koGou0yCFD2UUKKEWKmCeK6Hq9q9GfvUelBaBf4vDEJoe6/qiIzhDI7Wye3f5BvYAGX2gg1bcQBhgHCTJUSubswv2GHSeOFOSI1OUWtmvRjvgkQc+csm0rjt2iM2YKRRcQpJpRBpCxvusC3VDfeaBbsbjDxJ6aJQ27QTKlI90rP6eiJmn9dBzTafHsKdnvZH4n1ePsHPRjIUfRgg+nyzqRJJiQEdx0LZQwFEODWFcCXMr5T2mGEcTWsaE4My+PE8qxwXnrHB6c5IrXk7jSJN9ckDyxCHnpEiuSYmUCSeP5Jm8kjfryXqx3q2PSWvKms7skj+wPn8AdUiW6A==</latexit>q(✓|xo)<latexit sha1_base64="7iRIRqjbhZvJmXxix4/SldHgJ5s=">AAAB6HicbVDLSgNBEOyNrxhfUY9eBoPgKeyKr5MEvHhMwDwgWcLspJOMmZ1dZmbFsOQLvHhQxKuf5M2/cZLsQRMLGoqqbrq7glhwbVz328mtrK6tb+Q3C1vbO7t7xf2Dho4SxbDOIhGpVkA1Ci6xbrgR2IoV0jAQ2AxGt1O/+YhK80jem3GMfkgHkvc5o8ZKtaduseSW3RnIMvEyUoIM1W7xq9OLWBKiNExQrdueGxs/pcpwJnBS6CQaY8pGdIBtSyUNUfvp7NAJObFKj/QjZUsaMlN/T6Q01HocBrYzpGaoF72p+J/XTkz/2k+5jBODks0X9RNBTESmX5MeV8iMGFtCmeL2VsKGVFFmbDYFG4K3+PIyaZyVvcvyRe28VLnJ4sjDERzDKXhwBRW4gyrUgQHCM7zCm/PgvDjvzse8NedkM4fwB87nD+hFjQI=</latexit>x<latexit sha1_base64="w/TiuduNQ/PzIVdJYbhVTXgTFFI=">AAAB9HicbVDJSgNBEO2JW4xb1KOXxiBEkDAjbicJePEYwSyQDKGnU5M06VnsrgmEId/hxYMiXv0Yb/6NnWQOmvig4PFeFVX1vFgKjbb9beVWVtfWN/Kbha3tnd294v5BQ0eJ4lDnkYxUy2MapAihjgIltGIFLPAkNL3h3dRvjkBpEYWPOI7BDVg/FL7gDI3kljs4AGRdPKN42i2W7Io9A10mTkZKJEOtW/zq9CKeBBAil0zrtmPH6KZMoeASJoVOoiFmfMj60DY0ZAFoN50dPaEnRulRP1KmQqQz9fdEygKtx4FnOgOGA73oTcX/vHaC/o2bijBOEEI+X+QnkmJEpwnQnlDAUY4NYVwJcyvlA6YYR5NTwYTgLL68TBrnFeeqcvlwUareZnHkyRE5JmXikGtSJfekRuqEkyfyTF7JmzWyXqx362PemrOymUPyB9bnD5vqkVg=</latexit>(✓t,t)<latexit sha1_base64="7iptkgTBLDJPRBjG4Ba9pZ2COWU=">AAAB+nicbVDLSsNAFJ34rPWV6tLNYBEqSEnER5cFNy4r2Ae0IUym03boZBJmbqol9lPcuFDErV/izr9x2mahrQcuHM65l3vvCWLBNTjOt7Wyura+sZnbym/v7O7t24WDho4SRVmdRiJSrYBoJrhkdeAgWCtWjISBYM1geDP1myOmNI/kPYxj5oWkL3mPUwJG8u3CyE/h7HFS6sCAAfHh1LeLTtmZAS8TNyNFlKHm21+dbkSTkEmggmjddp0YvJQo4FSwSb6TaBYTOiR91jZUkpBpL52dPsEnRuniXqRMScAz9fdESkKtx2FgOkMCA73oTcX/vHYCvYqXchknwCSdL+olAkOEpzngLleMghgbQqji5lZMB0QRCiatvAnBXXx5mTTOy+5V+fLuolitZHHk0BE6RiXkomtURbeohuqIogf0jF7Rm/VkvVjv1se8dcXKZg7RH1ifPwCUk9A=</latexit>vt,x(✓t)<latexit sha1_base64="pmahqZ+AafOv/2HHqlPt0FnArQI=">AAAB63icbVDLSgNBEOyNrxhfUY9eBoPgKewGfBw8BETxIkQwD0iWMDuZTYbMzC4zs0JY8gtePCji1R/y5t84m+xBEwsaiqpuuruCmDNtXPfbKaysrq1vFDdLW9s7u3vl/YOWjhJFaJNEPFKdAGvKmaRNwwynnVhRLAJO28H4OvPbT1RpFslHM4mpL/BQspARbDLp9r5x0y9X3Ko7A1omXk4qkKPRL3/1BhFJBJWGcKx113Nj46dYGUY4nZZ6iaYxJmM8pF1LJRZU++ns1ik6scoAhZGyJQ2aqb8nUiy0nojAdgpsRnrRy8T/vG5iwks/ZTJODJVkvihMODIRyh5HA6YoMXxiCSaK2VsRGWGFibHxlGwI3uLLy6RVq3rn1bOHWqV+lcdRhCM4hlPw4ALqcAcNaAKBETzDK7w5wnlx3p2PeWvByWcO4Q+czx9iV43M</latexit>FMPE<latexit sha1_base64="/Efvfsn/3Tvgx98hvpg8zVYsZ4s=">AAAB6nicbVDLSgNBEOyNrxhfUY9eBoPgKewGfBw8BETwJBHNA5IlzE46yZDZ2WVmVghLPsGLB0W8+kXe/BsnyR40saChqOqmuyuIBdfGdb+d3Mrq2vpGfrOwtb2zu1fcP2joKFEM6ywSkWoFVKPgEuuGG4GtWCENA4HNYHQ99ZtPqDSP5KMZx+iHdCB5nzNqrPRwV7vpFktu2Z2BLBMvIyXIUOsWvzq9iCUhSsME1brtubHxU6oMZwInhU6iMaZsRAfYtlTSELWfzk6dkBOr9Eg/UrakITP190RKQ63HYWA7Q2qGetGbiv957cT0L/2UyzgxKNl8UT8RxERk+jfpcYXMiLEllClubyVsSBVlxqZTsCF4iy8vk0al7J2Xz+4rpepVFkcejuAYTsGDC6jCLdSgDgwG8Ayv8OYI58V5dz7mrTknmzmEP3A+fwDT/I19</latexit>NPEmatching, however, vt directly defines the velocity field of sample trajectories, which solve ordinary
differential equations (ODEs) and are deterministic. As a consequence, flow matching allows for
additional freedom in designing non-diffusion paths such as optimal transport, and provides direct
access to the density [16]. These differences are summarized in Tab. 1.

NPE

NPSE FMPE (Ours)

Tractable posterior density
Unconstrained network architecture
Network passes for sampling

Yes
No

No
Yes
Single Many

Yes
Yes
Many

Table 1: Comparison of posterior-estimation methods.

Our contributions are as follows:

• We adapt flow-matching to Bayesian inference, proposing FMPE. In general, the modeling
requirements of SBI are different from generative modeling. For the latter, sample quality
is critical, i.e., that samples lie in the support of a complex distribution (e.g., images). In
contrast, for SBI, p(θ
x) is typically less complex for fixed x, but x itself can be complex
|
and high-dimensional. We therefore consider pyramid-like architectures from x to vt, with
gated linear units to incorporate (θ, t) dependence, rather than the typical U-Net used for
images (Fig. 1). We also propose an alternative t-weighting in the loss, which improves
performance in many tasks.

• Under certain regularity assumptions, we prove an upper bound on the KL divergence
between the model and target posterior. This implies that estimated posteriors are mass-
covering, i.e., that their support includes all θ consistent with observed x, which is highly
desirable for scientific applications [17].

• We perform a number of experiments to investigate the performance of FMPE.2 Our two-
pronged approach, which involves a set of benchmark tests and a real-world problem,
is designed to probe complementary aspects of the method, covering breadth and depth
of applications. First, on an established suite of SBI benchmarks, we show that FMPE
performs comparably—or better—than NPE across most tasks, and in particular exhibits
mass-covering posteriors in all cases (Sec. 4). We then push the performance limits of FMPE
on a challenging real-world problem by turning to gravitational-wave inference (Sec. 5).
We show that FMPE substantially outperforms an NPE baseline in terms of training time,
posterior accuracy, and scaling to larger networks.

2 Preliminaries

Normalizing flows. A normalizing flow [1, 2] defines a probability distribution q(θ
rameters θ
q0(θ),

Rn in terms of an invertible mapping ψx : Rn

x) over pa-
|
Rn from a simple base distribution

→

∈

q(θ

|

x) = (ψx)

∗

1
q0(θ) = q0(ψ−

x (θ)) det

(cid:12)
(cid:12)
(cid:12)
(cid:12)

1
∂ψ−
x (θ)
∂θ

(cid:12)
(cid:12)
(cid:12)
(cid:12)

,

(1)

where (
)
denotes the pushforward operator, and for generality we have conditioned on additional
·
Rm. Unless otherwise specified, a normalizing flow refers to a discrete flow, where ψx
∗
context x
∈
is given by a composition of simpler mappings with triangular Jacobians, interspersed with shuffling
of the θ. This construction results in expressive q(θ

x) and also efficient density evaluation [2].
|

Continuous normalizing flows. A continuous flow [18] also maps from base to target distribution,
but is parametrized by a continuous “time” t
x).
|
For each t, the flow is defined by a vector field vt,x on the sample space.3 This corresponds to the
velocity of the sample trajectories,

x) = q0(θ) and q1(θ

[0, 1], where q0(θ

x) = q(θ
|

∈

|

d
dt

ψt,x(θ) = vt,x(ψt,x(θ)),

ψ0,x(θ) = θ.

(2)

2Code available here.
3In the SBI literature, this is also commonly referred to as “parameter space”.

3

We obtain the trajectories θt ≡

ψt,x(θ) by integrating this ODE. The final density is given by
(cid:90) 1

(cid:18)

(cid:19)

q(θ

x) = (ψ1,x)
|

∗

q0(θ) = q0(θ) exp

−

0

div vt,x(θt) dt

,

(3)

which is obtained by solving the transport equation ∂tqt + div(qtvt,x) = 0.

The advantage of the continuous flow is that vt,x(θ) can be simply specified by a neural network
taking Rn+m+1
Rn, in which case (2) is referred to as a neural ODE [18]. Since the density
is tractable via (3), it is in principle possible to train the flow by maximizing the (log-)likelihood.
However, this is often not feasible in practice, since both sampling and density estimation require
many network passes to numerically solve the ODE (2).

→

Flow matching. An alternative training objective for continuous normalizing flows is provided
by flow matching [16]. This directly regresses vt,x on a vector field ut,x that generates a target
probability path pt,x. It has the advantage that training does not require integration of ODEs, however
it is not immediately clear how to choose (ut,x, pt,x). The key insight of [16] is that, if the path is
chosen on a sample-conditional basis,4 then the training objective becomes extremely simple. Indeed,
given a sample-conditional probability path pt(θ
θ1), we
|
specify the sample-conditional flow matching loss as

θ1) and a corresponding vector field ut(θ
|

LSCFM = E

t

[0,1], x

x), θt∼
|
Remarkably, minimization of this loss is equivalent to regressing vt,x(θ) on the marginal vector
x) [16]. Note that in this expression, the x-dependence of vt,x(θ) is
field ut,x(θ) that generates pt(θ
|
picked up via the expectation value, with the sample-conditional vector field independent of x.

p(x), θ1∼

pt(θt|

θ1) ∥

∼U

−

p(θ

∼

vt,x(θt)

ut(θt|

θ1)
∥

2 .

(4)

There exists considerable freedom in choosing a sample-conditional path. Ref. [16] introduces the
family of Gaussian paths

(5)
where the time-dependent means µt(θ1) and standard deviations σt(θ1) can be freely specified
(subject to boundary conditions5). For our experiments, we focus on the optimal transport paths
defined by µt(θ1) = tθ1 and σt(θ1) = 1
σmin)t (also introduced in [16]). The sample-
(1
conditional vector field then has the simple form

µt(θ1), σt(θ1)2In),
|

θ1) =

pt(θ

(θ

N

−

−

|

.

ut(θ

−
−

(1
(1

θ1) =
|

σmin)θ
σmin)t

x) (usually a normalizing flow) to the posterior p(θ
|

θ1 −
1
−
Neural posterior estimation (NPE). NPE is an SBI method that directly fits a density estimator
x) [4–6]. NPE trains with the maximum
q(θ
|
x), using Bayes’ theorem to simplify the expecta-
θ) log q(θ
likelihood objective
|
tion value with E
|
θ). During training,
LNPE is estimated based on an empirical
p(x)p(θ
|
distribution consisting of samples (θ, x)
θ). Once trained, NPE can perform inference
p(θ)p(x
|
x), thereby amortizing the computational cost of simulation
for every new observation using q(θ
and training across all observations. NPE further provides exact density evaluations of q(θ
x). Both
of these properties are crucial for the physics application in section 5, so we aim to retain these
properties with FMPE.

E
LNPE =
−
E
x) →
|

p(θ)p(x

p(θ)p(x

(6)

∼

|

|

Related work

Flow matching [16] has been developed as a technique for generative modeling, and similar techniques
are discussed in [19–21] and extended in [22, 23]. Flow matching encompasses the deterministic ODE
version of diffusion models [10–12] as a special instance. Although to our knowledge flow matching
has not previously been applied to Bayesian inference, score-matching diffusion models have been
proposed for SBI in [8, 9] with impressive results. These studies, however, use stochastic formulations
via SDEs [15] or Langevin steps and are therefore not directly applicable when evaluations of the
posterior density are desired (see Tab. 1). It should be noted that score modeling can also be used

4We refer to conditioning on θ1 as sample-conditioning to distinguish from conditioning on x.
5The sample-conditional probability path should be chosen to be concentrated around θ1 at t = 1 (within a

small region of size σmin) and to be the base distribution at t = 0.

4

to parameterize continuous normalizing flows via an ODE. Extension of [8, 9] to the deterministic
formulation could thereby be seen as a special case of flow matching. Many of our analyses and the
practical guidance provided in Section 3 therefore also apply to score matching.

We here focus on comparisons of FMPE against NPE [4–6], as it best matches the requirements of
the application in section 5. Other SBI methods include approximate Bayesian computation [24–28],
neural likelihood estimation [29–32] and neural ratio estimation [33–39]. Many of these approaches
have sequential versions, where the estimator networks are specifically tuned to a specific observation
xo. FMPE has a tractable density, so it is straightforward to apply the sequential NPE [4–6] approaches
to FMPE. In this case, inference is no longer amortized, so we leave this extension to future work.

3 Flow matching posterior estimation

To apply flow matching to SBI we use Bayes’ theorem to make the usual replacement E
E
p(θ)p(x
FMPE loss

x) →
|
θ) in the loss function (4), eliminating the intractable expectation values. This gives the
|

p(x)p(θ

LFMPE = E

t

p(t), θ1∼

∼

p(θ),x

p(x
|

∼

θ1), θt∼

pt(θt|

θ1) ∥

vt,x(θt)

ut(θt|

θ1)

−

2 ,
∥

(7)

which we minimize using empirical risk minimization over samples (θ, x)
θ). In other
|
words, training data is generated by sampling θ from the prior, and then simulating data x correspond-
ing to θ. This is in close analogy to NPE training, but replaces the log likelihood maximization with
the sample-conditional flow matching objective. Note that in this expression we also sample t
p(t),
[0, 1] (see Sec. 3.3), which generalizes the uniform distribution in (4). This provides additional
t
freedom to improve learning in our experiments.

p(θ)p(x

∼

∼

∈

3.1 Probability mass coverage

x) can achieve excellent results in approxi-
As we show in our examples, trained FMPE models q(θ
|
x). However, it is not generally possible to achieve exact agreement
mating the true posterior p(θ
|
due to limitations in training budget and network capacity. It is therefore important to understand
how inaccuracies manifest. Whereas sample quality is the main criterion for generative modeling, for
scientific applications one is often interested in the overall shape of the distribution. In particular, an
x) is mass-covering, i.e., whether it contains the full support of
important question is whether q(θ
|
x). This minimizes the risk to falsely rule out possible explanations of the data. It also allows us
p(θ
to use importance sampling if the likelihood p(x
θ) of the forward model can be evaluated, which
can be used for precise estimation of the posterior [40, 41].

|

|

||

q(θ

x)
|

Consider first
the mass-covering property for NPE.
NPE directly minimizes the forward KL divergence
DKL(p(θ
x)), and thereby provides probability-
|
mass covering results. Therefore, even if NPE is not accu-
x) should cover the entire
rately trained, the estimate q(θ
|
support of the posterior p(θ
x) and the failure to do so can
|
be observed in the validation loss. As an illustration in an
unconditional setting, we observe that a unimodal Gaus-
sian q fitted to a bimodal target distribution p captures both
modes when using the forward KL divergence DKL(p
q),
but only a single mode when using the backwards direction
DKL(q

p) (Fig. 2).

||

For FMPE, we can fit a Gaussian flow-matching model
(ˆµ, ˆσ2) to the same bimodal target, in this case,
q(θ) =
parametrizing the vector field as

N

||

Figure 2: A Gaussian (blue) fitted to a
bimodal distribution (gray) with various
objectives.

vt(θ) =

(σ2

t + (tˆσ)2
−
(σ2
t + (tˆσ)2)
t

σt)θt + tˆµ

·

σt

·

(8)

(see Appendix A), we also obtain a mass-covering distribution when fitting the learnable parameters
(ˆµ, ˆσ) via (4). This provides some indication that the flow matching objective induces mass-covering
behavior, and leads us to investigate the more general question of whether the mean squared error

5

argminqDKL(q||p)argminqDKL(p||q)argminqLFMbetween vector fields ut and vt bounds the forward KL divergence. Indeed, the former agrees up to
constant with the sample-conditional loss (4) (see Sec. 2).

q0, pt = (ϕt)

q0. The
We denote the flows of ut, vt, by ϕt, ψt, respectively, and we set qt = (ψt)
∗
q1) by MSEp(u, v)α for some positive power
precise question then is whether we can bound DKL(p1||
α. It was already observed in [42] that this is not true in general, and we provide a simple example to
that effect in Lemma 1 in Appendix B. Indeed, it was found in [42] that to bound the forward KL
divergence we also need to control the Fisher divergence, (cid:82) pt(dθ)(
Here we show instead that a bound can be obtained under sufficiently strong regularity assumptions
on p0, ut, and vt. The following statement is slightly informal, and we refer to the supplement for the
complete version.
Theorem 1. Let p0 = q0 and assume ut and vt are two vector fields whose flows satisfy p1 = (ϕ1)
p0
∗
and q1 = (ψ1)
)
c(1 +
θ
|
|
∗
and ut and vt have bounded second derivatives. Then there is a constant C > 0 such that (for
MSEp(u, v) < 1))

q0. Assume that p0 is square integrable and satisfies

qt(θ))2.

ln p0(θ)

ln pt(θ)

− ∇

| ≤

|∇

∇

∗

DKL(p1||

q1)

≤

C MSEp(u, v)

1
2 .

(9)

The proof of this result can be found in appendix B. While the regularity assumptions are not
guaranteed to hold in practice when vt is parametrized by a neural net, the theorem nevertheless
gives some indication that the flow-matching objective encourages mass coverage. In Section 4
and 5, this is complemented with extensive empirical evidence that flow matching indeed provides
mass-covering estimates.

We remark that it was shown in [43] that the KL divergence of SDE solutions can be bounded by the
MSE of the estimated score function. Thus, the smoothing effect of the noise ensures mass coverage,
an aspect that was further studied using the Fokker-Planck equation in [42]. For flow matching,
imposing the regularity assumption plays a similar role.

3.2 Network architecture

Generative diffusion or flow matching models typically operate on complicated and high dimensional
data in the θ space (e.g., images with millions of pixels). One typically uses U-Net [44] like
architectures, as they provide a natural mapping from θ to a vector field v(θ) of the same dimension.
The dependence on t and an (optional) conditioning vector x is then added on top of this architecture.

For SBI, the data x is often associated with a complicated domain, such as image or time series data,
whereas parameters θ are typically low dimensional. In this context, it is therefore useful to build the
architecture starting as a mapping from x to v(x) and then add conditioning on θ and t. In practice,
one can therefore use any established feature extraction architecture for data in the domain of x, and
adjust the dimension of the feature vector to n = dim(θ). In our experiments, we found that the
(t, θ)-conditioning is best achieved using gated linear units [45] to the hidden layers of the network
(see also Fig. 1); these are also commonly used for conditioning discrete flows on x.

3.3 Re-scaling the time prior

U

[0, 1] in (4) distributes the training capacity uniformly across t. We observed that this
The time prior
is not always optimal in practice, as the complexity of the vector field may depend on t. For FMPE
we therefore sample t in (7) from a power-law distribution pα(t)
[0, 1], introducing
an additional hyperparameter α. This includes the uniform distribution for α = 0, but for α > 0,
assigns greater importance to the vector field for larger values of t. We empirically found this to
improve learning for distributions with sharp bounds (e.g., Two Moons in Section 4).

t1/(1+α), t

∝

∈

4 SBI benchmark

We now evaluate FMPE on ten tasks included in the benchmark presented in [46], ranging from
simple Gaussian toy models to more challenging SBI problems from epidemiology and ecology, with
[2, 100]). For
varying dimensions for parameters (dim(θ)
each task, we train three separate FMPE models with simulation budgets N
. We
}

[2, 10]) and observations (dim(x)

103, 104, 105

∈ {

∈

∈

6

use a simple network architecture consisting of fully connected residual blocks [47] to parameterize
the conditional vector field. For the two tasks with dim(x) = 100 (B-GLM-Raw, SLCP-D), we
condition on (t, θ) via gated linear units as described in Section 3.2 (Fig. 8 in Appendix C shows
the corresponding performance gain). For the remaining tasks with dim(x)
10 we concatenate
(t, θ, x) instead. We reserve 5% of the simulations for validation. See Appendix C for details.

≤

For each task and simulation budget, we evaluate the model with the lowest validation loss by
comparing q(θ
x) provided in [46] for ten different observations x
x) to the reference posteriors p(θ
|
|
in terms of the C2ST score [48, 49]. This performance metric is computed by training a classifier to
discriminate inferred samples θ
x). The C2ST score is
|
then the test accuracy of this classifier, ranging from 0.5 (best) to 1.0. We observe that FMPE exhibits
comparable performance to an NPE baseline model for most tasks and outperforms on several (Fig. 4).
In terms of the MMD metric (Fig. 6 in the Appendix), FMPE clearly outperforms NPE (but MMD
can be sensitive to its hyperparameters [46]). As NPE is one of the highest ranking methods for many
tasks in the benchmark, these results show that FMPE indeed performs competitively with other
existing SBI methods. We report an additional baseline for score matching in Fig. 7 in the Appendix.

x) from reference samples θ
|

p(θ

q(θ

∼

∼

As NPE and FMPE both directly target the posterior with a density estimator (in contrast to most
other SBI methods), observed differences can be primarily attributed to their different approaches for
density estimation. Interestingly, a great performance improvement of FMPE over NPE is observed
for SLCP with a large simulation budget (N = 105). The SLCP task is specifically designed to have
a simple likelihood but a complex posterior, and the FMPE performance underscores the enhanced
flexibility of the FMPE density estimator.

|

q(θ

p(θ

Finally, we empirically investigate the mass coverage
suggested by our theoretical analysis in Section 3.1.
We display the density log q(θ
x) of the reference
|
x) under our FMPE model q as
samples θ
p(θ
|
∼
a histogram (Fig. 3). All samples θ
x) fall
|
∼
x). This becomes appar-
into the support from q(θ
ent when comparing to the density log q(θ
x) for
|
samples θ
x) from q itself. This FMPE re-
sult is therefore mass covering. Note that this does
not necessarily imply conservative posteriors (which
is also not generally true for the forward KL diver-
gence [17, 50, 51]), and some parts of p(θ
x) may
still be undersampled. Probability mass coverage,
however, implies that no part is entirely missed (com-
pare Fig. 2), even for multimodal distributions such
as Two Moons. Fig. 9 in the Appendix confirms the
mass coverage for the other benchmark tasks.

∼

|

|

Figure 3: Histogram of FMPE densities
log q(θ
x)
x) for reference samples θ
|
|
(Two Moons task, N = 103). The estimate
q(θ

x) clearly covers p(θ

∼
x) entirely.
|

p(θ

|

5 Gravitational-wave inference

5.1 Background

Gravitational waves (GWs) are ripples of spacetime predicted by Einstein and produced by cata-
clysmic cosmic events such as the mergers of binary black holes (BBHs). GWs propagate across the
universe to Earth, where the LIGO-Virgo-KAGRA observatories measure faint time-series signals
embedded in noise. To-date, roughly 90 detections of merging black holes and neutron stars have
been made [52], all of which have been characterized using Bayesian inference to compare against
theoretical models.6 These have yielded insights into the origin and evolution of black holes [53],
fundamental properties of matter and gravity [54, 55], and even the expansion rate of the universe [56].
Under reasonable assumptions on detector noise, the GW likelihood is tractable,7 and inference is

6BBH parameters θ

R15 include black-hole masses, spins, and the spacetime location and orientation of
the system (see Tab. 4 in the Appendix). We represent x in frequency domain; for two LIGO detectors and
complex f

[20, 512] Hz, ∆f = 0.125 Hz, we have x

R15744.

∈

7Noise is assumed to be stationary and Gaussian, so for frequency-domain data, the GW likelihood p(x
|
(h(θ)
|

θ) =
Sn)(x). Here h(θ) is a theoretical signal model based on Einstein’s theory of general relativity, and Sn

N
is the power spectral density of the detector noise.

∈

∈

7

−4−2024logq(θ|x)θ∼p(θ|x)θ∼q(θ|x)Figure 4: Comparison of FMPE with NPE, a standard SBI method, across 10 benchmark tasks [46].

typically performed using tools [57–60] based on Markov chain Monte Carlo [61, 62] or nested
sampling [63] algorithms. This can take from hours to months, depending on the nature of the event
108 likelihood
and the complexity of the signal model, with a typical analysis requiring up to
evaluations. The ever-increasing rate of detections means that these analysis times risk becoming a
bottleneck. SBI offers a promising solution for this challenge that has thus been actively studied in
the literature [64–68, 7, 69, 70, 41]. A fully amortized NPE-based method called DINGO recently
achieved accuracies comparable to stochastic samplers with inference times of less than a minute
per event [7]. To achieve accurate results, however, DINGO uses group-equivariant NPE [7, 69]
(GNPE), an NPE extension that integrates known conditional symmetries. GNPE, therefore, does
not provide a tractable density, which is problematic when verifying and correcting inference results
using importance sampling [41].

∼

5.2 Experiments

We here apply FMPE to GW inference. As a baseline, we train an NPE network with the settings
described in [7] with a few minor changes (see Appendix D).8 This uses an embedding network [71]
to compress x to a 128-dimensional feature vector, which is then used to condition a neural spline
flow [72]. The embedding network consists of a learnable linear layer initialized with principal
components of GW simulations followed by a series of dense residual blocks [47]. This architecture
is a powerful feature extractor for GW measurements [7]. As pointed out in Section 3.2, it is
straightforward to reuse such architectures for FMPE, with the following three modifications: (1)
we provide the conditioning on (t, θ) to the network via gated linear units in each hidden layer;
(2) we change the dimension of the final feature vector to the dimension of θ so that the network
parameterizes the conditional vector field (t, x, θ)
vt,x(θ); (3) we increase the number and width
of the hidden layers to use the capacity freed up by removing the discrete normalizing flow.

→

106 simulations for 400 epochs using a batch size of
We train the NPE and FMPE networks with 5
2 days)
4096 on an A100 GPU. The FMPE network (1.9
≈
is larger than the NPE network (1.3
3 days), but trains
substantially faster. We evaluate both networks on GW150914 [73], the first detected GW. We
generate a reference posterior using the method described in [41]. Fig. 5 compares the inferred
posterior distributions qualitatively and quantitatively in terms of the Jensen-Shannon divergence
(JSD) to the reference.9

108 learnable parameters, training takes

108 learnable parameters, training takes

≈

·

·

·

FMPE substantially outperforms NPE in terms of accuracy, with a mean JSD of 0.5 mnat (NPE:
3.6 mnat), and max JSD < 2.0 mnat, an indistinguishability criterion for GW posteriors [59].
Remarkably, FMPE accuracy is even comparable to GNPE, which leverages physical symmetries

8Our implementation builds on the public DINGO code from https://github.com/dingo-gw/dingo.
9We omit the three parameters ϕc, ϕJL, θJN in the evaluation as we use phase marginalization in importance
sampling and the reference therefore uses a different basis for these parameters [41]. For GNPE we report the
results from [7], which are generated with slightly different data conditioning. Therefore, we do not display the
GNPE results in the corner plot, and the JSDs serve only as a rough comparison. The JSD for the tc parameter is
not reported in [7] due to a tc marginalized reference.

8

0.60.81.0GLGL-UGMTwoMoonsSLCP1031041050.60.81.0B-GLM103104105B-GLM-Raw103104105SLCP-D103104105SIR103104105LVNumberofSimulationsC2STNPEFMPEFigure 5: Results for GW150914 [73]. Left: Corner plot showing 1D marginals on the diagonal
and 2D 50% credible regions. We display four GW parameters (distance dL, time of arrival tc, and
sky coordinates α, δ); these represent the least accurate NPE parameters. Right: Deviation between
inferred posteriors and the reference, quantified by the Jensen-Shannon divergence (JSD). The FMPE
posterior matches the reference more accurately than NPE, and performs similarly to symmetry-
enhanced GNPE. (We do not display GNPE results on the left due to different data conditioning
settings in available networks.)

7667.958

0.006) and FMPE (log p(x) =

to simplify data. Finally, we find that the Bayesian evidences inferred with NPE (log p(x) =
0.005) are consistent within their statistical
7667.969
−
uncertainties. A correct evidence is only obtained in importance sampling when the inferred posterior
q(θ
x) [41], so this is another indication that FMPE indeed induces
mass-covering posteriors.

x) covers the entire posterior p(θ
|

±

−

±

|

5.3 Discussion

Our results for GW150914 show that FMPE substantially outperforms NPE on this challenging
problem. We believe that this is related to the network structure as follows. The NPE network
allocates roughly two thirds of its parameters to the discrete normalizing flow and only one third to
the embedding network (i.e., the feature extractor for x). Since FMPE parameterizes just a vector
field (rather than a collection of splines in the normalizing flow) it can devote its network capacity to
R15744. Hence, it scales better to larger networks and
the interpretation of the high-dimensional x
achieves higher accuracy. Remarkably, the performance iscomparable to GNPE, which involves a
much simpler learning task with likelihood symmetries integrated by construction. This enhanced
performance, comes in part at the cost of increased inference times, typically requiring hundreds of
network forward passes. See Appendix D for further details.

∈

In future work we plan to carry out a more complete analysis of GW inference using FMPE. Indeed,
GW150914 is a loud event with good data quality, where NPE already performs quite well. DINGO
with GNPE has been validated in a variety of settings [7, 69, 41, 74] including events with a larger
performance gap between NPE and GNPE [69]. Since FMPE (like NPE) does not integrate physical
symmetries, it would likely need further enhancements to fully compete with GNPE. This may require
a symmetry-aware architecture [75], or simply further scaling to larger networks. A straightforward
application of the GNPE mechanism to FMPE—GFMPE—is also possible, but less practical due to
the higher inference costs of FMPE. Nevertheless, our results demonstrate that FMPE is a promising
direction for future research in this field.

9

ReferenceNPEFMPE−808tc[ms]1.01.52.02.5α200400600dL[Mpc]−1.0−0.50.0δ−808tc[ms]1.01.52.02.5α−1.0−0.50.0δNPEFMPEGNPEm1m2a1a2t1t2φ12dLtcαδψ1.21.30.82.50.61.13.20.80.21.61.00.30.80.10.50.40.30.50.30.20.14.40.10.89.10.6–10.10.60.78.60.51.40.60.10.205101520JSD[mnat]6 Conclusions

We introduced flow matching posterior estimation, a new simulation-based inference technique based
on continuous normalizing flows. In contrast to existing neural posterior estimation methods, it
does not rely on restricted density estimation architectures such as discrete normalizing flows, and
instead parametrizes a distribution in terms of a conditional vector field. Besides enabling flexible
path specifications, while maintaining direct access to the posterior density, we empirically found
that regressing on a vector field rather than an entire distribution improves the scalability of FMPE
compared to existing approaches. Indeed, fewer parameters are needed to learn this vector field,
allowing for larger networks, ultimately enabling to solve more complex problems. Furthermore, our
architecture for FMPE (a straightforward ResNet with GLU conditioning) facilitates parallelization
and allows for cheap forward/backward passes.

We evaluated FMPE on a set of 10 benchmark tasks and found competitive or better performance
compared to other simulation-based inference methods. On the challenging task of gravitational-wave
inference, FMPE substantially outperformed comparable discrete flows, producing samples on par
with a method that explicitly leverages symmetries to simplify training. Additionally, flow matching
latent spaces are more naturally structured than those of discrete flows, particularly when using
paths such as optimal transport. Looking forward, it would be interesting to exploit such structure
in designing learning algorithms. This performance and flexibilty underscores the capability of
continuous normalizing flows to efficiently solve inverse problems.

Acknowledgements

We thank the DINGO team for helpful discussions and comments. We would like to particularly
acknowledge the contributions of Alessandra Buonanno, Jonathan Gair, Nihar Gupte and Michael
Pürrer. This material is based upon work supported by NSF’s LIGO Laboratory which is a major
facility fully funded by the National Science Foundation. This research has made use of data or
software obtained from the Gravitational Wave Open Science Center (gw-openscience.org), a service
of LIGO Laboratory, the LIGO Scientific Collaboration, the Virgo Collaboration, and KAGRA.
LIGO Laboratory and Advanced LIGO are funded by the United States National Science Foundation
(NSF) as well as the Science and Technology Facilities Council (STFC) of the United Kingdom, the
Max-Planck-Society (MPS), and the State of Niedersachsen/Germany for support of the construction
of Advanced LIGO and construction and operation of the GEO600 detector. Additional support for
Advanced LIGO was provided by the Australian Research Council. Virgo is funded, through the
European Gravitational Observatory (EGO), by the French Centre National de Recherche Scientifique
(CNRS), the Italian Istituto Nazionale di Fisica Nucleare (INFN) and the Dutch Nikhef, with
contributions by institutions from Belgium, Germany, Greece, Hungary, Ireland, Japan, Monaco,
Poland, Portugal, Spain. The construction and operation of KAGRA are funded by Ministry of
Education, Culture, Sports, Science and Technology (MEXT), and Japan Society for the Promotion
of Science (JSPS), National Research Foundation (NRF) and Ministry of Science and ICT (MSIT) in
Korea, Academia Sinica (AS) and the Ministry of Science and Technology (MoST) in Taiwan. M.D.
thanks the Hector Fellow Academy for support. J.H.M. and B.S. are members of the MLCoE, EXC
number 2064/1 – Project number 390727645 and the Tübingen AI Center funded by the German
Ministry for Science and Education (FKZ 01IS18039A).

References

[1] Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows.

In
International Conference on Machine Learning, pages 1530–1538, 2015. arXiv:1505.05770.

[2] George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, and Balaji
Lakshminarayanan. Normalizing flows for probabilistic modeling and inference. Journal of
Machine Learning Research, 22(57):1–64, 2021. URL: http://jmlr.org/papers/v22/
19-1028.html.

[3] Kyle Cranmer, Johann Brehmer, and Gilles Louppe. The frontier of simulation-based inference.
Proc. Nat. Acad. Sci., 117(48):30055–30062, 2020. arXiv:1911.01429, doi:10.1073/
pnas.1912789117.

10

[4] George Papamakarios and Iain Murray. Fast ε-free inference of simulation models with Bayesian
conditional density estimation. In Advances in neural information processing systems, 2016.
arXiv:1605.06376.

[5] Jan-Matthis Lueckmann, Pedro J Gonçalves, Giacomo Bassetto, Kaan Öcal, Marcel Non-
nenmacher, and Jakob H Macke. Flexible statistical inference for mechanistic models of
neural dynamics. In Proceedings of the 31st International Conference on Neural Information
Processing Systems, pages 1289–1299, 2017.

[6] David Greenberg, Marcel Nonnenmacher, and Jakob Macke. Automatic posterior transformation
for likelihood-free inference. In International Conference on Machine Learning, pages 2404–
2414. PMLR, 2019.

[7] Maximilian Dax, Stephen R. Green, Jonathan Gair, Jakob H. Macke, Alessandra Buonanno, and
Bernhard Schölkopf. Real-Time Gravitational Wave Science with Neural Posterior Estimation.
Phys. Rev. Lett., 127(24):241103, 2021. arXiv:2106.12594, doi:10.1103/PhysRevLett.
127.241103.

[8] Louis Sharrock, Jack Simons, Song Liu, and Mark Beaumont. Sequential neural score estima-
tion: Likelihood-free inference with conditional score based diffusion models. arXiv preprint
arXiv:2210.04872, 2022.

[9] Tomas Geffner, George Papamakarios, and Andriy Mnih. Score modeling for simulation-based

inference. arXiv preprint arXiv:2209.14249, 2022.

[10] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsuper-
vised learning using nonequilibrium thermodynamics. In International Conference on Machine
Learning, pages 2256–2265. PMLR, 2015.

[11] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data

distribution. Advances in neural information processing systems, 32, 2019.

[12] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances

in Neural Information Processing Systems, 33:6840–6851, 2020.

[13] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis.
In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors,
Advances in Neural Information Processing Systems, volume 34, pages 8780–8794. Curran
Associates, Inc., 2021. URL: https://proceedings.neurips.cc/paper_files/paper/
2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf.

[14] Jonathan Ho, Chitwan Saharia, William Chan, David J. Fleet, Mohammad Norouzi, and Tim
Salimans. Cascaded diffusion models for high fidelity image generation. J. Mach. Learn. Res.,
23:47:1–47:33, 2022. URL: http://jmlr.org/papers/v23/21-0635.html.

[15] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and
Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv
preprint arXiv:2011.13456, 2020.

[16] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow
matching for generative modeling. CoRR, abs/2210.02747, 2022. arXiv:2210.02747, doi:
10.48550/arXiv.2210.02747.

[17] Joeri Hermans, Arnaud Delaunoy, François Rozet, Antoine Wehenkel, and Gilles Louppe.
Averting a crisis in simulation-based inference. arXiv preprint arXiv:2110.06581, 2021.

[18] Tian Qi Chen, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. Neural or-
dinary differential equations.
In Samy Bengio, Hanna M. Wallach, Hugo Larochelle,
Kristen Grauman, Nicolò Cesa-Bianchi, and Roman Garnett, editors, Advances in Neu-
ral Information Processing Systems 31: Annual Conference on Neural Information
Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal, Canada,
pages 6572–6583, 2018. URL: https://proceedings.neurips.cc/paper/2018/hash/
69386f6bb1dfed68692a24c8686939b9-Abstract.html.

11

[19] Michael S Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic

interpolants. arXiv preprint arXiv:2209.15571, 2022.

[20] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate

and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

[21] Kirill Neklyudov, Daniel Severo, and Alireza Makhzani. Action matching: A variational method
for learning stochastic dynamics from samples. arXiv preprint arXiv:2210.06662, 2022.

[22] Aram Davtyan, Sepehr Sameni, and Paolo Favaro. Randomized conditional flow matching for

video prediction. arXiv preprint arXiv:2211.14575, 2022.

[23] Alexander Tong, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks,
Kilian Fatras, Guy Wolf, and Yoshua Bengio. Conditional flow matching: Simulation-free
dynamic optimal transport. arXiv preprint arXiv:2302.00482, 2023.

[24] Scott A Sisson, Yanan Fan, and Mark A Beaumont. Overview of abc.

In Handbook of

approximate Bayesian computation, pages 3–54. Chapman and Hall/CRC, 2018.

[25] Mark A Beaumont, Wenyang Zhang, and David J Balding. Approximate bayesian computation

in population genetics. Genetics, 162(4):2025–2035, 2002.

[26] Mark A Beaumont, Jean-Marie Cornuet, Jean-Michel Marin, and Christian P Robert. Adaptive

approximate bayesian computation. Biometrika, 96(4):983–990, 2009.

[27] Michael G. B. Blum and Olivier François.

Stat. Comput., 20(1):63–73, 2010.

Non-linear regression models for ap-
doi:10.1007/

proximate bayesian computation.
s11222-009-9116-0.

[28] Dennis Prangle, Paul Fearnhead, Murray P. Cox, Patrick J. Biggs, and Nigel P. French. Semi-
automatic selection of summary statistics for abc model choice, 2013. arXiv:1302.5624.

[29] Simon Wood. Statistical inference for noisy nonlinear ecological dynamic systems. Nature,

466:1102–4, 08 2010. doi:10.1038/nature09319.

[30] Christopher C Drovandi, Clara Grazian, Kerrie Mengersen, and Christian Robert. Approximat-

ing the likelihood in approximate bayesian computation, 2018. arXiv:1803.06645.

[31] George Papamakarios, David Sterratt, and Iain Murray. Sequential neural likelihood: Fast
likelihood-free inference with autoregressive flows. In The 22nd International Conference on
Artificial Intelligence and Statistics, pages 837–848. PMLR, 2019.

[32] Jan-Matthis Lueckmann, Giacomo Bassetto, Theofanis Karaletsos, and Jakob H Macke.
Likelihood-free inference with emulator networks. In Symposium on Advances in Approx-
imate Bayesian Inference, pages 32–53. PMLR, 2019.

[33] Rafael Izbicki, Ann Lee, and Chad Schafer. High-dimensional density ratio estimation with
extensions to approximate likelihood computation. In Artificial intelligence and statistics, pages
420–429. PMLR, 2014.

[34] Kim Pham, David Nott, and Sanjay Chaudhuri. A note on approximating abc-mcmc using

flexible classifiers. Stat, 3, 03 2014. doi:10.1002/sta4.56.

[35] Kyle Cranmer, Juan Pavez, and Gilles Louppe. Approximating likelihood ratios with calibrated

discriminative classifiers. arXiv preprint arXiv:1506.02169, 2015.

[36] Joeri Hermans, Volodimir Begy, and Gilles Louppe. Likelihood-free mcmc with approximate

likelihood ratios. arXiv preprint arXiv:1903.04057, 10, 2019.

[37] Conor Durkan, Iain Murray, and George Papamakarios. On contrastive learning for likelihood-
free inference. In International conference on machine learning, pages 2771–2781. PMLR,
2020.

[38] Owen Thomas, Ritabrata Dutta, Jukka Corander, Samuel Kaski, and Michael U. Gutmann.

Likelihood-free inference by ratio estimation, 2020. arXiv:1611.10242.

12

[39] Benjamin K Miller, Christoph Weniger, and Patrick Forré. Contrastive neural ratio estimation.

Advances in Neural Information Processing Systems, 35:3262–3278, 2022.

[40] Thomas Müller, Brian McWilliams, Fabrice Rousselle, Markus Gross, and Jan Novák. Neural

importance sampling. ACM Transactions on Graphics (TOG), 38(5):1–19, 2019.

[41] Maximilian Dax, Stephen R. Green, Jonathan Gair, Michael Pürrer, Jonas Wildberger, Jakob H.
Macke, Alessandra Buonanno, and Bernhard Schölkopf. Neural Importance Sampling for
Rapid and Reliable Gravitational-Wave Inference. Phys. Rev. Lett., 130(17):171403, 2023.
arXiv:2210.05686, doi:10.1103/PhysRevLett.130.171403.

[42] Michael S Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A

unifying framework for flows and diffusions. arXiv preprint arXiv:2303.08797, 2023.

[43] Yang Song, Conor Durkan, Iain Murray, and Stefano Ermon. Maximum likelihood training of
score-based diffusion models. Advances in Neural Information Processing Systems, 34:1415–
1428, 2021.

[44] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
for biomedical image segmentation. In Medical Image Computing and Computer-Assisted
Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9,
2015, Proceedings, Part III 18, pages 234–241. Springer, 2015.

[45] Yann N Dauphin, Angela Fan, Michael Auli, and David Grangier. Language modeling with
gated convolutional networks. In International conference on machine learning, pages 933–941.
PMLR, 2017.

[46] Jan-Matthis Lueckmann, Jan Boelts, David Greenberg, Pedro Goncalves, and Jakob Macke.
Benchmarking simulation-based inference. In International Conference on Artificial Intelligence
and Statistics, pages 343–351. PMLR, 2021.

[47] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image

recognition, 2015. arXiv:1512.03385.

[48] Jerome H Friedman. On multivariate goodness–of–fit and two–sample testing. Statistical

Problems in Particle Physics, Astrophysics, and Cosmology, 1:311, 2003.

[49] David Lopez-Paz and Maxime Oquab. Revisiting classifier two-sample tests. arXiv preprint

arXiv:1610.06545, 2016.

[50] Arnaud Delaunoy, Joeri Hermans, François Rozet, Antoine Wehenkel, and Gilles Louppe.
Towards reliable simulation-based inference with balanced neural ratio estimation, 2022. arXiv:
2208.13624.

[51] Arnaud Delaunoy, Benjamin Kurt Miller, Patrick Forré, Christoph Weniger, and Gilles
Louppe. Balancing simulation-based inference for conservative posteriors. arXiv preprint
arXiv:2304.10978, 2023.

[52] R. Abbott et al. GWTC-3: Compact Binary Coalescences Observed by LIGO and Virgo
During the Second Part of the Third Observing Run. arXiv preprint arXiv:2111.03606, 11 2021.
arXiv:2111.03606.

[53] R. Abbott et al. Population Properties of Compact Objects from the Second LIGO-Virgo
Gravitational-Wave Transient Catalog. Astrophys. J. Lett., 913(1):L7, 2021. arXiv:2010.
14533, doi:10.3847/2041-8213/abe949.

[54] B. P. Abbott et al. GW170817: Measurements of neutron star radii and equation of state. Phys.
Rev. Lett., 121(16):161101, 2018. arXiv:1805.11581, doi:10.1103/PhysRevLett.121.
161101.

[55] R. Abbott et al. Tests of general relativity with binary black holes from the second LIGO-Virgo
gravitational-wave transient catalog. Phys. Rev. D, 103(12):122002, 2021. arXiv:2010.14529,
doi:10.1103/PhysRevD.103.122002.

13

[56] B. P. Abbott et al. A gravitational-wave standard siren measurement of the Hubble constant.

Nature, 551(7678):85–88, 2017. arXiv:1710.05835, doi:10.1038/nature24471.

[57] J. Veitch, V. Raymond, B. Farr, W Farr, P. Graff, S. Vitale, et al. Parameter estimation for
compact binaries with ground-based gravitational-wave observations using the LALInference
software library. Phys. Rev., D91(4):042003, 2015. arXiv:1409.7215, doi:10.1103/
PhysRevD.91.042003.

[58] Gregory Ashton et al. BILBY: A user-friendly Bayesian inference library for gravitational-
wave astronomy. Astrophys. J. Suppl., 241(2):27, 2019. arXiv:1811.02042, doi:10.3847/
1538-4365/ab06fc.

[59] I. M. Romero-Shaw et al. Bayesian inference for compact binary coalescences with bilby:
validation and application to the first LIGO–Virgo gravitational-wave transient catalogue. Mon.
Not. Roy. Astron. Soc., 499(3):3295–3319, 2020. arXiv:2006.00714, doi:10.1093/mnras/
staa2850.

[60] Joshua S Speagle. dynesty: a dynamic nested sampling package for estimating Bayesian poste-
riors and evidences. Monthly Notices of the Royal Astronomical Society, 493(3):3132–3158,
Feb 2020. URL: http://dx.doi.org/10.1093/mnras/staa278, arXiv:1904.02180,
doi:10.1093/mnras/staa278.

[61] Nicholas Metropolis, Arianna W Rosenbluth, Marshall N Rosenbluth, Augusta H Teller, and
Edward Teller. Equation of state calculations by fast computing machines. The journal of
chemical physics, 21(6):1087–1092, 1953.

[62] W. K. Hastings. Monte Carlo sampling methods using Markov chains and their applica-
tions. Biometrika, 57(1):97–109, 04 1970. arXiv:https://academic.oup.com/biomet/
article-pdf/57/1/97/23940249/57-1-97.pdf, doi:10.1093/biomet/57.1.97.

[63] John Skilling. Nested sampling for general Bayesian computation. Bayesian Analysis, 1(4):833

– 859, 2006. doi:10.1214/06-BA127.

[64] Elena Cuoco, Jade Powell, Marco Cavaglià, Kendall Ackley, Michał Bejger, Chayan Chatterjee,
Michael Coughlin, Scott Coughlin, Paul Easter, Reed Essick, et al. Enhancing gravitational-
wave science with machine learning. Machine Learning: Science and Technology, 2(1):011002,
5 2020. arXiv:2005.03745, doi:10.1088/2632-2153/abb93a.

[65] Hunter Gabbard, Chris Messenger, Ik Siong Heng, Francesco Tonolini, and Roderick
Murray-Smith. Bayesian parameter estimation using conditional variational autoencoders
for gravitational-wave astronomy. Nature Phys., 18(1):112–117, 2022. arXiv:1909.06296,
doi:10.1038/s41567-021-01425-7.

[66] Stephen R. Green, Christine Simpson, and Jonathan Gair. Gravitational-wave parameter
estimation with autoregressive neural network flows. Phys. Rev. D, 102(10):104057, 2020.
arXiv:2002.07656, doi:10.1103/PhysRevD.102.104057.

[67] Arnaud Delaunoy, Antoine Wehenkel, Tanja Hinderer, Samaya Nissanke, Christoph Weniger,
Andrew R. Williamson, and Gilles Louppe. Lightning-Fast Gravitational Wave Parameter
Inference through Neural Amortization. In Third Workshop on Machine Learning and the
Physical Sciences, 10 2020. arXiv:2010.12931.

[68] Stephen R. Green and Jonathan Gair. Complete parameter inference for GW150914 using deep
learning. Mach. Learn. Sci. Tech., 2(3):03LT01, 2021. arXiv:2008.03312, doi:10.1088/
2632-2153/abfaed.

[69] Maximilian Dax, Stephen R. Green, Jonathan Gair, Michael Deistler, Bernhard Schölkopf, and
Jakob H. Macke. Group equivariant neural posterior estimation. In International Conference on
Learning Representations, 11 2022. arXiv:2111.13139.

[70] Chayan Chatterjee, Linqing Wen, Damon Beveridge, Foivos Diakogiannis, and Kevin Vinsen.
Rapid localization of gravitational wave sources from compact binary coalescences using deep
learning. arXiv preprint arXiv:2207.14522, 7 2022. arXiv:2207.14522.

14

[71] Stefan T. Radev, Ulf K. Mertens, Andreass Voss, Lynton Ardizzone, and Ullrich Köthe.
Bayesflow: Learning complex stochastic models with invertible neural networks, 2020.
arXiv:2003.06281.

[72] Conor Durkan, Artur Bekasov, Iain Murray, and George Papamakarios. Neural spline flows.
In Advances in Neural Information Processing Systems, pages 7509–7520, 2019. arXiv:
1906.04032.

[73] B.P. Abbott et al. Observation of Gravitational Waves from a Binary Black Hole Merger.
Phys. Rev. Lett., 116(6):061102, 2016. arXiv:1602.03837, doi:10.1103/PhysRevLett.
116.061102.

[74] Jonas Wildberger, Maximilian Dax, Stephen R. Green, Jonathan Gair, Michael Pürrer, Jakob H.
Macke, Alessandra Buonanno, and Bernhard Schölkopf. Adapting to noise distribution shifts
in flow-based gravitational-wave inference. Phys. Rev. D, 107(8):084046, 2023. arXiv:
2211.08801, doi:10.1103/PhysRevD.107.084046.

[75] Taco Cohen and Max Welling. Group equivariant convolutional networks. In Maria-Florina
Balcan and Kilian Q. Weinberger, editors, Proceedings of the 33nd International Conference
on Machine Learning, ICML 2016, New York City, NY, USA, June 19-24, 2016, volume 48
of JMLR Workshop and Conference Proceedings, pages 2990–2999. JMLR.org, 2016. URL:
http://proceedings.mlr.press/v48/cohenc16.html.

[76] Mark Hannam, Patricia Schmidt, Alejandro Bohé, Leïla Haegel, Sascha Husa, Frank Ohme,
Geraint Pratten, and Michael Pürrer. Simple model of complete precessing black-hole-
binary gravitational waveforms. Phys. Rev. Lett., 113:151101, Oct 2014. URL: https://
link.aps.org/doi/10.1103/PhysRevLett.113.151101, doi:10.1103/PhysRevLett.
113.151101.

[77] Sebastian Khan, Sascha Husa, Mark Hannam, Frank Ohme, Michael Pürrer, Xisco Jiménez
Forteza, and Alejandro Bohé. Frequency-domain gravitational waves from nonprecessing
black-hole binaries. II. A phenomenological model for the advanced detector era. Phys. Rev.,
D93(4):044007, 2016. arXiv:1508.07253, doi:10.1103/PhysRevD.93.044007.

[78] Alejandro Bohé, Mark Hannam, Sascha Husa, Frank Ohme, Michael Pürrer, and Patricia
Schmidt. PhenomPv2 – technical notes for the LAL implementation. LIGO Technical Document,
LIGO-T1500602-v4, 2016. URL: https://dcc.ligo.org/LIGO-T1500602/public.

[79] Benjamin Farr, Evan Ochsner, Will M. Farr, and Richard O’Shaughnessy. A more effective
coordinate system for parameter estimation of precessing compact binaries from gravitational
waves. Phys. Rev. D, 90(2):024018, 2014. arXiv:1404.7070, doi:10.1103/PhysRevD.90.
024018.

[80] J.R. Dormand and P.J. Prince. A family of embedded runge-kutta formulae.

Jour-
nal of Computational and Applied Mathematics, 6(1):19–26, 1980. URL: https://
www.sciencedirect.com/science/article/pii/0771050X80900133, doi:10.1016/
0771-050X(80)90013-3.

15

A Gaussian flow

We here derive the form of a vector field vt(θ) that restricts the resulting continuous flow to a one
dimensional Gaussian with mean ˆµ variance ˆσ2. With the optimal transport path µt(θ) = tθ1,
σt(θ) = 1

σt from [16], the sample-conditional probability path (5) reads

σmin)t

(1

−

−

≡

pt(θ

|

θ1) =

N

[tθ1, σ2

t ](θ).

(10)

We set our target distribution

N
To derive the marginal probability path and the marginal vector field we need two identities for
the convolution
of Gaussian densities. Recall that the convolution of two function is defined by
f

y)g(y) dy. We define the function

g(x) = (cid:82) f (x
∗

q1(θ1) =

[ˆµ, ˆσ2](θ1).

(11)

∗

−

gµ,σ2 (θ) = θ

· N

(cid:2)µ, σ2(cid:3) (θ).

Then the following holds
[µ1, σ2
1]

N

[µ2, σ2

2] =

∗ N

g0,σ2

1 ∗ N

[µ2, σ2

2] =

Marginal probability paths

1 + σ2
2]

N

[µ1 + µ2, σ2
σ2
(cid:16)
1
1 + σ2
σ2
2

gµ2,σ2

1 +σ2

2 −

Marginalizing over θ1 in (10) with (11), we find

(12)

(13)

(14)

[µ2, σ2

1 + σ2
2]

(cid:17)

µ2 N

(cid:90)

(cid:90)

(cid:90)

(cid:90)

pt(θ) =

=

=

=

N

N

pt(θ

θ1)q(θ1) dθ1

|
(cid:2)tθ1, σ2

t

(cid:3) (θ)

N

(cid:2)ˆµ, ˆσ2(cid:3) (θ1)dθ1

(cid:2)0, σ2

t

(cid:3) (θ

tθ1)

−

N

(tˆµ, (tˆσ)2)(tθ1)

t dθ1

·

(15)

t

(cid:3) (θ

(cid:2)0, σ2
N
(cid:2)tˆµ, σ2

θt
1)
−
N
t + (tˆσ)2(cid:3) (θ)

(cid:2)tˆµ, (tˆσ)2(cid:3) (θt

1) dθt
1

where we defined θt

=

N
1 = tθ1 and used (13).

Marginal vector field

We now calculate the marginalized vector field ut(θ) based on equation (8) in [16]. Using the
sample-conditional vector field (6) and the distributions (10) and (11) we find

(cid:90)

ut(θ) =

ut(θ

pt(θ

θ1)
|
(cid:90) (θ1 −
(cid:90) (θ1 −
(cid:90) (θ′1 −
(cid:90) (

−

(cid:90) (

−

=

=

=

=

=

1
pt(θ)
1
pt(θ)
1
pt(θ)
1
pt(θ)
1
pt(θ)

dθ1

θ1)q(θ1)
|
pt(θ)
(1

σmin)θ)

−
σt

σmin)θ)

(1

−
σt

·

σmin)t
t

(1
−
σt ·
θ′′1 + (1
(1
−
σt ·
θ)

−
t

θ′′1 + σt ·
t
σt ·

· N

(cid:2)tθ1, σ2

t

(cid:3) (θ)

(cid:2)ˆµ, ˆσ2(cid:3) (θ1) dθ1

· N

tθ1)

· N

θ′1)

−
(cid:2)0, σ2

t

· N
(cid:3) (θ′′1 )

· N
(cid:2)tˆµ, (tˆσ)2(cid:3) (θ

(cid:2)tˆµ, (tˆσ)2(cid:3) (tθ1)

·
(cid:2)tˆµ, (tˆσ)2(cid:3) (θ′1)

t dθ1

dθ′1

·
(cid:2)tˆµ, (tˆσ)2(cid:3) (θ

θ′′1 )

·

dθ′′1

−

θ′′1 )

·

−

dθ′′1

(16)

· N

· N
θ)

· N
σmin)t)

(cid:2)0, σ2

t

(cid:3) (θ

−
(cid:3) (θ

(cid:2)0, σ2

t

θ)

·

· N

(cid:2)0, σ2

t

(cid:3) (θ′′1 )

· N

16

where we used the change of variables θ′1 = tθ1 and θ′′1 = θ
using (12), then the identities (13) and (14) and the marginal probability (15)
(cid:2)tˆµ, (tˆσ)2(cid:3)(cid:17)

(cid:2)0, σ2

(θ) +

ut(θ) =

−

(cid:16)

(cid:0)

(cid:3)

g0,σ2

t

θ′1. Now we evaluate this expression

θ
pt(θ)
N
t + (tˆσ)2)(cid:3) (θ) +

t

·

t

1
−
σt ·
pt(θ)
·
1
−
σt ·
pt(θ)
·
t + (tˆσ)2)θ
(σ2
t
pt(θ)
t + (tˆσ)2
(σ2
t

(σ2

t

·

·

t ∗ N
tˆµ)

σ2
t

(θ
−
·
σ2
t + (tˆσ)2 · N
σt

(θ

tˆµ)
−
−
(σ2
t + (tˆσ)2)
σt)θ + tˆµ

·
−
t + (tˆσ)2)

·

σt

.

·

=

=

=

(cid:2)tˆµ, (σ2

pt(θ)

·

(cid:2)tˆµ, (tˆσ)2(cid:3)(cid:1) (θ)

∗ N

θ
pt(θ)

t N

·

(cid:2)tˆµ, (σ2

t + (tˆσ)2)(cid:3) (θ)

(17)

By choosing a vector field vt of the form (17) with learnable parameters ˆµ, ˆσ2, we can thus define a
continuous flow that is restricted to a one dimensional Gaussian.

B Mass covering properties of flows

In this supplement, we investigate the mass covering properties of continuous normalizing flows
trained using mean squared error and in particular prove Theorem 1. We first recall the notation from
the main part. We always assume that the data is distributed according to p1(θ). In addition, there is a
Rd
known and simple base distribution p0 and we assume that there is a vector field ut : [0, 1]
that connects p0 and p1 in the following sense. We denote by ϕt the flow generated by ut, i.e., ϕt
satisfies

Rd

→

×

Then we assume that (ϕ1)

∗

p0 = p1 and we also define the interpolations pt = (ϕt)

p0.

∗

∂tϕt(θ) = ut(ϕt(θ)).

(18)

We do not have access to the ground truth distributions pt and the vector field ut but we try to learn a
vector field vt approximating ut. We denote its flow by ψt and we define qt = (ψt)
q0 and q0 = p0.
We are interested in the mass covering properties of the learned approximation q1 of p1. In particular,
we want to relate the KL-divergence DKL(p1||

q1) to the mean squared error,

∗

(cid:90) 1

(cid:90)

MSEp(u, v) =

dt

pt(dθ)(ut(θ)

0

vt(θ))2,

−

(19)

of the generating vector fields. The first observation is that without any regularity assumptions on vt
it is impossible to obtain any bound on the KL-divergence in terms of the mean squared error.
Lemma 1. For every ε > 0 there are vector field ut and vt and a base distribution p0 = q0 such that

In addition we can construct ut and vt such that the support of p1 is larger than the support of q1.

MSEp(u, v) < ε and DKL(p1||

q1) =

.

∞

(20)

Proof. We consider the uniform distribution p0 = q0 ∼ U

([

−

1, 1]) and the vector fields

and

ut(θ) = 0

vt(θ) =

(cid:26)ε
0

θ < ε,

for 0
≤
otherwise.

(21)

(22)

As before, let ϕt denote the flow of the vector field ut and similarly ψt denote the flow of vt. Clearly
ϕt(θ) = θ. On the other hand

ψt(θ) =

(cid:26)min(θ + εt, ε)
otherwise.

θ

if 0

≤

θ < ε,

(23)

17

In particular

ψ1(θ) =

(cid:26)ε
θ

θ < ε,

if 0
≤
otherwise.

(24)

This implies that p1 = (ϕ1)
∗
[
1, 0]
−
∪
DKL(p1||

q0 has support in
[ε, 1]. In particular, the distribution of q1 is not mass covering with respect to p1 and
q1) =

1, 1]). On the other hand q1 = (ψ1)
∗

. Finally, we observe that the MSE can be arbitrarily small

p0 ∼ U

∞

−

([

(cid:90) 1

(cid:90)

MSEp(u, v) =

dt

pt(dθ)

0

ut(θ)
|

−

2 =
vt(θ)
|

(cid:90) 1

(cid:90) ε

0

0

1
2

ε2 =

ε3
2

.

(25)

Here we used that the density of pt(dθ) is 1/2 for

1

−

≤

θ

≤

1.

We see that an arbitrary small MSE-loss cannot ensure that the probability distribution is mass
covering and the KL-divergence is finite. On a high level this can be explained by the fact that
for vector fields vt that are not Lipschitz continuous the flow is not necessarily continuous, and
we can generate holes in the distribution. Note that we chose p0 to be a uniform distribution for
simplicity, but the result extends to any smooth distribution, in particular the result does not rely on
the discontinuity of p0.

Next, we investigate the mass covering property for Lipschitz continuous flows. When the flows ut
and vt are Lipschitz continuous (in θ) this ensures that the flows ψ1 and ϕ1 are continuous in x and it
is not possible to create holes in the distribution as shown above for non-continuous vector fields. We
show a weaker bound in this setting.
Lemma 2. For every 0
δ
vector fields ut and vt such that MSEp(u, v) = δ and

1 there is a base distribution p0 = q0 and the are Lipschitz-continuous

≤

≤

DKL(p1||

q1)

≥

1
2

MSEp(u, v)1/3.

Proof. We consider p0, q0 and ut as in Lemma 1, and we define

vt(θ) =






2θ
2ε
0

for 0
θ

≤
for ε
−
otherwise.

θ < ε,

θ < 2ε,

≤

Then we can calculate for 0

θ

≤

≤

e−

2ε that

ψt(θ) = θe2t.

Similarly we obtain for ε

θ

≤

≤

We find

2ε (solving the ODE f ′ = 2f )
2t.

ψt(θ) = 2ε

θ)e−

(2ε

−

−

ψ1(0) = 0, ψ1(e−

2ε) = ε, ψ1(ε) = 2

εe−

2; ψ2(2ε) = 2ε.

−

Next we find for the densities of q1 that

q1(ψ1(θ)) = q0(θ)

ψ′1(θ)
|
|

1 =

−

2

(cid:26)e−
e2

1
2

for 0

θ

e−

2ε,

for ε

≤
θ

≤
2ε.

≤

≤

Together with (30) this implies that the density of q1 is given by

Note that p1(θ) = 1/2 for

θ

1
≤
−
(cid:90) ε

ln

0

≤
p1(θ)
q1(θ)

q1(θ) =

1
2

2

(cid:26)e−
e2

for 0
for 2ε

≤
−
1 and therefore

θ
≤
2
εe−

ε,

θ

≤

≤

2ε.

p1(dθ) =

(cid:90) ε

0

ln(e2)

1
2

dθ = ε,

18

(26)

(27)

(28)

(29)

(30)

(31)

(32)

(33)

and

Moreover we note
(cid:90) 2ε

(cid:90) 2ε

εe−2

2ε

−

ln

p1(θ)
q1(θ)

p1(dθ) =

(cid:90) 2ε

2ε

εe−2

−

ln(e−

2)

1
2

dθ =

εe−

2.

−

(34)

εe−2

−

q1(dε) =

(cid:90) ε

e−2ε

q0(dε) =

1
2

ε(1

−

e−

2) =

(cid:90) 2ε

εe−2

−

ε

p1(dε),

(35)

ε

which implies (by positivity of the KL-divergence) that
(cid:19)

εe−2

(cid:90) 2ε

−

ε

ln

(cid:18) p1(θ)
q1(θ)

We infer using also p1(θ) = q1(θ) = 1/2 for θ

DKL(p1||
On the other hand we can bound

q1) =

(cid:90)

ln

1, 0]
(cid:19)

[
∈
−
(cid:18) p1(θ)
q1(θ)

p1(dθ)

0.

≥

[2ε, 1] that

∩

p1(dθ)

ε(1

−

≥

e−

2).

(cid:90) 1

(cid:90)

dt

0

pt(dθ)
|

vt(θ)

2 =
ut(θ)
|

−

We conclude that

DKL(p1||

q1)

≥

1
2

1
2

(cid:90) 1

(cid:90) 2ε

dt

0

0

ut(θ)
|
|

2 =

(cid:90) ε

0

s2 ds =

ε3
3

.

(MSEp(u, v))1/3 .

(36)

(37)

(38)

(39)

In particular, it is not possible to bound the KL-divergence by the MSE even when the vector fields
are Lipschitz continuous.

Let us put this into context. It was already shown in [42] that we can, in general, not bound the
forward KL-divergence by the mean squared error and our Lemmas 1 and 2 are concrete examples.
On the other hand, when considering SDEs the KL-divergence can be bounded by the mean squared
error of the drift terms as shown in [43]. Indeed, in [42] the favorable smoothing effect was carefully
investigated.

Here we show that we can alternatively obtain an upper bound on the KL-divergence when assuming
that ut, vt, and p0 satisfy additional regularity assumptions. This allows us to recover the mass
covering property from bounds on the means squared error for sufficiently smooth vector fields. The
scaling is nevertheless still weaker than for SDEs.

We now state our assumptions. We denote the gradient with respect to θ by
derivatives by

∇µ and second
2
µν. When applying the chain rule, we leave the indices implicit. We denote by

2 =

∇

=

(cid:16)(cid:80)

ij A2
ij

(cid:17)1/2

of a matrix. The Frobenius norm is submultiplicative,

and directly generalizes to higher order tensors.

=

∇

∇
the Frobenius norm
| · |
A
AB
i.e.,
|
Assumption 1. We assume that
ut| ≤

| ≤ |

A
|

| · |

|∇

B

|

|

L,

vt| ≤

|∇

L,

2ut| ≤

|∇

L′,

2vt| ≤

|∇

L′.

We require one further assumption on p0.
Assumption 2. There is a constant C1 such that
ln p0(θ)

We also assume that

|∇

C1(1 +

θ
|

).
|

| ≤

Ep0 |

θ

|

2 < C2 <

.

∞

(40)

(41)

(42)

. If we assume that

Note that (41) holds, e.g., if p0 follows a Gaussian distribution but also for smooth distribution with
is bounded the proof below simplifies slightly.
slower decay at
This is, e.g., the case if p0(θ)
We need some additional notation. It is convenient to introduce ϕs
time s to t (in particular ϕ0

t = ϕt) and similarly for ψ. We can now restate and prove Theorem 1.

ln p0(θ)
|
.
| → ∞

1, i.e., the flow from

t = ϕt ◦

|∇
θ
|

(ϕs)−

θ
e−|

| as

∞

∼

19

Theorem 2. Let p0 = q0 and assume ut and vt are two vector fields whose flows satisfy p1 = (ϕ1)
and q1 = (ψ1)
∗
there is a constant C > 0 depending on L, L′, C1, C2, and d such that (for MSEp(u, v) < 1))

p0
q0. Assume that p0 satisfies Assumption 2 and ut and vt satisfy Assumption 1. Then

∗

DKL(p1||

q1)

≤

C MSEp(u, v)

1
2 .

(43)

Remark 1. We do not claim that our results are optimal, it might be possible to find similar bounds
for the forward KL-divergence with weaker assumptions. However, we emphasize that Lemma 2
shows that the result of the theorem is not true without the assumption on the second derivative of vt
and ut.

Proof. We want to control DKL(p1||
Lemma 2.19 in [42] )

q1). It can be shown that (see equation above (25) in [43] or

(cid:90)

∂tDKL(pt||

qt) =

−

pt(dθ)(ut(θ)

vt(θ))

(
∇

·

−

ln pt(θ)

− ∇

ln qt(θ)).

(44)

Using Cauchy-Schwarz we can bound this by

(cid:18)(cid:90)

qt)

∂tDKL(pt||
We use the relation (see (3))

≤

pt(dθ)
|

(cid:19) 1

2 (cid:18)(cid:90)

ut(θ)

2
vt(θ)
|

−

pt(dθ)

|∇

ln pt(θ)

2

ln qt(θ)
|

− ∇

(cid:19) 1

2

.

(45)

ln(pt(ϕt(θ0)) = ln(p0(θ0))

(cid:90) t

−

0

(div us)(ϕs(θ0))ds,

which can be equivalently rewritten (setting θ = ϕtθ0) as
(cid:90) t

ln(pt(θ)) = ln(p0(ϕt

0θ))

(div us)(ϕt

sθ)ds.

−

0

We use the following relation for

ϕt
s

∇

ϕt

s(θ) = exp

∇

(cid:18)(cid:90) s

dτ (

∇

t

uτ )(ϕt

τ (θ))

(cid:19)

.

This relation is standard and can be directly deduced from the following ODE for

ϕt

s(θ) =

∂s∇

We can conclude that for 0

∇

≤

holds. We find

∂sϕt

s(θ) =

(us(ϕt

s(θ))) = (cid:0)(

us)(ϕt

s(θ))(cid:1)

· ∇

∇

s, t

≤

∇
1 the bound
ϕt

s(θ)

eL

| ≤

|∇

ϕt
s

∇
s(θ).

ϕt

ln(pt(θ))

|∇

(cid:12)
(cid:12)
(cid:12)
(cid:12)∇

=

|

ln(p0)(ϕt

0θ)

ln(p0)(ϕt

0θ)
|

≤ |∇

ϕt

0(θ)

· ∇
−
eL + L′eL,

(cid:90) t

(
∇

0

div us)(ϕt

sθ)

(cid:12)
(cid:12)
s(θ)ds
(cid:12)
(cid:12)

ϕt

· ∇

and a similar bound holds for qt. In words, we have shown that the score of pt at θ can be bounded by
the score of p0 of theta transported along the vector field ut minus a correction which quantifies the
change of score along the path. We now bound using the definition pt = (ϕt)
p0 and the assumption
∗
(41)
(cid:90)

(cid:90)

pt(dθ)

|∇

ln p0(ϕt

0(θ))

2 =
|

p0(dθ0)

|∇
Ep0(C1(1 +

ln p0(ϕt

)2)

θ0|
|

0ϕt(θ0))
|
1 (1 + Ep0 |

2 = Ep0 |∇
2)
θ0|

2C 2

≤

2

ln p0(θ0)
|
1 (1 + C 2

2C 2

2 ).

≤

≤

Similarly we obtain using q0 = p0

(cid:90)

pt(dθ)

ln q0(ψt

0θ)
|

2 =

|∇

p0(dθ0)

|∇

ln q0(ψt

0ϕtθ0)

2.
|

(cid:90)

20

(46)

(47)

(48)

(49)

(50)

(51)

(52)

(53)

0
(cid:90) t

eL

C 2
1

(cid:90)

C 2
1

≤

≤

≤

In words, to control the score of q integrated with respect to pt we need to control the distortion we
obtain when moving forward with u and backwards with v. We investigate ψt
0ϕt(θ0). We now show

First, by definition of ϕ, we find

∂hψt+h

t ϕt

t+h(θ)

|h=0 = ut(θ)

vt(θ).

−

|h=0 = ∂hϕt+hϕ−
To evaluate the second contribution we observe

t+h(θ)

∂hϕt

1
t (θ)

|h=0 = ut(ϕtϕ−

1

t (θ)) = ut(θ).

0 = ∂hθ

|h=0 = ∂hψt+h
1
= (∂hψt+h)ψ−
(θ)
t
= vt(θ) + ∂hψt+h

t

|h=0 = ∂hψt+hψ−
t+h)(θ)

t+h(θ)
|h=0 + ψt(∂hψ−
(θ)
|h=0

1

1
t+h(θ)
|h=0
|h=0 = vt(ψtψ−

t

1

(θ)) + ∂hψtψ−

1
t+h(θ)

|h=0

Now (54) follows from (55) and (56) together with ϕt

t = ψt
ψt

t = Id. Using (54) we find
((ut −

0)(ϕt(θ0))

·

vt)(ϕt(θ0))) .

|h=0 = (
∇

∂t(ψt

0ϕt)(θ0) = ∂h(ψt

0ψt+h

t ϕt

t+hϕt)(θ0)

Using (50) we conclude that

0ϕt(θ0)

ψt
|

θ0| ≤

−

(cid:12)
(cid:12)
(cid:12)
(cid:12)

(cid:90) t

∂sψs

(cid:12)
(cid:12)
0ϕs(θ0) ds
(cid:12)
(cid:12) ≤

(cid:90) t

(
∇
0 |

ψs

0)(ϕs(θ0))

us −

vs|

| · |

(ϕs(θ0)) ds

(54)

(55)

(56)

(57)

(58)

vs|
We use this and the assumption (41) to continue to estimate (53) as follows
(cid:90)

(ϕs(θ0)) ds.

us −

0 |

≤

(cid:90)

pt(dθ)

|∇

ln q0(ψt

2 =

0θ)
|

p0(dθ0)
(cid:90)

|∇

ln q0(ψt

0ϕt(θ0))

2

|

p0(dθ0)(1 +

ψt

0ϕt(θ0)

)2
|

|

p0(dθ0)(1 +
(cid:90)

|
p0(dθ0) (cid:0)

3C 2

1 + 3C 2
1

ψt

0ϕt(θ0)

θ0|

−

+

|

0ϕt(θ0)

ψt
|

−

θ0|

)2
θ0|
2 +

3C 2

1 (1 + Ep0 |

≤

2) + 3C 2

1 e2L

θ0|

(cid:90)

θ0|
|
(cid:18)(cid:90) t

2(cid:1)

(cid:19)2

p0(dθ0)

ds

us −

|

vs|

0

(ϕs(θ0))

.

(59)

Here we used (a + b + c)2
integral using Cauchy-Schwarz as follows

≤

3(a2 + b2 + c2) in the second to last step. We bound the remaining

(cid:90)

(cid:18)(cid:90) t

(cid:19)2

(cid:90)

(cid:18)(cid:90) t

p0(dθ0)

us −

0 |

(ϕs(θ0))

vs|

≤

t

≤

= t

(cid:90) t

0
(cid:90) t

ds

(cid:90)

(cid:90)

p0(dθ0)

us −
|

vs|

2(ϕs(θ0))

ds

ps(dθs)

us −
|

vs|

2(θs)

p0(dθ0)

ds

us −
|

vs|

0

(cid:19) (cid:18)(cid:90) t

2(ϕs(θ0))

(cid:19)

ds 12

0

0
(cid:90) 1

≤

0

(cid:90)

ds

ps(dθs)
|

us −

vs|

2(θs) = MSEp(u, v).

The last displays together imply
(cid:90)

pt(dθ)

|∇

ln q0(ψt

0θ)

2
|

≤

3C 2
1

(cid:0)1 + Ep0 |

θ0|

2 + e2L MSEp(u, v)(cid:1) .

21

(60)

(61)

Now we have all the necessary ingredients to bound the derivative of the KL-divergence. We control
the second integral in (45) using (51) (and again ((cid:80)4

4 (cid:80) a2

i ) as follows,

i=1 ai)2

≤

(cid:90)

pt(dθ)

|∇

ln pt(θ)

2

ln qt(θ)
|

− ∇

2

·

≤

22

·

L′

2e2L + 4e2L

(cid:90)

pt(dθ) (cid:0)

ln q0(ψt

0)θ)
|

2 +

|∇

|∇

ln p0(ϕt

0)θ)
|

2(cid:1) .

Using (52) and (61) we finally obtain

(cid:90)

pt(dθ)

|∇

ln pt(θ)

2

ln qt(θ)
|

8

·

≤

− ∇

L′

2e2L + C 2

1 e2L (cid:0)20(1 + C 2

2 ) + 12 MSEp(u, v)(cid:1)

for some constant C > 0. Finally, we obtain

C(1 + MSEp(u, v))

≤

DKL(p1||

q1) =

(cid:90) 1

0

dt ∂tDKL(pt||

qt)

(C(1 + MSEp(u, v)))

(C(1 + MSEp(u, v)))

1
2

1
2

(cid:90) 1

(cid:18)(cid:90)

dt

0
(cid:18)(cid:90) 1

0

(cid:90)

dt

ut(θ)
pt(dθ)
|

−

2

vt(θ)
|

ut(θ)
pt(dθ)
|

−

2

vt(θ)
|

(cid:19) 1

2

(cid:19) 1

2

(C(1 + MSEp(u, v)))

1

2 MSEp(u, v)

1

2 .

≤

≤

≤

(62)

(63)

(64)

C SBI Benchmark

In this section, we collect missing details and additional results for the analysis of the SBI benchmark
in Section 4.

C.1 Network architecture and hyperparameters

For each task and simulation budget in the benchmark, we perform a mild hyperparameter optimiza-
tion. We sweep over the batch size and learning rate (which is particularly important as the simulation
budgets differ by orders of magnitudes), the network size and the α parameter for the time prior
defined in Section 3.3 (see Tab. 2 for the specific values). We reserve 5% of the simulation budget for
validation and choose the model with the best validation loss across all configurations.

C.2 Additional results

We here provide various additional results for the SBI benchmark. First, we compare the performance
of FMPE and NPE when using the Maximum Mean Discrepancy metric (MMD). The results can
be found in Fig. 6. FMPE shows superior performance to NPE for most tasks and simulation
budgets. Compared to the C2ST scores in Fig. 4 the improvement shown by FMPE in MMD is more
substantial.

Fig. 7 compares the FMPE results with the optimal transport path from the main text with a comparable
score matching model using the Variance Preserving diffusion path [15]. The score matching results
were obtained using the same batch size, network size and learning rate as the FMPE network, while
optimizing for βmin ∈ {
. FMPE with the optimal transport path
}
clearly outperforms the score-based model on almost all configurations.

and βmax ∈ {

0.1, 1, 4
}

4, 7, 10

In Fig. 8 we compare FMPE using the architecture proposed in Section 3.2 with (t, θ)-conditioning
via gated linear units to FMPE with a naive architecture operating directly on the concatenated
(t, θ, x) vector. For the two displayed tasks the context dimension dim(x) = 100 is much larger
than the parameter dimension dim(θ)
, and there is a clear performance gain in using the
}
GLU conditioning. Our interpretation is that the low dimensionality of (t, θ) means that it is not
well-learned by the network when simply concatenated with x.

5, 10

∈ {

22

hyperparameter

hidden dimensions
number of blocks
batch size
learning rate
α (for time prior)

∈ {

4, . . . , 10

sweep values
2n for n
10, . . . , 18
2n for n
∈ {
1.e-3, 5.e-4, 2.e-4, 1.e-4
-0.25, -0.5, 0, 1, 4

2, . . . , 9

}

}

Table 2: Sweep values for the hyperparamters for the SBI benchmark. We split the configurations
according to simulation budgets, e.g. for 1000 simulations, we only swept over smaller values for
network size and batch size. The network architecture has a diamond shape, with increasing layer
width from smallest to largest and then decreasing to the output dimension. Each block consists of
two fully-connected residual layers.

Figure 6: Comparison of FMPE and NPE performance across 10 SBI benchmarking tasks [46]. We
here quantify the deviation in terms of the Maximum Mean Discrepancy (MMD) as an alternative
metric to the C2ST score used in Fig. 4. MMD can be sensitive to its hyperparameters [46], so we
use the C2ST score as a primary performance metric.

Fig. 9 displays the densities of the reference samples under the FMPE model as a histogram for all
tasks (extended version of Fig. 3). The support of the learned model q(θ
x) covers the reference
|
x), providing additional empirical evidence for the mass-covering behavior theoreti-
samples θ
|
cally explored in Thm. 1. However, samples from the true posterior distribution may have a small
density under the learned model, especially if the deviation between model and reference is high; see
Lotka-Volterra (bottom right panel). Fig. 10 displays P–P plots for two selected tasks.

p(θ

∼

Finally, we study the impact of our time prior re-weighting for one example task in Fig. 11. We
clearly see that our proposed re-weighting leads to increased performance by up-weighting samples
for t closer to 1 during training.

23

0.000.050.100.150.20GLGL-UGMTwoMoonsSLCP1031041050.00.20.40.6B-GLM103104105B-GLM-Raw103104105SLCP-D103104105SIR103104105LVNumberofSimulationsMMDNPEFMPEFigure 7: Comparison of FMPE with the optimal transport path (as used throughout the main paper)
with comparable models trained with a Variance Preserving diffusion path [15] by regressing on the
score (SMPE). Note that the SMPE baseline shown here is not directly comparable to NPSE [8, 9],
as this method uses Langevin steps, which reduces the dependence of the results on the vector field
for small t (at the cost of a tractable density).

Figure 8: Comparison of the architecture proposed in Section 3.2 with gated linear units for the
(t, θ)-conditioning (red) and a naive architecture based on a simple concatenation of (t, θ, x) (black).
FMPE with the proposed architecture performs substantially better.

24

0.60.81.0GL0.60.81.0GL-U0.60.81.0GM0.60.81.0TwoMoons0.60.81.0SLCP1031041050.60.81.0B-GLM1031041050.60.81.0B-GLM-Raw1031041050.60.81.0SLCP-D1031041050.60.81.0SIR1031041050.60.81.0LVNumberofSimulationsC2STSMPEFMPE1031041050.50.60.70.80.91.0B-GLM-Raw103104105SLCP-DNumberofSimulationsC2STGLUembeddingConcatenation∼

p(θ

Figure 9: Histogram of FMPE densities log q(θ
θ
|
reference samples θ
covering FMPE results. Nonetheless, reference samples may have a small density under q(θ
the validation loss is high, see Lotka-Volterra (LV).

x) and reference samples
|
x) for simulation budgets N = 103 (left), N = 104 (center) and N = 105 (right). The
x), indicating mass
x) are all within the support of the learned model q(θ
|
x), if

x) for samples θ

p(θ

q(θ

∼

∼

|

|

|

25

−20−15−10−5GLGL−20−15−10−5−20−15−10−5−15−10−5GL-UGL-U−15−10−5−15−10−50−6−4−202GMGM−5.0−2.50.02.55.0−505−5.0−2.50.02.5TwoMoonsTwoMoons−4−2024−2024−9−8−7−6SLCPSLCP−10.0−7.5−5.0−2.5−10−50−20−100B-GLMB-GLM−100−10−505−15−10−50B-GLM-RawB-GLM-Raw−10−505−10−505−10−8−6SLCP-DSLCP-D−10.0−7.5−5.0−2.5−10−50−7.5−5.0−2.50.0SIRSIR−6−4−202−6−4−202−10−505LVLV−10−505logq(θ|x)−5.0−2.50.02.55.0θ∼p(θ|x)θ∼q(θ|x)Figure 10: P-P plot for the marginals of the FMPE-posterior for the Two Moons (upper) and SLCP
(lower) tasks for training budgets of 103 (left), 104 (center), and 105 (right) samples.

Figure 11: Comparison of the time prior re-weighting proposed in Section 3.3 with a uniform prior
over t on the Two Moons task (Section 4). The network trained with the re-weighted prior clearly
outperforms the reference on all simulation budgets.

26

01p0.00.51.0CDF(p)01p0.00.51.0CDF(p)01p0.00.51.0CDF(p)01p0.00.51.0CDF(p)01p0.00.51.0CDF(p)01p0.00.51.0CDF(p)1031041050.50.60.70.80.91.0TwoMoonsNumberofSimulationsC2STpowerlawuniformD Gravitational-wave inference

We here provide the missing details and additional results for the gravitational wave inference problem
analyzed in Section 5.

D.1 Network architecture and hyperparameters

Compared to NPE with normalizing flows, FMPE allows for generally simpler architectures, since
the output of the network is simply a vector field. This also holds for NPSE (model also defined by a
vector) and NRE (defined by a scalar). Our FMPE architecture builds on the embedding network
developed in [7], however we extend the network capacity by adding more residual blocks (Tab. 3,
top panel). For the (t, θ)-conditioning we use gated linear units applied to each residual block, as
described in Section 3.2. We also use a small residual network to embed (t, θ) before applying the
gated linear units.

In this Appendix we also perform an ablation study, using the same embedding network as the
NPE network (Tab. 3, bottom panel). For this configuration, we additionally study the effect of
conditioning on (t, θ) starting from different layers of the main residual network.

D.2 Data settings

We use the data settings described in [7], with a few minor modifications. In particular, we use
the waveform model IMRPhenomPv2 [76–78] and the prior displayed in Tab. 4. Generation of the
training dataset with 5,000,000 samples takes around 1 hour on 64 CPUs. Compared to [7], we
reduce the frequency range from [20, 1024] Hz to [20, 512] Hz to reduce the computational load for
data preprocessing. We also omit the conditioning on the detector noise power spectral density (PSD)
introduced in [7] as we evaluate on a single GW event. Preliminary tests show that the performance
with PSD conditioning is similar to the results reported in this paper. All changes to the data settings
have been applied to FMPE and the NPE baselines alike to enable a fair comparison.

D.3 Additional results

Tab. 5 displays the inference times for FMPE and NPE. NPE requires only a single network pass
to produce samples and (log-)probabilities, whereas many forwards passes are needed for FMPE
to solve the ODE with a specific level of accuracy. A significant portion of the additional time
required for calculating (log-)probabilities in conjunction with the samples is spent on computing the
divergence of the vector field, see Eq. (3).

hyperparameter

residual blocks

residual blocks (t, θ) embedding
batch size
learning rate
α (for time prior)

residual blocks

residual blocks (t, θ) embedding
batch size
learning rate
α (for time prior)

values

3, 1024
3

×
3, 16

×

×

×
×

3, 2048
2048, 4096
128
3, 32
5, 64
16, 32, 64, 128, 256
4096
5.e-4
1

×

6, 512

8, 256

10,

×

×

×

4, 256

4, 128

4, 64

3,

×

×

×

×

4, 512

×

×

×

×
3, 16

2, 1024
3

2048
32
16, 32, 64, 128, 256
4096
5.e-4
1

Table 3: Hyperparameters for the FMPE models used in the main text (top) and in the ablation study
(bottom, see Fig. 12). The network is composed of a sequence of residual blocks, each consisting
of two fully-connected hidden layers, with a linear layer between each pair of blocks. The ablation
network is the same as the embedding network that feeds into the NPE normalizing flow.

27

Description

Parameter

Prior

3

5 /(m1 + m2)

component masses m1, m2
chirp mass
mass ratio
spin magnitudes
spin angles
time of coalescence
luminosity distance
reference phase
inclination
polarization
sky position

Mc = (m1m2)
q = m2/m1
a1, a2
θ1, θ2, ϕ12, ϕJL
tc
dL
ϕc
θJN
ψ
α, β

1
5

⊙

m2
, m1 ≥
(constraint)

[10, 120] M
[20, 120] M
⊙
[0.125, 1.0] (constraint)
[0, 0.99]
standard as in [79]
[
0.03, 0.03] s
−
[100, 1000] Mpc
[0, 2π]
[0, π] uniform in sine
[0, π]
uniform over sky

Table 4: Priors for the astrophysical binary black hole parameters. Priors are uniform over the
specified range unless indicated otherwise. Our models infer the mass parameters in the basis (Mc, q)
and marginalize over the phase parameter ϕc.

Fig. 12 presents a comparison of the FMPE performance using networks of the same hidden dimen-
sions as the NPE embedding network (Tab. 3 bottom panel). This comparison includes an ablation
study on the timing of the (t, θ) GLU-conditioning. In the top-row network, the (t, θ) conditioning is
applied only after the 256-dimensional blocks. In contrast, the middle-row network receives (t, θ)
immediately after the initial residual block. With FMPE we can achieve performance comparable to
1/3 of the network size (most of the NPE network parameters are in the
NPE, while having only
flow). This suggests that parameterizing the target distribution in terms of a vector field requires less
learning capacity, compared to directly learning its density. Delaying the (t, θ) conditioning until
the final layers impairs performance. However, the number of FLOPs at inference is considerably
reduced, as the context embedding can be cached and a network pass only involves the few layers
with the (t, θ) conditioning. Consequently, there’s a trade-off between accuracy and inference speed,
which we will explore in a greater scope in future work.

≈

Network Passes

Inference Time (per batch)

FMPE (sample only)
FMPE (sample and log probs)

NPE (sample and log probs)

248
350

1

26s
352s

1.5s

Table 5: Inference times per batch for FMPE and NPE on a single Nvidia A100 GPU, using the
training batch size of 4096. We solve the ODE for FMPE using the dopri5 discretization [80] with
absolute and relative tolerances of 1e-7. For FMPE, generation of the (log-)probabilities additionally
requires the computation of the divergence, see equation (3). This needs additional memory and
therefore limits the maximum batch size that can be used at inference.

Figure 12: Jensen-Shannon divergence between inferred posteriors and the reference posteriors for
GW150914 [73]. We compare two FMPE models with the same architecture as the NPE embedding
network, see Tab. 3 bottom panel. For the model in the first row, the GLU conditioning of (θ, t) is
only applied before the final 128-dim blocks. The model in the middle row is given the context after
the very first 2048 block.

28

m1m2a1a2t1t2φ12dLtcαδψFMPElateGLUFMPEearlyGLUNPE45.341.51.20.518.011.80.312.38.111.47.50.44.84.60.81.00.90.20.31.72.04.813.00.41.22.53.21.60.80.40.34.49.110.18.60.601020JSD[mnat]