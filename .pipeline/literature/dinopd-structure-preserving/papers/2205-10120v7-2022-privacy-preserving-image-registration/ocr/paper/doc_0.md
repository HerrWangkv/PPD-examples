Medical Image Analysis (2024)

Contents lists available at ScienceDirect

Medical Image Analysis

journal homepage: www.elsevier.com/locate/media

Privacy Preserving Image Registration

Riccardo Taielloa,b,c,∗, Melek ¨Onenb, Francesco Capanob, Olivier Humbertc, Marco Lorenzia,c

aEpione Research Group, Inria, Sophia Antipolis, France
bEURECOM, France
cUniversit´e Cˆote d’Azur, France

4
2
0
2

r
p
A
6
1

A R T I C L E I N F O

Article history:

]

V
C
.
s
c
[

Keywords: Image Registration,
Registration, Trustworthiness

Image

7
v
0
2
1
0
1
.
5
0
2
2
:
v
i
X
r
a

A B S T R A C T

Image registration is a key task in medical imaging applications, allowing to represent
medical images in a common spatial reference frame. Current approaches to image reg-
istration are generally based on the assumption that the content of the images is usually
accessible in clear form, from which the spatial transformation is subsequently esti-
mated. This common assumption may not be met in practical applications, since the
sensitive nature of medical images may ultimately require their analysis under privacy
constraints, preventing to openly share the image content. In this work, we formulate
the problem of image registration under a privacy preserving regime, where images are
assumed to be confidential and cannot be disclosed in clear. We derive our privacy
preserving image registration framework by extending classical registration paradigms
to account for advanced cryptographic tools, such as secure multi-party computation
and homomorphic encryption, that enable the execution of operations without leak-
ing the underlying data. To overcome the problem of performance and scalability of
cryptographic tools in high dimensions, we propose several techniques to optimize the
image registration operations by using gradient approximations, and by revisiting the
use of homomorphic encryption trough packing, to allow the efficient encryption and
multiplication of large matrices. We focus on registration methods of increasing com-
plexity, including rigid, affine, and non-linear registration based on cubic splines or
diffeomorphisms parametrized by time-varying velocity fields. In all these settings, we
demonstrate how the registration problem can be naturally adapted for accounting to
privacy-preserving operations, and illustrate the effectiveness of PPIR on a variety of
registration tasks.

© 2024 Elsevier B. V. All rights reserved.

1. Introduction

Image Registration is a crucial task in medical imaging ap-
plications, allowing to spatially align imaging features between
two or multiple scans. Registration methods are today a central
component of state-of-the-art methods for atlas-based segmen-
tation (Shattuck et al., 2009; Cardoso et al., 2013), morpholog-
ical and functional analysis (Dale et al., 1999; Ashburner and

∗Corresponding author: riccardo.taiello@inria.fr

Friston, 2000), multi-modal data integration (Heinrich et al.,
2011), and longitudinal analysis (Reuter et al., 2010; Ashburner
and Ridgway, 2013). Typical registration paradigms are based
on a given transformation model (e.g. affine or non-linear), a
cost function and an associated optimization routine. A large
number of image registration approaches have been proposed
in the literature over the last decades, covering a variety of as-
sumptions on the spatial transformations, cost functions, im-
age dimensionality and optimization strategy (Schnabel et al.,
Image registration is the workhorse of many real-life
2016).

 
 
 
 
 
 
2

Riccardo Taiello et al. / Medical Image Analysis (2024)

medical imaging software and applications, including public
web-based services for automated segmentation and labelling
of medical images. Using these services generally requires up-
loading and exchanging medical images over the Internet, to
subsequently perform image registration with respect to one or
multiple (potentially proprietary) atlases. Besides these classi-
cal medical imaging use-cases, emerging paradigms for collab-
orative data analysis, such as Federated Learning (FL) (McMa-
han et al., 2017), have been proposed to enable analysis of
medical images in multicentric scenarios for performing group
analysis (Gazula et al., 2021) and distributed machine learning
(Kaissis et al., 2021; Zerka et al., 2020). However, in these set-
tings, typical medical imaging tasks such as spatial alignment
and downstream operations are generally not possible without
disclosing the image information.

Due to the evolving juridical landscape on data protection,
medical image analysis tools need to be adapted to guarantee
compliance with regulations currently existing in many coun-
tries, such as the European General Data Protection Regulation
(GDPR) 1, or the US Health Insurance Portability and Account-
ability Act (HIPAA)2. Medical imaging information falls within
the realm of personal health data (Lotan et al., 2020) and its sen-
sitive nature should ultimately require the analysis under pri-
vacy preserving constraints, for instance by preventing to share
the image content in clear form.

Advanced cryptographic tools hold great potential in sensi-
tive data analysis problems (e.g., (Lauter, 2021)). Examples of
such approaches are Secure-Multi-Party-Computation (MPC)
(Yao, 1982) and Homomorphic Encryption (HE) (Rivest et al.,
1978). While MPC allows multiple parties to jointly compute
a common function over their private inputs and discover no
more than the output of this function, HE enables computation
on encrypted data without disclosing either the input data or the
result of the computation.

This work presents privacy-preserving image registration
(PPIR), a new methodological framework allowing image reg-
istration under privacy constraints. To this end, we reformulate
the image registration problem to integrate cryptographic tools,
namely MPC or FHE, thus preserving the privacy of the image
data. Due to the well-known scalability issues of such cryp-
tographic techniques, we investigate strategies for the practical
use of PPIR. In our experiments, we evaluate the effectiveness
of PPIR on a variety of registration tasks and medical imaging
modalities. Our results demonstrate the feasibility of PPIR and
pave the way for the application of secured image registration
in sensitive medical imaging applications.

2. Background

Given images I, J : Rd (cid:55)→ R, image registration (IR) aims
at estimating the parameters θ of a spatial transformation Wθ ∈
Rd (cid:55)→ Rd, either linear or non-linear, maximizing the spatial

1https://gdpr-info.eu/
2https://www.hhs.gov/hipaa/index.html

Algorithm 1 IR via Gauss-Newton optimization
Input:

▷ Moving image I
▷ Template image J
▷ Distance function f
▷ Spatial transformation Wθ, parameterized by θ
▷ convergence threshold ϵ

Output:

▷ Transformed image I(Wθ) after convergence is reached

1: function ImageRegistration(I, J, Wθ):
θ ←− InitializeParameters()
2:
repeat
3:
4:
5:

e ←− f (cid:2)I(Wθ), J(cid:3)
G ←− ∂ f
∂θ
H ←− ∂2 f
∂θ2
∆θ ←− H−1 · G
θ ←− θ + ∆θ

until ∥∆θ∥ ≤ ϵ

6:
7:
8:
9:

return I(Wθ), e

10:
11: end function

overlap between J and the transformed image I(Wθ), by mini-
mizing a registration loss function f :

θ∗ = argminθ f (I(Wθ(x)), J(x)) .

(1)

The loss f can be any similarity measure, e.g., the Sum of
Squared Differences (SSD), the negative Mutual Information
(MI), or normalized cross correlation (CC). Equation (1) can
be typically optimized through gradient-based methods, where
the parameters θ are iteratively updated until convergence. In
particular, when using a Gauss-Newton optimization scheme
(Algorithm 1), the update of the spatial transformation can be
computed through Equation (2):

∆θ = H−1 · G,

(2)

∂θ is the Jacobian and H = ∂2 f

where G = ∂ f
∂θ2 the Hessian of f .
Besides the Gauss-Newton schemes proposed in the field of IR
(Pennec et al., 1999; Modersitzki, 2009), gradient-based tech-
niques are classically adopted to solve the IR task, for exam-
ple in diffeomorphic image registration problems (Ashburner,
2007; Avants et al., 2011).

In all these cases we consider a scenario with two parties,
party1, and party2, whereby party1 owns image I and party2
owns image J. The parties wish to collaboratively optimize the
image registration problem without disclosing their respective
images to each other. We assume that only party1 has access
to the transformation parameters θ and that it is also responsi-
ble for computing the update at each optimization step. In what
follows, we introduce the basic notation to develop PPIR based
on different registration frameworks. We focus on registra-
tion methods of increasing complexity, including (i) rigid, (ii)
affine, and (iii) non-linear registration based on cubic splines or
diffeomorphisms parametrized by time-varying velocity fields
(large deformation diffeomorphic metric mapping, LDDMM)
(Beg et al., 2005). In all these settings, we demonstrate how the

Riccardo Taiello et al. / Medical Image Analysis (2024)

3

(a) PPIR(MPC)

(b) PPIR(FHE)-v1

Fig. 1: Optimization of SSD loss: proposed framework to compute matrix-vector multiplication S T · J based on PPIR(MPC) and PPIR(FHE)-v1.

registration problem can be naturally adapted for accounting to
privacy-preserving operations, and illustrate the effectiveness of
PPIR on a variety of registration tasks.

3. Analysis of classical IR loss functions under a privacy-

preserving perspective

In this section we review typical loss functions used in image
registration, and analyze the related requirements for privacy-
preserving optimization.

The solution to this problem requires the joint availability of
both images I and J, as well as of the gradients of I and of
Wθ. In a privacy-preserving setting, this information cannot be
disclosed, and the computation of Equation (2) is therefore im-
possible. We note that to calculate the update of the registra-
tion ∆θ of Equation (2), the only operation that requires the
joint availability of information from both parties is the term
R = (cid:80)
x S (x) · J(x), which can be computed a matrix-vector
multiplication of vectorized quantities R = S T · J.

3.2. Mutual Information

3.1. Optimization of SSD loss

A typical loss function to be optimized during the registra-
tion process is the sum of squared intensity differences (SSD)
evaluated on the set of image coordinates:

Mutual Information quantifies the joint information content
between the intensity distributions of the two images. This
is calculated from the joint probability distribution function
(PDF):

[I(Wθ(x)) − J(x)]2

(3)

MI(I, J, θ) = argminθ −

p(r, t; θ) log

(cid:32)

(cid:33)

p(r, t; θ)
p(r; θ)p(t)

(7)

(cid:88)

r,t

SSD(I, J, θ) = argminθ

(cid:88)

x

with Jacobian:

G =

(cid:88)

x

S (x) · (I(Wθ(x)) − J(x)),

where the quantity

S (x) = ∇I(x)

∂Wθ(x)
∂θ

quantifies image and transformation gradients, and

(cid:88)

(cid:32)
∇I(x)

H =

x

∂Wθ(x)
∂θ

(cid:33)T (cid:32)

∇I(x)

(cid:33)

∂Wθ(x)
∂θ

(4)

(5)

(6)

is the second order term obtained from Equation (3) through
linearization (Pennec et al., 1999; Baker and Matthews, 2004).

where, given Nr and Nt the maximum intensity for respectively
I and J, we define r ∈ [0; Nr − 1] ⊆ N and t ∈ [0; Nt − 1] ⊆ N
as the range of discretized intensity values of I and J, respec-
tively. A Parzen window (Parzen, 1961) is used to generate
continuous estimates of the underlying intensity distributions,
thereby reducing the effects of quantization from interpolation,
I : R (cid:55)→ [0, 1] be
and discretization from binning the data. Let ψ3
J : R (cid:55)→ [0, 1] be a zero-
a cubic spline Parzen window, and let ψ0
order spline Parzen window. The smoothed joint histogram of
I and J (Viola and Wells III, 1997; Mattes et al., 2003) is given
by:

p(r, t; θ) =
(cid:88)
1
Nx

ψ3
I

x

(cid:32)
r −

I(Wθ(x)) − I(Wθ(x))◦
∆br

(cid:33)

(cid:32)

· ψ0
J

t −

(cid:33)

J(x) − J◦
∆bt

4

Riccardo Taiello et al. / Medical Image Analysis (2024)

In the above formula, the intensity values of I and J are nor-
malized by their respective minimum (denoted by I(Wθ)◦ and
J◦), and by the bin size (respectively ∆br and ∆bt), to fit into
the specified number of bins (br or bt) of the intensity distribu-
tion. The final value for p(r, t; θ) is computed by normalizing by
Nx, the number of sampled voxels. Marginal probabilities are
simply obtained by summing along one axis of the PDF, that
is, p(r) = (cid:80)
r p(r, t; θ). Let the matrices
I ∈ RNx×Nr and B0
A3

t p(r, t; θ) and p(t) = (cid:80)

J ∈ RNx×Nt be defined as:

I (x, r; θ) = ψ3
A3

I

(cid:32)
r −

I(Wθ(x)) − I(Wθ)◦
∆br

(cid:33)

and

J(x, t) = ψ0
B0

J

(cid:32)

t −

J(x) − J◦
∆bt

(cid:33)

,

the discretized joint PDF can be rewritten in a matrix form via
the multiplication:

P = 1
Nx

· (A3

I )T · B0
J.

(8)

The first derivative of the joint PDF is calculated as follows

(Dowson and Bowden, 2006):

∂p(r, t; θ)
∂θ

= −

1
Nx

·

(cid:88)

x

B0
J(x, t) ·

∂ψ3
I (ϵ)
∂ϵ

·

∂ϵ
∂I(Wθ(x))

·

∂I(Wθ(x))
∂θ

where ϵ = ϵ(x, r; θ) = r − I(Wθ(x))−I(Wθ)◦
spline. We also introduce the tensor C3
I (x, r; θ) = ∂ψ3
I (ϵ)
∂I(Wθ(x)) · ∂I(Wθ(x))
C3
·
∂ϵ
first derivative as:

∆br

∂θ

∂ϵ

is the input of the cubic
I ∈ RNx×Nr×|θ| defined as
, to write the discretized

P′ = −

1
Nx

· (B0

J)T · C3
I ,

(9)

where P′ ∈ RNt×Nr×|θ|.

3.3. Cross Correlation with Advanced Normalization Tools

In the Advanced Normalization Tools (ANTs) introduced by
(Avants et al., 2008), the normalized cross-correlation (CC) loss
was specified in the context of diffeomorphic image registra-
tion. Let ϕ define a diffeomorphism over the domain Ω = Rd,
parameterized by a time-varying velocity field v(x, t).
In the
ANTs setting, inverse consistency is obtained by optimizing the
CC loss with respect to both forward and backward (inverse)
transformations, here denoted by ϕ1(x, t) and ϕ2(x, t), and pa-
rameterized by velocity fields v1(x, t) and v2(x, t) respectively.
In particular, both images I and J are simultaneously warped
towards a “half-way” space, to obtain I1 = I(ϕ1(x, 0.5)) and
J2 = J(ϕ2(x, 0.5)). The CC loss is thus defined as:

CC(x) =

(cid:80)

xi( ¯I(xi), ¯J(xi))2

(cid:80)
xi( ¯I(xi))2 (cid:80)

xi( ¯J(xi))2

= D2
EF

where ¯I(xi) = (cid:0)I1(xi) − µI1 (x)(cid:1) and ¯J(xi) = (cid:0)J2(xi) − µJ2 (x)(cid:1)
quantify the images appearance at location xi, with respect to
the average intensity µI1 and µJ2 measuread in a local window
of size M. Coherently with the LDDMM formulation, the vari-
ational optimization problem is defined as:

ECC = inf
ϕ1
(cid:90)

inf
ϕ2

(cid:90) 0.5

t=0

∥v1(x, t)∥2
L

+ ∥v2(x, t)∥2

L dt

+

CC(x) dΩ.

Ω

where L is a linear operator prescribing a norm on the velocity
fields acting as a regularizer. The equation for the derivative of
the forward update is given by:

∇ϕ1(x,0.5)CC(x) = 2Lv1(x, 0.5) + 2D
EF
(cid:19)
¯I(x)

¯J(x) −

×

(cid:18)

(10)

|Dϕ1|∇ ¯I(x),

The Jacobian of the MI is obtained from the chain rule and

takes the form:

while the derivative of the backward update is analogously
given by:

G = P′ log

(cid:33)

,

(cid:32)

P
PI

while the linearized Hessian (Dowson and Bowden, 2007) can
be written as:

H = P′T P′

(cid:32)

1
P

−

(cid:33)

,

1
PI

where PI = (cid:80)
marginal PDF of the moving image.

t p(r, t, θ) is a vector which defines the discretized

The derivatives can be easily calculated from the properties
of B-splines since we have
2 ). In a
privacy-preserving scenario, to calculate the update of the reg-
istration ∆θ of Equation (7), two operations require the joint
availability of information from both parties, which are the ma-
trix P of Equation (8) and the matrix P′ of Equation (9).

2 ) − ψ2

I (ϵ + 1

I (ϵ − 1

= ψ2

∂ψ3
I
∂ϵ

∇ϕ2(x,0.5)CC(x) = 2Lv2(x, 0.5) + 2D
EF
(cid:19)
¯J(x)

¯I(x) −

×

(cid:18)

(11)

|Dϕ2|∇ ¯J(x)

In privacy-preserving scenario, the sensitive terms carrying pri-
¯I(x)) and ( ¯I(x) −
vate image information are 2D
E
D
therefore be computed in a privacy-
F
preserving regime.

¯J(x)), which must

EF , ( ¯J(x) − D

4. Building blocks for Secure Computation

After introducing in Section 1 the IR optimization problem
and the related functionals addressed in this work, in this sec-
tion, we review the standard privacy-preserving techniques that
will be employed to develop PPIR.

D
E

D
F

4.1. Secure Multi-Party Computation

5.1. PPIR based on SSD

Riccardo Taiello et al. / Medical Image Analysis (2024)

5

Introduced by (Yao, 1982), MPC is a cryptographic tool that
allows multiple parties to jointly compute a common function
over their private inputs (secrets) and discover no more than the
output of this function. Among existing MPC protocols, addi-
tive secret sharing consists of first splitting every secret s into
additive shares ⟨s⟩i, such that (cid:80)n
i=1⟨s⟩i = s, where n is the num-
ber of collaborating parties. Each party i receives one share
⟨s⟩i and executes an arithmetic circuit in order to obtain the
final output of the function. In this paper, we adopt the two-
party computation protocol defined in SPDZ (Damgard et al.,
2011), whereby the actual function is mapped into an arithmetic
circuit and all computations are performed within a finite ring
with modulus Q. Additions consist of locally adding shares of
secrets, while multiplications require interaction between par-
ties. Following (Damgard et al., 2011), SPDZ defines: MPC-
Mul to compute element-wise multiplication, MPCDot to com-
pute matrix-vector multiplication and MPCMatMul to compute
matrix-matrix multiplication. These operations are performed
within an honest but curious protocol.

4.2. Homomorphic Encryption

Initially introduced by Rivest et al. in (Rivest et al., 1978),
Homomorphic Encryption (HE) enables the execution of oper-
ations over encrypted data without disclosing either the input
data or the result of the computation. Hence, party1 encrypts
the input with its public key and sends the encrypted input to
party2. In turn, party2 evaluates a circuit over this encrypted
input and sends the result, which still remains encrypted, back
to party1 which can finally decrypt them. Among various HE
schemes, CKKS (Cheon et al., 2017) supports the execution of
all operations on encrypted real values and is considered a lev-
elled homomorphic encryption (LHE) scheme. The supported
operations are: Sum (+), Element-wise multiplication (∗) and
DotProduct. With CKKS, an input vector is mapped to a poly-
nomial and further encrypted with a public key in order to ob-
tain a pair of polynomials c = (c0, c1). The original function
is further mapped into a set of operations that are supported by
CKKS, which are executed over c. The performance and se-
curity of CKKS depend on multiple parameters including the
degree of the polynomial N, which is usually sufficiently large
(e.g. N = 4096, or N = 8192).

5. Methods: from IR to PPIR

In this section, we describe the privacy preserving variants
of the three IR methods described in Section 3. We pro-
pose two versions of PPIR for SSD according to the underly-
ing cryptographic tool, namely PPIR(MPC), integrating MPC,
and PPIR(FHE), integrating FHE. In the case of MI and NCC,
we focus on the design of the MPC-variant, due to the non-
negligible computational overhead of FHE in these applica-
tions. Finally, we also study rigid point cloud registration and
describe its privacy-preserving variant in Appendix A 1.

As mentioned in Section 3.1, when optimizing the SSD cost,
the only sensitive operation that must be jointly executed by the
parties is the matrix-vector multiplication: R = S T · J, where
S T is only known to party1 and J to party2. Figure 1 illustrates
how cryptographic tools are employed to ensure privacy during
registration.

With MPC (Figure 1a), party1 secretly shares the matrix S T
to obtain (⟨S ⟩1, ⟨S ⟩2), while party2 secretly shares the image J
to obtain (⟨J⟩1, ⟨J⟩2). Each party also receives its correspond-
ing share, so that party1 holds (⟨S ⟩1, ⟨J⟩1) and party2 holds
(⟨S ⟩2, ⟨J⟩2). The parties execute a circuit with MPCMul oper-
ations to calculate the 2-party dot product between S T and J.
The parties further synchronize to allow party1 to obtain the
product and finally to calculate ∆θ (Equations (4) and (6)).

When using FHE (Figure 1b), party2 first uses a FHE key k to
encrypt J and obtains ⟦J⟧ ← Enc(k, J). This encrypted image is
sent to party1, who computes the encrypted matrix-vector mul-
tiplication ⟦R⟧. In this framework, only vector J is encrypted,
and therefore party1 executes scalar multiplications and addi-
tions in the encrypted domain only (which are less costly than
multiplications over two encrypted inputs). The encrypted re-
sult ⟦R⟧ is sent back to party2, which can obtain the result by
decryption: R = Dec(k, ⟦R⟧). Finally, party1 receives R in clear
form and can therefore compute ∆θ.

Thanks to the privacy and security guarantees of these cryp-
tographic tools, during the entire registration procedure, the
content of the image data S and J is never disclosed to the op-
posite party.

5.2. PPIR based on MI

= (⟨B0

J⟩2, ⟨A3

I )T = (⟨A3
J⟩1, ⟨B0

In the MPC variant to calculate the joint PDF P in a privacy-
preserving manner, the quantity A3
I is only known to party1,
while the quantity B0
J to party2 (Section 3.2). With refer-
ence to Supplementary Figure A1a, party1 secretly shares ma-
I ⟩1, ⟨A3
trix (A3
I ⟩2), while party2 does the same with
B0
J⟩2). Each party also receives its correspond-
J
I ⟩1, ⟨B0
ing share: party1 now holds (⟨A3
J⟩1), and party2 holds
I ⟩2). The parties execute a circuit with MPCMatMul
(⟨B0
operation to calculate the 2-party matrix multiplication between
(A3
J. The next operation carried out in the privacy-
preserving setting is the computation of the first derivative of
Equation (9). Supplementary Figure A1b illustrates the MPC
variant, where party1 only knows C3
I and party2 only knows B0
J.
= (⟨C3
Initially, party1 secretly shares the matrix C3
I ⟩1, ⟨C3
I ⟩2),
I
J)T = (⟨B0
while party2 does the same with (B0
J⟩1, ⟨B0
J⟩2). Each
party also receives its corresponding share, namely: party1
holds (⟨C3
J⟩1) and party2 holds (⟨C3
J⟩2). The par-
ties also execute a circuit with the MPCMatMul between (B0
J)T
and C3
I .

I )T and B0

I ⟩1, ⟨B0

I ⟩2, ⟨B0

5.3. PPIR based on ANTS CC loss

According to Section 3.3, party1 has access to ¯I and E,
whereas party2 has access to ¯J and F. The computation begins
with the optimization of the CC term, specifically the quantity

6

Riccardo Taiello et al. / Medical Image Analysis (2024)

Dataset

Dimension

Modality Registration Type

Loss function

PETs

2D Point Cloud
2D Whole body PET
2D Brain MRI
3D Brain MRI and PET
3D Abdomen MR and CT 192 × 160 × 192 voxels

Mono
193 points
1260 × 1090 voxels
Mono
121 × 121 voxels
Mono
180 × 256 × 256 and 160 × 160 × 96 voxels Multi
Multi

SSD
Rigid
Affine/Cubic splines
SSD
SSD
Cubic splines
Affine
MI
Diffeomorphic (ANTs) CC

MPC and FHE-v1
MPC+URS/GMS, FHE-v1 and v2 + URS/GMS
MPC, FHE-v1 and v2
MPC
MPC

Table 1: Overview of the datasets used in the study. PETs: Privacy Enhancing Technologies.

2D
EF . In the initial phase, as illustrated in Supplementary Fig-
ure A2a, party1 and party2 secretly share ¯I = (⟨ ¯I⟩1, ⟨ ¯I⟩2), and
¯J = (⟨ ¯J⟩1, ⟨ ¯J⟩2), respectively. The subsequent multiplication of
these shared values results in the computation of the shares of
D = (⟨D⟩1, ⟨D⟩2), which is never reconstructed. In the next step
of the protocol, party1 secretly shares 1
E , and party2 secretly
shares 1
E , and
F , both parties collectively obtain the final value of 2D
1
EF .

F . Through a multiplication of the shares of D, 1

The other two terms that need to be jointly computed are the
third term of Equation (10) and (11), namely ( ¯J − D
¯I) and ( ¯I −
E
¯J), reported in Supplementary Figure A2b. In the case of ( ¯J −
D
F
¯I), party1 secretly shares ¯I and 1
D
E , while party2 secretly shares
E
¯J. Since both parties already have access to the share of D from
the previous protocol, they proceed to multiply the secret shares
E , and ¯I. Subsequently, they subtract the result from ¯J to
of D, 1
¯J). The computation of ( ¯J − D
obtain the final value of ( ¯I − D
¯I)
F
E
is analogous to the one of ( ¯I − D
¯J), where party2 secretly shares
F
E , and both parties participate to the multiplication of D, 1
1
E , and
¯J. The resulting value is finally subtracted from ¯I, yielding the
final result ( ¯J − D
E

¯I).

5.4. Protocols enhancement for SSD loss

Effectively optimizing Equation (1) with MPC or FHE is
particularly challenging, due to the computational bottleneck
of these techniques when applied to large-dimensional objects
(Haralampieva et al., 2020; Benaissa et al., 2021), notably af-
fecting the computation time and the occupation of communi-
cation bandwidth between parties. Because cryptographic tools
introduce a non-negligible overhead in terms of performance
and scalability, in this section we introduce specific techniques
to optimize the underlying image registration operations.

5.4.1. Gradient sampling

Since the registration gradient is generally driven mainly by
a fraction of the image content, such as the image boundaries
in the case of SSD cost, a reasonable approximation of Equa-
tions (4) and (6) can be obtained by evaluating the cost only
on relevant image locations. This idea has been introduced in
medical image registration (Viola and Wells III, 1997; Mattes
et al., 2003; Sabuncu and Ramadge, 2004), and here is adopted
to optimize Equation (3) by reducing the dimensionality of the
arrays on which encryption is performed. We test two different
techniques: (i): Uniformly Random Selection (URS), proposed
by (Viola and Wells III, 1997; Mattes et al., 2003), in which a
random subset of dimension l ≤ d of spatial coordinates is sam-
pled at every iteration with uniform probabilities, p(x) = 1
d ;
and (ii): Gradient Magnitude Sampling (GMS) (Sabuncu and
Ramadge, 2004), which consists of sampling a subset of coor-
dinates with probability proportional to the norm of the image

gradient, p(x) ∼ ∥∇I(x)∥. We note that gradient sampling is
not necessary for computing the MI since in Equation (8) the
computation is already performed on a subsample of the image
voxels.

5.4.2. Matrix partitioning in FHE
We now describe additional

improvements dedicated to
(i)
PPIR(FHE) and propose two versions of this solution:
PPIR(FHE)-v1 implements an optimization of the matrix-
vector multiplication by partitioning the vector image into a
vector of submatrices, whereas (ii) PPIR(FHE)-v2 enhances the
workload of the parties by fairly distributing the computation
among them.

PPIR(FHE)-v1. We introduce here a novel optimization ded-
icated to PPIR with FHE, in particular when the CKKS algo-
rithm is adopted. CKKS allows multiple inputs to be packed
into a single ciphertext to decrease the number of homomor-
phic operations. To optimize matrix-vector multiplication, we
propose to partition the image vector J into k sub-arrays of di-
mension l, and the matrix S T into k sub-matrices of dimension
|θ| × l. Once all sub-arrays Ji are encrypted, we propose to itera-
tively apply DotProduct as proposed by Benaissa et al. (2021),
between each sub-matrix and corresponding sub-array; these
intermediate results are then summed to obtain the final result,
namely: ⟦R⟧ = (cid:80)K

i=0 DotProduct (cid:16)⟦JT

(cid:17) = S T · ⟦J⟧.

⟧, S i

i

PPIR(FHE)-v2. In addition to packing multiple inputs into a
single ciphertext, the application of FHE to PPIR can be op-
timized by more equally distributing the workload among the
two parties. We note that in PPIR(FHE)-v1, party1 is in charge
of computing the matrix-vector multiplication entirely while
party2 only encrypts the input and decrypts the result. Follow-
ing Supplementary Figure A4, PPIR(FHE)-v2 starts by splitting
the matrix S of party1 into two sub-matrices S 1 and S 2 using
the operation splith,K. This operation partitions the matrix into
K equally-sized sub-matrices. Next, the operation f latten is
1 and S ′
applied to S 1 and S 2, obtaining vectors S ′
2 respectively.
Then, party1 encrypts S ′
2 and sends it to party2, which subse-
quently applies to its vector J the operation splitv,K, obtaining
J1 and J2. This operation splits J into K equally-sized parti-
tions, and it executes the operation replicated to J1 and J2, ob-
taining J′
2 respectively. party2 encrypts J′
1 and sends it to
party1. Both parties then iteratively perform the element-wise
multiplication ∗ and sum up the results of the different partitions
using the primitive S umi,k. The protocol doesn’t rely anymore
on DotProduct and leads to a significant gain in computational
load.

1 and J′

Riccardo Taiello et al. / Medical Image Analysis (2024)

7

6. Experiments & Results

We demonstrate and assess the different versions of PPIR il-
lustrated in Section 3 on a variety of image registration prob-
lem, namely: (i) SSD for rigid transformation of point cloud
data, (ii) SSD with linear and non-linear alignment of whole
body positron emission tomography (PET) data; (iii) SSD and
MI for mono- and multimodal linear alignment of MRI and PET
brain scans; (iv) diffeomorphic non-linear registration with CC
of multimodal abdomen data from CT and MRI scans. Experi-
ments are carried out on 2D (mainly for the SSD case) and 3D
imaging data. In Table 1 is reported an overview of the datasets
used specifying their dimensions, modality, registration type,
loss functions, and the Privacy Enhancing Technologies (PETs)
employed.

6.1. Experimental data

Point Cloud Data. We showcase rigid registration on 2D
point cloud data representing the corpus callosum, as presented
in Vachet et al. (2012), with a set size n = 193. The regis-
tration loss here considered is SSD between point coordinates
(additional details are provided in Appendix A 1).

Whole body PET data. The dataset considered for linear
and non-linear registration with SSD consists of 18-Fluoro-
Deoxy-Glucose (18FDG) whole body PET scans. The images
are a frontal view of the maximum intensity projection recon-
struction, obtained by 2D projection of the voxels with the high-
est intensity across views (1260 × 1090 pixels).

Brain MRI and PET data. This dataset regroups brain
MRI and PET images obtained from the Alzheimer’s Disease
Neuroimaging Initiative (Mueller et al., 2005). MRI data were
processed via a standard processing pipeline to estimate gray
matter density maps (Ashburner and Friston, 2000). Non-linear
registration was carried out on the extracted mid-coronal slice,
of dimension 121 × 121 pixels. For 3D multimodal linear regis-
tration with MI, we use both MRI images and PET images, with
respective dimension of 180 × 256 × 256 and 160 × 160 × 96
voxels.

Abdomen MR and CT data. The multimodal dataset
Abdomen-MR-CT (Hering et al., 2022) was used for experi-
ments with ANTs registration based on CC. The data was com-
piled from public studies of the cancer imaging archive (TCIA)
Clark et al. (2013) that contains 8 paired scans of MRI and CT
from the same patients. The data have an isotropic resolution of
2mm and a voxel dimension of 192 × 160 × 192. They also pro-
vide 3D segmentation masks for the liver, spleen, and left and
right kidney. All scans were pre-aligned by groupwise affine
registration.

6.2. Experimental Details

In order to avoid local minima and to decrease computa-
tion time, we use a hierarchical multiresolution optimization
scheme. The scheme involves M resolution steps, denoted as
r1 . . . rM. At each resolution step rm, the input data is down-
sampled by a scaling factor m, where m ∈ [1 . . . M]. The qual-
ity of PPIR is assessed by comparing the registration results
with those obtained with standard registration on clear images

(Clear). The metrics considered are the difference in image
intensity at optimum (for SSD), the total number of iterations
required to converge, and the displacement root mean square
difference (RMSE) between Clear and PPIR. We also evaluate
the performance of PPIR in terms of average computation (run-
ning time) and communication (bandwidth) across iterations.
In the multimodal Abdomen data, the quality of the registration
result was assessed by the overlap across the labeled anatomical
regions, quantified by the DICE score.

Point Cloud Data. The registration protocol here adopted
is detailed in Appendix A 1. For MPC we set as the prime
modulus Q = 232. For PPIR(FHE), we define the polynomial
degree modulus as N = 4096.

Whole body PET data. Whole-body PET image alignment
was first performed by optimizing the transformation Wθ in
Equation (1) with respect to affine registration parameters. The
multiresolution steps used are r1, r5, r10, r20. A second whole-
body PET image alignment experiment was performed by non-
linear registration, without gradient approximation based on a
cubic spline model (one control point every four pixels along
both dimensions), with multiresolution steps r1,r2,r5,r10,r20 and
r30. Concerning the PPIR framework, transformations were op-
timized for both MPC and FHE by using gradient approxima-
tion (Section 5) using the same sampling seed for each test. For
MPC we set as the prime modulus Q = 232. For PPIR(FHE),
we define the polynomial degree modulus as N = 4096, and
set the resizing parameter D to optimize the trade-off between
run-time and bandwidth. Since D needs to be a divisor of the
image size image data we set D = 128.

Brain MRI and PET data. The registration of brain gray
matter density images was performed by non-linear registra-
tion based on SSD, without gradient approximation, based on a
cubic spline model (one control point every five pixels along
both dimensions), with multiresolution steps r1 and r2. For
PPIR(MPC) we use the same configuration defined in the pre-
vious section, while for PPIR(FHE) we use the same N and we
set D = 121.

We tested PPIR with MI for multi-modal 3D affine image
registration between PET and MRI brain scans where, in ad-
dition to varying the multi-resolution steps, a Gaussian blur-
ring filter is applied to the images with a kernel that narrows as
multi-resolution proceeds. The kernel size at different resolu-
tions, denoted with σ1 . . . σM, is used to control the amount of
blurring applied to the image at each step of the multi-resolution
process. The multiresolution steps applied are r5 and r10, with
10% of the image’s pixels utilized as the number of subsample
pixels (Nx). Gaussian image blurring is applied with a degree
of σ5 = 1 and σ10 = 3. For MPC we set as the prime modulus
Q = 264.

Abdomen MRI and CT data. We tested PPIR with CC for
multi-modal 3D ANTs image registration between MRI and CT
abdomen scans. The multiresolution steps applied are r3, r2 and
r1 and the CC window size M = 5 × 5 × 5.

Implementation Details. The PPIR framework for the SSD is
implemented using two state-of-the-art libraries: PySyft (Ryf-
fel et al., 2018), which provides SPDZ two-party computation,

8

Riccardo Taiello et al. / Medical Image Analysis (2024)

SSD Affine registration and efficiency metrics

Solution
Clear
PPIR(MPC)
Clear + URS
PPIR(MPC) + URS
PPIR(FHE)-v1 (D = 128) + URS
PPIR(FHE)-v2 (D = 128) + URS
Clear + GMS
PPIR(MPC) + GMS
PPIR(FHE)-v1 (D = 128) + GMS
PPIR(FHE)-v2 (D = 128) + GMS

Intensity Error (SSD) Num. Interation Displacement RMSE (mm) Time party1 (s) Time party2 (s) Comm. party1 (MB) Comm. party2 (MB)
4.34 ± 0.0
4.34 ± 0.0
4.38 ± 0.0
4.34 ± 0.0
4.34 ± 0.10
4.34 ± 0.10
4.34 ± 0.0
4.34 ± 0.0
4.34 ± 0.05
4.34 ± 0.05

118 ± 0.0
114.8 ± 4.0
61 ± 0.0
60.4 ± 6.85
61.80 ± 4.82
61.60 ± 7.21
63 ± 0.0
59.80 ± 6.20
60.40 ± 5.12
57.03 ± 4.07

-
0.13 ± 0.04
-
1.75 ± 0.19
13.47 ± 2.87
13.93 ± 4.28
-
0.93 ± 0.42
0.59 ± 0.35
0.50 ± 0.36

0.0
0.13
0.0
0.02
2.55
0.01
0.0
0.02
2.51
0.02

0.0
0.13
0.0
0.02
0.02
0.02
0.0
0.02
0.02
0.02

-
1.54
-
0.20
0.06
0.72
-
0.20
0.06
0.73

-
1.54
-
0.20
0.01
0.88
-
0.20
0.01
0.93

Table 2: Affine SSD registration test, comparison between Clear, PPIR(MPC), PPIR(FHE)-v1 and PPIR(FHE)-v2. Registration metrics are reported as mean and
standard deviation. Efficiency metrics in terms of average across iterations. RMSE: root mean square error.

SSD Cubic splines registration metrics

Solution
Clear
PPIR-MPC
PPIR(FHE)-v1(D = 121)
PPIR(FHE)-v2(D = 256)
Clear + URS
PPIR(MPC) + URS
PPIR(FHE)-v1(D = 128) + URS
PPIR(FHE)-v2(D = 128) + URS
Clear + GMS
PPIR(MPC) + GMS
PPIR(FHE)-v1(D = 128) + GMS
PPIR(FHE)-v2(D = 128) + GMS

Intensity Error (SSD) Num. Interation Displacement RMSE (mm) Time party1 (s) Time party2 (s) Comm. party1 (MB) Comm. party2 (MB)
0.65 ± 0.0
0.65 ± 0.0
0.64 ± 0.0
0.64 ± 0.0
0.02 ± 0.0
0.02 ± 0.00
0.02 ± 0.00
0.02 ± 0.00
0.02 ± 0.0
0.02 ± 0.04
0.02 ± 0.00
0.02 ± 0.00

413 ± 0.0
345.70 ± 91.22
224.7 ± 79.15
379.2 ± 75.82
101 ± 0.0
79.3 ± 1.88
105.40 ± 1.71
105.20 ± 2.54
103.00 ± 0.0
80.20 ± 1.62
105.70 ± 2.40
106.32 ± 1.30

-
7.31 ± 1.86
9.50 ± 4.34
11.02 ± 4.93
-
5.59 ± 0.39
7.63 ± 0.01
8.74 ± 1.90
-
6.17 ± 0.37
5.60 ± 2.22
9.11 ± 2.34

0.0
0.63
3.41
0.98
0.0
0.41
12.23
0.62
0.0
0.41
11.95
0.62

0.0
0.63
0.00
0.43
0.0
0.41
0.0
0.26
0.0
0.41
0.0
0.26

-
28.98
0.01
3.56
-
8.00
0.01
3.37

-
21.47
0.06
40.45
-
8.00
0.06
24.74

8.00
0.06
24.91

8.00
0.01
3.35

Table 3: Non-Linear SSD registration test comparison between Clear, PPIR(MPC), PPIR(FHE)-v1 and PPIR(FHE)-v2. The registration metrics are reported as
mean and standard deviation. Efficiency metrics in terms of average across iterations. RMSE: root mean square error.

and TenSeal (Benaissa et al., 2021), which implements the
CKKS protocol3.

PPIR based on MI and CC is implemented by extending the
Dipy framework of Garyfallidis et al. (2014) 4. Finally, PPIR
for point cloud data is released in a separated repository5.

All the experiments are executed on a machine with an
Intel(R) Core(TM) i7-7800X CPU @ (3.50GHz x 12) using
132GB of RAM. For each registration configuration, the op-
timization is repeated 10 times to account for the random gen-
eration of MPC shares and FHE encryption keys.

6.3. Results

Point Cloud Data. In Supplementary Table A1 we present
the registration metrics for PPIR(MPC) and PPIR(FHE)-v1.
The registration shows that PPIR(MPC) achieves the best re-
sults compared to PPIR(FHE), which exhibits not only a longer
computation time but also requires higher bandwidth, thanks to
its non-iterative algorithm. However, to carry out MatMul, a
sufficiently large N (4096) is required, and in this scenario, it
leads to a significant loss of chipertext slots compared to the
dimension of the point set n = 193. Finally, the qualitative
results reported in Figure A7 show negligible differences be-
tween point cloud transformed with Clear, PPIR(MPC) and
PPIR(FHE)-v1.

Whole body PET data: affine registration (SSD). Ta-
ble 2 compares Clear, PPIR(MPC), PPIR(FHE)-v1 and v2,
showcasing metrics resulting from the affine transformation

3https://github.com/rtaiello/pp_image_registration
4https://github.com/rtaiello/pp_dipy/tree/main
5https://github.com/rtaiello/ppir_pc

of whole-body PET images. Notably, registration through
PPIR(MPC) yields negligible differences compared to Clear
in terms of the number of iterations, intensity, and displace-
ment. In contrast, registering with PPIR(FHE) is not feasible
when considering entire images due to computational complex-
ity. Nevertheless, Supplementary Figure A5 shows that neither
MPC nor FHE decreases the overall quality of the affine reg-
istered images. A comprehensive assessment of the registra-
tion results is available in the Appendix. Table 2 (Efficiency
metrics) shows that PPIR(MPC) performed on full images re-
quires higher computation time and required communication
bandwidth compared to PPIR(MPC)+URS/GMS. These num-
bers improve sensibly when using URS or GMS (by factors
10× and 20× for time and bandwidth, respectively). Concern-
ing PPIR(FHE)-v1, we note the uneven computation time and
bandwidth usage between clients, due to the asymmetry of the
encryption operations and communication protocol (Figure 1).
PPIR(FHE)-v2, which shares the computational workload be-
tween the two parties and avoids DotProduct, allows obtain-
ing an important speed-up over PPIR(FHE)-v1 (100× faster).
Notably, this gain is obtained without affecting the quality of
the registration metrics, and improves the execution time of
PPIR(MPC). On the other side, although PPIR(FHE)-v1 is able
to improve communication with respect to PPIR(MPC), it still
suffers from the highest communication among the three pro-
posed solutions. This is due to the fact PPIR(FHE) protocols
can find different applications depending on the requirements
in terms of computational power or bandwidth.

Brain MRI data and whole body PET data: non-
linear registration (SSD). Table 3, comparing Clear and
PPIR(MPC), PPIR(FHE)-v1 and v2, showcases the metrics re-
sulting from spline-based non-linear registration between grey

Riccardo Taiello et al. / Medical Image Analysis (2024)

9

Solution
Clear
PPIR(MPC)

Intensity Error (MI) Num. Iteration Displacement RMSE (mm) Time party1 (s) Time party2 (s) Comm. party1 (MB) Comm. party2 (MB)

0.22 ± 0.00
0.22 ± 0.00

213 ± 0.0
264 ± 0.0

-
0.41 ± 0.04

0.0
1.02

0.0
1.02

-
15.00

-
15.00

MI Affine registration metrics

Table 4: Affine MI registration test, comparison between Clear and PPIR(MPC). Registration metrics are reported as mean and standard deviation. Efficiency metrics
in terms of average across iterations. RMSE: root mean square error.

CC ANTs registration registration metrics

Solution

Initial DICE score

Final DICE score Num. Iteration Displacement RMSE (mm) Time party1 (s) Time party2 (s) Comm. party1 (MB) Comm party2 (MB)

Clear
PPIR(MPC)

Clear
PPIR(MPC)

0.54 ± 0.13

0.54 ± 0.13

0.68 ± 0.19
0.67 ± 0.19

0.69 ± 0.19
0.68 ± 0.19

26 ± 0.0
26 ± 0.0

26 ± 0.0
26 ± 0.0

Forward

-
0.22 ± 0.04

Backward
-
0.22 ± 0.04

-
24.00

-
24.00

-
24.00

-
24.00

-
152.25

-
152.25

-
152.25

-
152.25

Table 5: ANTs registration with CC, comparison between Clear and PPIR(MPC). Registration metrics are reported as mean and standard deviation. Efficiency
metrics in terms of average across iterations. RMSE: root mean square error.

Fig. 2: Qualitative results for affine registration with MI over 3D medical images using ADNI dataset (Mueller et al., 2005). The images are presented in a 3 × 4
grid, with the first row representing the axial axis, the second row the coronal axis, and the third row the sagittal axis. In the first column of each row, the moving
image obtained using PET modality is shown, while in the second column, the fixed image obtained using MRI modality is displayed. The third column shows the
checkerboard alignment result using Clear, while the fourth column shows the result using PPIR(MPC). The different protocols are highlighted by red and green
frames, respectively.

10

Riccardo Taiello et al. / Medical Image Analysis (2024)

matter density images without the application of gradient ap-
proximation. Additionally, the table includes results for the reg-
istration between whole-body PET images when the gradient
approximation is applied.

Brain MRI data without gradient approximation. Regard-
ing the registration accuracy, we draw conclusions similar to
those of the affine case, where PPIR(MPC) leads to minimum
differences with respect to Clear, while PPIR(FHE)-v1 seems
slightly superior. PPIR(MPC) is associated with a lower ex-
ecution time and a higher computational bandwidth, due to
the larger number of parameters of the cubic splines, which
affects the size of the matrix S . Although PPIR(FHE)-v1
has a slower execution time, the demanded bandwidth is in-
ferior to the one of PPIR(MPC), since the encrypted image is
transmitted only once. PPIR(FHE)-v2, as in the affine case,
outperforms PPIR(FHE)-v1 (still about 100× faster) leading
to comparable values between registration metrics and is still
inferior to PPIR(MPC). Here, the limitations of PPIR(FHE)-
v1 on the bandwidth size are even more evident than in the
affine case, since the bandwidth increases according to the
number of parameters. This result gives a non-negligible bur-
den to the party1, due to the multiple sending of the flattened
and encrypted submatrices of updated parameters. Further-
more, in this case, PPIR(FHE)-v1 performs slightly worse than
PPIR(MPC) in terms of execution time.

Whole Body PET Data With Gradient Approximation. Incor-
porating gradient approximation for handling whole-body PET
data leads to similar conclusions as for the experiments on brain
data. Qualitative results, reported in Supplementary Figure A6,
show negligible differences between images transformed with
Clear+GMS, PPIR(MPC)+GMS, and PPIR(FHE)-v1+GMS.
Brain MRI and PET data: affine registration (MI). Ta-
ble 4 provides information on the computation cost of the pro-
tocol and the registration metrics for both the joint PDF and
the First Derivative of the Joint PDF, using both Clear and
PPIR(MPC). The results demonstrate a reasonable execution
time (completed in under 5 minutes for the entire process) and
noteworthy data transfer, totaling less than 4GB. Qualitative
results for the image registration are shown in Figure 2, indicat-
ing that there is no difference between the moving transformed
using Clear and PPIR(MPC).

Abdomen MRI and CT data: diffeomorphic non-linear
registration (CC). Table 5 presents the metrics for ANTs reg-
istration using both Clear and PPIR(MPC) methods. The ini-
tial DICE scores for both forward and backward transforma-
tions are consistent between Clear and PPIR(MPC), with slight
variations in the final DICE scores. The number of iterations
and displacement RMSE values also exhibit similar trends. In
terms of communication and computation times, PPIR(MPC)
demonstrates comparable performance with Clear, showcas-
ing its feasibility for secure registration. The computation times
and communication bandwith between the two parties are well
within reasonable limits. These qualitative result in Figure 3
indicates that the proposed PPIR(MPC) solution maintains the
quality of registration metrics while ensuring secure and private
communication between parties.

7. Discussion

This current work builds upon Taiello et al. (2022) by extend-
ing its application to include MI with linear registration (3D
images), CC using the ANTs framework (3D images), enhanc-
ing the FHE for the SSD loss function, namely PPIR(FHE)-v2,
and also integrating rigid point cloud registration. We recognize
specific limitations associated with the use of FHE in our PPIR
framework, which limited the effective use of this techniques
besides the optimization of the SSD loss. FHE’s computational
cost during homomorphic operations poses challenges, limiting
the scalability and real-time applicability of PPIR(FHE). This
is particularly evident when dealing with large datasets, such
as 3D images, or when employing advanced image registration
cost functions that demand significant computational resources.
To address this gap, researchers are actively exploring the opti-
mization of FHE through the integration of hardware accelera-
tors (Boemer et al., 2021).

For the SSD loss function, we provide comparison experi-
ments with both URS and GMS (Viola and Wells III, 1997;
Mattes et al., 2003; Sabuncu and Ramadge, 2004) for sake of
completeness and compatibility with subsampling approaches
in IR. We recognized that URS doesn’t bring substantial im-
provements with resepct to GRS, and this latter method should
be preferred in the considered application or testing scenario.

While PPIR focuses on the privacy-preserving formulation of
classical image registration methods based on gradient-based
optimization, throughout the past years the research commu-
nity has been steering the attention towards deep learning (DL)-
based image registration (Simonovsky et al., 2016; Krebs et al.,
2017; Yang et al., 2017; Balakrishnan et al., 2019). Among the
medical imaging application of privacy-preserving methodolo-
gies, Kaissis et al. (2021) discussed privacy-preserving FL with
Secure Aggregation (Bonawitz et al., 2017) and Differential Pri-
vacy (Abadi et al., 2016) for 2D medical image classification
tasks. However, as highlighted by (Kaissis et al., 2021), de-
ploying DL models for privacy-preserving inference nowadays
is predominantly achievable through Multi-Party Computation
(MPC). This process necessitates multiple servers and incurs
significant overhead, primarily attributed to the size of the DL
model, especially when handling 3D image registration tasks
within a DL-based framework (Balakrishnan et al., 2019). To
the best of our knowledge both DL and non-DL registration
methods available in the literature do not satisfy the PPIR re-
quirements investigated in our work, as they always require the
disclosure of the target and moving images in clear.

8. Conclusion and future works

This study introduces the novel paradigm of Privacy Preserv-
ing Image Registration, designed for allowing image registra-
tion in privacy-preserving scenarios where images are confi-
dential and cannot be shared in clear. Leveraging both secure
multi-party computation (MPC) and Fully Homomorphic En-
cryption (FHE), we propose in PPIR effective strategies inte-
grating cryptographic techniques into a variety of state-of-the-
art registration frameworks, encompassing different parameter-
ization and loss functions. We evaluate the framework’s per-

Riccardo Taiello et al. / Medical Image Analysis (2024)

11

formance across various registration benchmarks, conducting
quantitative and qualitative assessment for all the considered
image registration problems. Our future direction involve ex-
tending PPIR to encompass additional cost functions commonly
used in image registration, aiming to enhance the framework’s
versatility and applicability.

Acknowledgments

This work has been supported by the French government,
through the 3IA Cˆote d’Azur Investments in the Future project
managed by the National Research Agency (ANR) with the
reference number ANR-19-P3IA-0002, and by the ANR JCJC
project Fed-BioMed 19-CE45-0006-01.

References

Abadi, M., Chu, A., Goodfellow, I., McMahan, H.B., Mironov, I., Talwar, K.,
Zhang, L., 2016. Deep learning with differential privacy, in: Proceedings
of the 2016 ACM SIGSAC conference on computer and communications
security, pp. 308–318.

Arun, K.S., Huang, T.S., Blostein, S.D., 1987. Least-squares fitting of two 3-d
point sets. IEEE Transactions on pattern analysis and machine intelligence ,
698–700.

Ashburner, J., 2007. A fast diffeomorphic image registration algorithm. Neu-

roimage 38, 95–113.

Ashburner, J., Friston, K.J., 2000. Voxel-based morphometry—the methods.

Neuroimage 11, 805–821.

Ashburner, J., Ridgway, G.R., 2013. Symmetric diffeomorphic modeling of

longitudinal structural MRI. Frontiers in neuroscience 6, 197.

Avants, B.B., Epstein, C.L., Grossman, M., Gee, J.C., 2008. Symmetric dif-
feomorphic image registration with cross-correlation: evaluating automated
labeling of elderly and neurodegenerative brain. Medical image analysis 12,
26–41.

Avants, B.B., Tustison, N.J., Song, G., Cook, P.A., Klein, A., Gee, J.C., 2011. A
reproducible evaluation of ants similarity metric performance in brain image
registration. Neuroimage 54, 2033–2044.

Baker, S., Matthews, I., 2004. Lucas-Kanade 20 years on: A unifying frame-

work. International journal of computer vision 56, 221–255.

Balakrishnan, G., Zhao, A., Sabuncu, M.R., Guttag, J., Dalca, A.V., 2019. Vox-
elmorph: a learning framework for deformable medical image registration.
IEEE transactions on medical imaging 38, 1788–1800.

Beg, M.F., Miller, M.I., Trouv´e, A., Younes, L., 2005. Computing large de-
formation metric mappings via geodesic flows of diffeomorphisms. Interna-
tional journal of computer vision 61, 139–157.

Benaissa, A., Retiat, B., Cebere, B., Belfedhal, A.E., 2021. Tenseal: A li-
brary for encrypted tensor operations using homomorphic encryption. arXiv
preprint arXiv:2104.03152 .

Boemer, F., Kim, S., Seifu, G., DM de Souza, F., Gopal, V., 2021. Intel hexl:
Accelerating homomorphic encryption with intel avx512-ifma52, in: Pro-
ceedings of the 9th on Workshop on Encrypted Computing & Applied Ho-
momorphic Cryptography, pp. 57–62.

Bonawitz, K., Ivanov, V., Kreuter, B., Marcedone, A., McMahan, H.B., Patel,
S., Ramage, D., Segal, A., Seth, K., 2017. Practical secure aggregation
for privacy-preserving machine learning, in: proceedings of the 2017 ACM
SIGSAC Conference on Computer and Communications Security, pp. 1175–
1191.

Cardoso, M.J., Leung, K., Modat, M., Keihaninejad, S., Cash, D., Barnes, J.,
Fox, N.C., Ourselin, S., Initiative, A.D.N., et al., 2013. STEPs: Similarity
and truth estimation for propagated segmentations and its application to hip-
pocampal segmentation and brain parcelation. Medical image analysis 17,
671–684.

Cheon, J.H., Kim, A., Kim, M., Song, Y., 2017. Homomorphic encryption
for arithmetic of approximate numbers, in: International Conference on the
Theory and Application of Cryptology and Information Security, Springer.
pp. 409–437.

Clark, K., Vendt, B., Smith, K., Freymann, J., Kirby, J., Koppel, P., Moore, S.,
Phillips, S., Maffitt, D., Pringle, M., et al., 2013. The cancer imaging archive
(tcia): maintaining and operating a public information repository. Journal of
digital imaging 26, 1045–1057.

Dale, A.M., Fischl, B., Sereno, M.I., 1999. Cortical surface-based analysis: I.

segmentation and surface reconstruction. Neuroimage 9, 179–194.

Damgard, I., Pastro, V., Smart, N., Zakarias, S., 2011. Multiparty computa-
tion from somewhat homomorphic encryption. Cryptology, ePrint Archive,
Report 2011/535. https://ia.cr/2011/535.

Dowson, N., Bowden, R., 2006. A unifying framework for mutual information
methods for use in non-linear optimisation, in: European Conference on
Computer Vision, Springer. pp. 365–378.

Dowson, N., Bowden, R., 2007. Mutual information for lucas-kanade tracking
(milk): An inverse compositional formulation. IEEE transactions on pattern
analysis and machine intelligence 30, 180–185.

Garyfallidis, E., Brett, M., Amirbekian, B., Rokem, A., Van Der Walt, S., De-
scoteaux, M., Nimmo-Smith, I., Contributors, D., 2014. Dipy, a library for
the analysis of diffusion mri data. Frontiers in neuroinformatics 8, 8.

Gazula, H., Holla, B., Zhang, Z., Xu, J., Verner, E., Kelly, R., Jain, S., Bharath,
R.D., Barker, G.J., Basu, D., et al., 2021. Decentralized multisite vbm anal-
ysis during adolescence shows structural changes linked to age, body mass
index, and smoking: a coinstac analysis. Neuroinformatics 19, 553–566.
Haralampieva, V., Rueckert, D., Passerat-Palmbach, J., 2020. A systematic
comparison of encrypted machine learning solutions for image classifica-
tion, in: Proceedings of the 2020 workshop on privacy-preserving machine
learning in practice, pp. 55–59.

Heinrich, M.P., Jenkinson, M., Bhushan, M., Matin, T., Gleeson, F.V., Brady,
J.M., Schnabel, J.A., 2011. Non-local shape descriptor: A new similar-
ity metric for deformable multi-modal registration, in: International Con-
ference on Medical Image Computing and Computer-Assisted Intervention,
Springer. pp. 541–548.

Hering, A., Hansen, L., Mok, T.C., Chung, A.C., Siebert, H., H¨ager, S., Lange,
A., Kuckertz, S., Heldmann, S., Shao, W., et al., 2022. Learn2reg: compre-
hensive multi-task medical image registration challenge, dataset and evalua-
tion in the era of deep learning. IEEE Transactions on Medical Imaging 42,
697–712.

Kaissis, G., Ziller, A., Passerat-Palmbach, J., Ryffel, T., Usynin, D., Trask,
A., Lima Jr, I., Mancuso, J., Jungmann, F., Steinborn, M.M., et al., 2021.
End-to-end privacy preserving deep learning on multi-institutional medical
imaging. Nature Machine Intelligence 3, 473–484.

Krebs, J., Mansi, T., Delingette, H., Zhang, L., Ghesu, F.C., Miao, S., Maier,
A.K., Ayache, N., Liao, R., Kamen, A., 2017. Robust non-rigid registra-
tion through agent-based action learning, in: Medical Image Computing and
Computer Assisted Intervention- MICCAI 2017: 20th International Confer-
ence, Quebec City, QC, Canada, September 11-13, 2017, Proceedings, Part
I 20, Springer. pp. 344–352.

Lauter, K., 2021. Private AI: Machine Learning on Encrypted Data. Technical
Report. Eprint report https://eprint.iacr.org/2021/324.pdf.
Lotan, E., Tschider, C., Sodickson, D.K., Caplan, A.L., Bruno, M., Zhang,
B., Lui, Y.W., 2020. Medical imaging and privacy in the era of artificial
intelligence: myth, fallacy, and the future. Journal of the American College
of Radiology 17, 1159–1162.

Mattes, D., Haynor, D.R., Vesselle, H., Lewellen, T.K., Eubank, W., 2003.
Pet-ct image registration in the chest using free-form deformations. IEEE
transactions on medical imaging 22, 120–128.

McMahan, B., Moore, E., Ramage, D., Hampson, S., y Arcas, B.A., 2017.
Communication-efficient learning of deep networks from decentralized data,
in: Artificial intelligence and statistics, PMLR. pp. 1273–1282.

Modersitzki, J., 2009. FAIR: Flexible Algorithms for Image Registration.

SIAM, Philadelphia.

Mueller, S.G., Weiner, M.W., Thal, L.J., Petersen, R.C., Jack, C., Jagust, W.,
Trojanowski, J.Q., Toga, A.W., Beckett, L., 2005. The alzheimer’s disease
neuroimaging initiative. Neuroimaging Clinics 15, 869–877.

Parzen, E., 1961. Mathematical considerations in the estimation of spectra.

Technometrics 3, 167–190.

Pennec, X., Cachier, P., Ayache, N., 1999. Understanding the “Demon’s al-
gorithm”: 3D non-rigid registration by gradient descent, in: International
Conference on Medical Image Computing and Computer-Assisted Interven-
tion, Springer. pp. 597–605.

Reuter, M., Rosas, H.D., Fischl, B., 2010. Highly accurate inverse consistent

registration: a robust approach. Neuroimage 53, 1181–1196.

Rivest, R.L., Adleman, L., Dertouzos, M.L., et al., 1978. On data banks and

Riccardo Taiello et al. / Medical Image Analysis (2024)

1

privacy homomorphisms. Foundations of secure computation 4, 169–180.
Ryffel, T., Trask, A., Dahl, M., Wagner, B., Mancuso, J., Rueckert, D., Passerat-
Palmbach, J., 2018. A generic framework for privacy preserving deep learn-
ing. arXiv preprint arXiv:1811.04017 .

Sabuncu, M.R., Ramadge, P.J., 2004. Gradient based nonuniform subsampling
for information-theoretic alignment methods, in: The 26th Annual Interna-
tional Conference of the IEEE Engineering in Medicine and Biology Soci-
ety, IEEE. pp. 1683–1686.

Schnabel, J.A., Heinrich, M.P., Papie˙z, B.W., Brady, J.M., 2016. Advances and
challenges in deformable image registration: from image fusion to complex
motion modelling. Medical Image Analysis 33, 145–148.

Shattuck, D.W., Prasad, G., Mirza, M., Narr, K.L., Toga, A.W., 2009. On-
line resource for validation of brain segmentation methods. NeuroImage 45,
431–439.

Simonovsky, M., Guti´errez-Becker, B., Mateus, D., Navab, N., Komodakis,
N., 2016. A deep metric for multimodal registration, in: Medical Image
Computing and Computer-Assisted Intervention-MICCAI 2016: 19th Inter-
national Conference, Athens, Greece, October 17-21, 2016, Proceedings,
Part III 19, Springer. pp. 10–18.

Taiello, R., ¨Onen, M., Humbert, O., Lorenzi, M., 2022. Privacy preserving
image registration, in: Wang, L., Dou, Q., Fletcher, P.T., Speidel, S., Li,
S. (Eds.), Medical Image Computing and Computer Assisted Intervention –
MICCAI 2022, Springer Nature Switzerland, Cham. pp. 130–140.

Vachet, C., Yvernault, B., Bhatt, K., Smith, R.G., Gerig, G., Hazlett, H.C.,
Styner, M., 2012. Automatic corpus callosum segmentation using a de-
formable active fourier contour model, in: Medical Imaging 2012: Biomed-
ical Applications in Molecular, Structural, and Functional Imaging, SPIE.
pp. 79–85.

Viola, P., Wells III, W.M., 1997. Alignment by maximization of mutual infor-

mation. International journal of computer vision 24, 137–154.

Yang, X., Kwitt, R., Styner, M., Niethammer, M., 2017. Quicksilver: Fast
predictive image registration–a deep learning approach. NeuroImage 158,
378–396.

Yao, A.C., 1982. Protocols for secure computations, in: 23rd annual sympo-
sium on foundations of computer science (sfcs 1982), IEEE. pp. 160–164.
Zerka, F., Urovi, V., Vaidyanathan, A., Barakat, S., Leijenaar, R.T., Walsh,
S., Gabrani-Juma, H., Miraglio, B., Woodruff, H.C., Dumontier, M., et al.,
2020. Blockchain for privacy preserving and trustworthy distributed ma-
chine learning in multicentric medical imaging (c-distrim). Ieee Access 8,
183939–183951.

Solution
PPIR(MPC)
PPIR(FHE)

Displacement RMSE (mm) Time party1 (s) Time party2 (s) Comm. party1 (MB) Comm. party2 (MB)
0.02
0.18

1.11 ± 0.31
1.26 ± 0.37

0.03
35.93

0.03
1.42

0.02
1.10

Rigid Point Clouds Registration

Table A1: Rigid Point Clouds registration test, comparisson between
PPIR(MPC) and PPIR(FHE). Registration metrics are reported as mean and
standard deviation. Efficiency metrics in terms of average across iterations.
RMSE: root mean square error.

Appendix

A 1. Rigid Point Cloud Registration

Let {zi}, {z′

i} two finite n size point sets where zi, z′

i ∈ Rd ,
to continue ... A non-iterative least-squares approach to match
two sets of points, was proposed by Arun et al. (1987). The
method uses singular value decomposition (SVD) and is trying
to minimize the following cost function:

Σ2 =

n(cid:88)

i=1

||z′

i − (Rzi + t)||2,

(A1)

where R is the rotation matrix and t is the translation vector. Let
define ¯z and ¯z′ to represent the centroids of {zi} and {z′
i} respec-
tively. Let ˆR being the estimated rotation matrix, and ˆt being
the estimated translation. The method lies in the algorithm for
finding ˆR detailed below:

1. Calculate the following quantities:

qi = pi − ¯z
= p′
q′
i − ¯z′
i

for all 0 ≤ i ≤ n, which are distances from each point to
its centroid.

2. Calculate L ∈ Rd×d, L = (cid:80)n

i=1 qiq′T
i

, in vectorized form:

L = Q · Q′T ,

(A2)

where Q, Q′ ∈ Rd×n.

3. Find SVD of L, namely L = UΛV T ;
4. Calculate X = VUT ;
5. Check the determinant of X. If it equals to +1, then ˆR = X
If the determinant equals to −1, the

and ˆt = q′ − ˆRq.
algorithm has failed.

We note that the only operation that requires the joint availabil-
ity of information from both parties is Equation A2 which can
be computed with a matrix multiplication with PPIR(MPC) and
PPIR(FHE) as reported in Figure A7.

2

Riccardo Taiello et al. / Medical Image Analysis (2024)

Fig. 3: Qualitative results for diffeomorphic registration with CC between 3D medical images from the AbdomenMRCT dataset (Hering et al., 2022). The images
are presented in a 3 × 4 grid, with the first row representing the axial axis, the second row the coronal axis, and the third row the sagittal axis. First and second
column show respectively MRI and CT images. The third column shows the MRI transformed using Clear, while the fourth column shows the MRI transformed
using PPIR(MPC). The transformed images are highlighted by red and green frames, respectively.

(a) Computation P = 1
Nx

· (A3

I )T · B0

J with PPIR(MPC)

(b) Computation P′ = − 1
Nx

· (B0

J )T · C3

I with PPIR(MPC)

Fig. A1: Optimization of MI loss: proposed framework to calculate matrix multiplication P = 1
Nx

· (A3

I )T · B0

J and P′ = − 1
Nx

· (B0

J)T · C3

I based on PPIR(MPC).

Riccardo Taiello et al. / Medical Image Analysis (2024)

3

(a) PPIR(MPC) - 2D
EF

(b) PPIR(MPC) - ( ¯J − D
E

¯I)

Fig. A2: Optimization of ANTS NCC loss: proposed framework to calculate 2D

EF and ( ¯J − D
E

¯I) based on PPIR(MPC).

(a) PPIR(MPC)

(b) PPIR(FHE)

Fig. A3: Optimization of rigid point cloud: proposed framework to compute matrix multiplication L = Q · Q′T based on PPIR(MPC) and PPIR(FHE).

4

Riccardo Taiello et al. / Medical Image Analysis (2024)

Fig. A4: Proposed framework to compute matrix-vector multiplication S T · J based on PPIR(FHE)-v2.

Riccardo Taiello et al. / Medical Image Analysis (2024)

5

Fig. A5: Qualitative results for affine registration with SSD between 2D medical images. The red frame is the transformed moving image using Clear+URS
registration. Green and Yellow frames are the transformed images using respectively PPIR(MPC)+URS and PPIR(FHE)v1+URS.

Fig. A6: Qualitative results for Cubic splines registration with SSD between 2D medical images. The red frame is the transformed moving image using Clear+GMS
registration. Green and Yellow frames are the transformed images using respectively PPIR(MPC)+GMS and PPIR(FHE)v1+GMS.

6

Riccardo Taiello et al. / Medical Image Analysis (2024)

Fig. A7: Qualitative results for rigid point cloud registration between 2D corpus callosum point sets. The red frame is the transformed moving image using Clear
registration. Green and Yellow frames are the transformed images using respectively PPIR(MPC) and PPIR(FHE)v1.

