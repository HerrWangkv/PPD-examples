4
2
0
2

n
u
J

0
3

]

V
C
.
s
c
[

1
v
3
8
7
0
0
.
7
0
4
2
:
v
i
X
r
a

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

1

Diffusion Models and Representation Learning:
A Survey

Michael Fuest, Pingchuan Ma, Ming Gui, Johannes Schusterbauer, Vincent Tao Hu, Bj ¨orn Ommer

Abstract—Diffusion Models are popular generative modeling methods in various vision tasks, attracting significant attention. They can
be considered a unique instance of self-supervised learning methods due to their independence from label annotation. This survey
explores the interplay between diffusion models and representation learning. It provides an overview of diffusion models’ essential
aspects, including mathematical foundations, popular denoising network architectures, and guidance methods. Various approaches
related to diffusion models and representation learning are detailed. These include frameworks that leverage representations learned
from pre-trained diffusion models for subsequent recognition tasks and methods that utilize advancements in representation and
self-supervised learning to enhance diffusion models. This survey aims to offer a comprehensive overview of the taxonomy between
diffusion models and representation learning, identifying key areas of existing concerns and potential exploration. Github link:
https://github.com/dongzhuoyao/Diffusion-Representation-Learning-Survey-Taxonomy.

Index Terms—deep generative modeling, diffusion models, denoising diffusion models, score-based models, image generation,
representation learning.

✦

1 INTRODUCTION

Diffusion Models [68, 151, 154] have recently emerged as
the state-of-the-art of generative modeling, demonstrating
remarkable results in image synthesis [43, 67, 68, 141] and
across other modalities including natural language [9, 70, 77,
101], computational chemistry [6, 71] and audio synthesis
[80, 92, 109]. The remarkable generative capabilities of Dif-
fusion Models suggest that Diffusion Models learn both low
and high-level features of their input data, potentially mak-
ing them well-suited for general representation learning.
Unlike other generative models like Generative Adversarial
Networks (GANs) [22, 53, 84] and Variational Autoencoders
(VAEs) [88, 137], diffusion models do not contain fixed archi-
tectural components that capture data representations [124].
This makes diffusion model-based representation learning
challenging. Nevertheless, approaches leveraging diffusion
models for representation learning have seen increasing
interest, simultaneously driven by advancements in training
and sampling of Diffusion Models.

Current state-of-the-art self-supervised representation
learning approaches [8, 24, 33, 55] have demonstrated great
scalability. It is thus likely that diffusion models exhibit
similar scaling properties [159]. Controlled generation ap-
proaches like Classifier Guidance [43] and Classifier-free
Guidance [67] used to obtain state-of-the-art generation
results rely on annotated data, which represents a bottle-
neck for scaling up diffusion models. Guidance approaches
that leverage representation learning and that are thus
annotation-free offer a solution, potentially enabling dif-

• Michael Fuest is a Master’s student at the Technical University of
Munich. Pingchuan Ma, Ming Gui, and Johannes Schusterbauer are
PhD students at LMU Munich. Vincent Tao Hu, a PostDoc from LMU
Munich, is also the corresponding author.
E-mail: taohu620@gmail.com

• Bj¨orn Ommer is a full professor at LMU where he heads the Computer Vi-
sion & Learning Group (previously Computer Vision Group Heidelberg).

Fig. 1. Shows yearly numbers of both published and preprint papers on
diffusion models and representation learning. For 2024, the green bar
indicates the number of papers collected up to and including June 2024,
and the dashed grey bar indicates the projected number for the whole
year.

fusion models to train on much larger, annotation-free
datasets.

This survey paper aims to elucidate the relationship
and interplay between diffusion models and representation
learning. We highlight two central perspectives: Using diffu-
sion models themselves for representation learning and us-
ing representation learning for improving diffusion models.
We introduce a taxonomy of current approaches and derive
generalized frameworks that demonstrate commonalities
among current approaches.

Interest in exploring the representation learning capabil-
ities of diffusion models has been growing since the original
formulation of diffusion models by Ho et al. [68], Sohl-
Dickstein et al. [151], Song et al. [154]. As demonstrated

 
 
 
 
 
 
IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

in Fig. 1, we expect this trend to continue this year. The
increased volume of published works on diffusion models
and representation learning makes it more difficult for re-
searchers to identify state-of-the-art approaches and stay on
top of current developments. This can hinder progress in the
space, which is why we feel a comprehensive overview and
categorization is required.

Research on representation learning and diffusion mod-
els is in its infancy. Many of the current approaches rely
on using diffusion models solely trained for generative syn-
thesis for representation learning. We therefore hypothesize
that there are significant opportunities for further progress
in this area in the future and that diffusion models can
increasingly challenge the current state-of-the-art in rep-
resentation learning. Fig. 2 shows qualitative results from
existing methods. We hope that this survey can contribute
to advances in diffusion-based representation learning, by
clarifying commonalities and differences among current ap-
proaches. In summary, the main contributions of this paper
are the following:

• Comprehensive Overview: Offers a thorough survey
of the interplay between diffusion models and repre-
sentation learning, providing clarity on how diffu-
sion models can be used for representation learning
and vice versa.

• Taxonomy of Approaches: We introduce a taxonomy
of current approaches in diffusion-based represen-
tation learning, categorizing and highlighting com-
monalities and differences among them.

• Generalized Frameworks: The paper derives gener-
alized frameworks for both diffusion model feature
extraction and assignment-based guidance, offering
a structured view on a large number of works on
diffusion models and representation learning.
Future Directions: We identify key opportunities
for further progress in the field, encouraging the
exploration of diffusion models and flow matching
as a new state-of-the-art in representation learning.

•

2 BACKGROUND
The following section outlines the required mathematical
foundations of diffusion models. We also highlight current
architecture backbones of diffusion models and provide a
brief overview of sampling methods and conditional gener-
ation approaches.

2.1 Mathematical Foundations

Consider a set of training examples drawn from an under-
lying probability distribution p(x). The idea behind gener-
ative diffusion models is to learn a denoising process that
maps samples of random noise to novel images sampled
from p(x) [133]. To achieve this, images are corrupted by
gradually adding different levels of Gaussian noise. Given
an uncorrupted training sample x0 ∼ p(x), where index
0 denotes the fact that the sample is not corrupted, the
corrupted samples x1, x2..., xT are generated according to
a Markovian process. One common choice for the transition
kernel p(xt|xt−1) is the following:

p(xt|xt−1) = N

(cid:16)

xt; (cid:112)1 − βtxt−1, βtI
∀t ∈ {1, . . . , T },

(cid:17)

,

2

(1)

where T denotes the number of diffusion timesteps, βt
is a time-dependent variance schedule and I is an identity
matrix with dimensionality equal to x0 [37]. Note that other
parametrizations of the transition kernel p(xt|xt−1) are also
applicable in the same manner [87, 188]. We proceed with
the parametrization used in DDPMs [68] to simplify the dis-
cussion moving forward. A noisy image xt can be sampled
directly from x0 with the help of a reparametrization trick
[151] as follows:

p(xt|x0) = N

√

(cid:16)

xt;

¯αtx0; (1 − ¯αt)I

(cid:17)

,

(2)

where αt

:= 1 − βt and ¯αt

i=1 αi. Given the
original input image x0, we can now obtain xt in one step
by sampling Gaussian vector ϵt ∼ N (0, I) and applying:

:= (cid:81)t

√

xt =

¯αtx0 + (cid:112)(1 − ¯αt)ϵt.

(3)

We can generate novel samples from p(x0) starting
from a pure noise image xT ∼ π(xT ) = N (0, I)
with dimensionality equivalent to the data and sequen-
tially denoise it such that at every step, pθ(xt−1|xt) =
N (xt−1; µθ(xt, t), Σθ(xt, t)). In practice, this requires train-
ing a neural network pθ(xt−1|xt) that predicts the mean
µ0(xt, t) and the covariance Σθ(xt, t) given a diffusion
timestep t and the noisy input image xt [172]. Training
this neural network with a maximum likelihood objective
is intractable [37], so the objective is amended to minimize
a Variational Lower-Bound of the Negative Log-Likelihood
instead [68, 151]:

Lvlb = − log pθ(x0|x1) + DKL (p(xT |x0)∥π(xT ))

+

(cid:88)

t>1

DKL (p(xt−1|xt, x0)∥pθ(xt−1|xt)) ,

(4)

where DKL is the Kullback-Leibler divergence. This
objective ensures that the neural network is trained to
minimize the distance between pθ(xt−1|xt) and the true
posterior of the forward process when conditioned on x0.
The denoising network is generally applied to parametrize
the reverse mean µθ(x, t) of the distribution of the reverse
transition pθ(xt−1|xt) := N (xt−1; µθ(xt, t), Σθ(xt, t)) [27].
The true value of the reverse mean is a function of x0, which
is unknown in the reverse process and must therefore be
estimated using input timestep t and the noisy data xt.
Specifically, the reverse mean is formulated as the following:

µ(xt, t) :=

√

¯αt−1(1 − ¯αt−1)xt +
1 − ¯αt

√

¯αt−1(1 − αt)x0

,

(5)

where the original data x0 is unavailable in the reverse
process and must therefore be estimated. We denote the
denoising network’s prediction of the original data as ˆx0.
This prediction ˆx0 can then be used to obtain µθ(xt, t) using
Equation 5. Parametrizing with ˆx0 directly is beneficial
at the beginning of sampling, since predicting ˆx0 directly
helps the denoising network to learn higher-level structural
features [115].

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

3

Fig. 2. Left: Shows qualitative generation results from diffusion models conditioned using self-supervised guidance signals. Right: Shows qualitative
results of downstream image tasks that leverage representations learned in training diffusion models. Adapted from Li et al. [100], Hu et al. [73],
Pan et al. [130], Baranchuk et al. [15], Yang and Wang [173].

[68] suggest fixing the covariance Σθ(xt, t) to a constant
value, which enables rewriting the parametrized reverse
mean as a function of the added noise ϵ(xt, t) instead of
x0:

µθ(xt, t) =

(cid:18)

xt −

1
√
αt

1 − αt
√
1 − ¯αt

(cid:19)

ϵθ(xt, t).

(6)

This reparametrization allows for the derivation of a
simplification of the objective Lvlb which we denote Lsimple
that measures the distance between the predicted noise
ϵθ(xt, t) and the actual noise ϵt as follows:

Lsimple = Et∼[1,T ]Ex0∼p(x0)Eϵt∼N (0,I) ∥ϵt − ϵθ(xt, t)∥2 .

(7)
Instead of predicting the mean and covariance directly,
the network is now parametrized to predict the added noise
for a diffusion timestep and noisy image input. The reverse
mean is obtained using Equation 6, and the covariance is
fixed. Noise prediction networks have the benefit of being
able to recover xt−1 from xt in the final sampling stages
by predicting zero noise [79]. This is more difficult for direct
parametrizations of ˆx0. There is therefore a tradeoff between
the two, where direct parametrizations can be more bene-
ficial for very noisy inputs in the initial sampling stages,
and noise prediction parametrization can be beneficial in
the latter sampling stages [27].

In efforts to improve sampling efficiency, Salimans and
Ho [143] introduce velocity prediction as a further alterna-
tive parametrization. Velocity is a linear combination of the
denoised input and the added noise, commonly defined as:

v = ¯αtϵ − (1 − ¯αt)xt.

(8)

This parametrization combines benefits of both data and
noise parametrizations, allowing the denoising network to
flexibly learn noise prediction as well as reconstruction dy-
namics based on the signal-to-noise ratio. This parametriza-
tion has led to stable results in diffusion distillation ap-
proaches [143], and can speed up generation [19].

Recently, several works [32, 133, 153, 154] further pro-
pose to think of the noise in terms of continuous instead of

discrete timesteps. Here, the diffusion process is expressed
as a continuous time-dependent function σ(t). Noise is
gradually added whenever a sample x moves forward
in time, and gradually removed if the image follows the
reverse trajectory. More specifically, the diffusion process
can be expressed using an It ˆo Stochastic Differential Equa-
tion (SDE) [83], where the vector-valued drift coefficient
f (·, t) : Rd → Rd and the scalar-valued diffusion coefficient
g(·) : R → R need to be selected when implementing a
diffusion model:

dx = f (x, t)dt + g(t)dw,

(9)

where w is the standard Wiener process. There are two
widely used choices of the SDE formulation used to model
the diffusion process. The first is the Variance-Preserving
(VP) SDE, used in the work of Ho et al. [68] which is
2 β(t)x and g(t) = (cid:112)β(t), where
given by f (x, t) = − 1
β(t) = βt as T goes to infinity. Note that this is equivalent to
the continuous formulation of the DDPM parametrization
in Equation 1. The second is the Variance-Exploding (VE)
SDE [153], resulting from a choice of f (x, t) = 0 and
g(t) =
. The VE SDE gets its name since the
variance continually increases with increasing t, whereas the
variance in the VP SDE is bounded [154]. Anderson [7] de-
rives an SDE that reverses a diffusion process, which results
in the following when applied to the Variance Exploding
SDE:

2σ(t) dσ(t)
dt

(cid:113)

dx = −2σ(t)

dσ(t)
dt

∇x log p(x; σ(t)) dt +

2σ(t)

(cid:114)

dσ(t)
dt

dw.

(10)
∇x log p(x; σ(t)) is known as the score function. This
score function is generally not known, so it needs to be
approximated using a neural network. A neural network
D(x; σ) that minimizes the L2-denoising error can be used
the score function since ∇x log p(x; σ(t)) =
to extract
D(x;σ)−x
. This idea is known as Denoising Score Matching
σ2
[161].

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

4

Fig. 3. Left: An exemplary visualization of the U-Net architecture [140]. Consists of an encoder and a decoder, with residual connections that
preserve gradient flow and low-level input details. Adapted from [135]. Right: An exemplary visualization of the DiT architecture. Shows the high-
level architecture, as well as a breakdown of the adaLN-Zero DiT block. Adapted from Peebles and Xie [132].

2.2 Backbone Architectures

We outline the mathematical foundations of diffusion mod-
els in Section 2.1. Since denoising prediction networks are
generally parametrized by parameters θ, we discuss the
formulation of θ by several neural network architectures
in the following section. All of these network architectures
map from the same input space to the same output space.

Ho et al. [68] use a U-Net backbone similar to an un-
masked PixelCNN++ [144] to approximate the score func-
tion. This U-Net architecture, originally used in semantic
segmentation approaches [30, 31, 113, 140], is based on
a Wide ResNet [182] and takes a noisy image and the
diffusion timestep t as input, encodes the image to a lower-
dimensional representation, and outputs the noise predic-
tion for that image and noise level. The U-Net consists
of an encoder and a decoder with residual connections
between blocks that preserve gradient flow and help re-
cover fine-grained details lost in the compressed represen-
tation. The encoder consists of a series of residual and self-
attention blocks and downsamples the input image to a
low-dimensional representation. The decoder mirrors this
structure, gradually upsampling the low-dimensional rep-
resentation to match the input dimensionality. The diffusion
timestep t is specified by adding a sinusoidal positional em-
bedding in each residual block [68] that scales and shifts the
input features, enhancing the network’s ability to capture
temporal dependencies.

DDPMs operate in the pixel space, making their training
and inference computationally expensive. Rombach et al.
[138] address this by proposing Latent Diffusion Models
(LDMs), which operate in the latent space of a pre-trained
variational autoencoder. The diffusion process is applied to
the generated representation as opposed to the image di-
rectly, leading to computational benefits without sacrificing
generation quality. While the authors introduce additional
cross-attention mechanisms to allow for more flexible condi-
tioned generation, the denoising network backbone remains
very close to the DDPM U-Net architecture.

Recent advances in the use of transformer architectures
for vision tasks like ViT [45] have led to the adoption of
transformer-based architectures for diffusion models. Pee-
bles and Xie [132] propose Diffusion Transformers (DiT),
a diffusion model backbone architecture that is largely in-
spired by ViTs, and demonstrates state-of-the-art generation
performance on ImageNet when combined with the LDM
framework. Following ViT, DiTs work by transforming in-
put images into a sequence of patches, which are converted

TABLE 1
An overview of different diffusion model guidance approaches.
Self-guidance [75] and [73] are both classifier and annotation-free, and
online guidance facilitates online learning.

Approach

Classifier-Free Annotation-Free Online Learning

Classifier Guidance [42]
Classifier-free Guidance [67]
Self-guidance [75, 100]
Online guidance [73]

✗
✓
✓
✓

✗
✗
✓
✓

✗
✗
✗
✓

into a sequence of tokens using a ”patchify” layer. After
adding ViT-style positional embeddings to all input tokens,
the tokens are fed through a series of transformer blocks.
These blocks are equivalent to standard ViT blocks that
take additional conditional information such as the diffusion
timestep t and a conditioning signal c as inputs. A detailed
overview of their structure can be seen in Fig 3.

U-ViTs [12] combine the U-Net and ViT backbones into a
unified backbone. U-ViTs follow the design methodology
of transformers in tokenizing time, conditioning and im-
age inputs, but additionally employ long skip connections
between shallow and deep layers. These skip connections
provide shortcuts for low-level features and therefore sta-
bilize training of the denoising network [12]. Works utiliz-
ing U-ViT-based backbones [13, 72] achieve results on par
with U-Net CNN-based architectures, demonstrating their
potential as a viable alternative to other denoising network
backbones.

2.3 Diffusion Model Guidance

Recent improvements in image generation results have
largely been driven by improved guidance approaches. The
ability to control generation by passing user-defined con-
ditions is an important property of generative models, and
guidance describes the modulation of the strength of the
conditioning signal within the model. Conditioning signals
can have a wide range of modalities, ranging from class
labels, to text embeddings to other images. A simple method
to pass spatial conditioning signals to diffusion models
is to simply concatenate the conditioning signal with the
denoising targets and then pass the signal through the
denoising network [12, 75]. Another effective approach uses
cross-attention mechanisms, where a conditioning signal c
is preprocessed by an encoder to an intermediate projection
E(c), and then injected into the intermediate layer of the
denoising network using cross-attention [76, 142]. These
conditioning approaches alone do not leave the possibility

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

5

to regulate the strength of the conditioning signal within the
model. Diffusion model guidance has recently emerged as
an approach to more precisely trade-off generation quality
and diversity.

Dhariwal and Nichol [42] use classifier guidance, a
compute-efficient method leveraging a pre-trained noise-
robust classifier to improve sample quality. Classifier guid-
ance is based on the observation that a pre-trained diffusion
model can be conditioned using the gradients of a classifier
parametrized by ϕ outputting pϕ(c|xt, t). The gradients of
the log-likelihood of this classifier ∇xt log pϕ(c|xt, t) can be
used to guide the diffusion process towards generating an
image belonging to class label y. The score estimator for
p(x|c) can be written as

∇xt log (pθ(xt)pϕ(c|xt)) = ∇xt log pθ(xt)+∇xt log pϕ(c|xt).

(11)
By using Bayes’ theorem, the noise prediction network

can then be rewritten to estimate:

ˆϵθ(xt, c) = ϵθ(xt, c) − wσt∇xt log pϕ(c|xt),

(12)

where the parameter w modulates the strength of the
conditioning signal. Classifier guidance is a versatile ap-
proach that increases sample quality, but it is heavily reliant
on the availability of a noise-robust pre-trained classifier,
which in turn relies on the availability of annotated data,
which is not available in many applications.

To address this limitation, Classifier-free guidance
(CFG) [67] eliminates the need for a pre-trained classifier.
CFG works by training an unconditional diffusion model
parametrized by ϵθ(xt, t, ϕ) together with a conditional
model parametrized by ϵθ(xt, t, c). For the unconditional
model, a null input token ϕ is used as a conditioning signal
c. The network is trained by randomly dropping out the
conditioning signal with probability puncond. Sampling is
then performed using a weighted combination of condi-
tional and unconditional score estimates:

˜ϵθ(xt, c) = (1 + w)ϵθ(xt, c) − wϵθ(xt, ϕ).

(13)

This sampling method does not rely on the gradients of a
pre-trained classifier but still requires an annotated dataset
to train the conditional denoising network. Fully uncondi-
tional approaches have yet to match classifier-free guidance,
though recent works using diffusion model representations
for self-supervised guidance show promise [73, 100]. These
methods do not need annotated data, allowing the use of
larger unlabelled datasets.

Table 1 shows the requirements of current guidance
methods. While classifier and classifier-free guidance im-
prove generation results, they require annotated training
data. Self-guidance and online guidance are fully self-
supervised alternatives that achieve competitive perfor-
mance without annotations.

Classifier and classifier-free guidance are controlled gen-
eration methods that rely on conditional training. Training-
free approaches modify the generation process of a pre-
trained model by binding multiple diffusion processes [14]
or using time-independent energy functions [179]. Other
controlled generation methods take a variational perspec-
tive [54, 119, 146, 164], treating controlled generation as a
source point optimization problem [17]. The goal is to find

samples x that minimize a loss function L(x) and are likely
under the model’s distribution p. The optimization is formu-
lated as minx0 L(x), where x0 is the source noise point. The
loss function L(x) can be modified for conditional sampling
to generate a sample belonging to a particular class y.

3 METHODS
Having covered the main preliminaries for diffusion mod-
els, we outline a series of methods related to diffusion
models and representation learning in the following sec-
tion. In subsection 3.1 we describe and categorize current
frameworks utilizing representations learned by pre-trained
diffusion models for downstream recognition tasks. In sub-
section 3.2, we describe methods that leverage advances in
representation learning to improve diffusion models them-
selves.

3.1 Diffusion Models for Representation Learning

Learning useful representations is one of the main moti-
vations for designing architectures like VAEs [88, 89] and
GANs [22, 84]. Contrastive learning approaches, where the
goal is to learn a feature space in which representations of
similar images are very close together, and vice versa for
dissimilar images (e.g. SimCLR [34], MoCo [60]), have also
led to significant advances in representation learning. These
contrastive methods are not fully self-supervised however,
since they require supervision in the form of augmentations
that preserve the original content of the image.

Diffusion models offer a promising alternative to these
approaches. While diffusion models are primarily designed
for generation tasks, the denoising process encourages the
learning of semantic image representations [15], that can
be used for downstream recognition tasks. The diffusion
model learning process is similar to the learning process of
Denoising Autoencoders (DAE) [18, 162], which are trained
to reconstruct images corrupted by adding noise. The main
difference is that diffusion models additionally take the
diffusion timestep t as input, and can thus be viewed as
multi-level DAEs with different noise scales [169]. Since
DAEs learn meaningful representations in the compressed
latent space, it is intuitive that diffusion models exhibit
similar representation learning capabilities. We outline and
discuss current approaches in the following section.

3.1.1 Leveraging intermediate activations

Baranchuk et al. [15] investigate the intermediate activations
from the U-Net network that approximates the Markov
step of the reverse diffusion process in DDPMs [42]. They
show that for certain diffusion timesteps, these intermediate
activations capture semantic information that can be used
for downstream semantic segmentation. The authors take
a noise-predictor network ϵθ(xt, t) trained on the LSUN-
Horse [177] and FFHQ-256 [84] datasets and extract feature
maps produced by one of the network’s 18 decoder blocks
for label-efficient downstream segmentation tasks. Selecting
the ideal diffusion timestep and decoder block activation
to extract is non-trivial. To understand the efficacy of pixel-
level representations of different decoder blocks, the authors
train a multi-layer perceptron (MLP) to predict the semantic

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

6

TABLE 2
Summary of the methods using diffusion models for representation learning.

Paradigm

Generative Augmentation

Leveraging Intermediate Activations

Diffusion Model Reconstruction

Downstream Task

Classification

Semantic Segmentation

Classification

Semantic Segmentation

Panoptic Segmentation

Semantic Correspondence

Depth Estimation

Image Editing

Classification

Semantic Segmentation

Image Editing

Image Interpolation

Diffusion Model Knowledge Transfer

Classification

Joint Diffusion Models

Classification

Semantic Segmentation

Method

Generative Augmentation [10]
MA-ZSC [150]

ScribbleGen [148]

GDC [125]
DifFormer [126]
DDAE [169]

DDPM-Seg [15]
VDM [187]

ODISE [170]

DIFT [157]
SD+DINO [183]
Diffusion Hyperfeatures [116]
SD4Match [103]
USCSD [62]

VDM [187]

P2PCAC [65]
Plug-and-Play Diffusion Features [160]

SODA [82]
l-DAE [35]
DiffMAE [166]

MDM [130]

DiffAE [134]
PDAE [186]

InfoDiffusion [165]
SmoothDiffusion [58]

DiffusionClassifier [95]
RepFusion [173]
DreamTeacher [96]

JDM [40]
HybViT [174]

ADDP [158]

label from features produced by different decoder blocks
on a specific diffusion step t. The representations from a
fixed set of blocks B of the pre-trained U-Net decoder and
higher diffusion timesteps are upsampled to the image size
using bilinear interpolation and concatenated. The obtained
feature vectors are then used to train an ensemble of in-
dependent MLPs which predict a semantic label for each
pixel. The final prediction is obtained by majority voting.
This method, denoted DDPM-Seg, outperforms baselines
that exploit alternative generative models and achieves seg-
mentation results competitive with MAE [61], illustrating
that intermediate denoising network activations contain
semantic image features.

Xiang et al. [169] extend this approach to further ar-
chitectures and image recognition on CIFAR-10 and Tiny-
ImageNet. They investigate the discriminative efficacy of
extracted features for different backbones (U-Net and DiT
[132]) under different frameworks (DDPM and EDM [85]).
The relationship between feature quality and layer-noise
combinations is evaluated through grid search, where the
quality of feature representations is determined using linear
probing. The best-performing features lie in the middle of
up-sampling using relatively small noising levels, which is
in line with conclusions drawn in DDPM-Seg [15]. Bench-

mark comparisons against diffusion-based methods like Hy-
bViT [174] and SBGC [190] on CIFAR-10 and Tiny-ImageNet
[41] show that EDM-based Denoising Diffusion Autoen-
coders (DDAEs) outperform previous supervised and un-
supervised diffusion-based methods on both generation
and recognition, especially after fine-tuning. Benchmarking
against contrastive learning methods shows that the EDM-
based DDAE is comparable with Sim-CLRs considering
model sizes, and outperforms SimCLRs with comparable
parameters on CIFAR-10 and Tiny-ImageNet.

ODISE [170] is a related approach that unites text-
to-image diffusion models with discriminative models to
perform panoptic segmentation [90, 91], a segmentation
approach unifying instance and semantic segmentation into
a common framework for comprehensive scene understand-
ing. ODISE extracts the internal features of a pre-trained
text-to-image diffusion model. These features are input to
a mask generator trained on annotated masks. A mask
classification module then categorizes each generated bi-
nary mask into an open vocabulary category by relating the
predicted mask’s diffusion features with text embeddings of
object category names. The authors use the Stable Diffusion
U-Net DDPM backbone and extract features by computing
a single forward pass and extracting the intermediate ac-

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

7

tivations f = UNet(xt, τ (s), t) where τ (s) is an encoded
representation of the image caption s obtained leveraging a
pre-trained text encoder τ . Interestingly, the authors obtain
the best results using t = 0, whereas previous methods
obtain better results using higher diffusion timesteps. To
overcome reliance on available image captions, Xu et al.
[170] additionally train an MLP-based implicit captioner
that computes an implicit text embedding from the image
itself. ODISE establishes a new state-of-the-art in open-
vocabulary segmentation and is a further example of the
rich semantic representations learned by denoising diffusion
models.

Mukhopadhyay et al. [125] also propose leveraging in-
termediate activations from the unconditional ADM U-Net
architecture [42] for ImageNet classification. The method-
ology for layer and timestep selection is similar to previ-
ous approaches. Additionally, the impact of different sizes
for feature map pooling is evaluated and several different
lightweight architectures for classification (including linear,
MLP, CNN, and attention-based classification heads) are
used. Feature quality is found to be mostly insensitive to
pooling size, and is mostly dependent on time steps and
the selected block number. Their approach, which we term
guided diffusion classification (GDC), achieves competitive
performance against other unified models, namely BigBi-
GAN [44] and MAGE [99]. The attention-based classification
heads perform best on ImageNet-50, but perform poorly on
Fine-Grained Visual Classification datasets, indicating their
reliance on a large amount of available data.

In a continuation of their previous work, Mukhopad-
hyay et al. [126] extend this approach by introducing two
methods for more fine-grained block and denoising time
step selection. The first is DifFormer [126], an attention
mechanism replacing the fixed pooling and linear classi-
fication head from [125] with an attention-based feature
fusion head. This fusion head is designed to replace the
fixed flattening and pooling operation required to generate
vector feature representations from the U-Net CNN used
in the GDC approach with a learnable pooling mechanism.
The second mechanism is DifFeed [126], a dynamic feedback
mechanism that decouples the feature extraction process
into two forward passes. In the first forward pass, only
the selected decoder feature maps are stored. These are
fed to an auxiliary feedback network that learns to map
decoder features to a feature space suitable for adding them
to the encoder blocks of corresponding blocks. In the second
forward pass, the feedback features are added to the encoder
features, and the DifFeed attention head is used on top
of those second forward pass features. These additional
improvements further increase the quality of learned rep-
resentations and improve ImageNet and fine-grained visual
classification performance.

The previously described diffusion representation learn-
ing methods focus on segmentation and classification,
which are only a subset of downstream recognition tasks.
Correspondence tasks are another subset that generally in-
volves identifying and matching points or features between
different images. The problem setting is as follows: Consider
two images I1 and I2 and a pixel location p1 in I1. A
correspondence task involves finding the corresponding
pixel location p2 in I2. The relationship between p1 and

p2 can be semantic (pixels that contain similar semantics),
geometrical (pixels that contain different views of an object)
or temporal (pixels that contain the same object deforming
over time). DIFT (Diffusion Features) [157] is an approach
leveraging pre-trained diffusion model representations for
correspondence tasks. DIFT also relies on extracting dif-
fusion model features. Similarly to previous approaches,
diffusion timestep and network layer numbers used for
extraction are an important consideration. The authors ob-
serve more semantically meaningful features for large dif-
fusion timesteps and earlier network layer combinations,
whereas lower-level features are captured in smaller dif-
fusion timesteps and later denoising network layers. DIFT
is shown to outperform other self-supervised and weakly-
supervised methods across a range of correspondence tasks,
showing on-par performance with state-of-the-art methods
on semantic correspondence specifically.

Zhang et al. [183] evaluate how learned diffusion fea-
tures relate across multiple images, instead of focusing on
downstream tasks for single images. To investigate this, they
employ Stable Diffusion features for semantic correspon-
dence as well. The authors observe that Stable Diffusion
features have a strong sense of spatial layout, but some-
times provide inaccurate semantic matches. DINOv2 [128],
a method for self-supervised representation learning using
knowledge distillation and vision transformers, produces
more sparse features that provide more accurate matches.
Zhang et al. [183] therefore propose to combine the two fea-
tures and employ zero-shot evaluation of nearest neighbor
search on the combined features to achieve state-of-the-art
performance on several semantic correspondence datasets
like SPair-71k and TSS.

SD4Match [103] builds on this approach by using various
prompt tuning and conditioning techniques. One method,
SD4Match-Class, fine-tunes prompt embedding Θ for each
semantic class using a semantic matching loss [102]. Given
t and IB
images IA
t , the Stable Diffusion U-Net f (·) extracts
t and FB
feature maps FA
t by Ft = f (It, t, Θ). Correspon-
dence points are predicted by normalizing feature maps and
computing a correlation map, which is converted to a prob-
ability distribution using a softmax operation. Additionally,
Li et al. [103] propose conditioning prompts on input im-
ages using a Conditional Prompting Module (CPM), which
includes a DINOv2 feature extractor, linear layers, and an
adaptive MaxPooling layer. The conditioning embedding
Θcond is formed by concatenating feature representations
and projecting them to the prompt embedding dimension.
The final prompt ΘAB is obtained by appending Θcond to
a global prompt Θglobal. This method sets new benchmark
accuracies on SPair-71k [122], PF-Willow, and PF-Pascal [59],
surpassing methods like DIFT [157] and SD+DINO [183]

Luo et al. [116] introduce Diffusion Hyperfeatures, a
framework designed to consolidate multiple intermediate
activation maps across diffusion timesteps for downstream
recognition. Activations are consolidated using an inter-
pretable aggregation network, that takes the collection of
intermediate feature maps as input and produces a single
feature descriptive feature map as output. While other
approaches manually select fixed diffusion timesteps and
activations from a pre-determined number of intermediate
network layers, Diffusion Hyperparameters cache all feature

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

8

maps across all layers and timesteps in the diffusion process
to generate a dense set of activations. This high dimensional
set of activations is upsampled, passed through a bottleneck
layer B and weighed with a unique learnable mixing weight
wl,s for each layer and timestep combination. The final
diffusion hyperfeatures take on the form

S
(cid:88)

L
(cid:88)

s=0

l=1

wl,sBl(rl,s),

(14)

where L is the number of layers, S is a subsample
of the number of diffusion timesteps and r is an activa-
tion feature map. Bottleneck layers and mixing weights
are finetuned on the specific downstream task. Similar to
previous approaches, Diffusion Hyperfeatures is used for
semantic correspondence. The authors extract activations
from Stable-Diffusion and tune the aggregation network on
a subset of SPair-71k. Diffusion Hyperfeatures outperforms
models that use self-supervised descriptors or supervised
hypercolumns on the SPair-71k and CUB datasets.

Hedlin et al. [62] focus on optimizing the prompt embed-
dings by exploiting intermediate attention maps specifically.
Given a certain input text prompt, these attention activation
maps correspond to the semantics of the prompt. Instead
of optimizing a global or a class-dependent prompt embed-
ding Θ using the semantic loss, Hedlin et al. [62] optimize
the embedding to maximize the cross-attention at the loca-
tion of interest. Locating corresponding points in a second
image then comes down to conditioning on the optimized
prompt, and selecting the point with the pixel attaining
the maximum attention map value within the target image.
Note that this approach does not utilize supervised training
specific to semantic correspondence. However, they require
test-time optimization which is costly. Text prompts are
optimized using an off-the-shelf diffusion model without
fine-tuning. Several further works building on aforemen-
tioned approaches [120, 184] exist, showing that exploiting
pre-trained diffusion models for semantic correspondence
remains a promising application of diffusion models.

Zhao et al. [187] propose Visual Perception with a pre-
trained Diffusion Model (VDM), a framework closely re-
lated to USCSD [62] that employs a text feature refinement
network as well as an additional recognition encoder for
semantic segmentation and depth estimation. Here, the de-
noising network is fed with refined text representations as
well as an input image, and the resulting feature maps as
well as the cross-attention maps between the text and image
features are used to provide guidance for a decoder. To
achieve this, the prediction model is written as pϕ(y|x, S),
where S represents the set of all category labels of the
downstream task. The prediction model is implemented as
the following:

pϕ(y|x, S) = pϕ3 (y|F)pϕ2 (F|x, C)pϕ1(C|S),

(15)

where F denotes the set of feature maps and C denotes
the text features. Here, pϕ1 (C|S) denotes a text adapter con-
sisting of a two-layer MLP that refines the text features ob-
tained by applying the CLIP text encoder to a text template
of "a photo of a [CLS]". pϕ2 (F|x) extracts the feature
maps from the denoising network given the input image x
and the set of refined text features C. The authors use t = 0

when feeding the denoising network the latent represen-
tation of the input image generated by using the VQGAN
encoder [47] to obtain feature maps F . Finally, pϕ3(y|F)
serves as a light-weight prediction head implemented as
a semantic feature pyramid network [90] that is adapted
to the downstream task. VDM is evaluated on semantic
segmentation and depth estimation, and achieves highly
competitive performance and fast convergence compared to
methods with other pre-training paradigms.

A more indirect application of text-to-image diffusion
model representations is instructional image editing [23, 51,
98], where the desired image edit is described by a natural
language instruction rather than a description of the desired
new image [81]. Prompt-based image editing is challenging
since small changes in the textual prompt can lead to vastly
different generation outcomes. [65] propose a textual editing
method for pre-trained text-conditioned diffusion models
that leverages the semantic strength of the intermediate
cross-attention layers in the denoising backbone. This ap-
proach is based on a key observation also employed in
[62, 187]: Cross-attention maps contain rich information on
the spatial layout and geometry of the generated image. In-
jecting the cross-attention layers obtained when generating
an image I into the generation process of the edited image
I ∗ ensures that the edited image preserves the original
spatial layout. Hertz et al. [65] use Imagen [141] to conduct
experiments and demonstrate promising results on text-only
localized editing, global editing, and real image editing. Fol-
lowing works like Plug-and-play Diffusion Features [160]
further improve upon this by leveraging all intermediate
activation maps to enable instructional image editing. Other
techniques like TokenFlow [52] and work by Yatim et al.
[175] have extended this idea to the video space, using
diffusion features to enable prompt-based video editing
text-driven motion transfer.

3.1.2 A general representation extraction framework

Many of the methods outlined in the previous section follow
a similar procedure in leveraging learned representations of
pre-trained diffusion models for downstream vision tasks.
In this section, we aim to consolidate these approaches to a
common three-step framework. We do this to provide clarity
on the relationship between diffusion models and their use
for downstream predictive tasks. To leverage intermediate
activations for downstream tasks, a selection methodology
that outputs the ideal diffusion timestep input as well as the
intermediate layer number(s) whose activation maps have
the highest predictive performance when upsampled and
linearly probed must be applied. This can be a trainable
model [116], a grid search procedure [169] or a learning
agent [173]. The goal of this methodology is generally to
select timestep t ∈ T and a set of decoder block numbers
B that maximize predictive performance on a downstream
task. Given a set of possible timesteps T and a set of decoder
blocks B, the goal is to find:

(t∗, B∗) = arg min

t∈T,B⊆B

Ldiscr(t, B)

(16)

where Ldiscr(t, B) represents the discriminative loss at
timestep t when the blocks in B are used for downstream
prediction. Generally, discriminative tasks will require more

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

9

Fig. 4. A high-level overview of a framework for extracting representations from pre-trained diffusion models for downstream tasks.

high-level features corresponding to structural elements and
shapes, whereas generative tasks mapping random noise to
images will require the computation of lower-level features.
The ideal intermediate layer number as well as the optimal
diffusion timestep will largely depend on the exact down-
stream prediction task, the dataset, and the architecture of
the diffusion model used.

Once the ideal timestep and layer number are deter-
mined, an input image and the selected diffusion timestep
are passed to the diffusion model, and the intermediate acti-
vations in the selected decoder blocks computed in the for-
ward pass are extracted and generally concatenated and pre-
processed depending on the downstream task (e.g. through
upsampling, pooling, etc.). Finally, a classification head is
trained on the annotated dataset, taking the preprocessed
features extracted from the diffusion model as input. This
classification head can be an MLP, a CNN, or an attention-
based network depending on the availability of labeled data
and predictive performance on the dataset. The diffusion
model weights are usually frozen in this probing process,
but additional fine-tuning regimes can increase discrimina-
tive performance for certain datasets and architectures (see
e.g., Xiang et al. [169]). Fig. 4 shows an overview of the
generalized framework.

3.1.3 Knowledge transfer

Aside from leveraging intermediate activations from pre-
trained diffusion models directly as inputs to a recognition
network, several recent approaches propose a more indirect
method of reusing learned representations for downstream
tasks. We summarize these under the term knowledge transfer
methods. This reflects the common idea of distilling repre-
sentations from pre-trained diffusion models and then trans-
ferring them to auxiliary networks in a way that is distinct
from simply providing aggregated feature activation maps
as input. Several of these approaches are discussed in the
following section.

Yang and Wang [173] propose RepFusion, a knowledge
distillation approach that dynamically extracts intermediate
representations at different time steps using a reinforcement
learning framework, and uses the extracted representations
as auxiliary supervision for student networks. Given an
input x with label y, the authors extract a pair of features,
one from the diffusion probabilistic model (DPM) and one
from the student model, where z(t) is the diffusion model

representation and z is the student model representation.
The distance between the two is minimized during training
using a loss function Lkd. After the distillation, the student
network is reapplied as a feature extractor and fine-tuned
on the available task labels. Previous approaches for using
diffusion model representations rely on grid-search to deter-
mine which diffusion timestep to use for feature extraction.
Here, the authors formulate a reinforcement learning envi-
ronment where the action space is the set of all possible
timesteps t available for selection, and the reward function
is the negative task loss −Ltask(y, g(z(t); θg)). Given the
input x, a policy network πθπ (t|x) is trained to determine
which timestep t to use for representation extraction. Once
the timestep is selected, the authors use the feature represen-
tations in the mid-block of the DPM for the selected timestep
t∗ to obtain z(t∗). After the distillation phase, the student
network is used as a feature extractor and subsequently fine-
tuned on the task label y.

i }N

i , f is trained by distilling f g

Li et al. [96] introduce DreamTeacher, a knowledge distil-
lation method using a feature regressor module that distills
the learned representations of a generative model G into a
target image recognition backbone f . Given a feature dataset
D = {xi, f g
i=1 consisting of images x and extracted
features f g
i into the intermediate
features of f (xi). The features are extracted from G by
running a forward diffusion process for T timesteps and
conducting a single denoising step to extract f g
from the
i
intermediate layers of the U-Net backbone. The extracted
features are distilled using a feature regressor module with
a top-down architecture containing lateral skip connections
that aligns the image backbone features with the generative
features. Intermediate CNN encoder features f e
l at layers
l and regressor outputs f r
l are used to compute an MSE
feature regression loss inspired by FitNet [139]:

LMSE =

1
L

L
(cid:88)

l=1

∥f r

l − W(f e

l )∥2

2

(17)

W is a non-learnable operator implemented as Lay-
erNorm [11]. This loss is combined with the activation-
based Attention Transfer (AT) objective [181], which distills
a one dimensional ”attention map” for each spatial feature.
DreamTeacher is evaluated on a range of downstream recog-
nition tasks by fine-tuning the pre-trained backbone with
additional classification heads for each task. DreamTeacher

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

10

outperforms existing contrastive and masking-based self-
supervised methods on the COCO [106], ADE20k [189] and
BDD100K [178] benchmarks.

Both RepFusion and DreamTeacher are inspired by ear-
lier works on knowledge distillation [66, 139]. Li et al. [95]
propose a slightly different knowledge transfer approach:
Diffusion Classifier, a method for zero-shot classification
that leverages conditional density estimates from text-to-
image diffusion models. This classifier converts the diffu-
sion model into a classifier by computing class conditional
likelihoods pθ(x|ci) and using Bayes’ theorem to obtain
predicted class probabilities p(ci|x). Since direct computa-
tion of pθ(x|ci) is intractable, they use the Evidence Lower
Bound (ELBO) in its place. The classifier is derived by
adding noise repeatedly and estimating noise reconstruction
losses for each class using Monte Carlo methods. While
Diffusion Classifier suffers from high inference time, it gen-
erally outperforms DDPM-Seg Baranchuk et al. [15] on most
datasets and is competitive with CLIP ResNet-50 [136] and
OpenCLIP ViT-H/14 [36].

3.1.4 Reconstructing diffusion models

Previous diffusion representation learning techniques do
not propose making fundamental modifications to diffu-
sion model architectures and training methodologies. While
these techniques often show encouraging performance for
downstream tasks, they fail to generate deep insights into
the architectural components and techniques required to
learn useful representations. It remains largely unclear for
example whether the representation learning abilities of
diffusion models are driven by the diffusion process, or
by the model’s denoising capabilities. It is also unclear
what architectural and optimization choices can improve
diffusion models’ representation learning capabilities.

Chen et al. [35] investigate these questions by decon-
structing a denoising diffusion model (DDM), modifying
individual model components to turn a DDM into a De-
noising Autoencoder. The deconstruction process consists
of three stages. In the first stage, the DDM is reoriented
for self-supervised learning. This entails the removal of
class conditioning and a reconstruction of the VQGAN
tokenizer [47] used in the DiT baseline. Both the perceptual
and adversarial loss terms rely on annotated data and are
thus removed. This essentially converts the VQGAN to a
VAE. The second stage consists of simplifying the VAE
tokenizer even further, replacing it with different autoen-
coder variants. Surprisingly, the authors find that using
simpler autoencoder variants, like patch-wise PCA, does
not degrade performance substantially. The authors con-
clude that the dimensionality per token of the latent space
has a much larger impact on probing accuracy than the
chosen autoencoder. The final deconstruction step includes
converting the DDM to predict the denoised input instead
of the added noise and removing input scaling, as well as
changing the diffusion model to operate directly in the pixel
space. This final stage results in what the authors call the
latent Denoising Autoencoder (l-DAE). They conclude that
representation learning abilities are largely driven by the
denoising-driven process rather than the diffusion process.
l-DAE is inspired by the observation that diffusion
models resemble hierarchical autoencoders with varying

noise scales. This insight is also applied in DiffAE [134],
which uses diffusion models for representation learning
via autoencoding. Preechakul et al. [134] separate latent
representations into a compact semantic representation and
a stochastic representation. DiffAE consists of a semantic
encoder, that generates a semantic representation zsem, as
well as a conditional DDIM [152]. This DDIM acts both
as the stochastic encoder, which maps x0 to xT , and as
the decoder, which maps xT to x0. xT represents the
stochastic representation and captures low-level variation,
whereas zsem encodes higher-level semantics. During infer-
ence, [134] fit a second latent DDIM to zsem, and sample
from this DDIM and xT to facilitate unconditional sampling.
Variations in xT with fixed zsem result in minor changes
in generated images, while varying z leads to different
reconstructions, showing DiffAE’s efficiency in generating
semantically meaningful and decodable representations. In-
foDiffusion [165] extends DiffAE, supporting custom priors
and improving latent representations zsem via mutual infor-
mation regularization.

Zhang et al. [186] observe that there is a gap between
the true and the predicted posterior mean of xt−1 when
predicting from xt in the diffusion reverse process. Clas-
sifier guidance can be viewed as reconstructing informa-
tion lost in the diffusion forward process by shifting the
posterior mean to fill that gap. They propose Pre-trained
DPM AutoEncoding (PDAE), a method for adapting DPMs
to decoders for image reconstruction. Instead of using a
class label y to fill this information gap, PDAE employs a
model to predict mean shift according to encoded repre-
sentations z, ensuring that z contains as much information
as possible from x0. Specifically, Zhang et al. [186] employ
an encoder Ephi(x0) = z along with a gradient estimator
Gψ(xt, z, t) that simulates ∇xt log(p(z|xt) to modify the
conditional DPM training objective. This modified objective
forces the predicted mean shift to fill the aforementioned
posterior mean gap. With a trained Gψ(xt, z, t), the score
of the implicit classifier p(z|xt) can be used analogously to
classifier-guided sampling. PDAE is evaluated using similar
experiments as used in [134] and exhibits improved training
efficiency and performance.

Pan et al. [130] propose a different method for DDM
reconstruction. They introduce a masked diffusion model
(MDM), designed for self-supervised semantic segmenta-
tion. MDM substitutes the conventional diffusion process
with a masking mechanism inspired by the masked autoen-
coder [61]. The representations learned by the pre-trained
MDM are extracted following Baranchuk et al. [15]. The
proposed MDM is a variant of a time-dependent denoising
autoencoder, that takes a masked input image and subse-
quently reconstructs the uncorrupted image. While other
DDMs and MAE use an MSE reconstruction loss, Pan et al.
[130] propose using the structural similarity index (SSIM)
loss. This is done to narrow the gap between reconstruction
and subsequent segmentation tasks. MDM is pre-trained
on a set of unlabeled images using the described self-
supervised approach. The learned representations are then
extracted to train an MLP-based classification head on a
smaller labeled dataset. Features based on specific block
setting B are extracted by selecting the activation maps from
each of the specified blocks, upsampling activation maps

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

11

to match the image size, and concatenating the activations.
The method achieves state-of-the-art results against existing
supervised segmentation methods on multiple benchmark
datasets even when only 10% of labels are available. Diff-
MAE [166] is a similar approach that uses a conditional
generative objective, where the distribution of the masked
pixels xm
0 conditioned on the visible pixels xv
0 is modeled,
and diffusion is only applied to masked regions.

Hudson et al. [82] introduce a novel view generation
learning goal as well as a bottleneck layer to aid repre-
sentation learning. They present SODA, a self-supervised
diffusion model that consists of an encoder and a denoising
decoder. The encoder produces a concise latent represen-
tation, which is used for denoising decoder guidance by
modulation of the decoder activations. The encoder E(x)
converts an input view x into a compressed latent rep-
resentation z, which is used to generate a novel output
view x′ relating to the input x. x′ is created through a
diffusion process conditioned on the latent representation
z via feature modulation. In addition to this, the authors
use layer modulation, where the latent representation is
partitioned, with each partition zi modulating a specific
pair of layer activations. This enables further specialization
among the latent subvectors, where some are optimized
to capture finer levels of granularity than others. During
training, Hudson et al. [82] opt to randomly zero out a
subset of the latent subvectors, effectively implementing
a layer-wise generalization of classifier-free guidance. This
further increases control over the generative process since
the trained model can then be conditioned using a curated
subset of latent subvectors.

SmoothDiffusion [58] is a work focusing on improving
the smoothness of the latent space of diffusion models,
which refers to the consistency of perturbations in the latent
and the image space. SmoothDiffusion enforces smooth-
ness over its latent space by proposing a novel step-wise
variation regularization method in training. The resulting
smoothed latents benefit a wide range of image interpola-
tion, image inversion and image editing tasks.

3.1.5 Joint diffusion models

Many current diffusion-based representation learning meth-
ods focus on using the diffusion model’s latent variables to
benefit the training of a separate recognition network. These
frameworks are conceptually equivalent to constructing hy-
brid models that solely concentrate on synthesis in the pre-
training stage, and on downstream recognition in the post-
training/fine-tuning phase. The recognition head and the
diffusion denoising network do not share a parametrization,
and the recognition head is often trained separately while
keeping the weights of the denoising network frozen. A
natural question that arises is whether this separation is nec-
essary and whether approaches that optimize a generative
and a discriminative objective simultaneously in a shared
parametrization can improve representation learning.

HybViT [174] is an approach that establishes a direct con-
nection between diffusion models and vision transformers
by training a single hybrid model for both image classifica-
tion and image generation. This hybrid model uses a shared
parametrization for image classification and reconstruction.
The authors use a ViT backbone to train a model with a

combined loss L consisting of a standard cross-entropy loss
to train p(y|x) and the simple denoising loss to train p(x).
HybViT provides stable training and outperforms previous
hybrid models on both generative and discriminative tasks,
but lags behind generative-only models in generation qual-
ity. HybViT also requires more training iterations to achieve
high classification performance, and the sampling speed
during inference is slow.

Joint Diffusion Models (JDM) [40] is a related work that
produces meaningful representations across generative and
discriminative tasks. Using a U-Net backbone, JDM consists
of an encoder eν , a decoder dψ, and a classifier gω. The
encoder maps an input xt to feature vectors Zt = eν(xt).
The decoder reconstructs these into a denoised sample
xt−1 = dψ(Zt), and the classifier predicts the target class
ˆy = gω(Zt). The combined training objective includes cross-
entropy loss Lclass and the noise prediction network’s sim-
plified objective Lt,diff(ν, ψ), resulting in the following loss:

L(ν, ψ, ω) = Lclass(ν, ω)−L0(ν, ψ)−

T
(cid:88)

t=2

Lt,diff(ν, ψ)−LT (ν, ψ).

JDM also enables a simplification of classifier guidance.
By applying the classifier to noisy images xt, the classifier
is effectively augmented to be robust to noise. To guide the
generated sample towards a target label, representations Zt
are optimized according to the classifier gradient, giving
Z′
t = Zt − α∇zt log gω(y|Zt). JDM achieves state-of-the-
art performance for joint models on CIFAR and CelebA
datasets, outperforming HybViT.

Tian et al. [158] propose the Alternating Denoising Diffu-
sion Process (ADDP). ADDP alternately denoises pixels and
VQ tokens. Given an image x0, a pre-trained VQ Encoder
[26] maps time image to VQ tokens z0. The alternating
diffusion process masks regions of z0 with a Markov chain
according to diffusion timestep t, producing zt. Unreliable
tokens ¯zt are generated by a token predictor and fed into a
VQ Decoder to synthesize xt, replacing the masked regions
of z0. A pixel-to-token generation network is then trained
to approximate the distribution of ¯zt−1. During sampling,
ADDP starts with a representation of pure unreliable tokens
¯zT and iteratively denoises the token sequence by predicting
¯zt−1. For recognition, the representations learned by the
pixel-to-token generation network can be forwarded to dif-
ferent task-specific recognition heads. ADDP with the VQ-
GAN tokenizer [47] MAGE-Large [99] token predictor and
ViT-Large [45] pixel-to-token encoder, outperforms previous
unified models in image classification, object detection, se-
mantic segmentation, and unconditional generation.

3.1.6 Generative augmentation

A lot of state-of-the-art representation learning methods
[33, 55, 60] rely on a fixed set of data augmentations to define
positive labels for learning representations. This approach
encourages encoders to learn to map the original and the
augmented image to similar embedding space representa-
tions [10]. These augmentations should not alter the seman-
tics of the image, and they should not render the image
unrealistic in a real-world setting. A set of standard trans-
formations might not adequately capture the distribution
of real-world data, raising the question of how to design

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

12

transformations that create diverse images and improve the
generalization of learned representations.

Ayromlou et al. [10] propose using latent diffusion mod-
els [138] to generate novel views of the original image that
preserve the semantic content, while closely following the
distribution of real images. This augmentation method is
denoted by:

T0(x) =

(cid:40)

G(z; ϕ(x))
x

if p ≤ p0
otherwise,

(18)

where G denotes a conditional generative model taking
noise vector z ∼ N (0, I) and condition vector ϕ(x) as
inputs. ϕ is a pre-trained image encoder such as CLIP
[136], p ∈ [0, 1] is a random number and p0 is a hyper-
parameter specifying the probability of applying the aug-
mentation. Ayromlou et al. [10] show that using generative
augmentation leads to consistent improvements in learned
representations over standard transformations across other
representation learning techniques.

Shipard et al. [150] take this approach one step further,
using Stable Diffusion to generate a fully synthetic dataset to
improve model-agnostic zero-shot classification (MA-ZSC).
They use Stable Diffusion, employing several variations of
prompts designed to increase the diversity of the synthetic
dataset. An image classifier is subsequently trained on this
synthetic dataset, and zero-shot classification results on CI-
FAR10, CIFAR100, and EuroSAT [64] are evaluated. Shipard
et al. [150] observe substantial classification architecture-
agnostic improvements on the aforementioned datasets,
achieving comparable performance to state-of-the-art zero-
shot classification methods like CLIP.

Moving beyond classification, Schnell et al. [148] apply
similar ideas to scribble-supervised segmentation [104, 129],
a weakly-supervised form of semantic segmentation that
uses sparse annotations in the form of scribbles drawn
over the images. They introduce ScribbleGen, a diffusion
model conditioned on semantic scribbles that generates syn-
thetic training images for data augmentation. ScribbleGen
utilizes a ControlNet [185] denoising diffusion model for
noise prediction given xt and conditioning signal c. The
number of classes is denoted by different color scribbles in
RGB images, and the conditioning signal c is supplemented
by a text prompt stating all classes in the image. Schnell
et al. [148] trade-off photorealism and image diversity by
introducing an encode ratio λ ∈ [0, 1]. This diffusion param-
eter controls the number of noise-adding forward diffusion
steps, where λ = 1 leads to no change but λ < 1 leads
to λ · T steps, meaning less noise is added to the input
image. The authors evaluate both a fixed and an adaptive
λ, where the encoding ratio is gradually increased to pro-
vide increasingly diverse synthetic images during training.
ScribbleGen achieves state-of-the-art performance on the
PASCAL VOC12 segmentation dataset [48] using scribbles
from Scribblesup [104].

DiffuMask [167] is another generative augmentation
method designed to improve downstream semantic seg-
mentation tasks. The idea here is to exploit cross-attention
maps between text prompts and generated images to extend
image synthesis to semantic mask generation. Synthetically
generated masks are used for data augmentation to improve

downstream segmentation performance. Individual token
attention maps of all layers are averaged and converted to
binary masks using an adaptive threshold mechanism based
on an AffinityNet [4]. Additionally, a noise-learning module
prunes low-quality segmentation masks, and the authors
employ several prompt engineering and static image trans-
formations to further enhance the diversity of the generated
images and corresponding segmentation masks.

3.2 Representation Learning for Diffusion Model Guid-
ance

Despite the remarkable performance of generative mod-
els, there exists a gap in quality between conditional and
unconditional image generation approaches [25]. This is
especially the case for GANs [53], which suffer from mode
collapse when trained in a fully unsupervised setting [110].
Unconditional GANs often fail to accurately model multi-
modal distributions, e.g. not being able to generate all digits
for MNIST [110]. Class-conditional GANs [22] [123] miti-
gate this issue, but require labeled data. Recent approaches
like self-conditioned GANs [110] and instance-conditioned
GANs [25] attempt to train conditional GANs without re-
quiring labeled data, and are able to achieve competitive
generation results.

Diffusion models have since surpassed the image gen-
eration capabilities of GANs [42], but suffer from a simi-
lar performance discrepancy between conditional and fully
self-supervised approaches. Current state-of-the-art diffu-
sion models are conditional models that rely on guidance
approaches that also require annotated data. Self-supervised
guidance approaches can leverage much larger unlabeled
datasets for pre-training, and thus have the potential to tran-
scend current image generation approaches. One intuitive
approach for leveraging representation learning to facilitate
these guidance methods is to explore methods that assign
labels to unlabeled data, e.g. through clustering and clas-
sification approaches. We introduce several approaches in
the following section. Fig. 5 shows a proposed taxonomy of
representation learning techniques for diffusion guidance.

3.2.1 Assignment-based guidance
Sheynin et al. [149] propose kNN-Diffusion, an efficient
text-to-image diffusion model trained without large-scale
image text pairings. To facilitate text-guided image gener-
ation without paired text-image data, a shared text-image
encoder mapping text-image pairs into the same latent space
is required. The authors use CLIP to achieve this, a pre-
trained encoder trained using contrastive loss on a large-
scale text-image pair dataset. kNN-Diffusion leverages k-
Nearest-Neighbors search to generate k embeddings from a
retrieval model. The retrieval model uses the input image
representation during training, and the text prompt rep-
resentation curing inference. This approach eliminates the
need for annotated data but still requires a pre-trained en-
coder like CLIP, which in turn requires a large-scale dataset
of text-image embeddings for pre-training.

Blattmann et al. [20] propose retrieval-augmented dif-
fusion models (RDM), which equip diffusion models with
an image database for composing new scenes based
on retrieved images. Inspired by advances in retrieval-
augmented NLP [21, 168], RDM enhances performance with

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

13

Fig. 5. A hierarchical overview of current diffusion model training frameworks that leverage representation learning techniques for conditional
generation and guidance.

fewer parameters and computational resources. Despite be-
ing trained only on images, RDM allows conditional synthe-
sis due to the shared image-text feature space of CLIP [136].
RDM includes a trainable conditional latent diffusion model
pθ, an external image database D, and a fixed sampling
strategy ξk that selects a subset M(k)
D of D based on a
query x. One strategy ξk(x, D) is to retrieve the k nearest
neighbors using a distance function d(·, x). The retrieved
data is processed through a frozen image encoder ϕ and
used to condition pθ. During training, ξk retrieves k nearest
neighbors for a query image x using cosine similarity in
CLIP’s image feature space as the distance function d(x, y).
This approach ensures that retrieved image representations
are useful for generation tasks and allows for text condi-
tioning due to CLIP’s shared feature space. The dataset
D and retrieval strategy ξk can be changed at test time,
adding flexibility for different conditioning modalities and
adaptability to other data distributions.

Hu et al. [75] propose a method also motivated by elim-
inating the need for annotated data. Self-guided diffusion
is a framework encompassing a feature extraction function
gϕ and a self-annotation function fψ. The feature extraction
function is a self-supervised feature extractor that maps the
input data x ∈ D to a feature space H, where D denotes
the dataset. This feature representation is an input of fψ,
which maps feature representation gϕ(x; D) to a guidance
signal k. This framework can be applied to achieve self-
labeled guidance, where k is a one-hot embedding derived
using k-means clustering as the self-annotation function f
on compacted features generated by gϕ. More fine-grained
spatial guidance is achieved by self-boxed guidance, which
uses a mapping from feature space H to a bounding box
as the self-annotation function f , as well as self-segmented
guidance, which uses a mapping to a segmentation mask
to generate guidance signals by clustering. Self-guidance
significantly outperforms unconditional diffusion models,
and even outperforms classifier-free guided diffusion mod-
els that use ground-truth annotations on image generation.
This suggests that the clusters are potentially more aligned
with the visual similarity of the images, and are better
guidance signals than ground-truth labels alone. While this
approach is self-supervised, it still relies on an external pre-

trained feature extractor to generate feature representations
for clustering.

For this reason, Hu et al. [73] extend their work to
propose an online feature clustering method using the
Sinkhorn-Knopp algorithm. This is challenging since the
idea requires obtaining conditioning signals for clustering
during training from a diffusion model that is dependent on
this conditioning. This issue is solved by introducing a zero
vector into the conditional diffusion model for the signals
used to identify the clustering. For each image example, the
conditional diffusion model conditioned on this zero vector
undergoes a fully-connected feature prediction head used to
compute features that are mapped to a set of learnable pro-
totypes denoted M . This method uses a combination of the
diffusion training loss and a Sinkhorn-Knopp loss to achieve
guidance signals c that are based on clustering features us-
ing M . The promise of this method is high, with self-guided
diffusion outperforming related unconditional generation
baseline comparisons on ImageNet256 and LSUN-Churches
while being competitive with class guidance methods that
rely on ground truth labels. The online approach specifically
does not rely on ground truth labels or any external pre-
trained models. Adaloglou et al. [2] build on the aforemen-
tioned cluster-based guidance approaches by utilizing EDM
[85], TEMI clustering [1] and a method for deriving an upper
cluster bound for feature-based clustering.

Other approaches to diffusion model guidance rely on
generating pseudo-labels for unlabeled data. You et al. [176]
propose dual pseudo training (DPT), which uses a classifier
trained on limited labeled data to generate pseudo-labels.
These are then used to condition a diffusion model to
generate pseudo images, which are in turn used as data
augmentation to retrain a classifier on a mix of pseudo
and real images. DPT involves three stages. First, a semi-
supervised classifier is trained on partially labeled data to
predict pseudo-labels ˆy for all images x ∈ X . Second,
a conditional generative model is trained on the dataset
S1 = {(x, y)|x ∈ X } with pseudo-labels. Finally, the classi-
fier is retrained on real data that is augmented by the gener-
ated data. DPT achieves highly competitive performance on
ImageNet classification and generation with as little as five
labels per class, outperforming several supervised diffusion

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

14

model benchmarks like ADM [42] and LDM [138].

3.2.2 A generalized framework for assignment-based guid-
ance

Assignment-based guidance approaches all rely on assign-
ing annotation to inputs during training, which enables
controlled generation during inference when conditioning
on this annotation. We therefore propose to formulate a
generalized framework that encapsulates all assignment-
based guidance approaches discussed here. This framework
consists of three main components. The first is a self-
supervised image encoder E(x), that maps inputs to a low-
dimensional feature representation z. Using a multi-modal
feature extractor like CLIP has the advantage of enabling
text-based as well as image-based conditioning, but other
feature extractors can be used, provided they generate se-
mantically meaningful image representations.

The second is a self-annotation function f (z), which uses
the image representation to produce annotation c for input
image x. In the simplest case, this self-annotation function
is an external pre-trained image classifier that generates
pseudo-class labels from image representations, similar to
the approach employed in DPT [176], where the external
classifier is subsequently re-trained on the conditionally
generated images. In other cases, the self-annotation func-
tion is a retrieval model, which uses a distance function d
to retrieve images similar to the training image, and uses
representations of the retrieved images for generating the
guidance signal c.

The final component is a denoising network Dθ(xt, c, t),
which takes the noisy image xt, the diffusion timestep t and
the guidance signal c as input, and denoises the image. Dur-
ing inference, controlled generation is enabled by passing an
initial guidance signal k (which can be multi-modal as long
as the embedding space of the encoder E is shared between
modalities) through the encoder to generate representation
z = E(k). The conditioning signal c is then generated by
passing z to the self-annotation function f where c = f (z).
Passing xt, c and t to the denoising network Dθ now enables
synthesis of novel images semantically similar to the initial
guidance signal k.

One of the main motivations behind the design of
assignment-based guidance methods is the reliance on exist-
ing methods on labeled data. While it could be argued that
the aforementioned assignment-based guidance approaches
are indirectly reliant on annotated data through the pre-
trained image encoder, it is important to note that this
encoder can be replaced with a fully self-supervised encoder
as well. CLIP relies on the availability of a large-scale dataset
of image-caption pairs and is thus not fully self-supervised,
but other representation learning methods are also able to
generate semantic representations. CLIP is used in many
approaches to facilitate both text prompt-based and image
conditioning during inference, which may no longer be pos-
sible when using primarily image-based feature extractors.
A summary of the training and inference methodology can
be found in Fig. 6.

3.2.3 Representation-based guidance

Li et al. [100] present Representation-Conditioned Image
Generation (RCG), a framework conditioning diffusion

models on a self-supervised representation distribution
mapped from the image distribution using a pre-trained en-
coder. The idea is to train a Representation Diffusion Model
(RDM) on the representations generated by a pre-trained
encoder to generate low-dimensional image representations.
After this, a pixel generator conditioned on the representa-
tion is trained to map noise distributions to image distri-
butions. RCG consists of three main components. The first
is a pre-trained image encoder, which converts the original
image distribution into a representation distribution. The
authors propose using self-supervised contrastive learning
methods (e.g. MoCo v3) for generating this representation
distribution. The second is a representation generator in the
form of an RDM, which learns to generate representations
from Gaussian noise following the DDIM [152] sampling
process. The final component is a pixel generator that crafts
image pixels conditioned on image representations. RCG
can easily incorporate classifier-free guidance for uncondi-
tional generation tasks, since the pixel generator is condi-
tioned on self-supervised representations. RCG emerges as
a highly promising method for bridging the gap between
conditional and unconditional image generation, outper-
forming pre-existing unconditional generation approaches
on ImageNet, and exhibiting competitive performance with
current state-of-the-art class-conditional approaches.

Readout Guidance (RG) [117] makes use of auxiliary
readout heads trained on top of a frozen diffusion model to
extract properties of the generated image that can be used
for guidance. These properties can include human pose,
depth maps, edges, and even higher-order properties like
similarity to another image. During sampling, the properties
extracted by the readout heads can be compared to user-
defined control targets, and used in a methodology similar
to classifier guidance [43] to guide generation.

Lin and Yang [105] identified a novel self-perceptual
objective that enhances diffusion models, enabling them to
generate more realistic samples. Contrary to the conven-
tional approach of training or employing an image encoder,
the authors demonstrate that a pre-trained diffusion model
inherently functions as a perceptual network and can be
used to generate perceptual representations. The perceptual
loss facilitates the model’s ability to generate more realistic
images even with unconditional synthesis.

Also inspired by the downsides of classifier guidance
and classifier-free guidance, Hong et al. [69] introduce Self-
Attention Guidance (SAG). SAG adversarially blurs regions
that contain salient information by leveraging intermediate
self-attention activation maps, using the residual informa-
tion as guidance. This increases the generation quality with-
out requiring external information or additional training.
The self-attention mechanism, contained in both U-Net and
DiT diffusion backbones, allows the noise predictor to at-
tend to the most informative features of the input. The self-
attention maps AS
t ∈ RN ×(HW )×(HW ) are aggregated and
reshaped to dimension RH×W using global average pooling
and nearest-neighbor upsampling to match the resolution
of xt. The difference between the blurred image ˜xt and xt
is used as conditioning, thereby retaining the information
masked in this process.

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

15

Fig. 6. A generalization of assignment-based guidance training and sampling pipelines. Samples are conditioned on annotations generated by a
self-annotation function f , using features extracted by a pre-trained image encoder (e.g., CLIP [136]).

3.2.4 Objective-based guidance

Many of the previous outlined approaches focus on elim-
inating the need for pre-trained classifiers, encoders and
dataset annotations for training conditional diffusion mod-
els. Other recent works [46, 86] have demonstrated that
internal diffusion model representations can be used to
improve generation control over the structural and semantic
composition of generated images.

One such approach is Self-guidance for Controllable Im-
age Generation [46] (which we denote SGCIG to distinguish
it from [75]). SGCIG is a zero-shot method designed to
increase user control over structural and semantic elements
of objects in images generated by text-to-image diffusion
models. Incorporating similar ideas as [65], the authors of
SGCIG leverage representations from intermediate activa-
tions and attention maps to steer the generation process. SG-
CIG works by adding a series of guidance terms to the ob-
jective of the denoising network that each define a series of
properties that can be used to perform image manipulations.
Image edits can then be carried out by guiding properties to
change in the pixel generation process. While the method
is limited to the manipulation of objects explicitly stated
in the conditioning text prompt, it represents a promising
first step towards increased control over generated images.
Diffusion Handles [131] extend this to 3D object editing,
using manipulated diffusion model activations to produce
plausible edits.

Depth-aware guidance (DAG) [86] is a related method
that uses semantic information from intermediate denoising
network layers for improved depth-aware image synthe-
sis. Kim et al. [86] propose training depth predictors with
limited depth-labeled data using internal U-Net backbone
representations, similar to DDPM-Seg [15]. The used depth
predictors are pixel-wise shallow MLP regressors estimat-
ing depth values from intermediate U-Net features ft at
timestep t. Features are concatenated across layers to form
gt, with depth maps dt = MLP(gt, t) generated using
an appended time-embedding block. This depth predictor
is trained using a limited depth-labeled dataset. To now
guide the diffusion process toward depth-aware generation,
two guidance strategies are introduced: Depth consistency
guidance uses pseudo-labels with a consistency loss Ldc

between weak and strong depth predictors, guiding the
generation process using the gradient of Ldc with respect to
xt in a methodology similar to [42]. Depth prior guidance
employs an additional small-resolution diffusion U-Net on
the depth domain, adding noise to depth predictions and
using a denoising objective Ldp. The gradient of Ldp is
treated like an external classifier gradient and added to
the image generation objective. Combining both methods
during training results in enhanced depth semantics in
generated images.

Perturbed Attention Guidance (PAG) [3] is a sampling
guidance method that improves generation quality for both
conditional and unconditional settings. PAG does not re-
quire additional training or external pre-trained models.
Instead, Ahn et al. [3] introduce an implicit discriminator
D that differentiates between desirable and undesirable
samples during the diffusion process, where y is a desirable
and ˆy is an undesirable sample. The diffusion sampling
process is then redefined to incorporate the derivative of
the discriminator loss LD. The score with undesirable label
ˆy cannot be approximated using the existing denoising net-
work ϵθ(xt). Thus the score is estimated by perturbing the
forward pass of a pre-trained denoising network, denoted
by ˆϵθ. PAG works by perturbing the self-attention maps in
the diffusion U-Net, replacing them with an identity matrix
to guide the sampling process away from degraded samples.
The final noise prediction is obtained by feeding xt into
both ϵθ(·) and ˆϵθ(·) to get the final noise prediction ˜ϵθ.
PAG improves generation quality in both conditional and
unconditional settings, and can be combined with existing
guidance methods like classifier guidance.

4 CHALLENGES & FUTURE DIRECTIONS
4.1 General Challenges

Diffusion model-based representation learning is a novel
research field with a lot of potential for theoretical and prac-
tical improvements. Improving synergies between represen-
tation learning and generative models is akin to a chicken-
and-egg problem, where better diffusion models simulta-
neously lead to higher quality image representations, and
better representation learning methods improve generative
quality of diffusion models when applied to self-supervised

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

16

guidance methods. Improved online-bootstrapping meth-
ods that provide guidance to diffusion models during train-
ing can be beneficial here.

To conserve computation in diffusion models [114], the
sampling process has been significantly reduced to just a
few steps [143, 145] or even a single step [63, 118, 155].
However, maintaining the potential of representation learn-
ing with few sampling steps presents a challenge.

4.2 Potential Future Directions

In many works discussed, the quality of representations
learned by diffusion models is evaluated indirectly using
task-specific metrics from auxiliary models. Interpretability
and disentanglement are other important ways to evaluate
representation efficacy and are currently underexplored.
Methods enhancing the interpretability of the latent space
can improve generation control and benefit a wide range
of recognition tasks. We look towards methods on inter-
pretable direction discovery as have been proposed for
GANs in [163] for inspiration, and see similar approaches
for diffusion models as promising. While there are some re-
cent works focusing on disentangled and interpretable rep-
resentation learning in diffusion models (e.g., [28, 93, 180]),
we feel that this area remains underserved.

Current diffusion-based representation learning frame-
works use U-Net and DiT backbones, which were primarily
designed for generative tasks. Developing novel architec-
tures tailored for representation learning is a promising
area of research. Current transformer-based backbones are
popular due to their scalability and performance, but their
inability for parallel inference and the quadratic complexity
of the attention mechanism are significant downsides, lim-
iting their use for high-resolution images and long videos
[76]. Techniques like windowing [112], sliding [16], and ring
attention [108] help mitigate these issues, but complexity
limitations remain. Recent works [49, 76, 171] have begun to
utilize state-space diffusion models [56, 127], which offer
linear complexity with respect to token sequence length,
and are thus well suited to long token sequence mod-
eling for both text [121] and images/video [29, 97]. The
representation-learning capabilities of these models are yet
to be fully analyzed, but we expect that conclusions drawn
from diffusion models can also be applied to state-space
models and their representation learning capabilities.

We also see significant room for further research in
using other generative models for representation learning.
Flow Matching models [5, 107, 111] have recently gained
prominence for their ability to maintain straight trajecto-
ries during generation. This characteristic results in faster
inference, making Flow Matching a suitable alternative for
addressing trajectory issues encountered in diffusion mod-
els. Their versatility has been demonstrated across various
applications, including image [38, 78], video [39], depth [57],
human motion [74], audio [94], boosting diffusion mod-
els [50, 147, 156], and even text generation [77]. The close
relationship between Diffusion and Flow Matching models
suggests that many of the diffusion representation learning
frameworks can also be applied to Flow Matching models.

ACKNOWLEDGMENTS
We would like to thank Yuki Asano, Stefan Andreas Bau-
mann, Timy Phan, and Frank Fundel for providing addi-
tional related literature.

BIBLIOGRAPHY
[1] N. Adaloglou, F. Michels, H. Kalisch, and M. Kollmann,
“Exploring the limits of deep image clustering using
pretrained models,” in BMVC. BMVA, 2023.

[2] N. Adaloglou, T. Kaiser, F. Michels, and M. Koll-
mann, “Rethinking cluster-conditioned diffusion mod-
els,” arXiv, 2024.

[3] D. Ahn, H. Cho, J. Min, W. Jang, J. Kim, S. Kim, H. H.
Park, K. H. Jin, and S. Kim, “Self-Rectifying Diffusion
Sampling with Perturbed-Attention Guidance,” arXiv,
2024.
J. Ahn and S. Kwak, “Learning pixel-level semantic affin-
ity with image-level supervision for weakly supervised
semantic segmentation,” in CVPR, 2018, pp. 4981–4990.

[4]

[5] M. S. Albergo and E. Vanden-Eijnden, “Building normal-

izing flows with stochastic interpolants,” in ICLR, 2023.

[7]

[6] N. Anand and T. Achim, “Protein Structure and Sequence
Generation with Equivariant Denoising Diffusion Proba-
bilistic Models,” arXiv, 2022.
B. D. O. Anderson, “Reverse-time diffusion equation
models,” Stochastic Processes and their Applications, vol. 12,
no. 3, pp. 313–326, 1982.
Y. M. Asano, C. Rupprecht, and A. Vedaldi, “Self-
labelling via simultaneous clustering and representation
learning,” arXiv, 2019.
J. Austin, D. D. Johnson, J. Ho, D. Tarlow, and R. van den
Berg, “Structured Denoising Diffusion Models in Discrete
State-Spaces,” in NeurIPS, vol. 34, 2021.

[9]

[8]

[10] S. Ayromlou, A. Afkanpour, V. R. Khazaie, and
F. Forghani, “Can Generative Models Improve Self-
Supervised Representation Learning?” arXiv, 2024.
J. L. Ba, J. R. Kiros, and G. E. Hinton, “Layer Normaliza-
tion,” arXiv, 2016.

[11]

[12] F. Bao, S. Nie, K. Xue, Y. Cao, C. Li, H. Su, and J. Zhu, “All
are worth words: A vit backbone for diffusion models,”
in CVPR, 2023, pp. 22 669–22 679.

[13] F. Bao, S. Nie, K. Xue, C. Li, S. Pu, Y. Wang, G. Yue, Y. Cao,
H. Su, and J. Zhu, “One transformer fits all distributions
in multi-modal diffusion at scale,” in ICML. PMLR, 2023,
pp. 1692–1717.

[14] O. Bar-Tal, L. Yariv, Y. Lipman, and T. Dekel, “Multi-
Diffusion: Fusing diffusion paths for controlled image
generation,” in ICML, vol. 202. PMLR, 2023, pp. 1737–
1752.

[15] D. Baranchuk, A. Voynov, I. Rubachev, V. Khrulkov, and
A. Babenko, “Label-efficient semantic segmentation with
diffusion models,” in ICLR, 2022.
I. Beltagy, M. E. Peters, and A. Cohan, “Longformer: The
Long-Document Transformer,” arXiv, 2020.

[16]

[17] H. Ben-Hamu, O. Puny, I. Gat, B. Karrer, U. Singer, and
Y. Lipman, “D-Flow: Differentiating through Flows for
Controlled Generation,” arXiv, 2024.

[18] Y. Bengio, L. Yao, G. Alain, and P. Vincent, “General-
ized denoising auto-encoders as generative models,” in
NeurIPS, vol. 26, 2013.

[19] Y. Benny and L. Wolf, “Dynamic dual-output diffusion

models,” in CVPR, 2022, pp. 11 482–11 491.

[20] A. Blattmann, R. Rombach, K. Oktay, and B. Ommer,

“Retrieval-augmented diffusion models,” arXiv, 2022.

[21] S. Borgeaud, A. Mensch, J. Hoffmann, T. Cai, E. Ruther-
ford, K. Millican, G. B. Van Den Driessche, J.-B. Lespiau,
B. Damoc, A. Clark et al., “Improving language models

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

17

by retrieving from trillions of tokens,” in ICLR. PMLR,
2022, pp. 2206–2240.

[22] A. Brock, J. Donahue, and K. Simonyan, “Large scale
gan training for high fidelity natural image synthesis,”
in ICLR, 2019.

[23] T. Brooks, A. Holynski, and A. A. Efros, “Instructpix2pix:
Learning to follow image editing instructions,” in CVPR,
2023, pp. 18 392–18 402.

[24] M. Caron, H. Touvron, I. Misra, H. J´egou, J. Mairal,
P. Bojanowski, and A. Joulin, “Emerging properties in
self-supervised vision transformers,” in ICCV, 2021, pp.
9650–9660.

[25] A. Casanova, M. Careil,

J. Verbeek, M. Drozdzal,
and A. Romero Soriano, “Instance-conditioned gan,” in
NeurIPS, 2021.

[26] H. Chang, H. Zhang, L. Jiang, C. Liu, and W. T. Free-
man, “Maskgit: Masked generative image transformer,”
in CVPR, 2022.

[27] Z. Chang, G. A. Koulieris, and H. P. H. Shum, “On the
Design Fundamentals of Diffusion Models: A Survey,”
arXiv, 2023.

[28] H. Chefer, O. Lang, M. Geva, V. Polosukhin, A. Shocher,
M. Irani, I. Mosseri, and L. Wolf, “The Hidden Language
of Diffusion Models,” in ICLR, 2024.

[29] G. Chen, Y. Huang, J. Xu, B. Pei, Z. Chen, Z. Li, J. Wang,
K. Li, T. Lu, and L. Wang, “Video Mamba Suite: State
Space Model as a Versatile Alternative for Video Under-
standing,” arXiv, 2024.

[30] L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and
A. L. Yuille, “Semantic Image Segmentation with Deep
Convolutional Nets and Fully Connected CRFs,” arXiv,
2016.

[31] ——, “Deeplab: Semantic image segmentation with deep
convolutional nets, atrous convolution, and fully con-
nected crfs,” IEEE Transactions on Pattern Analysis & Ma-
chine Intelligence, vol. 40, no. 4, pp. 834–848, 2017.
[32] N. Chen, Y. Zhang, H. Zen, R. J. Weiss, M. Norouzi, and
W. Chan, “WaveGrad: Estimating Gradients for Wave-
form Generation,” in ICLR, 2021.

[33] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A
simple framework for contrastive learning of visual rep-
resentations,” in ICML. PMLR, 2020, pp. 1597–1607.

[34] T. Chen, S. Kornblith, K. Swersky, M. Norouzi, and G. E.
Hinton, “Big self-supervised models are strong semi-
supervised learners,” in NeurIPS, 2020.

[35] X. Chen, Z. Liu, S. Xie, and K. He, “Deconstructing De-
noising Diffusion Models for Self-Supervised Learning,”
arXiv, 2024.

[36] M. Cherti, R. Beaumont, R. Wightman, M. Wortsman,
G.
Ilharco, C. Gordon, C. Schuhmann, L. Schmidt,
and J. Jitsev, “Reproducible scaling laws for contrastive
language-image learning,” in CVPR, 2023, pp. 2818–2829.
[37] F. Croitoru, V. Hondru, R. Ionescu, and M. Shah, “Dif-
fusion models in vision: A survey,” IEEE Transactions on
Pattern Analysis & Machine Intelligence, vol. 45, no. 09, pp.
10 850–10 869, 2023.

[38] Q. Dao, H. Phung, B. Nguyen, and A. Tran, “Flow match-

ing in latent space,” arXiv, 2023.

[39] A. Davtyan, S. Sameni, and P. Favaro, “Efficient video
prediction via sparsely conditioned flow matching,” in
ICCV, 2023, pp. 23 263–23 274.

[40] K. Deja, T. Trzci ´nski, and J. M. Tomczak, “Learning
data representations with joint diffusion models,” in Joint
European Conference on Machine Learning and Knowledge
Discovery in Databases. Springer, 2023, pp. 543–559.
J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei,
“Imagenet: A large-scale hierarchical image database,” in
CVPR, 2009.

[41]

[42] P. Dhariwal and A. Nichol, “Diffusion models beat gans

on image synthesis,” in NeurIPS, 2021.

[43] ——, “Diffusion Models Beat GANs on Image Synthesis,”

[44]

in NeurIPS, vol. 34, 2021.
J. Donahue and K. Simonyan, “Large scale adversarial
representation learning,” in NeurIPS, 2019.

[45] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn,
X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer,
G. Heigold, S. Gelly et al., “An image is worth 16x16
words: Transformers for image recognition at scale,”
ICLR, 2021.

[46] D. Epstein, A. Jabri, B. Poole, A. Efros, and A. Holynski,
“Diffusion self-guidance for controllable image genera-
tion,” in NeurIPS, vol. 36, 2023, pp. 16 222–16 239.
[47] P. Esser, R. Rombach, and B. Ommer, “Taming transform-
ers for high-resolution image synthesis,” in CVPR, 2021,
pp. 12 873–12 883.

[48] M. Everingham, L. Van Gool, C. K. Williams, J. Winn,
and A. Zisserman, “The pascal visual object classes (voc)
challenge,” International journal of computer vision, vol. 88,
pp. 303–338, 2010.

[49] Z. Fei, M. Fan, C. Yu, and J. Huang, “Scalable Diffusion

[50]

Models with State Space Backbone,” arXiv, 2024.
J. Schusterbauer, M. Gui, P. Ma, N. Stracke, S. A. Bau-
mann, and B. Ommer, “Boosting latent diffusion with
flow matching,” arXiv, 2023.

[51] Z. Geng, B. Yang, T. Hang, C. Li, S. Gu, T. Zhang,
J. Bao, Z. Zhang, H. Li, H. Hu et al., “Instructdiffusion: A
generalist modeling interface for vision tasks,” in CVPR,
2024, pp. 12 709–12 720.

[52] M. Geyer, O. Bar-Tal, S. Bagon, and T. Dekel, “Token-
flow: Consistent diffusion features for consistent video
editing,” in ICLR, 2024.
I. Goodfellow,
J. Pouget-Abadie, M. Mirza, B. Xu,
D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio,
“Generative adversarial nets,” in NeurIPS, vol. 27, 2014.

[53]

[54] A. Graikos, N. Malkin, N. Jojic, and D. Samaras, “Diffu-

[55]

sion models as plug-and-play priors,” in NeurIPS, 2022.
J.-B. Grill, F. Strub, F. Altch´e, C. Tallec, P. Richemond,
E. Buchatskaya, C. Doersch, B. Avila Pires, Z. Guo,
M. Gheshlaghi Azar, B. Piot, k. kavukcuoglu, R. Munos,
and M. Valko, “Bootstrap Your Own Latent - A New Ap-
proach to Self-Supervised Learning,” in NeurIPS, vol. 33,
2020, pp. 21 271–21 284.

[56] A. Gu and T. Dao, “Mamba: Linear-Time Sequence Mod-

eling with Selective State Spaces,” arXiv, 2024.

[58]

[57] M. Gui, J. Schusterbauer, U. Prestel, P. Ma, D. Kotovenko,
O. Grebenkova, S. A. Baumann, V. T. Hu, and B. Ommer,
“Depthfm: Fast monocular depth estimation with flow
matching,” arXiv, 2024.
J. Guo, X. Xu, Y. Pu, Z. Ni, C. Wang, M. Vasu, S. Song,
G. Huang, and H. Shi, “Smooth diffusion: Crafting
smooth latent spaces in diffusion models,” in CVPR, 2024.
[59] B. Ham, M. Cho, C. Schmid, and J. Ponce, “Proposal flow:
Semantic correspondences from object proposals,” IEEE
Transactions on Pattern Analysis & Machine Intelligence,
vol. 40, no. 7, pp. 1711–1725, 2017.

[60] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, “Mo-
mentum contrast for unsupervised visual representation
learning,” in CVPR, 2020.

[61] K. He, X. Chen, S. Xie, Y. Li, P. Doll´ar, and R. Girshick,
“Masked autoencoders are scalable vision learners,” in
CVPR, 2022.

[62] E. Hedlin, G. Sharma, S. Mahajan, H. Isack, A. Kar,
A. Tagliasacchi, and K. M. Yi, “Unsupervised semantic
correspondence using stable diffusion,” in NeurIPS, 2023.
J. Heek, E. Hoogeboom, and T. Salimans, “Multistep
consistency models,” arXiv, 2024.

[63]

[64] P. Helber, B. Bischke, A. Dengel, and D. Borth, “Eurosat:
A novel dataset and deep learning benchmark for land
use and land cover classification,” IEEE Journal of Selected
Topics in Applied Earth Observations and Remote Sensing,

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

18

vol. 12, no. 7, pp. 2217–2226, 2019.

[65] A. Hertz, R. Mokady,

J. Tenenbaum, K. Aberman,
Y. Pritch, and D. Cohen-Or, “Prompt-to-Prompt Image
Editing with Cross Attention Control,” arXiv, 2022.
[66] G. Hinton, O. Vinyals, and J. Dean, “Distilling the Knowl-

[67]

[68]

edge in a Neural Network,” arXiv, 2015.
J. Ho and T. Salimans, “Classifier-free diffusion guid-
ance,” in NeurIPS Workshop, 2021.
J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion proba-
bilistic models,” in NeurIPS, 2020.

[69] S. Hong, G. Lee, W. Jang, and S. Kim, “Improving Sample
Quality of Diffusion Models Using Self-Attention Guid-
ance,” in ICCV, 2023.

[70] E. Hoogeboom, D. Nielsen, P.

Jaini, P. Forr´e, and
M. Welling, “Argmax flows and multinomial diffusion:
Learning categorical distributions,” in NeurIPS, 2021.
[71] E. Hoogeboom, V. G. Satorras, C. Vignac, and M. Welling,
“Equivariant Diffusion for Molecule Generation in 3D,”
in Proceedings of the 39th International Conference on Ma-
chine Learning. PMLR, 2022, pp. 8867–8887.

[72] E. Hoogeboom, J. Heek, and T. Salimans, “simple diffu-
sion: End-to-end diffusion for high resolution images,” in
Proceedings of the 40th International Conference on Machine
Learning. PMLR, 2023, pp. 13 213–13 232.

[73] V. T. Hu, Y. Chen, M. Caron, Y. M. Asano, C. G. M. Snoek,
and B. Ommer, “Guided Diffusion from Self-Supervised
Diffusion Features,” arXiv, 2023.

[74] V. T. Hu, W. Yin, P. Ma, Y. Chen, B. Fernando, Y. M. Asano,
E. Gavves, P. Mettes, B. Ommer, and C. G. M. Snoek,
“Motion flow matching for human motion synthesis and
editing,” arXiv, 2023.

[75] V. T. Hu, D. W. Zhang, Y. M. Asano, G. J. Burghouts, and
C. G. Snoek, “Self-guided diffusion models,” in CVPR,
2023, pp. 18 413–18 422.

[76] V. T. Hu, S. A. Baumann, M. Gui, O. Grebenkova, P. Ma,
J. Schusterbauer, and B. Ommer, “ZigMa: A DiT-style
Zigzag Mamba Diffusion Model,” arXiv, 2024.

[77] V. T. Hu, D. Wu, Y. M. Asano, P. Mettes, B. Fernando,
B. Ommer, and C. G. M. Snoek, “Flow matching for
conditional text generation in a few sampling steps,” in
EACL, 2024.

[78] V. T. Hu, W. Zhang, M. Tang, P. Mettes, D. Zhao, and
C. Snoek, “Latent space editing in transformer-based
flow matching,” in Proceedings of the AAAI Conference on
Artificial Intelligence, vol. 38, no. 3, 2024, pp. 2247–2255.

[79] C.-W. Huang, J. H. Lim, and A. Courville, “A variational
perspective on diffusion-based generative models and
score matching,” in NeurIPS, 2021.

[80] R. Huang, J. Huang, D. Yang, Y. Ren, L. Liu, M. Li,
Z. Ye, J. Liu, X. Yin, and Z. Zhao, “Make-An-Audio: Text-
To-Audio Generation with Prompt-Enhanced Diffusion
Models,” in Proceedings of the 40th International Conference
on Machine Learning. PMLR, 2023, pp. 13 916–13 932.

[81] Y. Huang, J. Huang, Y. Liu, M. Yan, J. Lv, J. Liu, W. Xiong,
H. Zhang, S. Chen, and L. Cao, “Diffusion Model-Based
Image Editing: A Survey,” arXiv, 2024.

[82] D. A. Hudson, D. Zoran, M. Malinowski, A. K. Lampinen,
A. Jaegle, J. L. McClelland, L. Matthey, F. Hill, and A. Ler-
chner, “Soda: Bottleneck diffusion models for representa-
tion learning,” in CVPR, 2024.

[83] K. It ˆo, “Stochastic differential equations in a differentiable
manifold,” Nagoya Mathematical Journal, vol. 1, pp. 35–47,
1950.

[84] T. Karras, S. Laine, and T. Aila, “A style-based genera-
tor architecture for generative adversarial networks,” in
ICCV, 2019, pp. 4401–4410.

[85] T. Karras, M. Aittala, T. Aila, and S. Laine, “Elucidating
the design space of diffusion-based generative models,”
in NeurIPS, 2022.

[86] G. Kim, W. Jang, G. Lee, S. Hong, J. Seo, and S. Kim,

“Depth-aware guidance with self-estimated depth repre-
sentations of diffusion models,” Pattern Recognition, vol.
153, p. 110474, 2024.

[87] D. Kingma, T. Salimans, B. Poole, and J. Ho, “Variational

Diffusion Models,” in NeurIPS, vol. 34, 2021.

[88] D. P. Kingma and M. Welling, “Auto-encoding variational

bayes,” in ICLR, 2014.

[89] ——, “An Introduction to Variational Autoencoders,”
Foundations and Trends® in Machine Learning, vol. 12, no. 4,
pp. 307–392, 2019.

[90] A. Kirillov, R. Girshick, K. He, and P. Doll´ar, “Panoptic
feature pyramid networks,” in CVPR, 2019, pp. 6399–
6408.

[91] A. Kirillov, K. He, R. Girshick, C. Rother, and P. Doll´ar,
“Panoptic segmentation,” in CVPR, 2019, pp. 9404–9413.
[92] Z. Kong, W. Ping, J. Huang, K. Zhao, and B. Catanzaro,
“DiffWave: A Versatile Diffusion Model for Audio Syn-
thesis,” arXiv, 2021.

[93] M. Kwon, J. Jeong, and Y. Uh, “Diffusion models already

have a semantic latent space,” in ICLR, 2023.

[94] M. Le, A. Vyas, B. Shi, B. Karrer, L. Sari, R. Moritz,
M. Williamson, V. Manohar, Y. Adi, J. Mahadeokar et al.,
“Voicebox: Text-guided multilingual universal speech
generation at scale,” arXiv, 2023.

[95] A. C. Li, M. Prabhudesai, S. Duggal, E. Brown, and
D. Pathak, “Your diffusion model is secretly a zero-shot
classifier,” in ICCV, 2023.

[96] D. Li, H. Ling, A. Kar, D. Acuna, S. W. Kim, K. Kreis,
A. Torralba, and S. Fidler, “Dreamteacher: Pretraining
image backbones with deep generative models,” in ICCV,
2023, pp. 16 698–16 708.

[97] K. Li, X. Li, Y. Wang, Y. He, Y. Wang, L. Wang, and
Y. Qiao, “VideoMamba: State Space Model for Efficient
Video Understanding,” arXiv, 2024.

[98] S. Li, C. Chen, and H. Lu, “MoEController: Instruction-
based Arbitrary Image Manipulation with Mixture-of-
Expert Controllers,” arXiv, 2024.

[99] T. Li, H. Chang, S. Mishra, H. Zhang, D. Katabi, and
D. Krishnan, “Mage: Masked generative encoder to unify
representation learning and image synthesis,” in CVPR,
2023, pp. 2142–2152.

[100] T. Li, D. Katabi, and K. He, “Return of Unconditional
Generation: A Self-supervised Representation Generation
Method,” arXiv, 2024.

[101] X. L. Li, J. Thickstun, I. Gulrajani, P. Liang, and T. B.
Hashimoto, “Diffusion-LM Improves Controllable Text
Generation,” arXiv, 2022.

[102] X. Li, K. Han, X. Wan, and V. A. Prisacariu, “SimSC: A
Simple Framework for Semantic Correspondence with
Temperature Learning,” arXiv, 2023.

[103] X. Li, J. Lu, K. Han, and V. A. Prisacariu, “Sd4match:
Learning to prompt stable diffusion model for semantic
matching,” in CVPR, 2024, pp. 27 558–27 568.

[104] D. Lin, J. Dai, J. Jia, K. He, and J. Sun, “Scribblesup:
Scribble-supervised convolutional networks for semantic
segmentation,” in CVPR, 2016, pp. 3159–3167.

[105] S. Lin and X. Yang, “Diffusion Model with Perceptual

Loss,” arXiv, 2024.

[106] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona,
D. Ramanan, P. Doll´ar, and C. L. Zitnick, “Microsoft coco:
Common objects in context,” in ECCV, 2014.

[107] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel,
and M. Le, “Flow matching for generative modeling,” in
ICLR, 2023.

[108] H. Liu, M. Zaharia, and P. Abbeel, “Ringattention with
blockwise transformers for near-infinite context,” in
ICLR, 2024.

[109] J. Liu, C. Li, Y. Ren, F. Chen, and Z. Zhao, “DiffSinger:
Singing Voice Synthesis via Shallow Diffusion Mecha-
nism,” in AAAI, vol. 36, no. 10, 2022, pp. 11 020–11 028.

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

19

[110] S. Liu, T. Wang, D. Bau, J.-Y. Zhu, and A. Torralba,
“Diverse Image Generation via Self-Conditioned GANs,”
in CVPR, 2020, pp. 14 274–14 283.

[111] X. Liu, C. Gong, and Q. Liu, “Flow straight and fast:
Learning to generate and transfer data with rectified
flow,” in ICLR, 2023.

[112] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and
B. Guo, “Swin Transformer: Hierarchical Vision Trans-
former using Shifted Windows,” in ICCV, 2021, pp. 9992–
10 002.

[113] J. Long, E. Shelhamer, and T. Darrell, “Fully convolu-
tional networks for semantic segmentation,” in CVPR,
2015, pp. 3431–3440.

[114] A. S. Luccioni, Y. Jernite, and E. Strubell, “Power hungry
processing: Watts driving the cost of ai deployment?”
arXiv, 2023.

[115] C. Luo, “Understanding Diffusion Models: A Unified

Perspective,” arXiv, 2022.

[116] G. Luo, L. Dunlap, D. H. Park, A. Holynski, and T. Dar-
rell, “Diffusion hyperfeatures: Searching through time
and space for semantic correspondence,” in NeurIPS,
2023.

[117] G. Luo, T. Darrell, O. Wang, D. B. Goldman, and
A. Holynski, “Readout Guidance: Learning Control from
Diffusion Features,” in CVPR, 2024.

[118] W. Luo, T. Hu, S. Zhang, J. Sun, Z. Li, and Z. Zhang,
“Diff-instruct: A universal approach for transferring
knowledge from pre-trained diffusion models,” NeurIPS,
vol. 36, 2024.

[119] M. Mardani, J. Song, J. Kautz, and A. Vahdat, “A vari-
ational perspective on solving inverse problems with
diffusion models,” in ICLR, 2024.

[120] O. Mariotti, O. Mac Aodha, and H. Bilen, “Improving se-
mantic correspondence with viewpoint-guided spherical
maps,” in CVPR, 2024, pp. 19 521–19 530.

[121] H. Mehta, A. Gupta, A. Cutkosky, and B. Neyshabur,
“Long range language modeling via gated state spaces,”
in ICLR, 2023.

[122] J. Min, J. Lee, J. Ponce, and M. Cho, “SPair-71k: A Large-
scale Benchmark for Semantic Correspondence,” arXiv,
2019.

[123] M. Mirza and S. Osindero, “Conditional generative ad-

versarial nets,” arXiv, 2014.

[124] S. Mittal, K. Abstreiter, S. Bauer, B. Sch ¨olkopf, and
A. Mehrjou, “Diffusion based representation learning,”
in ICML. PMLR, 2023.

[125] S. Mukhopadhyay, M. Gwilliam, V. Agarwal, N. Pad-
manabhan, A. Swaminathan, S. Hegde, T. Zhou, and
A. Shrivastava, “Diffusion Models Beat GANs on Image
Classification,” arXiv, 2023.

[126] S. Mukhopadhyay, M. Gwilliam, Y. Yamaguchi, V. Agar-
wal, N. Padmanabhan, A. Swaminathan, T. Zhou, and
A. Shrivastava, “Do text-free diffusion models learn dis-
criminative visual representations?” arXiv, 2023.

[127] E. Nguyen, K. Goel, A. Gu, G. Downs, P. Shah, T. Dao,
S. Baccus, and C. R´e, “S4ND: Modeling images and
videos as multidimensional signals with state spaces,” in
NeurIPS, 2022.

[128] M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo,
M. Szafraniec, V. Khalidov, P. Fernandez, D. HAZIZA,
F. Massa, A. El-Nouby, M. Assran, N. Ballas, W. Galuba,
R. Howes, P.-Y. Huang, S.-W. Li, I. Misra, M. Rabbat,
V. Sharma, G. Synnaeve, H. Xu, H. Jegou, J. Mairal, P. La-
batut, A. Joulin, and P. Bojanowski, “DINOv2: Learning
robust visual features without supervision,” Transactions
on Machine Learning Research, 2024.

[129] Z. Pan, P. Jiang, Y. Wang, C. Tu, and A. G. Cohn,
“Scribble-supervised semantic segmentation by uncer-
tainty reduction on neural representation and self-
supervision on neural eigenspace,” in ICCV, 2021, pp.

7416–7425.

[130] Z. Pan, J. Chen, and Y. Shi, “Masked Diffusion as Self-
supervised Representation Learner,” arXiv, 2024.
[131] K. Pandey, P. Guerrero, M. Gadelha, Y. Hold-Geoffroy,
K. Singh, and N. J. Mitra, “Diffusion handles enabling 3d
edits for diffusion models by lifting activations to 3d,” in
CVPR, 2024.

[132] W. Peebles and S. Xie, “Scalable diffusion models with

transformers,” in ICCV, 2023.

[133] R. Po, W. Yifan, V. Golyanik, K. Aberman, J. T. Barron,
A. H. Bermano, E. R. Chan, T. Dekel, A. Holynski,
A. Kanazawa, C. K. Liu, L. Liu, B. Mildenhall, M. Nießner,
B. Ommer, C. Theobalt, P. Wonka, and G. Wetzstein,
“State of the Art on Diffusion Models for Visual Com-
puting,” arXiv, 2023.

[134] K. Preechakul, N. Chatthee, S. Wizadwongsa, and
S. Suwajanakorn, “Diffusion autoencoders: Toward a
meaningful and decodable representation,” in CVPR,
2022.

[135] S. J. Prince, Understanding Deep Learning. The MIT Press,

2023.

[136] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh,
S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark
et al., “Learning transferable visual models from natural
language supervision,” in ICML, 2021.

[137] D. J. Rezende, S. Mohamed, and D. Wierstra, “Stochas-
tic backpropagation and approximate inference in deep
generative models,” in ICML, 2014.

[138] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and
B. Ommer, “High-resolution image synthesis with latent
diffusion models,” in CVPR, 2022.

[139] A. Romero, N. Ballas, S. E. Kahou, A. Chassang, C. Gatta,
and Y. Bengio, “FitNets: Hints for Thin Deep Nets,” arXiv,
2015.

[140] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolu-
tional networks for biomedical image segmentation,” in
MICCAI. Springer, 2015, pp. 234–241.

[141] C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. L. Den-
ton, K. Ghasemipour, R. Gontijo Lopes, B. Karagol Ayan,
T. Salimans, J. Ho, D. J. Fleet, and M. Norouzi, “Photore-
alistic text-to-image diffusion models with deep language
understanding,” in NeurIPS, vol. 35, 2022, pp. 36 479–
36 494.

[142] C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. L. Den-
ton, K. Ghasemipour, R. Gontijo Lopes, B. Karagol Ayan,
T. Salimans et al., “Photorealistic text-to-image diffusion
models with deep language understanding,” in NeurIPS,
2022.

[143] T. Salimans and J. Ho, “Progressive distillation for fast

sampling of diffusion models,” in ICLR, 2022.

[144] T. Salimans, A. Karpathy, X. Chen, and D. P. Kingma,
“Pixelcnn++: A pixelcnn implementation with discretized
logistic mixture likelihood and other modifications,” in
ICLR, 2017.

[145] T. Salimans, T. Mensink, J. Heek, and E. Hoogeboom,
“Multistep distillation of diffusion models via moment
matching,” arXiv, 2024.

[146] D. Samuel, R. Ben-Ari, S. Raviv, N. Darshan, and
G. Chechik, “Generating images of rare concepts using
pre-trained diffusion models,” in AAAI, vol. 38, no. 5,
2024, pp. 4695–4703.

[147] A. Sauer, F. Boesel, T. Dockhorn, A. Blattmann, P. Esser,
and R. Rombach, “Fast high-resolution image synthesis
with latent adversarial diffusion distillation,” arXiv, 2024.
[148] J. Schnell, J. Wang, L. Qi, V. T. Hu, and M. Tang, “Scribble-
Gen: Generative Data Augmentation Improves Scribble-
supervised Semantic Segmentation,” arXiv, 2024.
[149] S. Sheynin, O. Ashual, A. Polyak, U. Singer, O. Gafni,
E. Nachmani, and Y. Taigman, “kNN-diffusion: Image
generation via large-scale retrieval,” in ICLR, 2023.

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

20

learner,” in ICCV, 2023.

[174] X. Yang, S.-M. Shih, Y. Fu, X. Zhao, and S. Ji, “Your ViT
is Secretly a Hybrid Discriminative-Generative Diffusion
Model,” arXiv, 2022.

[175] D. Yatim, R. Fridman, O. Bar-Tal, Y. Kasten, and
T. Dekel, “Space-Time Diffusion Features for Zero-Shot
Text-Driven Motion Transfer,” in CVPR, 2024.

[176] Z. You, Y. Zhong, F. Bao, J. Sun, C. Li, and J. Zhu,
“Diffusion models and semi-supervised learners benefit
mutually with few labels,” in NeurIPS, 2023.

[177] F. Yu, Y. Zhang, S. Song, A. Seff, and J. Xiao, “Lsun:
Construction of a large-scale image dataset using deep
learning with humans in the loop,” arXiv, 2015.

[178] F. Yu, H. Chen, X. Wang, W. Xian, Y. Chen, F. Liu,
V. Madhavan, and T. Darrell, “Bdd100k: A diverse driv-
ing dataset for heterogeneous multitask learning,” in
CVPR, 2020, pp. 2636–2645.

[179] J. Yu, Y. Wang, C. Zhao, B. Ghanem, and J. Zhang, “Free-
dom: Training-free energy-guided conditional diffusion
model,” ICCV, 2023.

[180] Z. Yue, J. Wang, Q. Sun, L. Ji, E. I. Chang, and H. Zhang,
“Exploring diffusion time-steps for unsupervised repre-
sentation learning,” in ICLR, 2024.

[181] S. Zagoruyko and N. Komodakis, “Paying more attention
to attention: Improving the performance of convolutional
neural networks via attention transfer,” in ICLR, 2017.

[182] ——, “Wide residual networks,” in BMVC, 2016.
[183] J. Zhang, C. Herrmann, J. Hur, L. P. Cabrera, V. Jampani,
D. Sun, and M.-H. Yang, “A tale of two features: Stable
diffusion complements DINO for zero-shot semantic cor-
respondence,” in NeurIPS, 2023.

[184] J. Zhang, C. Herrmann, J. Hur, E. Chen, V. Jampani,
D. Sun, and M.-H. Yang, “Telling left from right: Identify-
ing geometry-aware semantic correspondence,” in CVPR,
2024, pp. 3076–3085.

[185] L. Zhang, A. Rao, and M. Agrawala, “Adding conditional
control to text-to-image diffusion models,” in ICCV, 2023,
pp. 3836–3847.

[186] Z. Zhang, Z. Zhao, and Z. Lin, “Unsupervised represen-
tation learning from pre-trained diffusion probabilistic
models,” in NeurIPS, 2022.

[187] W. Zhao, Y. Rao, Z. Liu, B. Liu, J. Zhou, and J. Lu,
“Unleashing text-to-image diffusion models for visual
perception,” in ICCV, 2023, pp. 5729–5739.

[188] K. Zheng, C. Lu, J. Chen, and J. Zhu, “Improved tech-
niques for maximum likelihood estimation for diffusion
odes,” in ICML. PMLR, 2023.

[189] B. Zhou, H. Zhao, X. Puig, T. Xiao, S. Fidler, A. Barriuso,
and A. Torralba, “Semantic understanding of scenes
through the ade20k dataset,” IJCV, vol. 127, pp. 302–321,
2019.

[190] R. S. Zimmermann, L. Schott, Y. Song, B. A. Dunn,
and D. A. Klindt, “Score-based generative classifiers,”
in NeurIPS 2021 Workshop on Deep Generative Models and
Downstream Applications, 2021.

[150] J. Shipard, A. Wiliem, K. N. Thanh, W. Xiang, and
C. Fookes, “Diversity is definitely needed: Improving
model-agnostic zero-shot classification via stable diffu-
sion,” in CVPR, 2023, pp. 769–778.

[151] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and
S. Ganguli, “Deep unsupervised learning using nonequi-
librium thermodynamics,” in ICML, 2015.

[152] J. Song, C. Meng, and S. Ermon, “Denoising diffusion

implicit models,” in ICLR, 2021.

[153] Y. Song and S. Ermon, “Generative Modeling by Esti-
mating Gradients of the Data Distribution,” in NeurIPS,
vol. 32, 2019.

[154] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Er-
mon, and B. Poole, “Score-based generative modeling
through stochastic differential equations,” in ICLR, 2021.
[155] Y. Song, P. Dhariwal, M. Chen, and I. Sutskever, “Consis-

tency models,” arXiv, 2023.

[156] Y. Song, A. Keller, N. Sebe, and M. Welling, “Flow factor-

ized representation learning,” in NeurIPS, vol. 36, 2024.

[157] L. Tang, M. Jia, Q. Wang, C. P. Phoo, and B. Hariha-
ran, “Emergent correspondence from image diffusion,”
in NeurIPS, vol. 36, 2023.

[158] C. Tian, C. Tao, J. Dai, H. Li, Z. Li, L. Lu, X. Wang, H. Li,
G. Huang, and X. Zhu, “ADDP: Learning general rep-
resentations for image recognition and generation with
alternating denoising diffusion process,” in ICLR, 2024.

[159] K. Tian, Y. Jiang, Z. Yuan, B. Peng, and L. Wang, “Visual
autoregressive modeling: Scalable image generation via
next-scale prediction,” arXiv, 2024.

[160] N. Tumanyan, M. Geyer, S. Bagon, and T. Dekel, “Plug-
and-play diffusion features for text-driven image-to-
image translation,” in CVPR, 2023, pp. 1921–1930.
[161] P. Vincent, “A connection between score matching and
denoising autoencoders,” Neural computation, 2011.
[162] P. Vincent, H. Larochelle, Y. Bengio, and P.-A. Manzagol,
“Extracting and composing robust features with denois-
ing autoencoders,” in ICML, 2008, pp. 1096–1103.
[163] A. Voynov and A. Babenko, “Unsupervised discovery of
interpretable directions in the gan latent space,” in ICML.
PMLR, 2020, pp. 9786–9796.

[164] B. Wallace, A. Gokul, S. Ermon, and N. Naik, “End-
to-end diffusion latent optimization improves classifier
guidance,” in ICCV, 2023, pp. 7246–7256.

[165] Y. Wang, Y. Schiff, A. Gokaslan, W. Pan, F. Wang, C. De Sa,
and V. Kuleshov, “Infodiffusion: Representation learn-
ing using information maximizing diffusion models,” in
ICML. PMLR, 2023.

[166] C. Wei, K. Mangalam, P.-Y. Huang, Y. Li, H. Fan, H. Xu,
H. Wang, C. Xie, A. Yuille, and C. Feichtenhofer, “Diffu-
sion models as masked autoencoders,” in ICCV, 2023.

[167] W. Wu, Y. Zhao, M. Z. Shou, H. Zhou, and C. Shen, “Dif-
fumask: Synthesizing images with pixel-level annotations
for semantic segmentation using diffusion models,” in
ICCV, 2023, pp. 1206–1217.

[168] Y. Wu, M. N. Rabe, D. Hutchins, and C. Szegedy, “Mem-

orizing transformers,” in ICLR, 2022.

[169] W. Xiang, H. Yang, D. Huang, and Y. Wang, “Denoising
diffusion autoencoders are unified self-supervised learn-
ers,” in ICCV, 2023.

[170] J. Xu, S. Liu, A. Vahdat, W. Byeon, X. Wang, and
S. De Mello, “Open-vocabulary panoptic segmentation
with text-to-image diffusion models,” in CVPR, 2023, pp.
2955–2966.

[171] J. N. Yan, J. Gu, and A. M. Rush, “Diffusion models
without attention,” in CVPR, 2024, pp. 8239–8249.
[172] L. Yang, Z. Zhang, Y. Song, S. Hong, R. Xu, Y. Zhao,
W. Zhang, B. Cui, and M.-H. Yang, “Diffusion models:
A comprehensive survey of methods and applications,”
ACM Computing Surveys, vol. 56, no. 4, pp. 1–39, 2023.

[173] X. Yang and X. Wang, “Diffusion model as representation

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE

21

is a research intern in the
Michael Fuest
Computer Vision & Learning Group at Ludwig-
Maximilians-Universit ¨at M ¨unchen (LMU). He re-
cently received his master’s degree in Manage-
ment & Technology with a major in Computer
Science from the Technical University of Munich,
and is currently a visiting researcher at the MIT
Laboratory for Information & Decision Systems.

Bj ¨orn Ommer is a professor at LMU and leads
the Computer Vision & Learning Group. He was
previously with Heidelberg University’s Depart-
ment of Mathematics and Computer Science,
IWR, and HCI. He studied computer science and
physics at the University of Bonn, completed
his Ph.D. at ETH Zurich where his dissertation
received the ETH Medal, and held a post-doc po-
sition with Jitendra Malik at UC Berkeley. He is a
member of the Bavarian AI Council, an editor for
IEEE T-PAMI, an ELLIS Fellow, faculty of ELLIS
unit Munich, a PI at the Munich Center for Machine Learning (MCML),
and has held various roles at numerous CVPR, ICCV, ECCV, and
NeurIPS conferences. He delivered the opening keynote at NeurIPS’23.

Pingchuan Ma is a Ph.D. student in the Com-
puter Vision & Learning Group at Ludwig Maxi-
milian University of Munich (LMU) and a Munich
Center for Machine Learning (MCML) member.
He previously received his master’s degree in
Applied Computer Science from Heidelberg Uni-
versity, where he developed an interest in deep
metric learning and style transfer. He served as
a reviewer for CVPR 2024 and NeurIPS 2024.
His current research focuses on leveraging gen-
erative models for tasks beyond generation and

exploring multi-modality representation learning.

Ming Gui is currently a PhD researcher at the
Computer Vision & Learning Group at Ludwig
Maximilian University of Munich (LMU). He re-
ceived his bachelor’s and master’s degree in
electrical and computational engineering from
the Technical University of Munich, where he de-
veloped interest in deep learning and computer
vision. His research is presently centered around
the development and enhancement of scalable
generative models.

Johannes Schusterbauer is a PhD Student at
the University of Munich’s Computer Vision &
Learning Group. He did his undergraduate stud-
ies in Psychology and Computer Science, both
at the University of Munich, followed by a Mas-
ter’s degree in Intelligent
Interactive Systems
from Pompeu Fabra University in Barcelona. Be-
sides works in the field of generative modeling,
Johannes’ research focuses on the adaptability
of Diffusion and Flow-based models for general
image understanding.

Tao Hu is a postdoctoral research fellow at
Ludwig Maximilian University of Munich (LMU)
where they are developing next generation of
Stable Diffusion models. He obtained his Ph.D.
degree in computer science from the University
of Amsterdam in 2023 under the supervision
of Cees Snoek. He was an intern at Megvii,
Amazon AWS in 2017 and 2020. His Ph.D. re-
search has been selected for the CVPR2023
and ICCV2023 Doctoral Consortium. He co-
organized the ECCV 2024 Workshop on Audio-
Visual Generative Learning. He has also served as the Area Chair for
the CVPR AI4CC workshop and the CVPR 2024 Efficient Large Vision
Model workshop. He has been a reviewer for top-tier computer vision
and machine learning conferences for several years. His current re-
search interests lie in scalable, flexible, and efficient generative models.

