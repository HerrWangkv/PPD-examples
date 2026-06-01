Published in Computer Vision and Image Understanding, September 2025 (CVIU 2025)

STYLE TRANSFER WITH DIFFUSION MODELS FOR
SYNTHETIC-TO-REAL DOMAIN ADAPTATION

5
2
0
2

p
e
S
8
1

]

V
C
.
s
c
[

2
v
0
6
3
6
1
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

Estelle Chigot
ISAE-Supaero, University of Toulouse, France
Airbus, France
estelle.chigot2@isae.fr

Dennis G. Wilson
ISAE-Supaero, University of Toulouse, France
dennis.wilson@isae.fr

Meriem Ghrib
Airbus, France
meriem.ghrib@airbus.com

Thomas Oberlin
ISAE-Supaero, University of Toulouse, France
thomas.oberlin@isae.fr

ABSTRACT

Semantic segmentation models trained on synthetic data often perform poorly on real-world images
due to domain gaps, particularly in adverse conditions where labeled data is scarce. Yet, recent
foundation models enable to generate realistic images without any training. This paper proposes
to leverage such diffusion models to improve the performance of vision models when learned on
synthetic data. We introduce two novel techniques for semantically consistent style transfer using
diffusion models: Class-wise Adaptive Instance Normalization and Cross-Attention (CACTI) and its
extension with selective attention Filtering (CACTIF). CACTI applies statistical normalization selec-
tively based on semantic classes, while CACTIF further filters cross-attention maps based on feature
similarity, preventing artifacts in regions with weak cross-attention correspondences. Our methods
transfer style characteristics while preserving semantic boundaries and structural coherence, unlike
approaches that apply global transformations or generate content without constraints. Experiments us-
ing GTA5 as source and Cityscapes/ACDC as target domains show that our approach produces higher
quality images with lower FID scores and better content preservation. Our work demonstrates that
class-aware diffusion-based style transfer effectively bridges the synthetic-to-real domain gap even
with minimal target domain data, advancing robust perception systems for challenging real-world
applications. The source code is available at: https://github.com/echigot/cactif.

Keywords · Semantic segmentation · Synthetic data · Domain adaptation · Style transfer · Diffusion model

1

Introduction

Semantic segmentation represents a fundamental computer vision task that assigns a class label to each pixel in an
image, enabling detailed scene understanding. This capability forms a critical foundation for numerous applications
including autonomous driving, robotics, and medical image analysis. While deep learning approaches have achieved
remarkable results in this domain, their success depends heavily on the availability of large, accurately labeled datasets.

The creation of such datasets requires extensive manual annotation, a process that is both time-consuming and costly.
For example, annotating a single image in the Cityscapes dataset of urban views from a car takes approximately 90
minutes of human effort [1]. Synthetic data generated from graphic engines offer a compelling alternative, providing
automatically labeled images at scale. However, models trained exclusively on synthetic data typically perform poorly
when evaluated on real-world images due to the domain gap—differences in visual characteristics between synthetic
and real domains including lighting conditions, texture details, and object appearances.

This synthetic-to-real domain gap has motivated extensive research in domain adaptation techniques, which aim to
transfer knowledge from a labeled source domain (synthetic) to an unlabeled or sparsely labeled target domain (real).

 
 
 
 
 
 
Style Transfer with Diffusion Models for Synthetic-to-Real Domain Adaptation

Existing approaches typically fall into three categories: feature-level adaptation that aligns representations in the
model’s latent space [2, 3], output-level adaptation that focuses on prediction consistency [4], and input-level adaptation
that transforms source images to resemble the target domain [5, 6, 7].

Among input-level methods, style transfer is a promising approach. By transferring the visual characteristics of target
domain images to source domain content, these techniques can generate training data that maintains the accurate labels
of the source while exhibiting the appearance of the target. Traditional style transfer methods have focused primarily on
artistic applications, with limited exploration of their potential for domain adaptation in semantic segmentation tasks
[8, 9, 10].

Recent advances in diffusion models have greatly improved image generation capabilities, demonstrating unprecedented
quality and flexibility in content creation. These models, which generate images through an iterative denoising process,
offer powerful new mechanisms for style transfer that could potentially address the synthetic-to-real domain gap
more effectively than previous approaches. However, applying diffusion-based style transfer specifically for domain
adaptation presents unique challenges that remain largely unexplored.

In this paper, we investigate diffusion-based style transfer for synthetic-to-real domain adaptation in semantic segmenta-
tion, with a particular focus on few-shot scenarios where only limited target domain data is available. We identify two
key limitations in existing approaches: (1) standard style transfer methods apply global transformations that ignore
semantic class boundaries, leading to unrealistic appearances within object categories, and (2) direct application of
cross-attention mechanisms can create artifacts when structural correspondence between domains is weak.

To address these challenges, we propose two novel techniques: First, Class-wise Adaptive Instance Normalization
and Cross-attenTIon (CACTI) applies statistical normalization selectively based on semantic classes, ensuring that
style characteristics are transferred appropriately within semantic boundaries. Second, CACTI with Selective Attention
Filtering (CACTIF) extends this approach by selectively applying cross-attention based on feature similarity, preventing
artifacts in regions where direct style transfer might produce inconsistencies.

Our experiments demonstrate that these techniques not only produce visually coherent images but also improve
semantic segmentation performance when used for domain adaptation. Particularly noteworthy is their effectiveness
in challenging adverse weather conditions such as fog, rain, snow, and nighttime, where the domain gap is especially
pronounced. When combined with state-of-the-art segmentation architectures, our approach improves over existing
methods, highlighting the potential of semantically-aware diffusion-based style transfer for addressing the synthetic-to-
real domain gap.

2 Related work

2.1 Style Transfer with Diffusion Models

Diffusion models generate images through an iterative denoising process that progressively refines Gaussian noise into
coherent visual outputs. Recent advancements such as DDPM [11] and DALL·E 2 [12] have demonstrated the ability to
produce images of high quality and diversity. Stable Diffusion (SD) [13] further improves efficiency by performing
the denoising process in the latent space of a Variational Autoencoder (VAE), using a UNet-based architecture. This
design significantly reduces computational cost while maintaining high fidelity in generated images. Building on this
foundation, ControlNet [14] introduces a conditioning mechanism that enables control over image generation using
inputs such as human poses or segmentation maps. By incorporating zero-convolution layers during training, ControlNet
aligns conditional inputs with the denoising process, allowing for fine-grained manipulation of image attributes.

A subclass of image generation methods focuses on transferring style characteristics from a reference image to a target
content image—a process known as style transfer. Early approaches, such as the method proposed by Gatys et al. [15],
utilized Convolutional Neural Networks (CNNs) to separately extract content and style features by analyzing activations
at different convolutional layers. To improve flexibility and efficiency, Adaptive Instance Normalization (AdaIN) [16]
was introduced. AdaIN aligns the channel-wise mean and variance of the style features with those of the content image
in the latent space of a CNN, enabling arbitrary style transfer in real-time. In parallel, Generative Adversarial Networks
(GANs) [17] have also been widely adopted for style transfer tasks [18, 19, 20], at the expense of training stability and
with a risk of mode collapse.

Developments in diffusion models have made them suitable to perform style transfer tasks. Several methods leverage
text-based algorithms to stylize images by prompting a pretrained diffusion model without the need for finetuning.
These approaches use textual descriptions to guide the generation process and achieve style transfer through prompt
engineering and latent manipulation [21, 22, 23, 10]. Beyond text-based guidance, other techniques enable style
transfer by directly blending features from two input images during inference. Cross-image attention [8], Z* [24] and

2

Style Transfer with Diffusion Models for Synthetic-to-Real Domain Adaptation

StyleID [25] use KV injection in the self-attention layers of Stable-Diffusion’s UNet. Eye-for-an-eye [9] rearranges the
features in those self-attention layers following a computed semantic correspondence. Additionally, some recent works
incorporate Low-Rank Adaptation (LoRA) [26], a parameter-efficient finetuning technique, to enable more flexible
style-content fusion. These methods, such as B-LoRA [27] and ZipLoRA [28], fine-tune lightweight layers to blend
styles from multiple reference images.

Those methods are generally used to generate artistic images, and only few of them are actually employed as data
generation pipelines for other vision tasks. However, due to their large training datasets and generalization capabilities,
diffusion based style transfer methods seem appropriate to bridge the synthetic to real domain gap.

2.2 Synthetic to Real Domain Adaptation

Synthetic to real, or Sim-to-real, domain adaptation refers to the attempt of closing the gap between training on
simulated data and testing on real-life data. This topic is particularly relevant when data is scarse, expensive to annotate
and scenes are complex such as in robotics or autonomous driving. Classical domain adaptation techniques typically
rely on access to large, labeled datasets in both source and target domains. However, more recent approaches focus
on one-shot or zero-shot adaptation, using limited to no target domain data. In the context of semantic segmentation,
DAFormer [3] introduces a transformer-based architecture that leverages strong feature encoding and pseudo-label
refinement, achieving state-of-the-art performance on synthetic-to-real benchmarks. Extending this work, HRDA
[29] enhances performance by incorporating high-resolution feature fusion at multiple scales, allowing the model to
better capture fine-grained details and spatial consistency across domains. PØDA [30] uses CLIP to integrate prompts
embeddings to align the source domain and the target domain, without having access to target annotations.

Leveraging the generation capacity of diffusion models, some methods have integrated those into domain adaptation
pipelines. SGG [31] uses stable diffusion to generation intermediate domains during the segmentation training
mechanism. Gong et al. [32] plug a segmentation head onto a diffusion backbone, making use of the disantangled
representations of style and content learned by diffusion models.

DGInStyle [6] finetunes Dreambooth [33] on the source domain, then train ControlNet to generate images conditioned
by segmentation maps. Finally, they apply a style swap operation, switching the source specific UNet to a generalist
pretrained one. In this manner they only retain the segmentation conditioning without style leaking from the source
dataset during the ControlNet training. DATUM [5] also uses Dreambooth to finetune stable diffusion on only one
target image for a few iterations. Then, they prompt the model to generate one target object in the style of the target
image. Those two generation methods can be used to generate datasets tailored to the domain adaptation use-case, in a
zero-shot setting for DGInStyle or one-shot setting for DATUM.

Diffusion based style transfer has not yet been used commonly to generate datasets for domain adaptation. DoGE
[7] computes the mean difference of CLIP embeddings pairs between the source and target dataset (Domain Gap
Embedding). Then, it adds this Domain Gap Embedding to the latent representation of a source image, before
reconstructing the image with Stable UnCLIP. This method can use ControlNet, allowing the model to generate
datasets conditioned on semantic segmentation maps. In this work, we propose to use Cross-image attention [8] as our
style-transfer method to generate datasets for its zero-shot generation capability.

3 Methodology

We introduce two novel techniques to address these limitations: Class-wise Adaptive Instance Normalization with
Attention (CACTI), possibly combined with Selective Attention Filtering (CACTIF).

3.1 Class-wise Adaptive Instance Normalization

In its standard form, AdaIN computes global statistics across entire feature maps in order to align them between
images. This global calculation can be problematic when dealing with domain adaptation scenarios where class-specific
appearances are critical for downstream tasks such as semantic segmentation.

AdaIN is a common brick in style transfer methods. It is used as a way to match feature statistics between two images.
It can be used in a supervised manner by training a feature extractor from a dataset of paired images, or in a zero-shot
setting with a pre-trained auto-encoder. We use the latter, within the backbone of a latent diffusion model.

As in standard applications of AdaIN with diffusion models, at each step t of the denoising process we apply the AdaIN
operation between the features of the style image zstyle

:

t

and transferred image zout
, zstyle
t

).

t

zout
t ← AdaIN(zout

t

(1)

3

Style Transfer with Diffusion Models for Synthetic-to-Real Domain Adaptation

Figure 1: Overview of our proposed method for style transfer with diffusion models. Given a content image and a style
image, we first perform DDPM inversion (left) to obtain their respective latent representations. These representations
are then processed through our cross-attention mechanism, where our semantic filtering approach (middle) determines
when to apply cross-image attention. When the KV injection yields coherent feature correspondences, cross-image
attention transfers style features from the style image to the content image; otherwise, the original content features are
preserved. Finally, our class-wise AdaIN module (right) refines the style transfer by computing class-specific statistics
using segmentation masks from both source and content images.

This operation makes the features statistics from zout
t
distributions and contrasts. Specifically, the AdaIN operation is defined as:

t match those of zstyle

, effectively transferring global color

AdaIN(x, y) = σ(y)

(cid:19)

(cid:18) x − µ(x)
σ(x)

+ µ(y),

(2)

where µ(x) and σ(x) denote the mean and standard deviation computed across spatial dimensions for each channel.
Visually, this ensures a transfer of texture and color information.

In our proposed CACTI method, we leverage the semantic segmentation masks available for both the synthetic content
images and real style images to perform a class-specific statistical alignment. Rather than computing global statistics,
we compute and apply statistics separately for each semantic class. This approach prevents dominant classes in the
style image (such as sky or vegetation) from inappropriately influencing the appearance of other classes.

To implement CACTI, we first resize the segmentation masks to match the latent dimensions of Stable Diffusion’s VAE.
Then, for each semantic class c present in both images, we compute:

µc(z) =

(cid:80)

i,j Mc(i, j) · z(i, j)
(cid:80)
i,j Mc(i, j)

,

σc(z) =

(cid:115) (cid:80)

i,j Mc(i, j) · (z(i, j) − µc(z))2
i,j Mc(i, j)

(cid:80)

,

(3)

(4)

where Mc is a binary mask indicating pixels belonging to class c and (i, j) are spatial coordinates. We then apply the
class-specific AdaIN operation:

zout
t

(i, j) ← σc(zstyle

t

)

(cid:18) zout
t

(i, j) − µc(zout

t

)

(cid:19)

σc(zout
t

)

+ µc(zstyle

t

),

(5)

for all positions (i, j) where Mc(i, j) = 1.

This class-specific approach ensures that statistical features from the style image influence only corresponding semantic
regions in the content image. For example, the appearance of a "road" class in the style image affects only road pixels

4

Style Transfer with Diffusion Models for Synthetic-to-Real Domain Adaptation

in the output image, preserving more realistic color distributions and reducing cross-class artifacts that can occur with
global AdaIN. This is particularly important for domain adaptation scenarios where semantic consistency is crucial for
downstream tasks.

3.2 Selective Attention Filtering

Several methods use a mechanism of cross-attention, or KV injection for style transfer, as a means of mixing feature
information from a content and style image together [8, 24, 25], leveraging the self-attention layers of the denoising
UNet. When fed to a self-attention layer, a feature map F is linearly projected into query Q, key K and value V . Those
projections, of dimension d will enable to learn the interactions between different spatial positions of the feature maps.
The standard self-attention operation processes these Q, K and V projections as follows:

Attention(Q, K, V ) = softmax

(cid:18) Q · K T
√

d

(cid:19)

· V,

(6)

where the softmax is applied independently on all rows of the matrix Q · K T . In cross-image attention [8], on which
we base our method, the attention is computed using the content query Qc and the style key Ks and value Vs, hence the
term KV injection. Their hypothesis is that Q encapsulates the content within an image, while K and V represent the
style information.

The style transfer of cross-image attention therefore modifies the standard attention calculation as such:

Attention(Qc, Ks, Vs) = softmax

(cid:19)

(cid:18) Qc · K T
s√
d

· Vs.

(7)

Cross-image attention establishes correspondences between semantically similar regions of source and target images,
regardless of their spatial positions. However, in the context of synthetic-to-real domain adaptation, this approach faces
a significant challenge. The cross-attention maps can be diffuse, with a single query in the source image attending to
multiple semantically similar regions in the target image. This diffusion effect can lead to inconsistent style transfer and
diminished adaptation performance.

To address this limitation, we propose cross-attention filtering, CACTIF, as an extension of CACTI. This filtering
refines the cross-image attention mechanism by keeping only the style features whose value vectors are similar to the
one at the maximum attention position, reducing noise and ensuring more consistent and stable style transfer.

In the cross-attention mechanism, at each diffusion step and self-attention layer, the feature map is projected into queries
(Q), keys (K), and values (V ). The cross-attention maps Ac,s ∈ R(h×w)2

are computed as:

Ac,s = Qc · K T
s ,

(8)

where Qc represents queries from the content image, Ks keys from the style image, h and w being the height and width
of the self-attention layer. These attention maps indicate, for each position in the content features, which regions in the
style features are most relevant.

In our proposed CACTIF method, we selectively filter these attention maps based on feature similarity. For each position
i in the content image, we first identify the position mi in the style image that receives maximum attention:

mi = arg max

j

(Qc(i) · Ks(j)T ).

(9)

After identifying these maximum-attention correspondences, we evaluate whether the style transfer at each position
is likely to produce coherent results. We compute the cosine similarity between the value vectors at corresponding
positions:

s(i) = cos (Vc(i), Vs(mi)) =

Vs(i) · Vs(mi)
∥Vc(i)∥∥Vs(mi)∥

,

(10)

where Vc(i) is the value vector at position i in the content image and Vs(mi) is the value vector at the corresponding
position of maximum attention in the style image.

We then set a threshold τ based on a percentile p of the similarity distribution:

τ = Percentile({s(i)}, p).

(11)

5

Style Transfer with Diffusion Models for Synthetic-to-Real Domain Adaptation

For positions where the similarity falls below this threshold, indicating potential inconsistencies in the transfer, we
retain the original content features rather than applying style transfer:

Aout(i) =

Vout(i) =

(cid:26)Ac,s(i)
Ac(i)
(cid:26)Vs(i)
Vc(i)

if s(i) ≥ τ
if s(i) < τ

,

if s(i) ≥ τ
if s(i) < τ

.

(12)

(13)

This selective approach preserves content structure in regions where direct style transfer might produce artifacts. The
parameter p controls the strictness of the filtering, with higher values resulting in more conservative style transfer.
Through empirical testing, we determined that values of p between 0.1 and 0.3 provide a good balance between style
transfer effectiveness and content preservation.

3.3 Combined Approach

Our complete methodology, CACTIF, combines both class-wise AdaIN and attention filtering in a complementary
fashion. The class-wise AdaIN ensures appropriate appearance transfer within semantic regions, while attention filtering
prevents artifacts in areas where the cross-attention correspondence between content and style is weak.

In practice, we implement our approach within the Stable Diffusion framework, applying these techniques during the
denoising process. As shown in Figure 1, at each diffusion step:

• We compute the cross-attention maps between content queries and style keys;

• We apply attention filtering based on value vector similarity;

• We perform the modified cross-attention operation;

• We apply class-wise AdaIN to refine feature statistics.

We empirically find that applying these operations at multiple resolution levels in the UNet decoder (32×32 and 64×64
feature maps) yields the best results. For the percentile threshold in attention filtering, we use p = 0.25 as our default
setting based on preliminary experiments.

To evaluate the effectiveness of our proposed techniques, we conduct two complementary studies. First, we perform
a qualitative analysis of the style transfer results, examining how CACTI and CACTIF reduce artifacts and improve
visual coherence compared to existing methods. This analysis helps isolate the specific contributions of class-wise
AdaIN and attention filtering to the overall quality of the generated images.

Second, we conduct a quantitative evaluation in the context of synthetic-to-real domain adaptation for semantic
segmentation. By testing our generated images as training data for segmentation models under various adverse weather
conditions, we demonstrate the practical utility of our approach for downstream computer vision tasks. This evaluation
framework allows us to assess not only the visual quality of our style transfer but also its effectiveness in bridging the
domain gap between synthetic and real data distributions.

Together, these two studies highlighting how class-based AdaIN improves color consistency and class-specific appear-
ance transfer, while attention filtering preserves structural integrity and reduces artifacts. The following sections detail
our experimental setup and results for both evaluations.

4 Content-preserving style transfer

The quality of synthetically generated images is crucial for effective domain adaptation in semantic segmentation tasks.
In this section, we evaluate how our proposed methods—CACTI and CACTIF—improve image quality compared
to existing approaches. We specifically focus on reducing diffusion artifacts and enhancing alignment between
segmentation masks and the corresponding generated images, which is essential for maintaining semantic consistency
when transferring style from real to synthetic domains.

4.1 Datasets

We evaluate our methods using widely adopted datasets in the domain adaptation literature, focusing on driving scenes
with semantic segmentation annotations.

6

Style Transfer with Diffusion Models for Synthetic-to-Real Domain Adaptation

GTA5
[34] serves as our synthetic source domain. This dataset comprises 24,966 densely annotated images rendered
from the Grand Theft Auto V video game environment, with a resolution of 1914 × 1052. The synthetic nature of this
dataset provides perfect ground truth labels but exhibits the characteristic domain gap when used to train models for
real-world deployment.

Cityscapes
[1] represents our primary real-world target domain. It consists of 5,000 densely labeled images captured
in 50 European cities during daytime and under good weather conditions. These images have a resolution of 2048 ×
1024 and feature typical urban driving scenarios.

ACDC [35] provides our adverse condition target domains. This dataset contains 4,006 densely annotated images
from driving scenarios in Switzerland, specifically focused on challenging weather conditions. The images are equally
distributed across four adverse conditions: fog, nighttime, rain, and snow. Each image has a resolution of 1912 × 1024.

In our experimental framework, we address the synthetic-to-real domain adaptation scenario, using GTA5 as the source
domain and either Cityscapes or ACDC as the target domain. For style transfer, we adopt a few-shot setting where
we assume access to a single representative image from each target domain condition, which aligns with realistic
constraints in practical applications where target domain data may be scarce.

4.2 Experimental Setup

We compare our proposed methods against several state-of-the-art approaches for style transfer and domain adaptation:

AdaIN diffusion represents the baseline approach where standard Adaptive Instance Normalization is applied at each
step of the diffusion process without class-specific modifications or cross-attention mechanisms.

Cross-Image Attention implements the method from [8], which uses cross-image attention with standard AdaIN to
transfer appearance between images while preserving content structure.

DATUM [5] employs personalized diffusion models for one-shot unsupervised domain adaptation, fine-tuning a
text-to-image diffusion model on a single target sample to generate target-like images.

DGInStyle
a generative model rather than a style transfer approach.

[6] generates images in the target domain style without a content reference image, operating primarily as

CACTI
guide the feature statistic matching during the diffusion process.

(our method) enhances Cross-Image Attention with class-specific AdaIN, leveraging segmentation labels to

CACTIF
reducing artifacts in regions where direct style transfer might produce inconsistencies.

(our method) extends CACTI by adding selective attention filtering based on feature similarity, further

For all methods based on cross-image attention (Cross-Image Attention, CACTI, and CACTIF), we generate images
at 512 × 1024 resolution using Stable Diffusion v1.5 with an empty prompt. To balance generation quality with
computational efficiency, we skip 30 out of 50 denoising steps. For AdaIN-based methods, we apply the normalization
at every step throughout the diffusion process.

While we first measure the capacity of these methods to transfer styles, we note that DATUM and DGInStyle do not
fully preserve content through segmentation mask guidance. These methods usually generate images showing artifacts
and undefined objects. By using style transfer and not image generation without prior, we manage to keep strong
structural coherence.

4.3 Evaluation Metrics

We first employ two complementary metrics to evaluate the quality of generated images, studying the style transfer
capacity of the proposed methods.

The Fréchet Inception Distance (FID) [17] measures the statistical similarity between generated images and real target
domain images. Lower FID values indicate better alignment with the target domain distribution, reflecting successful
style transfer. For FID computation, we generate 5,000 images per method and compare against the Cityscapes dataset.

7

Style Transfer with Diffusion Models for Synthetic-to-Real Domain Adaptation

(a) Content (GTA5)

(b) CACTIF

(c) DGInStyle

(d) DATUM

(e) Content (GTA5)

(f) CACTIF

(g) DGInStyle

(h) DATUM

Figure 2: Qualitative comparison between dataset generation methods, with style transfer (ours) or generation without
base image. Using the GTA5 dataset as base allows to keep object consistency and image structure. Note that DATUM
does not rely on semantic segmentation masks and can therefore generate an image with multiple scenes, as shown here.

FIDcityscapes ↓ LPIPSGTA5 ↓

AdaIN diffusion
Cross-image attention
DATUM
DGInStyle
CACTI
CACTIF

73.26
65.31
113.12
86.94
54.56
55.30

0.30
0.41
-
-
0.39
0.36

Table 1: Quantitative evaluation of style transfer quality. FIDcityscapes measures similarity to the target domain (lower
is better), while LPIPSGTA5 measures content preservation (lower is better). LPIPS is not calculated for methods that
don’t use content reference images.

The Learned Perceptual Image Patch Similarity (LPIPS) [36] quantifies the perceptual distance between pairs of
images, providing a measure of content preservation. We compute LPIPS between each synthetic GTA5 image and its
corresponding generated version. Lower LPIPS values indicate better preservation of the original content structure.

4.4 Results and Analysis

Figure 2 provides a visual comparison between different methods. The limitations of approaches that do not use a
content reference image become immediately apparent. Methods like DGInStyle and DATUM exhibit notable artifacts
and poor spatial coherence, with objects appearing distorted (as seen with the red car in Figure 2c) or showing unrealistic
spatial organization (Figure 2h).

In contrast, our proposed methods maintain object localization and structural integrity while successfully transferring
the target domain style. This preservation of spatial relationships is crucial for downstream semantic segmentation
tasks, as it ensures alignment between generated images and their corresponding labels.

The quantitative results in Table 1 confirm the visual observations. Our CACTI method achieves the lowest FID score
(54.56) compared to all other approaches, indicating superior style transfer quality and closest resemblance to the real
Cityscapes distribution. The CACTIF variant, which adds attention filtering, shows a slight increase in FID (55.30) but
significantly improves content preservation as measured by LPIPS (0.36 compared to 0.39 for CACTI).

Standard Cross-Image Attention shows reasonable style transfer capabilities (FID = 65.31) but struggles with content
preservation (LPIPS = 0.41). Basic AdaIN diffusion achieves the best content preservation (LPIPS = 0.30) but at
the expense of style transfer quality (FID = 73.26). We note that there is no cross-image attention in this baseline,
explaining the good content preservation as only AdaIN is used for transfer between the images.

8

Style Transfer with Diffusion Models for Synthetic-to-Real Domain Adaptation

We do not calculate LPIPS for methods that do not use a content reference image, DATUM and DGInStyle, as LPIPS is
calculated over image pairs. However, we note that DGInStyle shows poor performance on FID (86.94), highlighting
the challenge of generating domain-specific images without structural guidance.

These results demonstrate a fundamental trade-off between style transfer fidelity and content preservation. Our proposed
methods, particularly CACTIF, achieve a favorable balance between these objectives, generating images that both
resemble the target domain style and maintain the structural information necessary for semantic segmentation.

The class-specific approach in CACTI prevents dominant classes in the style image (such as sky or road) from
inappropriately influencing the appearance of other classes, resulting in more realistic color distributions. The
selective attention mechanism in CACTIF further enhances content preservation by identifying and filtering potentially
problematic attention mappings.

These improvements in image quality and content preservation directly impact the effectiveness of the generated datasets
for domain adaptation in semantic segmentation tasks. By reducing artifacts and maintaining structural coherence,
our methods produce training data that better represents the target domain while preserving the semantic information
from the source domain. This balance is essential for successful domain adaptation, as demonstrated in our subsequent
semantic segmentation experiments.

5 Style transfer for domain adaptation

While the visual quality of generated images is important, the ultimate goal of our approach is to improve performance
on downstream semantic segmentation tasks through effective domain adaptation.

In practical scenarios, access to target domain data is often limited, particularly for adverse weather conditions which
may be difficult or dangerous to capture. Our method addresses this constraint by generating realistic target-domain-like
images using only a small number of reference style images, enabling training of robust segmentation models without
extensive target domain data collection.

In Section 4, we evaluated the visual quality of images generated by our proposed methods. Now, we focus on assessing
how these generated images perform as training data for semantic segmentation models under various domain adaptation
scenarios. This section examines whether the improvements in style transfer quality translate to better segmentation
performance when adapting from synthetic to real domains, particularly under challenging adverse weather conditions.

5.1 Experimental Setup

We continue using the same datasets described in Section 4: GTA5 as our synthetic source domain and both Cityscapes
and ACDC as our real target domains. For the few-shot setting, we select 1 representative image from each target
domain condition (normal, fog, night, rain, and snow) to serve as style references. This reflects a practical scenario
where a limited number of target domain samples might be available.

We explore domain adaptation in a standard scenario with the Cityscapes dataset, as well as more challenging settings
with adverse weather conditions as demonstrated in the ACDC dataset.

Network Architectures We evaluate our generated datasets using three network configurations:

• DAFormer [3] is a state-of-the-art domain adaptation framework for semantic segmentation that employs a
transformer-based encoder with adaptive context-aware fusion. We use DAFormer with its default MiT-B5
backbone.

• HRDA [29] extends DAFormer with a hierarchical multi-resolution approach specifically designed to handle
high-resolution input images, which can be particularly beneficial for detecting small objects and fine details
in adverse conditions.

• Segformer [37] is a transformer-based segmentation model without domain adaptation components. We
include this model to evaluate whether our generated datasets can improve performance even without explicit
domain adaptation techniques.

Training Protocol For all experiments, we follow a consistent training protocol. We first generate a dataset of 250
images for each target domain condition using the corresponding style reference image. For DAFormer and HRDA,
we employ their standard training configurations, including self-training with pseudo-labels and curriculum learning
strategies. For Segformer, we train directly on the generated images without domain adaptation components. All models
are trained for 40,000 iterations with the AdamW optimizer, following learning rate and augmentation settings from

9

Style Transfer with Diffusion Models for Synthetic-to-Real Domain Adaptation

(a) Content (GTA5)

(b) Style (ACDC-snow)

(c) DGInStyle

(d) Cross-image attention

(e) CACTI

(f) CACTIF

Figure 3: Style transfer comparison from GTA5 synthetic content (a) to ACDC-snow style (b). The results show:
DGInStyle (c), Cross-Image Attention with standard AdaIN (d), Cross-Image Attention with class-based AdaIN
(CACTI) (e), and complete CACTIF (f). In dataset generation, images of 1024 by 512 pixels are used when possible.
For DGInStyle, due to incompatibility with rectangular images, we show two images of 512 by 512 side-by-side. Note
the artifacts in (d) (blue coloration in trees), color bleeding in mountains in (d) and (e), and the more realistic snow
appearance on the road in our proposed methods.

their respective original implementations. We evaluate performance on the validation sets of Cityscapes and ACDC
using the mean Intersection over Union (mIoU) metric.

Baselines We compare our methods against several approaches, which we enumerate here. Source Only is trained
only on the original synthetic GTA5 dataset without any adaptation. Source + Style uses the source dataset augmented
with 1 style reference image from each target domain. AdaIN Diffusion uses images generated with standard AdaIN
during the diffusion process without class-specific modifications. We also compare with images generated from the
baseline methods Cross-Image Attention [8], DATUM [5], and DGInStyle [6].

5.2 Qualitative Analysis

Figures 3 and 4 provide visual comparisons of our methods when transferring styles from adverse weather conditions
to synthetic images. In Figure 3, we observe how different methods transfer the snowy appearance from an ACDC
image to a synthetic GTA5 scene. The Cross-Image Attention approach with standard AdaIN (Figure 3d) introduces
several artifacts, including unrealistic blue coloration in trees and yellow color bleeding into mountain regions. Adding
class-specific AdaIN as in our CACTI method (Figure 3e) partially mitigates these issues but still exhibits some color
bleeding.

Our CACTIF method (Figure 3f) produces the most coherent results, successfully transferring the snowy appearance
while maintaining the structural integrity of the original scene. Notably, the snow accumulation on the road surface is
realistic, and the color distribution across semantic classes is consistent with real-world expectations. The selective
attention filtering prevents inappropriate transfers between dissimilar regions, which is particularly evident when
comparing the rendering of distant objects like mountains and trees.

Figure 4 illustrates the challenge of transferring nighttime appearance, which involves significant lighting changes.
The standard Cross-Image Attention approach (Figure 4d) produces an image with unnatural contrasts and loss of
detail in darker regions. Class-based AdaIN (Figure 4e) improves the results in highlighting the vehicle and darkening
the sky. Our CACTIF method (Figure 4f) achieves a more balanced nighttime appearance, where the overall scene is
appropriately darkened but important semantic elements remain discernible like the edges of the vegetation. This is

10

Style Transfer with Diffusion Models for Synthetic-to-Real Domain Adaptation

(a) Content (GTA5)

(b) Style (ACDC-night)

(c) DGInStyle

(d) Cross-image attention

(e) CACTI

(f) CACTIF

Figure 4: Style transfer results from GTA5 content (a) to ACDC-nighttime style (b), using the same methods as in
Figure 3. Our methods better preserve visibility of important semantic elements while successfully transferring the
nighttime appearance, maintaining better contrast and detail in critical regions.

crucial for downstream segmentation tasks, as preserving the visibility and correct appearance of traffic participants and
infrastructure elements enables better model training. The selective attention filtering is particularly effective in this
challenging lighting condition, preventing the loss of semantic information while still transferring the nighttime style
characteristics.

5.3 Quantitative Results

Table 2 presents the quantitative results for semantic segmentation performance across different models and generation
methods. When using the DAFormer architecture, both of our proposed methods (CACTI and CACTIF) consistently
outperform the baseline approaches across most conditions. Specifically, CACTI achieves the best performance on fog
and night conditions, with mIoU scores of 53.23% and 22.82% respectively, surpassing the Source + Style baseline by
+3.33% and +4.84%. DATUM performs slightly better on rain and snow conditions, but our CACTI method achieves the
highest mean performance across all ACDC conditions (42.80%), demonstrating its robustness across diverse adverse
conditions.

When comparing different model architectures, we observe that HRDA consistently outperforms DAFormer across all
conditions when trained with both our CACTI and CACTIF methods. The improvement is particularly significant for
fog conditions, where CACTI with HRDA achieves 58.24% mIoU, representing a +5.01% improvement over the same
method with DAFormer. This suggests that the hierarchical multi-resolution approach of HRDA is particularly effective
at exploiting the high-quality and semantically consistent images generated by our method.

Examining performance across different adverse conditions reveals interesting patterns. The night condition remains the
most challenging for all methods, with substantially lower mIoU scores than other conditions. However, our methods
still provide considerable improvements over the baselines in this difficult scenario. For example, CACTI with HRDA
achieves 26.07% mIoU on night conditions, representing a +4.30% improvement over the Source + Style baseline with
the same architecture.

Perhaps most notably, our CACTIF method enables even non-domain-adaptive architectures like Segformer to achieve
competitive performance. Segformer trained with CACTIF-generated images achieves 36.71% mean mIoU across
ACDC conditions and 48.40% on Cityscapes, outperforming the Source Only baseline by +1.93% and +4.41%
respectively. This demonstrates that our approach helps bridge the domain gap even without explicit domain adaptation
techniques in the segmentation model.

11

Style Transfer with Diffusion Models for Synthetic-to-Real Domain Adaptation

Source + style
AdaIN diffusion
Cross-image Attention
DATUM
DGInStyle
CACTI
CACTIF
Source + style
CACTI
CACTIF
Source only
CACTI
CACTIF

#TS
5
5
5
5
0
5
5
5
5
5
5
5
5

Model
DAFormer
DAFormer
DAFormer
DAFormer
DAFormer
DAFormer
DAFormer
HRDA
HRDA
HRDA
Segformer
Segformer
Segformer

Fog
49.90
51.01
51.80
52.70
49.70
53.23
52.88
53.02
58.24
58.17
44.32
45.79
44.53

Night
17.98
16.40
19.37
22.07
21.06
22.82
21.39
21.77
26.07
25.61
16.02
18.23
21.55

Rain
43.22
45.72
45.20
47.81
44.04
47.78
45.93
45.63
50.04
51.36
39.03
39.03
36.79

Snow Mean (ACDC) Cityscapes
39.85
43.45
43.80
46.00
44.28
44.49
43.82
41.68
45.44
45.11
37.88
38.88
38.78

37.91
37.88
40.05
42.16
40.18
42.80
41.13
40.74
45.66
45.60
34.78
36.62
36.71

53.18
50.31
52.36
55.04
52.42
53.47
54.39
54.15
53.81
54.82
43.99
43.30
48.40

Table 2: Semantic segmentation performance (mIoU %) on synthetic-to-real domain adaptation across different methods
and models. We report results for each of the four adverse conditions in ACDC (Fog, Night, Rain, Snow) as well as
their average (Mean ACDC) and normal conditions (Cityscapes). Three network architectures are evaluated: DAFormer
(domain adaptation transformer), HRDA (high-resolution domain adaptation), and Segformer (without explicit domain
adaptation components). #TS indicates the total number of target samples used for generation. All results are averaged
over three independent trials with different random seeds. Bold values indicate the best performance in each model
category and condition. Our proposed methods (CACTI and CACTIF) consistently outperform baselines, with CACTIF
showing particular strength when paired with the HRDA architecture.

Overall, these results confirm that our proposed methods not only generate visually appealing images but also produce
semantically meaningful data that enables effective domain adaptation for semantic segmentation. The class-specific
approach of CACTI ensures appropriate style transfer for different semantic regions, while the selective attention
filtering in CACTIF preserves critical structural information. Together, these techniques enable the generation of
high-quality synthetic datasets that significantly improve segmentation performance across diverse target domains and
model architectures.

6 Conclusion and perspectives

In this work, we proposed two novel methods for synthetic-to-real domain adaptation: Class-wise Adaptive Instance
Normalization (CACTI) and Selective Attention Filtering (CACTIF). These methods leverage diffusion models for
style transfer, generating high-quality images that maintain strong structural coherence while adapting to target domain
characteristics. Our approach specifically targets the challenging scenario of few-shot domain adaptation, where only
a limited number of target domain images are available. The results demonstrate that class-aware style transfer can
significantly improve semantic segmentation performance across various adverse weather conditions.

While our methods show clear improvements over standard style transfer approaches, the comparison with generative
methods like DATUM reveals interesting trade-offs. DATUM achieves slightly better performance in some conditions
using DAFormer, but it fundamentally transforms the content rather than preserving it. In contrast, our style transfer
approach maintains structural coherence and semantic alignment with source images, which is critical when annotation
preservation is required. The integration of our methods with HRDA demonstrates particularly strong results, achieving
up to 58.24% mIoU on fog conditions and a mean of 45.66% across all adverse conditions, showing the potential of
style transfer for bridging the synthetic-to-real gap.

Our experiments highlight the significant challenges that remain in domain adaptation for adverse weather conditions.
The nighttime scenario presents the most difficulty, with our best method (CACTI with HRDA) achieving only 26.07%
mIoU. This underperformance can be attributed to the substantial appearance gap between synthetic and real nighttime
scenes, as GTA5 contains fewer night examples and their visual characteristics differ significantly from real nighttime
images in Cityscapes and ACDC. These results underscore the importance of continued research in domain adaptation
techniques specifically tailored for extreme lighting and weather conditions.

Several technical aspects of our approach warrant further investigation. The current filtering method relies on maximum
attention values, which may not capture the full complexity of attention map distributions. Alternative pixel matching
strategies, such as those employed in Eye for an Eye, could potentially improve correspondence quality, albeit at higher

12

Style Transfer with Diffusion Models for Synthetic-to-Real Domain Adaptation

computational cost. Additionally, integrating ControlNet into the pipeline could provide more explicit guidance using
segmentation masks, potentially improving structural coherence in challenging regions. Future work should also explore
the integration of textual information through prompts to further guide the generation process.

This research demonstrates that class information can be highly valuable in guiding image generation and style
transfer for domain adaptation. By leveraging semantic labels to constrain both statistical normalization and attention
mechanisms, we can generate more realistic and semantically consistent target domain images. As autonomous
systems continue to face deployment challenges in adverse conditions, approaches like ours that can effectively adapt
models using limited target data will be increasingly important. Our work represents a step toward more robust
perception systems that can operate reliably across diverse environmental conditions, though significant challenges
remain, particularly for extreme conditions like nighttime driving.

References

[1] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe
Franke, Stefan Roth, and Bernt Schiele. The Cityscapes Dataset for Semantic Urban Scene Understanding. In
2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 3213–3223, June 2016.

[2] Judy Hoffman, Dequan Wang, Fisher Yu, and Trevor Darrell. FCNs in the Wild: Pixel-level Adversarial and

Constraint-based Adaptation. arXiv preprint arXiv:1612.02649, December 2016.

[3] Lukas Hoyer, Dengxin Dai, and Luc Van Gool. DAFormer: Improving network architectures and training strategies
for domain-adaptive semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR), pages 9924–9935, 2022.

[4] Prithvijit Chattopadhyay, Bharat Goyal, Boglarka Ecsedi, Viraj Prabhu, and Judy Hoffman. Augcal: Improving
sim2real adaptation by uncertainty calibration on augmented synthetic images. In International Conference on
Learning Representations, 2024.

[5] Yasser Benigmim, Subhankar Roy, Slim Essid, Vicky Kalogeiton, and Stéphane Lathuilière. One-shot unsupervised
domain adaptation with personalized diffusion models. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 698–708, 2023.

[6] Yuru Jia, Lukas Hoyer, Shengyu Huang, Tianfu Wang, Luc Van Gool, Konrad Schindler, and Anton Obukhov.
Dginstyle: Domain-generalizable semantic segmentation with image diffusion models and stylized semantic
control. In European Conference on Computer Vision, pages 91–109, 2024.

[7] Yinong Oliver Wang, Younjoon Chung, Chen Henry Wu, and Fernando De La Torre. Domain Gap Embeddings for
Generative Dataset Augmentation. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), pages 28684–28694, June 2024.

[8] Yuval Alaluf, Daniel Garibi, Or Patashnik, Hadar Averbuch-Elor, and Daniel Cohen-Or. Cross-image attention for

zero-shot appearance transfer. In ACM SIGGRAPH 2024 Conference Papers, 2024.

[9] Sooyeon Go, Kyungmook Choi, Minjung Shin, and Youngjung Uh. Eye-for-an-eye: Appearance transfer with

semantic correspondence in diffusion models. arXiv preprint arXiv:2406.07008, 2024.

[10] Amir Hertz, Andrey Voynov, Shlomi Fruchter, and Daniel Cohen-Or. Style aligned image generation via shared
attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages
4775–4785, 2024.

[11] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural

information processing systems, 33:6840–6851, 2020.

[12] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image

generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.

[13] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-Resolution Image
Synthesis with Latent Diffusion Models. In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, 2022.

[14] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models.

In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023.

[15] Leon A Gatys, Alexander S Ecker, and Matthias Bethge. Image style transfer using convolutional neural networks.
In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2414–2423, 2016.

[16] Xun Huang and Serge Belongie. Arbitrary style transfer in real-time with adaptive instance normalization. In

Proceedings of the IEEE international conference on computer vision, pages 1501–1510, 2017.

13

Style Transfer with Diffusion Models for Synthetic-to-Real Domain Adaptation

[17] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained
by a two time-scale update rule converge to a local nash equilibrium. In Proceedings of the 31st International
Conference on Neural Information Processing Systems, page 6629–6640, 2017.

[18] Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation using
cycle-consistent adversarial networks. In Proceedings of the IEEE international conference on computer vision,
pages 2223–2232, 2017.

[19] Xinyuan Chen, Chang Xu, Xiaokang Yang, Li Song, and Dacheng Tao. Gated-gan: Adversarial gated networks

for multi-collection style transfer. IEEE Transactions on Image Processing, 28(2):546–560, 2019.

[20] Che-Tsung Lin, Sheng-Wei Huang, Yen-Yi Wu, and Shang-Hong Lai. Gan-based day-to-night image style transfer
for nighttime vehicle detection. IEEE Transactions on Intelligent Transportation Systems, 22(2):951–963, 2021.

[21] Yuxin Zhang, Nisha Huang, Fan Tang, Haibin Huang, Chongyang Ma, Weiming Dong, and Changsheng Xu.
Inversion-based style transfer with diffusion models. In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, pages 10146–10156, 2023.

[22] Tianhao Qi, Shancheng Fang, Yanze Wu, Hongtao Xie, Jiawei Liu, Lang Chen, Qian He, and Yongdong Zhang.
In Proceedings of the

Deadiff: An efficient stylization diffusion model with disentangled representations.
IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8693–8702, 2024.

[23] Jaeseok Jeong, Junho Kim, Yunjey Choi, Gayoung Lee, and Youngjung Uh. Visual style prompting with swapping

self-attention. arXiv preprint arXiv:2402.12974, 2024.

[24] Yingying Deng, Xiangyu He, Fan Tang, and Weiming Dong. Z*: Zero-shot style transfer via attention reweighting.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6934–6944,
2024.

[25] Jiwoo Chung, Sangeek Hyun, and Jae-Pil Heo. Style Injection in Diffusion: A Training-Free Approach for
Adapting Large-Scale Diffusion Models for Style Transfer. In 2024 IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR), pages 8795–8805, June 2024.

[26] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu
Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Represen-
tations, 2022.

[27] Yarden Frenkel, Yael Vinker, Ariel Shamir, and Daniel Cohen-Or. Implicit style-content separation using b-lora.

In European Conference on Computer Vision, pages 181–198, 2024.

[28] Viraj Shah, Nataniel Ruiz, Forrester Cole, Erika Lu, Svetlana Lazebnik, Yuanzhen Li, and Varun Jampani. Ziplora:
Any subject in any style by effectively merging loras. In European Conference on Computer Vision, pages
422–438, 2024.

[29] Lukas Hoyer, Dengxin Dai, and Luc Van Gool. Hrda: Context-aware high-resolution domain-adaptive semantic

segmentation. In European conference on computer vision, pages 372–391, 2022.

[30] Mohammad Fahes, Tuan-Hung Vu, Andrei Bursuc, Patrick Pérez, and Raoul De Charette. Poda: Prompt-driven
zero-shot domain adaptation. In Proceedings of the IEEE/CVF International Conference on Computer Vision,
pages 18623–18633, 2023.

[31] Duo Peng, Ping Hu, Qiuhong Ke, and Jun Liu. Diffusion-based image translation with label guidance for domain
adaptive semantic segmentation. In Proceedings of the IEEE/CVF international conference on computer vision,
pages 808–820, 2023.

[32] Rui Gong, Martin Danelljan, Han Sun, Julio Delgado Mangas, Nikolay Marin, and Luc Van Gool. Prompting
diffusion representations for cross-domain semantic segmentation. In 35th British Machine Vision Conference,
BMVC, 2024.

[33] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth:
Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, pages 22500–22510, 2023.

[34] Stephan R. Richter, Vibhav Vineet, Stefan Roth, and Vladlen Koltun. Playing for data: Ground truth from

computer games. In European Conference on Computer Vision (ECCV), volume 9906, pages 102–118, 2016.

[35] Christos Sakaridis, Dengxin Dai, and Luc Van Gool. ACDC: The adverse conditions dataset with correspondences
for semantic driving scene understanding. In Proceedings of the IEEE/CVF International Conference on Computer
Vision (ICCV), October 2021.

14

Style Transfer with Diffusion Models for Synthetic-to-Real Domain Adaptation

[36] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness
of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern
recognition, pages 586–595, 2018.

[37] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple
and efficient design for semantic segmentation with transformers. Advances in neural information processing
systems, 34:12077–12090, 2021.

15

