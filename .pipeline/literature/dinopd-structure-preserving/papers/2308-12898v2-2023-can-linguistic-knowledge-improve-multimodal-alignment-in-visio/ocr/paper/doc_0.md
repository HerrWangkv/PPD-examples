JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015

1

Can Linguistic Knowledge Improve Multimodal
Alignment in Vision-Language Pretraining?

Fei Wang, Liang Ding, Member, IEEE, Jun Rao, Ye Liu, Li Shen, Changxing Ding, Member, IEEE

3
2
0
2

g
u
A
5
2

]

M
M

.
s
c
[

2
v
8
9
8
2
1
.
8
0
3
2
:
v
i
X
r
a

Abstract—The multimedia community has shown a significant
interest in perceiving and representing the physical world with
multimodal pretrained neural network models, and among them,
the visual-language pertaining (VLP) is, currently, the most
captivating topic. The common practice for pretraining the
visual-language backbone is supervising the training process with
paired image-text data. However, there have been few endeavors
dedicated to the exploration of 1) whether essential linguistic
knowledge (e.g., semantics and syntax) can be extracted during
VLP, and 2) how such linguistic knowledge impact or enhance the
multimodal alignment. In response, here we aim to elucidate the
impact of comprehensive linguistic knowledge, including semantic
expression and syntactic structure, on multimodal alignment.
Specifically, we design and release the SNARE, the first large-
scale multimodal alignment probing benchmark, to detect the
vital linguistic components, e.g., lexical, semantic, and syntax
knowledge, containing four tasks: Semantic structure, Negation
logic, Attribute ownership, and Relationship composition. Based
on our proposed probing benchmark, our holistic analyses of
five advanced VLP models (i.e., BLIP, CLIP, Flava, X-VLM,
and BLIP2) illustrate that the VLP model: i) shows insensi-
tivity towards complex syntax structures and relies on content
words for sentence comprehension;
ii) demonstrates limited
comprehension of combinations between sentences and negations;
iii) faces challenges in determining the presence of actions or
information and struggles
spatial relationships within visual
with verifying the correctness of triple combinations. Given the
above findings, we suggest that, to improve the multimodal
alignment, 1) using the large generative language model as the
language backbone in VLP to understand complex sentences;
2) establishing high-quality datasets by highlighting the content
words and using simple syntax (e.g., short-distance semantic com-
position) to improve multimodal alignment; and 3) incorporating
more fine-grained visual knowledge (e.g., spatial relationships)
into pretraining objectives. We make our benchmark and code
available at https://github.com/WangFei-2019/SNARE/.

Index Terms—Multimodal Learning, Visual-Language Pre-

training, Alignment Probing

I. INTRODUCTION

V ISUAL-LANGUAGE pretraining (VLP) is a technology

that aims to learn and align multimodal knowledge
from large-scale pretraining datasets using carefully designed
architecture [1], [2], [3], [4]. After fine-tuning the pretrained
VLP models, they exhibit better comprehension, cognitive, and

Fei Wang, Ye Liu, and Changxing Ding are with the School of Future Tech-
nology, South China University of Technology, Guangzhou 511436, China (e-
mail: ft feiw@mail.scut.edu.cn; yliu03@scut.edu.cn; chxding@scut.edu.cn).
Liang Ding is with the School of Computer Science, University of Sydney,

NSW 2006, Australia (e-mail: liangding.liam@gmail.com)

Jun Rao is with the School of Computer Science and Technology, Harbin In-
stitute of Technology, Shenzhen 518055, China (e-mail: rao7jun@gmail.com).
Li Shen is with the JD Explore Academy at JD.com, Beijing 100176,

China (e-mail: mathshenli@gmail.com).

reasoning ability in downstream tasks, such as multimodal
machine translation [5], [6],
image-text retrieval [7], [8],
multimodal reasoning [9], [10]. The multimodal alignment
knowledge from pretraining image-text pairs is the key factor
determining VLP models’ generalization and downstream per-
formance. Recently, the development of large language models
(LLMs) [11], [12] have propelled VLP methods to a new
paradigm, commonly known as multimodal large language
models (MLLMs) [13], [14]. However, despite being built
upon LLMs that contain rich linguistic knowledge, MLLMs
still face limitations in recognizing complex visual content
and generating logically coherent responses conditioned on
the vision information [15]. This has sparked our curiosity
about the potential influence of linguistic knowledge on
multimodal alignment in VLP, as it plays a crucial role in
improving visual understanding, cross-modality reasoning, and
generation.

Compositionality is a fundamental presupposition to ro-
bustly and accurately represent, understand, reason, and gen-
erate linguistic knowledge [16], where syntax governs the rule
of compositionality and semantics response the outcomes [17].
Prior studies of VLP probing [18], [19] primarily concentrate
on the representation and richness of semantic knowledge
within the models and can be divided into three primary
categories. First, probing whether knowledge from multimodal
training is better than unimodal one: Yun et al. [20] studied
whether VLP improves linguistic knowledge comprehension.
Salin et al. [21] compared knowledge learned from visual/ tex-
tual models to multimodal models. Second, probing whether
VLP models can infer semantic relationships in images and
text: Shekhar et al. [22] and Hendricks et al. [23] studied
noun and verb comprehension, respectively. R¨osch et al. [24]
focused on learning and reasoning about location information
in VLP models. Thrush et al. [25] formed a manual dataset
named Winoground to probe VLP models’ ability on recogniz-
ing similar semantics. The third category involves exploring
the complementarity of different modality knowledge. Liu et
al. [26] and Alper et al. [27] studied the complementarity of
visual knowledge to textual knowledge. These methods pri-
marily rely on the model’s comprehension of semantic features
in vision or language, and limited research has been dedicated
to exploring the influence of diverse linguistic knowledge,
particularly syntax, on multimodal alignment.

To examine the influence of semantics and syntax on mul-
timodal alignment, we design and introduce SNARE, a pio-
neer multimodal alignment probing benchmark, which encom-
passes four tasks: a) Semantic Structure, b) Negation Logic,
c) Attribute Ownership, and d) Relationship Composition.

 
 
 
 
 
 
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015

2

Fig. 1. Samples in the SNARE benchmark for each task.
Relationship, a sub-task of Relationship Composition, the options need to fill in ‘ ? ’ to form a complete sentence and try to align with the image.

indicate the matching or not between the image and text. In the Multi-spatial

and

In addition to exploring rich syntactic knowledge, SNARE
provides a comprehensive investigation at the semantic level
compared to previous studies and we show the comparison
with previous approaches in Tab. I. For a better understanding
of our SNARE benchmark, we show samples of tasks in Fig. 1.

In a) Semantic Structure, we partition words based on
the part of speech and then separately shuffle the positions
of content words (words with specific meanings) and others
to disrupt compositionality, including semantics and syntax. It
aims to investigate the VLP model’s dependence on a partic-
ular type of words and whether it exhibits sensitivity to order.
b) Negation Logic adds negation words (“not”) to sentences
to test the model’s understanding of negation logic. In c)
Attribute Ownership, we introduce sentences with different
syntactic expressions, including short- and long-distance se-
mantic combinations, e.g., short one (distance between “white/
black” and “cat/ dog” is 1): “the white cat and the black
dog” vs. long one (distances of “white/ black” and “cat/ dog”
become 5/ 4, respectively): “the cat and the dog are white and
black, respectively”. Also, we count sentences that contain the
exact words but have different semantic combinations that do
not match the image, such as “the black dog and the white
cat”. This Attribute Ownership aims to evaluate the model’s
understanding of different syntactic forms of expression. In
d) Relationship Composition, we construct sentences using
triplet that include two objects and a relationship. Specifically,
we generate sentences with correct syntactic expressions (e.g.,
“the girl is wearing the shirt”), sentences with incorrect ex-
pressions where the positions of the two objects are exchanged
(e.g., “the shirt is wearing the girl”), and sentences without
the relationship word (e.g., “the girl and the shirt”). These
sentences are used to investigate whether the VLP model
comprehends the relationship element in sentences accurately
and whether it understands the syntactic combination of object
elements and relationship elements within the sentences. In

Relationship Composition, we further extract information to
construct a sub-test set called “Multi-Spatial Relationship”
to explore the model’s understanding of spatial relationships
between objects.

With the carefully designed probing benchmark SNARE,
we evaluate four state-of-the-art VLP models,
including
BLIP [14], CLIP [28], Flava [29] and X-VLM [30]. We
also extend SNARE to be compatible with MLLMs and test
BLIP2 [31]. Through extensive analyses and experiments, we
conclude some consistent and important findings:

❶ On the lexical level, VLP models prefer simple content
words rather than more precise and complete function
words that could make the sentence semantically legal
(e.g., “girl wearing shirt” instead of “the girl is wearing
the shirt”).

❷ On the syntactic level, VLP models can easily compre-
hend short-distance syntactic combinations and simple
relations (e.g., “the white cat and the black dog”), while
they have difficulty in understanding long and relatively
complicated syntactic combinations (e.g., “The cat and
the dog are white and black, respectively”).

❸ On the semantic level, VLP models encounter diffi-
culties in a) comprehending the semantics of negation
(e.g., “is not”), b) precisely discerning spatial relations
between objects (particularly “left” and “right”), and
c) maintaining sensitivity to word order changes that
could alter the overall semantic composition (such as the
difference between sentence with correct order and with
shuffling order).

This paper is an early step in probing the linguistic knowl-
edge representation in VLP multimodal alignment, covering
low-level lexicon, middle-level syntax, and high-level semantic
and reasoning knowledge. To our knowledge, our SNARE is
the first alignment probing benchmark for VLP, which could
snare and reveal the shortages of current VLP models. We

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015

3

TABLE I
BASED ON THE SEMANTICS LEVEL, WE COMPARE THE EXISTING PROBING METHODS BY ADOPTING AN APPROXIMATE CATEGORIZATION OF THEIR
PROBING TARGETS (E.G., IN THE FOIL, THE PROBING TARGET IS THE CORRESPONDENCE BETWEEN NOUNS AND THEIR MODIFIERS IN BOTH THE SCENE
AND THE SENTENCE). THE METHODS MARKED WITH †ARE IN THE FORM OF DATASETS, WHILE THE OTHERS ARE BENCHMARKS. THE VL-CHECKLIST
DOES NOT MENTION THE BENCHMARK SIZE, AND WE CALCULATE THE NUMBER OF ITS BASE DATASETS AS A SUBSTITUTE.

Relation
Source

Methods

FOIL † [22]
SVO-Probes † [23]

Salin et al. [21]

Liu et al. [26]

VL-Checklist [33]

Winoground [25]
ARO [37]
SNARE

Object↔Modifier

Object↔Object

Sentence⟲

Attribute
Ownership

Relationship
Composition

Special
Relationship

Semantic
Structure (Order)

Logic
Relationship

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
✓

✓

✓ (Negation)

Base Dataset

Size

COCO
human annotators
Flickr30k, COCO,
Flower-102 [32]
human annotators
VG, VAW [34],
HAKE [35], SWiG [36]
human annotators
VG, COCO, Flickr30k
VG, COCO, Flickr30k

123K
14K

6K

2K

>410K

<1K
58K
76K

hope the probed disadvantages in state-of-the-art VLP models
could promote the development of multimodal pretraining.

The remaining sections are organized as follows: Sec. II
presents related work, introducing existing language knowl-
edge probing and vision-language knowledge probing ap-
proaches. Sec. III describes the collection and processing of
the data and elaborates on the construction of SNARE. Sec. IV
outlines the experimental setup. We show the experimental
results and suggestions in Sec. V and VI, respectively. Finally,
we conclude in Sec. VII.

II. RELATED WORK

A. Language Knowledge Probing

There are a series of studies in the field of natural language
processing (NLP) exploring how to probe the linguistic knowl-
edge (e.g., surface-, lexical-, syntactic-, and semantic knowl-
edge) implicitly learned by the neural network models, e.g.,
language models [38], [39] and machine translation [40], [41],
[42]. The studies conducted by Pham et al. [43] highlighted
that fine-tuning the BERT [44] on the representative language
understanding benchmark – GLUE [45] may overlook the
word order information. Sinha et al. [46] confirmed that word
order information is not important during the pretraining of the
large language models. O’Connor et al. [47] pointed out that
for long-range contexts, Transformers [48] use co-occurrence
statistics of content words to predict the next words. Ettinger et
al. [49] used psychological tasks to evaluate the contextual
information of BERT and found that BERT is insensitive
to negation factors, a characteristic that we also observe in
VLP models. Parthasarathi et al. [50] and Sinha et al. [51]
studied how models recognize syntax, while Krishna et al. [52]
and Warstadt et al. [53] investigated the complex interaction
between syntax and semantic categories in language models. In
our work, we draw on language evaluation methods, including
word shuffling and semantic reversal, to construct the probing
benchmark for probing the alignment of multimodal VLP
models. We observe similarities between the performance of
VLP models and that of the pretrained language models, such
as the low sensitivity of multimodal alignment to word order.

B. Vision-Language Knowledge Probing

studies

Previous

in the multilingual NLP field have
shown that learning accurate alignment (word-, phrase-, and
structural-level) between the source-target pairs could bring
significantly better source-side language understanding and
target-side language generation [54], [55]. Similarly, one of
the keys to cross-modality learning is to develop accurate
vision-language knowledge alignment. To this end, how to
appropriately probe such cross-modality alignment becomes
important.

This research direction has evolved from studying the
mutual interactions between features to the large-scale, rich
feature alignment. Choi et al. [18] found that contextual
information in images affects the model’s understanding of
the text. Cao et al. [19] noted that pretraining models em-
phasize textual information during inference and that there
are potential correspondences between image regions and text
words in the attention matrices. Frank et al. [56] found that the
sharing of information between text and vision is unbalanced,
with feature representations of the text encoder being more
influenced by visual features. Parcalabescu et al. [57] found
that vision language models have a poor perception of object
quantity information in visual input. Thrush et al. [25] formed
a 400-sample Winoground dataset using a manual approach
to investigate the perception of features such as objects, ac-
tions, and symbolic representations in visual language models.
Yuksekgonul et al. [37] developed a large-scale Attribution,
Relation, and Order (ARO) benchmark that consists of 50,000
examples designed to evaluate relationships and attributes with
fine granularity.

We follow the ARO benchmark and extend it to reflect
the semantic and syntax level knowledge required in VLP
alignment. Therefore, our SNARE benchmark offers chal-
lenging probing tasks (with finer-granular options) without
sacrificing sample simplicity. Apart from feature alignment,
we also focus on exploring linguistic knowledge, including
semantics, syntax, and so on, to determine how much they
impact and enhance VLP alignment.

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015

4

TABLE II
THE AVERAGE CLASS PROBABILITY AND STANDARD DEVIATION OF THREE RANDOM EXPERIMENTS ON SEMANTIC STRUCTURE.

Relation

Random

Flickr30k - Semantic Structure

COCO - Semantic Structure

BLIP

CLIP

Flava

X-VLM

BLIP

CLIP

Flava

X-VLM

Correct
shuffle non-content words
shuffle content words
shuffle all

25.0
25.0
25.0
25.0

26.1±0.7
43.7±0.6
18.1±0.9
12.1±0.5

65.9±0.3
13.0±0.2
15.8±0.7
5.4±0.7

20.3±0.8
27.7±0.7
19.0±2.1
33.0±2.0

49.6±1.0
26.0±1.2
14.0±0.7
10.4±0.8

29.3±0.4
37.7±0.6
18.7±0.6
14.3±0.3

53.9±0.5
16.9±0.2
20.6±0.3
8.7±0.3

8.9±0.1
29.3±0.4
15.0±0.5
46.7±0.5

42.5±0.4
28.0±0.3
14.6±0.3
15.0±0.2

TABLE III
THE CLASS PROBABILITY ON SNARE (NEGATION LOGIC, ATTRIBUTE OWNERSHIP, AND RELATIONSHIP COMPOSITION). WE RESPECTIVELY REMOVED
THE Sep CLASS IN ATTRIBUTE OWNERSHIP AND THE None CLASS IN RELATIONSHIP COMPOSITION, AS REPRODUCTION OF VG-ATTRIBUTE AND
VG-RELATIONSHIP TASKS IN THE ARO [37] .

Models

Random

BLIP
CLIP
FLAVA
X-VLM

Negation Logic

Attribution Ownership

Cor ↑

Cor ↑

Sep ↑

Exc ↓

VG-Attribution [37] ↑

50.0

79.0
47.3
12.9
48.1

33.3

48.9
40.0
67.9
54.0

33.3

45.2
36.1
1.6
39.1

33.3

5.9
23.9
30.5
6.9

50.0

85.3
61.7
68.7
85.6

Relationship Composition

Cor ↑

Exc ↓

None↑

33.3

41.2
38.3
0.8
30.0

33.3

35.1
36.8
1.9
23.9

33.3

23.7
25.0
97.3
46.0

VG-Relation [37] ↑

50.0

54.6
51.7
43.6
57.1

III. SNARE BENCHMARK CONSTRUCTION

Previous probing benchmarks [21], [25], [22], [33] provide
two options in one sample to assess the effectiveness of
semantic level multimodal alignment and usually overlook
syntactic level. Following the multi-choice question-answering
approach from Li et al. [58], we devise more options in
tasks of SNARE to explore the impact of syntactic- and
semantic-level alignment (§III-B, III-D, III-E). In addition,
considering the significance of reasoning [59], [60], we present
the Negation Logic task (§III-C).

Firstly, we introduce how we obtain and process the data.
Then, we explain why and how to structure the four tasks
(B. Semantic Structure, C. Negation Logic, E. Attribute Own-
ership, and D. Relationship Composition) in the SNARE
benchmark. The detailed samples are shown in Fig. 1 to
facilitate understanding.

A. Data Collection

In the SNARE benchmark, fine-grained image-text fea-
tures are required. We obtain explicit features from three
commonly-used high-quality multimodal datasets, including
Visual Genome [52], COCO [61], and Flickr30k [62], and
process them through the method proposed by Yuksekgonul et
al. [37], and the Spacy toolkit [63].

To process the Visual Genome dataset, we extract explicit
visual and textual features through the following steps: 1)
traversing through the scene graphs annotated in GQA [64]
and identifying the objects with bounding box; 2) to ensure
salience, discarding objects whose bounding box’s width or
height is less than 1/4 of that of the image; 3) randomly
selecting two different objects X, Y , where X = Y , and
extracting their corresponding attribute A, B or the space/verb
relationships R between them, where X is the subject and Y
is object; 5) to reduce interference from the rich visual content,

extracting a minimal bounding box containing X and Y from
the scene as image I; 6) finally, obtaining the nouns-relation
dataset, whose samples contain features {I, X, Y , R}, and
the nouns-attributes dataset, whose samples contain fea-
tures {I, X, Y , A, B}.

To tag the part of speech ai of words ti in the text T from
the COCO and Flickr30k dataset, we employ the Spacy [63]
toolkit to parse the sentences. We obtain a set of sample
features {I, T , a1, . . . , al}, where l represents the text length
and ai ∈ Spos. Spos = {noun, adjective, verb, . . . } is a set
including all POS tags1 in the Spacy toolkit.

B. Dataset for Semantic Structure

The Semantic Structure2 task aims to investigate whether
the syntax structure (words order) and semantics composition
(combination between different parts of speech) influence the
alignment. It is constructed by disrupting the syntax order
and retaining discrete semantics. We re-organize the pro-
cessed COCO and Flickr30k data features {I, T , a1, ..., al}.
We define a part of speech set E ⊆ Spos and mark ti
whenever ai ∈ E. The function f (T , a1, ..., al, E) is defined
to shuffle the marked ti in the text sequence T . The sample
representation is obtained as follows:

I and f (T , ai|l

1, E)


Origin:
E = ∅

E = C
Shuffle content words:
Shuffle non-content words: E = C

Shuffle all:

E = Spos

,

(1)

1For additional detailed information on the POS tags of the Spacy toolkit,

please refer to https://universaldependencies.org/u/pos/.

2“Semantic Structures” means the conceptual structure of the sentence and
its lexical and syntactic expression [65]. Shuffling word order destroys all
structure in the sentence and we use “Semantic Structure” as the task name.

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015

5

bike noun

with adposition

where C = {noun, adjective, verb} represents the type set
of content words3. For instance, we provide an example in
in Fig. 1. The “Origin” is
the “Semantic Structure” part
“a determiner girl noun sits verb on determiner a determiner
decorated verb
a determiner
boy noun while subordinating conjunction
younger adjective
takes verb a determiner
another determiner girl determiner
picture noun”, which is a reference sentence T from COCO
or Flickr30k without shuffling. The subscripts indicate part of
speech ai. Words are annotated in green whose ai ∈ C, and
the others whose ai ∈ C are annotated in blue. In “shuffle
content words”, we shuffle the green part in T , like “a girl
(girl) boy (sit) on a bike (decorated) picture (bike) with a
younger (younger) sits (boy) while another decorated (girl)
takes (takes) a girl (picture)”, where the original words are
provided in parentheses. Similarly,
in “shuffle non-content
words” and “shuffle all”, we shuffle the blue part and all
words in T respectively.

We construct samples by utilizing the nouns-attributes
dataset. In Cor, noun and attribute pairs are closely connected
(with distance 1), and easy to merge semantics, like “the blue
(A) sky (X) and the large (B) building (Y )”. The Sep has
a longer semantic expression (with distance 5/ 6), like “the
sky (X) and the building (Y ) are blue (A) and large (B)
respectively”. It is less frequently used but still enables humans
to easily understand the image content. The Exc swaps the
position of attributes and nouns in Cor, resulting in a mismatch
semantic with the image, like “the large (B) sky (X) and
the blue (A) building (Y )”. Three classes in samples are
illustrated below:

I and






Cor:
Sep:
Exc:

the A X and the B Y
the X and the Y are A and B respectively
the B X and the A Y

(3)

C. Dataset for Negation Logic

E. Dataset for Relationship Composition

Humans can infer missing visual pieces by understanding
negation commands, such as “Not” and “No”, in multimodal
information processing [59], [60]. By introducing the syntax
rules, the Negation Logic task incorporates the negation word
(“not”) into the nouns-attributes dataset to evaluate logical
reasoning abilities.

In Negation Logic, each sample comprises an image, a pos-
itive statement (Correct class, Cor), and a negative statement
(Wrong class, Wro). The sample structure is illustrated below:

I and

(cid:40)

Cor:
Wro:

the X is A and the Y is B
the X is not A and the Y is not B

(2)

We show an example in Fig.1 in “Negation Logic” part,
where the Cor sentence is “the bus (X) is white (A) and the
road (Y ) is black (B)” and the Wro sentence is “the bus (X)
is not white (A) and the road (Y ) is not black (B)”.

D. Dataset for Attribute Ownership

Humans construct complex semantics by combining adjec-
tives and nouns utilizing various syntax forms [66], enabling
them to envision elaborate visual scenes [17]. The Attribute
Ownership task aims to assess the VLP models’ ability to the
semantic match between vision and language and understand-
ing of syntax (short- and long-distance). For two sentences
conveying the same semantic meaning, we label the sentence,
in which nouns have a shorter distance to adjectives than
the other, as a short-distance syntax sentence, and the other
as a long-distance syntax sentence. Each sample in Attribute
Ownership includes a short- (Correct class, Cor) and a long-
distance (Separate class, Sep) syntax sentence, and a sentence
mismatching with I (Exchange class, Exc).

3Content words, in linguistics, are words that possess semantic content
and contribute to the meaning of the sentence in which they occur. We
simply selected nouns, adjectives, and verbs that prominently represent visual
semantics as content words.

The Relationship Composition assesses whether the VLP
model can accurately comprehend the relationship between
two objects and whether it is sensitive to more multi-element
(two or three) word combinations.

We combine features from the nouns-relation dataset and
devise three different sentences for each sample. The Correct
class sentence (Cor) describes the relationship composition
between X and Y accurately (triplet. X is the initiator of
R and Y is the recipient, for example, “the girl (X) is
wearing (R) the shirt (Y )”). In the Exchange class sentence
(Exc), the position between X and Y is exchanged, like “the
shirt (Y ) is wearing (R) the girl (X)”. The None class (None)
removes the relationship word R and becomes a binary tuple
comprised of nouns, like “the girl (X) and the shirt (Y )”.
The sample structure is as follows:

I and






Cor:
Exc:
None:

the X is R the Y
the Y is R the X
the X and the Y

(4)

We develop a sub-task from Relationship Composition,
named the Multi-spatial Relationship. It focuses on evaluating
the ability to distinguish the spatial relationship in vision. we
filter data from the nouns-relation dataset when

R ∈ {“to the left of ”, “to the right of ”, “on”, “below”}, (5)

representing four different direction relationships. The sample
structure is as follows:

I and





the X is to the left of the Y
the X is to the right of the Y
the X is on the Y
the X is below the Y

,

(6)

where only the sentence with the correct spatial relationship
R which matches with the I is the right class.

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015

6

(a) BLIP (Acc=31.4%)

(b) CLIP (Acc=40.8%)

(c) Flava (Acc=10.8%)

(d) X-VLM ((Acc=34.1%)

Fig. 2. Performance on the Multi-spatial Relationship. The vertical axis indicates the correct spatial relations in the sentence. The legend indicates the class.
CLIP and Flava display a preference towards specific spatial relationships (“to the right of” and “on”).

TABLE IV
PERFORMANCE ON EACH SUB-CATEGORY IN RELATIONSHIP COMPOSITION. WHEN MODELS HAVE LOW PERFORMANCE IN DISTINGUISHING
RELATIONSHIPS ( Cor - Exc< 5% , NO STATISTICALLY SIGNIFICANT DIFFERENCE), WE HIGHLIGHT THE CORRESPONDING SCORES. WHEN THERE ARE
THREE MODELS ACHIEVING LOW PERFORMANCE ON THE SAME SUB-CATEGORY, WE HIGHLIGHT THE CATEGORY NAME. WE ONLY SHOW
SPATIAL-BASED RELATIONSHIPS WHOSE OCCURRENCES ARE LARGER THAN 25 AND VERB-BASED ONES WHOSE OCCURRENCES ARE LARGER THAN 20.

Relation

Acc(25 <Freq< 5112)
above
at
behind
below
in
in front of
inside
of
on
on top of
to the left of
to the right of
under

Acc(25 <Freq< 425)
covered by
covering
eating
holding
looking at
lying on
parked on
riding
sitting at
sitting in
sitting on
standing in
standing on
watching
wearing

BLIP

CLIP

Flava

X-VLM

Cor↑

Exc↓

None↑

Cor↑

Exc↓

None↑

Cor↑

Exc↓

None↑

Cor↑

Exc↓

None↑

52.9
43.5
54.7
56.8
49.3
58.8
50.3
56.9
42.2
54.8
50.7
37.7
36.2
49.2

52.5
38.9
45.5
61.9
34.5
45.2
56.7
57.1
43.1
61.5
65.2
61.1
71.2
63.5
40.9
41.5

22.3
26.8
32.0
16.4
25.4
18.9
36.4
29.3
23.4
19.4
18.4
38.5
38.5
19.7

31.1
30.6
18.2
33.3
51.4
25.8
26.7
14.3
51.0
38.5
30.4
24.6
15.3
11.5
40.9
47.1

24.8
29.7
13.3
26.8
25.4
22.3
13.3
13.8
34.3
25.9
30.8
23.8
25.3
31.1

16.4
30.6
36.4
4.8
14.1
29.0
16.7
28.6
5.9
0.0
4.3
14.3
13.6
25.0
18.2
11.4

Spatial-based Relationship
0.6
0.0
2.7
0.7
1.0
0.7
0.3
0.0
1.1
0.4
0.5
1.0
1.1
0.8

45.4
38.3
42.7
46.2
41.6
50.8
39.6
29.3
29.7
49.7
48.8
17.5
11.4
50.0

26.1
33.5
21.3
24.0
23.4
18.5
27.9
36.2
41.4
25.4
29.4
41.0
44.6
16.7

28.5
28.3
36.0
29.8
34.9
30.6
32.5
34.5
28.9
24.9
21.9
41.5
44.0
33.3

Verb-based Relationship

36.0
13.9
15.2
57.1
28.9
38.7
38.3
57.1
58.8
34.6
26.1
36.0
45.8
38.5
27.3
31.1

21.9
30.6
27.3
9.5
18.3
3.2
30.0
19.0
21.6
34.6
21.7
20.6
15.3
34.6
27.3
25.6

42.0
55.6
57.6
33.3
52.8
58.1
31.7
23.8
19.6
30.8
52.2
43.4
39.0
26.9
45.5
43.3

0.3
2.8
0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
1.7
0.0
0.0
0.4

5.4
0.4
6.7
0.9
0.0
14.8
0.0
5.2
12.3
5.7
2.0
0.9
1.1
0.8

1.3
0.0
0.0
0.0
0.0
0.0
0.0
19.0
0.0
0.0
0.0
2.9
0.0
1.9
0.0
0.1

94.0
99.6
90.7
98.4
99.0
84.5
99.7
94.8
86.6
93.9
97.5
98.1
97.8
98.5

98.4
97.2
100.0
100.0
100.0
100.0
100.0
81.0
100.0
100.0
100.0
97.1
98.3
98.1
100.0
99.5

41.0
41.3
29.3
46.3
48.8
32.3
46.4
46.6
30.2
41.7
44.8
26.5
25.2
45.5

43.9
33.3
27.3
28.6
22.5
29.0
41.7
57.1
56.9
50.0
39.1
63.4
35.6
69.2
27.3
24.9

13.1
6.3
30.7
10.6
13.9
17.9
22.3
20.7
26.2
6.1
9.5
27.1
28.9
13.6

12.6
36.1
42.4
9.5
14.8
0.0
3.3
4.8
21.6
23.1
13.0
2.9
18.6
5.8
13.6
21.7

45.9
52.4
40.0
43.0
37.3
49.7
31.3
32.8
43.6
52.1
45.8
46.4
45.9
40.9

43.5
30.6
30.3
61.9
62.7
71.0
55.0
38.1
21.6
26.9
47.8
33.7
45.8
25.0
59.1
53.4

Freq

4865
269
75
574
209
708
588
58
367
1684
201
7741(-)
7741(-)
132

752
36
33
21
142
31
60
21
51
26
23
175
59
52
22
949 (-)

IV. EXPERIMENT SETUP

A. Models

We use the state-of-the-art VLP models from the past two
years including BLIP (“base”4 variant), CLIP (“VIT-B/32”5
variant), Flava (“flava-full”6 variant), X-VLM (“base”7 vari-
ant) and BLIP2 (“blip2 t5”8 variant). The experiment code

4https://github.com/salesforce/BLIP
5https://github.com/openai/CLIP
6https://huggingface.co/facebook/flava-full
7https://github.com/zengyan-97/X-VLM
8https://github.com/salesforce/LAVIS/tree/main/projects/blip2

is modified from Yuksekgonul et al. [37]9, with input text
lengths limited to 30. For part-of-speech tagging, we employed
the “en core web sm” annotation model from the Spacy
library [63].

We focus on the linguistic knowledge acquired from VLP.
Hence, we have not fine-tuned any models on downstream
tasks. We calculate the multimodal similarity scores of BLIP
and X-VLM using the fully connected layer obtained during
model pretraining. The CLIP model calculated the cosine

9https://github.com/mertyg/when-and-why-vlms-bow

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015

7

Algorithm 1 Scoring Method for MLLMs
Input: {I, T }:

image-text pair; Fm: BLIP2 model; Tp:
prompt = ”Describe the image.”; Fswi: switch sentence
to yes/ no questions.

Output: score

1: function SCORINGMETHOD({I, T })
2:
3:

Tq ← Fswi(T )
Tdes ← Fm(I, “Question:{Tp} Answer:”)
Tout ← Fm(I,“Question:{Tp} Answer:{Tdes}

Question:{Tq} Answer:”)

if “no” in Tout then
score ← 0

else

score ← 1

4:
5:
6:
7:
8:
9:

end if
return score

10:
11:
12: end function

distance between visual and text features as the similarity
scores. Flava computes them with the fusion module and the
fully connected layer classifier.

B. Evaluation Metrics

1) Accuracy for VLP Models: The four tasks in SNARE
require the VLP model to rank the image-text matching confi-
dence for each sentence. We report the per-class accuracy
score on each class. For the Semantic Structure task, we
conduct repeated experiments using three different random
seeds and present the average and standard deviation as the
outcome.

2) Binary Classification Accuracy for MLLMs: The eval-
uation of MLLMs models is still incomplete [67]. Taking
inspiration from MME [68], we develop a scoring method that
utilizes simple prompts to guide the model’s output, shown as
a binary classification-like Alg. III-E. Binary classification
accuracy scores for the same class of sentences are used as
the evaluation criteria. The scoring is based on the presence
of “yes” or “no” in the model’s answers. It is important to
note that there are occasional instances where these words
are absent in the model’s responses, which are regarded as
negative examples. In cases where the sentence T in text-
image pairs does not contain a copula verb, we transform it
into a yes/ no question using the structure “are there + T in
the image”.

V. RESULT AND ANALYSIS

We analyze the performance of the VLP model on SNARE
across three levels:
lexical (§V-A), syntactic (§V-B), and
semantic (§V-C). Separately, we analyze the performance of
the MLLMs model (§V-D), a new paradigm of VLP models.

A. Lexical Level Probing Results

The content words contain effective words semantic
information in constructing sentence semantics, and func-
tional words play a less significant role.

(a) Cor ↑

(b) Wro ↓

Fig. 3. The BLIP2 performance on the Negation Logic. For BLIP,
CLIP, Flava, and X-VLM, we present the sub-classification accuracy of the
corresponding class. Compared to models that do not utilize LLM as the main
component, BLIP2 performs better in understanding negation content.

(a) Cor ↑

(b) Sep ↑

(c) Exc ↓

Fig. 4. The BLIP2 performance on the Attribute Ownership. BLIP, CLIP,
Flava, and X-VLM show the sub-classification accuracy of the corresponding
class (“Cor”, “Sep”, and “Exc”) in the task.

Tab. II presents the model performance on the Semantic
Structure. CLIP (65.9% and 53.9%) and X-VLM (49.6% and
42.5%) exhibit a higher probability of selecting “correct”,
in which content and function words play significant roles
in composing sentence semantics. BLIP displays a higher
probability (43.7% and 37.7%) to select “shuffle non-content
words” where the order of function words is shuffled while that
of the content words is retained. It indicates that BLIP prefers
content contributing to regularization. Flava has a similar
performance to BLIP (27.7% and 29.3% on “shuffle non-
content words” > 20.3% and 8.9% on “correct”). However, it
is prone to selecting thoroughly unordered sentences (33.0%
and 46.7% on “shuffle all”). This suggests Flava lacks an
advantage in assembling discrete words into more intricate
semantic expressions. Notably, when the order of content
words is disrupted (“shuffle content words” and “shuffle all”),
all models except Flava exhibit lower selection probabilities.
This observation underscores the crucial role of content
words in conveying sentence semantics regarding part of
speech. In contrast, the significance of function words remains
relatively secondary, playing a role similar to that of a regu-
larization term. In terms of word order, there exist models that

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015

8

prefer discrete semantics on the word level (Flava), and most
models rely on word order to capture complex and accurate
semantics (BLIP, CLIP, and X-VLM). This does not align with
previous research [37]. VLP models do not always behave
like bag-of-words models, and word order is necessary for
structuring complex semantics using discrete content words.

B. Syntactic Level Probing Results

The VLP models are inclined to deal with compre-
hending short-distance syntactic combinations and simple
syntactic relationships.

In Tab. III, we present the models’ performance on Attribute
Ownership and Relationship Composition evaluations. In At-
tribution Ownership, all models tend to choose sentences with
short-distance syntactic expressions (Cor). Flava has a particu-
larly remarkable performance (67.9%), and it shows a very low
probability of selecting long-distance syntactic (1.6% in Sep).
In Relationship Composition, both Flava and X-VLM tend to
describe images using single nouns without relationship words
(97.3% and 46.0% in None respectively). Moreover, all models
have a better performance (with a higher score on Cor and
a lower one on Exc) on Attribute Ownership (combination
between two content words) than Relationship Composition
(combination between three content words). From this, it can
be observed that VLP models have learned short-distance
syntactic combinations and superficial syntactic relationships
but struggle to handle complex syntactic relationships. The
performance of comprehending simple semantic features is
consistent with the model’s performance in the VG-Attribution
(superficial) and VG-Relation tasks (complex) in ARO [37],
where the former achieves a higher score compared to the
latter.

In Tab. IV, we separately provide the models’ performance
on each relation category and categorize them into two classes:
spatial-based and verb-based relationships [37]. To alleviate
the class imbalance problem, we exclude the categories whose
sample numbers are more than 25% of the total samples. BLIP,
CLIP, and X-VLM do not consistently perform badly on sub-
categories, particularly regarding verb-based relationships. In
spatial-based relationships, they only exhibit similar perfor-
mance issues when distinguishing between “left” and “right”.
Furthermore, in instances where the models demonstrate poor
performance, they do not display a preference for selecting
None classes lacking spatial/verb-based relationships; instead,
they tend to choose the Exc and Cor classes. This suggests that
the models comprehend the semantics of relationship words
using the current syntax but struggle to accurately discern the
initiators and recipients of these relationships.

We observe that the performance of VLP models in Tab. IV
is consistent with their performance on the Semantic Struc-
ture (Tab. II). When models (BLIP) are sensitive to semantic
and syntactic relationships of content words (with the highest
probability choosing “shuffle non-content words” and the
secondary probability choosing “Correct” in Semantic Struc-
ture), they become easier to differentiate relationships between
nouns (easier to choose Cor). When models (CLIP) have high
sensitivity to syntactic relationships across all words (with the

highest probability choosing “Correct” in Semantic Structure),
they can pose challenges in distinguishing relationships be-
tween nouns and verbs and tend to more equitably probable
choices of Cor and Exc. When models (Flava) only depend on
discrete words semantic without syntax (selecting all classes
with similar probabilities in Semantic Structure), they tend
to opt for the None that excludes intricate relationships. This
indicates that syntax is indispensable for the VLP model’s
acquisition of linguistic knowledge. However, utilizing func-
tion words (non-content words) to construct intricate syntactic
structures may not effectively enhance the understanding of
relationships between words.

TABLE V
THE BLIP2 PERFORMANCE ON THE RELATIONSHIP COMPOSITION.
BLIP2 DEMONSTRATES STRONG PERFORMANCE ON VERB-BASED
RELATIONSHIPS. HOWEVER, IT EXHIBITS SLIGHTLY WEAKER
PERFORMANCE ON SPATIAL-BASED ONES. WE HIGHLIGHT THE
SUB-CATEGORIES IF THE MODEL ACHIEVES POOR PERFORMANCE
( Exc> 50% ). WE ONLY SHOW THE RELATIONSHIPS WHOSE SAMPLE
NUMBERS ARE LARGER THAN 25.

Relation

Cor↑

Exc↓

None↑

Freq

Spatial-based Relationship

Acc(25 <Freq< 5087)
above
at
behind
below
in
in front of
inside
of
on
on top of
to the left of
to the right of
under

81.9
50.6
88.0
80.7
72.7
82.5
91.8
87.9
79.6
85.1
82.1
72.5
73.9
78.0

43.5
27.1
60.0
33.6
39.2
51.7
81.6
15.5
75.8
30.6
14.9
73.3
73.3
34.9

Verb-based Relationship

Acc(Freq> 25)
covered by
covering
holding
looking at
lying on
riding
sitting at
sitting on
standing in
standing on
wearing

88.3
80.6
78.8
90.1
77.4
86.7
86.3
88.5
89.7
84.8
84.6
89.4

13.8
55.6
15.2
5.6
22.6
5.0
2.0
0.0
1.7
3.4
7.7
17.8

98.6
99.3
97.3
98.6
99.0
98.6
99.2
96.6
98.6
98.4
98.5
96.4
96.6
98.5

98.8
97.2
93.9
100.0
100.0
100.0
98.0
96.2
100.0
100.0
100.0
98.4

4865
269
75
574
209
708
588
58
367
1684
201
7741(-)
7741(-)
132

1614
36
33
142
31
60
51
26
175
59
52
949

C. Semantic Level Probing Results

1) VLP models cannot distinguish negation logic in
multimodal alignment. In the Negation Logic task in Tab. III,
only the BLIP model achieves an accuracy (79.0%) above
the random level. The performance of CLIP (47.3%) and X-
VLM (48.1%) is close to the random level. Flava, on the
other hand, tends to select sentences with negation words.
This indicates that it is challenging for the pretraining process
to transfer the understanding of negation words from the
linguistic knowledge in datasets and language models to VLP
models.

2) VLP models exhibit poor perception of spatial rela-
tionships, making it difficult for them to correctly identify

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015

9

simple spatial relationships, especially “left” and “right”.
In Tab. IV, for “to the left of” and “to the right of” sub-
category in the spatial-based Relationships, all models exhibit
similar probabilities of selecting Cor or Exc (p < 5%). This
suggests that the models are confusing the primary/ secondary
objects in the spatial relationships. In Fig. 2, we show the
model performance on the Multi-spatial Relationship task. To
our surprise, all models struggle to differentiate positional
relationships. In samples whose right relationships match
the images are “to the left of” and “to the right of”, the
models show similar probabilities distribution to select the
four relationship labels and cannot distinguish between “left”
and “right”. CLIP prefers the relationship “to the right of”,
while Flava prefers “on”. However, these relationships are not
the correct options for the given sample. This indicates that
VLP models are mainly incapable of accurately discerning
spatial relationships within images or do not understand the
positional relationships. To some extent,
is noteworthy
that BLIP and X-VLM showcase the ability to distinguish
between “on” and “below” and otherspatialrelationships in
the Relationship Composition task (Tab. IV). During the VLP
process, BLIP integrates visual features into the text encoder.
In contrast, X-VLM employs a pretraining objective,
the
Bounding Box Prediction loss, linked to spatial relationships.
This underscores the significance of incorporating spatial-
related information during pretraining, which enhances the
VLP model’s perception of the physical world.

it

D. Analysis on MLLMs

MLLMs exhibit excellent alignment ability and ef-
ficiently transfer linguistic knowledge, showcasing their
competence in handling intricate syntactic and semantic
understanding, including negation comprehension and the
composition of triplet relationships. Nonetheless, MLLMs
continue to encounter challenges in accurately learning
spatial relationships and exhibit confusion in dealing with
the compositional aspects of nouns and attributes within
sentences.

The Fig. 3, Fig. 4, and Tab. V show the BLIP2 performance
on Negation Logic, Attribute Ownership, and Relationship
Composition, respectively.

In the Negation Logic task (Fig. 3), BLIP2 demonstrates
a good understanding of the semantics of negation words
when answering questions in the Wro class. This observation
signifies that BLIP2 can comprehend the meaning of negation
and engage in reasoning.

However, BLIP2 exhibits similar confusion in the Cor class
of the Negation Logic task and the Sep class of the Attribute
Ownership task. (Both of them put nouns behind adjectives
in the form of “is the X A and is the Y B” or “are the
X and the Y A and B respectively”. It is different from
the form of “are there the A X and the B Y ”. The former
emphasizes nouns, while the latter emphasizes attributes.) This
could be attributed to BLIP2’s better alignment of nouns in
the multimodal domain but its inability to accurately deter-
mine if the attributes belong to the noun. In the Attribute
Ownership task (Fig. 4), BLIP2 has a similar performance.

When answering questions in the Cor class, BLIP2 accurately
determines whether the sentence semantic matches the image.
However, even when the nouns and adjectives are mismatched
in the Exc class questions, BLIP2 still answers “yes” with a
high probability. This also indicates that BLIP2 is sensitive
to nouns and struggles to judge the semantic combination
of attributes and nouns accurately. This result highlights the
existing limitation that the multimodal alignment in MLLMs
is a coarse-grained alignment of entities while overlooking
fine-grained alignment.

In Relationship Composition task (Tab. V), BLIP2 exhibits
excellent alignment of nouns (97.0% and 98.8% on None)
and understanding of relationships between entities (75.3%
and 88.3% on Cor), showcasing its visual-language alignment
capability and the rich linguistic knowledge obtained from
LLMs. However, it exhibits slightly weaker performance in
distinguishing spatial-based relationships (Exc) such as “at”
(60%), “of” (75.8% ), “in front of” (81.6%), “to the left of”
(73.3%), and “to the right of” (73.3%). This suggests that
similar to VLP models, MLLMs also encounter challenges in
learning precise spatial relationships.

VI. SUGGESTION

LLMs have gained rich linguistic knowledge, physical world
knowledge, and impressive reasoning abilities from pretraining
on a vast amount of text [11], [12], [69], thus showing decent
performance on a bunch of downstream tasks [70], [71],
[72]. However, despite the prevalence of multimodal data in
real-life scenarios, acquiring and training high-quality image-
text pairs remains challenging. Current research on MLLMs
addresses this issue by leveraging the knowledge within LLMs
to align visual features and textual representations [13]. But
MLLMs still have shortcomings in feature representation,
comprehension, and reasoning [68]. To promote the progress
of the multimodal community, based on our findings, we offer
the following recommendations for future MLLMs research
and development:

Utilizing LLMs as the language backbone can facil-
itate the comprehension of text encompassing intricate
semantics, syntax, and logic. Comparing the performance of
traditional VLP models and the MLLMs on SNARE, the latter
can better accomplish multimodal tasks by leveraging the rich
linguistic knowledge from LLMs. For example, BLIP2 can
understand negation semantics, and its syntactic knowledge
can help to construct better complex relationships between
words (e.g., the relationship between nouns and adjectives and
triplet relationships in the Relationship Composition).

Focusing on content words and simplifying the sen-
tence’s syntactic structure may be an important approach
to constructing high-quality datasets and improving the
effect of multimodal alignment. In the Attribute Ownership
task, we find that both VLP models and MLLMs easily under-
stand concise syntactic relationships (short-distance combina-
tions between nouns and adjectives). In the Semantic Structure
task, VLP models do not exhibit an obvious preference for the
sentence without shuffling. In contrast, BLIP, CLIP, and X-
VLM prefer sentences that maintain the syntactic relationship

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015

10

between content words. This suggests that complex syntactic
structures may not be the key factors driving better multi-
modal alignment performance, and content words are critical
elements. Hence, employing more content words to establish
clearer and simpler syntactic relationships within sentences
might contribute to creating higher-quality datasets and enable
more effective multimodal alignment.

The quality of multimodal modeling largely depends
on better fine-grained semantic relationships. Although
MLLMs and some VLP models (such as Flava) can accu-
rately align entity features between images and text, they still
struggle to differentiate whether the attributes belong to the
noun accurately. Most VLP models struggle to differentiate
verb- or spatial-based relationships between nouns. Therefore,
multimodal models need to learn more fine-grained align-
ment to structure complex relationships between nouns and
attributes accurately. In future work,
to improve the fine-
grained relationships in the training data, we may explore 1)
the refinement of the sentences conditioned on the fine-grained
visual information, which can be seen as a kind of knowledge
distillation [73], [74]; and 2) the bidirectional refinement of
the paired data [75], [76], i.e., refine the text conditioned on
the image, and vice versa.

Complex visual knowledge mining should be considered
in the VLP process. Both VLP models and MLLMs struggle
to accurately determine the spatial relationships of objects in
the visual context, especially “left” and “right”. BLIP and X-
VLM achieve better results in understanding location infor-
mation by incorporating visual features into the text encoder
and using position-related pretraining objectives. Recently,
MLLMs like Kosmos-2 [77] have also improved multimodal
alignment performance by combining location information
into pretraining objectives. Therefore, it is meaningful for
future research to explore pretraining objectives that facilitate
the learning of fine-grained visual knowledge. Besides, the
dynamic learning process, e.g., curriculum learning [78] and
progressive learning [79], can be employed for the VLP, where
the training starts with simple patterns and gradually goes into
complex patterns.

VII. CONCLUSION

In this paper, we introduce the first comprehensive multi-
modal alignment probing benchmark – SNARE for evaluating
the impact of linguistic knowledge, e.g., lexicon, syntax, and
semantics, for the VLP models. We carefully designing four
tasks: semantic structure, negation logic, attribute ownership,
and relationship composition. By evaluating the state-of-the-
art VLP models, including BLIP, CLIP, Flava, X-VLM, and
BLIP2, we show that current VLP models are capable of
understanding simple semantics but struggle with complex
syntactic relationships and negation logic and lack the model-
ing of fine-grained information (e.g., spatial relationship and
attribute ownership) in visual features. To enhance the cross-
modal alignment modeling, we suggest 1) using LLMs that
own rich linguistic knowledge as the language backbone of
VLP to improve the understanding and generation of the
sentences with difficult semantics and logic, 2) constructing

high-quality datasets by closely aligning the visual objectives
with the content words in the sentence, and making the
syntactic structure simpler, and 3) mining the fine-grained and
complex visual knowledge by carefully designing better learn-
ing objectives. We hope that our benchmark and conclusions
will facilitate the development of VLP models in the future.

REFERENCES

[1] L. H. Li, M. Yatskar, D. Yin, C.-J. Hsieh, and K.-W. Chang, “Visualbert:
A simple and performant baseline for vision and language,” arXiv
preprint, 2019.

[2] J. Lu, D. Batra, D. Parikh, and S. Lee, “Vilbert: Pretraining task-agnostic
visiolinguistic representations for vision-and-language tasks,” Advances
in neural information processing systems, 2019.

[3] A. Radford, J. W. Kim, C. Hallacy et al., “Learning transferable visual
models from natural language supervision,” in International conference
on machine learning, 2021.

[4] X. Chen, X. Wang, S. Changpinyo et al., “Pali: A jointly-scaled

multilingual language-image model,” arXiv preprint, 2022.

[5] O. Caglayan, P. S. Madhyastha, L. Specia, and L. Barrault, “Probing the
need for visual context in multimodal machine translation,” in North
American Chapter of the Association for Computational Linguistics,
2019.

[6] B. Li, C. Lv, Z. Zhou, T. Zhou, T. Xiao, A. Ma, and J. Zhu, “On vision
features in multimodal machine translation,” in Annual Meeting of the
Association for Computational Linguistics, 2022.

[7] J. Rao, F. Wang, L. Ding, S. Qi, Y. Zhan, W. Liu, and D. Tao, “Where
does the performance improvement come from? -a reproducibility con-
cern about image-text retrieval,” in Proceedings of the 45th International
ACM SIGIR Conference on Research and Development in Information
Retrieval, 2022.

[8] J. Rao, L. Ding, S. Qi, M. Fang, Y. Liu, L. Shen, and D. Tao, “Dynamic
contrastive distillation for image-text retrieval,” IEEE Transactions on
Multimedia, 2023.

[9] E. Bugliarello, R. Cotterell, N. Okazaki, and D. Elliott, “Multimodal pre-
training unmasked: A meta-analysis and a unified framework of vision-
and-language berts,” Transactions of the Association for Computational
Linguistics, 2021.

[10] T. Iki and A. Aizawa, “Effect of visual extensions on natural language
understanding in vision-and-language models,” in Proceedings of the
2021 Conference on Empirical Methods in Natural Language Process-
ing, 2021.

[11] OpenAI, “Gpt-4 technical report,” 2023.
[12] H. Touvron, T. Lavril, G. Izacard et al., “Llama: Open and efficient

foundation language models,” arXiv preprint, 2023.

[13] S. Yin, C. Fu, S. Zhao, K. Li, X. Sun, T. Xu, and E. Chen, “A survey

on multimodal large language models,” arXiv preprint, 2023.

[14] J. Li, D. Li, C. Xiong, and S. Hoi, “Blip: Bootstrapping language-image
pre-training for unified vision-language understanding and generation,”
in International Conference on Machine Learning, 2022.

[15] Z. Yin, J. Wang, J. Cao, Z. Shi, D. Liu, M. Li, L. Sheng, L. Bai,
X. Huang, Z. Wang et al., “Lamm: Language-assisted multi-modal
instruction-tuning dataset, framework, and benchmark,” arXiv preprint
arXiv:2306.06687, 2023.

[16] M. J. Cresswell, Logics and languages. Routledge, 2016.
[17] C.-Y. Hsieh, J. Zhang, Z. Ma, A. Kembhavi, and R. Krishna, “Sugar-
crepe: Fixing hackable benchmarks for vision-language compositional-
ity,” arXiv preprint arXiv:2306.14610, 2023.

[18] M. J. Choi, A. Torralba, and A. S. Willsky, “Context models and out-

of-context objects,” Pattern Recognition Letters, 2012.

[19] J. Cao, Z. Gan, Y. Cheng, L. Yu, Y.-C. Chen, and J. Liu, “Behind the
scene: Revealing the secrets of pre-trained vision-and-language models,”
in Computer Vision–ECCV 2020: 16th European Conference, Glasgow,
UK, August 23–28, 2020, Proceedings, Part VI 16, 2020.

[20] T. Yun, C. Sun, and E. Pavlick, “Does vision-and-language pretraining

improve lexical grounding?” arXiv preprint arXiv:2109.10246, 2021.

[21] E. Salin, B. Farah, S. Ayache, and B. Favre, “Are vision-language trans-
formers learning multimodal representations? a probing perspective,” in
Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36,
no. 10, 2022, pp. 11 248–11 257.

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015

11

[22] R. Shekhar, S. Pezzelle, Y. Klimovich, A. Herbelot, M. Nabi,
E. Sangineto, and R. Bernardi, “Foil it! find one mismatch between
image and language caption,” in ACL 2017 The 55th Annual Meeting
of the Association for Computational Linguistics: Proceedings of the
Conference, Vol. 1 (Long Papers).
Association for Computational
Linguistics, 2017, pp. 255–265.

[23] L. A. Hendricks and A. Nematzadeh, “Probing image-language trans-
formers for verb understanding,” in Findings of the Association for
Computational Linguistics: ACL-IJCNLP 2021, 2021, pp. 3635–3644.
[24] P. J. R¨osch and J. Libovick`y, “Probing the role of positional information
in vision-language models,” in Findings of the Association for Compu-
tational Linguistics: NAACL 2022, 2022, pp. 1031–1041.

[25] T. Thrush, R. Jiang et al., “Winoground: Probing vision and language
the
models for visio-linguistic compositionality,” in Proceedings of
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2022.

[26] X. Liu, D. Yin, Y. Feng, and D. Zhao, “Things not written in text:
Exploring spatial commonsense from visual signals,” in Proceedings
of
the Association for Computational
Linguistics (Volume 1: Long Papers), 2022, pp. 2365–2376.

the 60th Annual Meeting of

[27] M. Alper, M. Fiman, and H. Averbuch-Elor, “Is bert blind? exploring
the effect of vision-and-language pretraining on visual language under-
standing,” arXiv preprint arXiv:2303.12513, 2023.

[28] A. Radford, J. W. Kim et al., “Learning transferable visual models from
natural language supervision,” in International conference on machine
learning, 2021.

[29] A. Singh, R. Hu, V. Goswami, G. Couairon, W. Galuba, M. Rohrbach,
and D. Kiela, “Flava: A foundational language and vision alignment
model,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2022, pp. 15 638–15 650.

[30] Y. Zeng, X. Zhang, and H. Li, “Multi-grained vision language
pre-training: Aligning texts with visual concepts,” arXiv preprint
arXiv:2111.08276, 2021.

[31] J. Li, D. Li, S. Savarese, and S. Hoi, “BLIP-2: bootstrapping language-
image pre-training with frozen image encoders and large language
models,” in ICML, 2023.

[32] M.-E. Nilsback and A. Zisserman, “Automated flower classification over
a large number of classes,” in Indian Conference on Computer Vision,
Graphics and Image Processing, Dec 2008.

[33] T. Zhao, T. Zhang, M. Zhu, H. Shen, K. Lee, X. Lu, and J. Yin, “Vl-
checklist: Evaluating pre-trained vision-language models with objects,
attributes and relations,” arXiv preprint arXiv:2207.00221, 2022.
[34] K. Pham, K. Kafle, Z. Lin, Z. Ding, S. Cohen, Q. Tran, and A. Shrivas-
tava, “Learning to predict visual attributes in the wild,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2021, pp. 13 018–13 028.

[35] Y.-L. Li, L. Xu, X. Liu, X. Huang, Y. Xu, M. Chen, Z. Ma, S. Wang,
H.-S. Fang, and C. Lu, “Hake: Human activity knowledge engine,” arXiv
preprint arXiv:1904.06539, 2019.

[36] S. Pratt, M. Yatskar, L. Weihs, A. Farhadi, and A. Kembhavi, “Grounded
situation recognition,” in Computer Vision–ECCV 2020: 16th European
Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IV
16. Springer, 2020, pp. 314–332.

[37] M. Yuksekgonul, F. Bianchi, P. Kalluri, D. Jurafsky, and J. Zou, “When
and why vision-language models behave like bags-of-words, and what
to do about it?” in The Eleventh International Conference on Learning
Representations, 2022.

[38] A. Conneau, G. Kruszewski, G. Lample, L. Barrault, and M. Baroni,
“What you can cram into a single $&!#* vector: Probing sentence
embeddings for linguistic properties,” in Proceedings of the 56th Annual
Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers). Melbourne, Australia: Association for Computational
Linguistics, Jul. 2018, pp. 2126–2136.

[39] Q. Zhong, L. Ding, J. Liu, B. Du, and D. Tao, “E2s2: Encoding-
enhanced sequence-to-sequence pretraining for language understanding
and generation,” arXiv preprint arXiv:2205.14912, 2022.

[40] L. Ding, L. Wang, D. Wu, D. Tao, and Z. Tu, “Context-aware cross-
attention for non-autoregressive translation,” in Proceedings of the 28th
International Conference on Computational Linguistics, 2020, pp. 4396–
4402.

[41] L. Ding, L. Wang, X. Liu, D. F. Wong, D. Tao, and Z. Tu, “Understand-
ing and improving lexical choice in non-autoregressive translation,” in
International Conference on Learning Representations, 2021.

[42] Z. Zhang, L. Ding, D. Cheng, X. Liu, M. Zhang, and D. Tao, “Bliss:
Robust sequence-to-sequence learning via self-supervised input repre-
sentation,” arXiv preprint arXiv:2204.07837, 2022.

[43] T. Pham, T. Bui, L. Mai, and A. Nguyen, “Out of order: How important
is the sequential order of words in a sentence in natural language
understanding tasks?” in Findings of ACL, 2021.

[44] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training
of deep bidirectional transformers for language understanding,” arXiv
preprint arXiv:1810.04805, 2018.

[45] A. Wang, A. Singh et al., “Glue: A multi-task benchmark and analysis
platform for natural language understanding,” in Proceedings of the 2018
EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural
Networks for NLP, 2018.

[46] K. Sinha, R. Jia, D. Hupkes, J. Pineau, A. Williams, and D. Kiela,
“Masked language modeling and the distributional hypothesis: Order
word matters pre-training for little,” in Empirical Methods in Natural
Language Processing, 2021.

[47] J. O’Connor and J. Andreas, “What context features can transformer
the Association for

language models use?” in Annual Meeting of
Computational Linguistics, 2021.

[48] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in
neural information processing systems, vol. 30, 2017.

[49] A. Ettinger, “What bert is not: Lessons from a new suite of psycholin-
guistic diagnostics for language models,” Transactions of the Association
for Computational Linguistics, 2020.

[50] P. Parthasarathi, K. Sinha, J. Pineau, and A. Williams, “Sometimes we
want ungrammatical translations,” in Findings of the Association for
Computational Linguistics: EMNLP 2021, 2021, pp. 3205–3227.
[51] K. Sinha, P. Parthasarathi, J. Pineau, and A. Williams, “Unnatural
language inference,” in Annual Meeting of the Association for Com-
putational Linguistics, 2021.

[52] R. Krishna, Y. Zhu, O. Groth et al., “Visual genome: Connecting
language and vision using crowdsourced dense image annotations,”
International journal of computer vision, 2017.

[53] A. Warstadt, A. Parrish, H. Liu, A. Mohananey, W. Peng, S.-F. Wang,
and S. R. Bowman, “Blimp: The benchmark of linguistic minimal
pairs for english,” Transactions of the Association for Computational
Linguistics, 2020.

[54] L. Ding, L. Wang, and D. Tao, “Self-attention with cross-lingual position
representation,” in Annual Meeting of the Association for Computational
Linguistics, 2020.

[55] D. Wu, L. Ding, S. Yang, and M. Li, “Mirroralign: A super lightweight
unsupervised word alignment model via cross-lingual contrastive learn-
ing,” in International Workshop on Spoken Language Translation, 2021.
[56] S. Frank, E. Bugliarello, and D. Elliott, “Vision-and-language or vision-
for-language? on cross-modal influence in multimodal transformers,” in
Proceedings of the 2021 Conference on Empirical Methods in Natural
Language Processing, 2021.

[57] L. Parcalabescu, A. Gatt, A. Frank, and I. Calixto, “Seeing past
words: Testing the cross-modal capabilities of pretrained v&l models
on counting tasks,” in Proceedings of the 1st Workshop on Multimodal
Semantic Representations (MMSR), 2021.

[58] B. Li, R. Wang, G. Wang, Y. Ge, Y. Ge, and Y. Shan, “Seed-bench:
Benchmarking multimodal llms with generative comprehension,” arXiv
preprint arXiv:2307.16125, 2023.

[59] H. Zeijlstra, “Negation in natural language: On the form and meaning
of negative elements,” Language and Linguistics Compass, vol. 1, no. 5,
pp. 498–518, 2007.

[60] I. Orenes, D. Beltr´an, and C. Santamar´ıa, “How negation is understood:
Evidence from the visual world paradigm,” Journal of memory and
language, vol. 74, pp. 36–45, 2014.

[61] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan,
P. Doll´ar, and C. L. Zitnick, “Microsoft coco: Common objects in
context,” in European Conference of Computer Vision. Springer, 2014.
[62] P. Young, A. Lai, M. Hodosh, and J. Hockenmaier, “From image
descriptions to visual denotations: New similarity metrics for semantic
inference over event descriptions,” Transactions of the Association for
Computational Linguistics, 2014.

[63] M. Honnibal and I. Montani, “spacy 2: Natural language understanding
with bloom embeddings, convolutional neural networks and incremental
parsing,” To appear, vol. 7, no. 1, pp. 411–420, 2017.

[64] D. A. Hudson and C. D. Manning, “Gqa: A new dataset for real-world
visual reasoning and compositional question answering,” in Proceedings
of the IEEE/CVF conference on computer vision and pattern recognition,
2019, pp. 6700–6709.

[65] R. S. Jackendoff, Semantic structures. MIT press, 1992, vol. 18.
[66] A. Fyshe, G. Sudre, L. Wehbe, N. Rafidi, and T. M. Mitchell, “The
lexical semantics of adjective–noun phrases in the human brain,” Human
brain mapping, vol. 40, no. 15, pp. 4457–4469, 2019.

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015

12

[67] W. X. Zhao, K. Zhou, J. Li, T. Tang et al., “A survey of large language

models,” arXiv preprint, 2023.

[68] C. Fu, P. Chen, Y. Shen, Y. Qin, M. Zhang, X. Lin, Z. Qiu,
W. Lin, J. Yang, X. Zheng et al., “Mme: A comprehensive evaluation
large language models,” arXiv preprint
benchmark for multimodal
arXiv:2306.13394, 2023.

[69] T. L. Scao, A. Fan, C. Akiki, E. Pavlick, S.

Ili´c, D. Hesslow,
R. Castagn´e, A. S. Luccioni, F. Yvon, M. Gall´e et al., “Bloom: A 176b-
parameter open-access multilingual language model,” arXiv preprint
arXiv:2211.05100, 2022.

[70] Q. Zhong, L. Ding, J. Liu, B. Du, and D. Tao, “Can chatgpt understand
too? a comparative study on chatgpt and fine-tuned bert,” arXiv preprint,
2023.

[71] Q. Lu, B. Qiu, L. Ding, L. Xie, and D. Tao, “Error analysis prompting
enables human-like translation evaluation in large language models: A
case study on chatgpt,” arXiv preprint, 2023.

[72] K. Peng, L. Ding, Q. Zhong, L. Shen, X. Liu, M. Zhang, Y. Ouyang, and
D. Tao, “Towards making the most of chatgpt for machine translation,”
arxiv preprint, 2023.

[73] J. Rao, X. Meng, L. Ding, S. Qi, and D. Tao, “Parameter-efficient
and student-friendly knowledge distillation,” ArXiv, vol. abs/2205.15308,
2022.

[74] H. Deng, L. Ding, X. Liu, M. Zhang, D. Tao, and M. Zhang, “Improving
simultaneous machine translation with monolingual data,” in AAAI
Conference on Artificial Intelligence, 2023.

[75] L. Ding, D. Wu, and D. Tao, “Improving neural machine translation by
bidirectional training,” in Conference on Empirical Methods in Natural
Language Processing, 2021.

[76] L. Ding and D. Tao, “The usyd-jd speech translation system for
iwslt2021,” in International Workshop on Spoken Language Translation,
2021.

[77] Z. Peng, W. Wang, L. Dong, Y. Hao, S. Huang, S. Ma, and F. Wei,
“Kosmos-2: Grounding multimodal large language models to the world,”
ArXiv, vol. abs/2306, 2023.

[78] Y. Bengio, J. Louradour, R. Collobert, and J. Weston, “Curriculum
learning,” in International Conference on Machine Learning, 2009.
[79] L. Ding, L. Wang, X. Liu, D. F. Wong, D. Tao, and Z. Tu, “Progressive
multi-granularity training for non-autoregressive translation,” in Findings
of Annual Meeting of the Association for Computational Linguistics,
2021.

