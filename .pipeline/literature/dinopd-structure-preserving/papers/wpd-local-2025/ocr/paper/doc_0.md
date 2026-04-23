001

002

003

004

005

006

007

008

009

010

011

012

013

014

015

016

017

Wavelet Phase Diffusion for Structurally and
Semantically Consistent Sim-to-Real Translation

Anonymous ECCV 2026 Submission

Paper ID #6276

t
u
p
n
I

]
2
5
[

r
e
t
s
a
m
e
R
l
a
r
u
e
N

]
1
[

5
.
2

)
s
r
u
O
(
D
P
-
ψ

r
e
f
s
n
a
r
T
-
s
o
m
s
o
C

Fig. 1: Wavelet Phase Diffusion (ψ-PD) preserves structure and semantics
in sim-to-real translation. Compared to prior baselines, ψ-PD better maintains
structural (red) and semantic consistency (yellow) without auxiliary control modules.

Abstract. High-fidelity simulation-to-reality translation seeks to bridge
appearance gaps while preserving structural and semantic consistency in
images and videos. Conditioning-based generative methods enforce spa-
tial alignment but introduce computationally expensive auxiliary control
modules. Corruption-based phase-preserving diffusion is model-agnostic,
but Fourier-domain phase manipulation induces artifacts that under-
mine structural consistency, a limitation that stems from the global
support of Fourier bases. To address this, we introduce Wavelet Phase
Diffusion (ψ-PD), a phase-preserving diffusion framework that operates
in the Dual-Tree Complex Wavelet Packet Transform (DT-CWPT) do-
main. DT-CWPT yields localized, oriented complex wavelet packets with

001

002

003

004

005

006

007

008

009

010

011

012

013

014

015

016

017

018

019

020

021

022

023

024

025

026

027

028

029

030

031

032

033

034

035

036

037

038

039

040

041

042

043

044

045

046

047

048

049

050

051

052

053

054

055

056

057

2

ECCV 2026 Submission #6276

approximately shift-invariant phase, enabling spatially adaptive phase
constraints with reduced global spectral coupling. For image transla-
tion, ψ-PD outperforms prior methods in realism and semantic consis-
tency while maintaining strong structural alignment. Furthermore, we
measure downstream performance with Vision-Language Model (VLM)-
based drivability evaluations on translated driving videos. Unlike prior
baselines, which degrade driving performance, ψ-PD succeeds in pre-
serving critical structural and semantic cues while improving realism.
Compared to simulator image input, this results in a reduction of ADE
and FDE by 5.5% and 5.2%, respectively.1

Keywords: Style Transfer · Sim-to-Real · Diffusion Models

1

Introduction

Image-to-image translation and related visual synthesis aim to transform vi-
sual appearance while preserving semantic and geometric structure. Such prob-
lems arise in image restoration, re-rendering, stylization, and sim-to-real trans-
fer, where realistic generation requires synthesizing high-frequency texture while
maintaining low-frequency structural consistency. Prior work in signal process-
ing [28] establishes that phase predominantly encodes geometric structure, whereas
magnitude governs texture statistics, motivating recent diffusion-based approaches
that preserve phase information to improve spatial alignment during generation.
Recent phase-preserving diffusion methods [52] implement this principle in
the Fourier domain by constraining phase while randomizing magnitude. While
effective at enforcing global structural consistency, Fourier-based representations
suffer from an inherent limitation: Fourier basis functions are globally supported.
As a result, frequency-domain constraints introduce global spectral coupling,
whereby enforcing structural rigidity in a localized region implicitly affects the
entire image. This lack of spatial locality limits the ability to impose region-
specific constraints and often leads to artifacts such as ringing, grid-like interfer-
ence, and boundary leakage, particularly in scenes with heterogeneous structural
requirements.

To address this limitation, we propose ψ-PD, a phase-preserving diffusion
framework operating in the Dual-Tree Complex Wavelet Packet Transform (DT-
CWPT) domain. Unlike globally supported Fourier bases, DT-CWPT yields
localized, approximately shift-invariant, orientation-selective complex wavelet
packets, allowing ψ-PD to avoid global spectral coupling and control the realism–
consistency trade-off in a finer manner.

To promote robustness and avoid overfitting to domain-specific artifacts, ψ-
PD is trained on diverse open-domain visual data rather than simulator-specific
datasets. This training strategy encourages the model to learn a general photore-
alistic appearance prior while relying on localized phase constraints to preserve

1 We will release an open-source implementation of our algorithm and checkpoints

upon acceptance.

018

019

020

021

022

023

024

025

026

027

028

029

030

031

032

033

034

035

036

037

038

039

040

041

042

043

044

045

046

047

048

049

050

051

052

053

054

055

056

057

ECCV 2026 Submission #6276

3

structure and semantics. Importantly, the proposed approach does not depend
on semantic annotations or task-specific conditioning and can be integrated with
arbitary diffusion backbones without architectural modification.

We evaluate the proposed method on sim-to-real image and video translation
benchmarks, which provide a stringent test of structural fidelity. Datasets such
as SYNTHIA and CARLA impose strong geometric constraints, as even mi-
nor spatial inconsistencies can significantly impact downstream perception and
planning tasks. Performance under these conditions therefore serves as a rigor-
ous validation of the proposed structural priors, while the method itself remains
applicable to general visual translation tasks.

The contributions of this work are as follows:

– We introduce ψ-PD, a phase-preserving diffusion framework operating in
the DT-CWPT domain that overcomes the global coupling limitations of
Fourier-domain methods, enabling image style translation while preserving
structural and semantic information.

– We achieve state-of-the-art results on both SYNTHIA image translation
and sim-to-real driving video translation, improving realism and consistency
while reducing downstream Vision–Language Model planning errors.

– We demonstrate the flexibility of ψ-PD’s spatially adaptive phase control

through applications like instance-level style transfer.

2 Related Work

2.1 Diffusion Models

Diffusion models have emerged as a powerful class of generative models, achiev-
ing state-of-the-art performance across image [3, 36, 47], video [4, 43, 44], and
3D generation [45, 46, 54]. Early formulations define a forward noising process
and learn a reverse denoising procedure through variational inference [16, 19, 39]
or score matching [40, 41]. Recent work has increasingly adopted flow match-
ing [10, 22, 23, 29], which replaces stochastic diffusion with deterministic vec-
tor fields, substantially improving sampling efficiency and scalability in large
transformer-based models. Despite their success, both diffusion and flow-based
generative processes progressively corrupt data into noise, leading to structural
and semantic degradation when strong appearance changes or high noise levels
are required.

2.2 Structure-Aligned Generation with Diffusion

Existing approaches for structure-aligned generation with diffusion can be broadly
categorized into conditioning-based and corruption-based methods. Conditioning-
based methods enforce structural alignment by introducing explicit conditioning
signals or adaptation modules into the diffusion backbone. Representative ex-
amples include ControlNet [55] and its lightweight variants [27, 30, 49, 56], which
inject structural guidance through auxiliary control architectures, as well as

058

059

060

061

062

063

064

065

066

067

068

069

070

071

072

073

074

075

076

077

078

079

080

081

082

083

084

085

086

087

088

089

090

091

092

093

094

095

096

097

058

059

060

061

062

063

064

065

066

067

068

069

070

071

072

073

074

075

076

077

078

079

080

081

082

083

084

085

086

087

088

089

090

091

092

093

094

095

096

097

4

ECCV 2026 Submission #6276

instruction-based and multimodal image editing models, e.g., InstructPix2Pix [5]
and QWen-Edit [47]. In sim-to-real translation, Cosmos-Transfer 2.5 [1] further
combines multiple control branches to achieve multi-modal alignment across
appearance, structure, and semantics. While effective, conditioning-based ap-
proaches increase architectural complexity, incur additional computational over-
head, and typically require task-specific supervision or annotations. In contrast,
corruption-based methods preserve structure by directly constraining the diffu-
sion corruption process without introducing auxiliary networks. SDEdit [25] im-
plicitly retains structural content by injecting noise at an intermediate diffusion
step and limiting the diffusion horizon. Recently proposed NeuralRemaster [52]
explicitly preserves structure by fixing phase during diffusion, enabling model-
agnostic structure-aligned image and video generation without additional param-
eters. However, existing corruption-based methods impose global constraints and
offer limited flexibility in enforcing spatially adaptive structural control.

2.3 Wavelet-Based Approaches

Wavelet representations provide localized decompositions [24] and have been
widely used for image compression [42], restoration [7,17,51], super-resolution [21,
26], and synthesis [11, 12, 14, 31]. In the context of diffusion models, recent work
has explored wavelet-based formulations primarily to improve computational
efficiency or frequency-aware generation by performing diffusion in a wavelet-
transformed domain, decomposing images into low- and high-frequency subbands
for reduced-resolution sampling or enhanced detail synthesis. However, existing
wavelet-based diffusion approaches operate on real-valued wavelet coefficients
and focus on efficiency or perceptual quality, rather than explicitly preserving
structural and semantic information during the diffusion process. In particu-
lar, they do not exploit phase information or provide mechanisms for spatially
adaptive control over the corruption process, leaving open the question of how
localized spectral representations can be leveraged to impose semantics-aware
structural constraints.

3 Preliminary: Phase-Preserving Diffusion

The 2D Fourier Transform F can be used to decompose an image I ∈ RH×W
into magnitude |F(I)| and phase ∠F(I):

F(I)(u, v) =

I(x, y) exp

(cid:16)

−j2π

(cid:16) ux
W

+

vy
H

(cid:17)(cid:17)

(cid:88)

x,y

= |F(I)(u, v)| exp (j∠F(I)(u, v)) .

(1)
As established in the seminal research of Oppenheim et al. [28], phase acts as the
primary carrier of spatial structure, while magnitude predominantly controls tex-
ture statistics. Building on this principle, phase-preserving diffusion approaches,
e.g., NeuralRemaster [52], construct structured noise ˆϵ by adopting the phase of
the source image I while retaining the magnitude of a Gaussian noise ϵ ∼ N (0, I).

098

099

100

101

102

103

104

105

106

107

108

109

110

111

112

113

114

115

116

117

118

119

120

121

122

123

124

125

126

127

128

129

130

131

132

133

134

135

098

099

100

101

102

103

104

105

106

107

108

109

110

111

112

113

114

115

116

117

118

119

120

121

122

123

124

125

126

127

128

129

130

131

132

133

134

135

ECCV 2026 Submission #6276

5

(a) Input

(b) Fourier(r=32) (c) Wavelet(r=32) (d) Fourier(r=64) (e) Wavelet(r=64)

Fig. 2: Fourier- vs. wavelet-domain low-pass filtering. We compare Fourier-
domain filtering to DT-CWPT filtering at cutoff radii r = 32 and r = 64. Fourier
reconstructions exhibit non-local ringing near high-contrast boundaries (see zoomed
insets), while DT-CWPT preserves local geometry and edge structure more faithfully.

To regulate the degree of structural preservation, a global frequency selection is
applied. By defining a radial cutoff frequency r, we construct a low-pass filtering
mask Mr that selects the source-image phase in the masked region and injects
noise phase in the complementary region:

ˆϵ = F −1(cid:16)

|F(ϵ)| exp (cid:0)j · (cid:0)Mr ⊙ ∠F(I) + (1 − Mr) ⊙ ∠F(ϵ)(cid:1)(cid:1) (cid:17)

,

(2)

where ⊙ means element-wise multiplication. This mechanism keeps the phase
in the masked (low-frequency) region consistent with the source image while
the high-frequency components, corresponding to textures, are randomized. For
videos, structured noise is constructed along the time dimension and conditioned
on the re-rendered first frame.

4 Method

4.1 Wavelet Analysis Fundamentals

Fourier-domain phase preservation is fundamentally limited by the global sup-
port of the Fourier basis functions: since each coefficient influences the entire
spatial domain, any spectral modification affects the reconstruction globally
rather than locally. As a consequence, low-pass filtering induces non-local in-
terference despite the use of smooth spectral attenuation, manifesting as Gibbs
phenomenon [13] and boundary leakage [15], particularly around high-contrast
edges as shown in Figs. 2b and 2d.

To overcome these limitations, we adopt the Dual-Tree Complex Wavelet
Packet Transform (DT-CWPT) [38] as our foundational representation. Built
from finite-support dual-tree filter banks, DT-CWPT provides spatially local-
ized, approximately analytic responses via Hilbert-pair wavelets [20], which sub-
stantially reduces non-local interference compared with Fourier bases.

We use DT-CWPT to decompose an image into a single real-valued low-
frequency packet L and L complex-valued high-frequency packets {Hl}L
l=1, in-
dexed from high to low frequency. Each packet Hl occupies a dyadic frequency
band [f min

].

, f max
l

l

136

137

138

139

140

141

142

143

144

145

146

147

148

149

150

151

152

153

154

155

156

157

158

159

160

161

162

163

136

137

138

139

140

141

142

143

144

145

146

147

148

149

150

151

152

153

154

155

156

157

158

159

160

161

162

163

6

ECCV 2026 Submission #6276

Fig. 3: Overview of the Wavelet Phase Diffusion Algorithm. The process sep-
arates low-frequency components for global phase injection and high-frequency com-
ponents for packet-wise local phase injection.

164

165

166

167

168

169

170

171

172

173

174

175

176

177

178

179

180

181

182

183

184

185

186

4.2 Wavelet Phase Diffusion (ψ-PD)

We construct structured noise by injecting source phase into a Gaussian noise
sample in the DT-CWPT domain. An overview is depicted in Fig. 3.

Following NeuralRemaster, we control structure preservation via a cutoff ra-
dius in radial frequency coordinates. In the simplest setting we use a single
global scalar radius r; for spatially adaptive control, we use a radius map R.
Concretely, we convert radii to a unitless cutoff map by normalizing with the
Nyquist frequency fNyq:

164

165

166

167

168

169

170

171

(cid:18)

F = min

max

(cid:18) R
fNyq

(cid:19)

(cid:19)

, 0

, 1

,

(cid:18)

f = min

max

(cid:18) r

fNyq

(cid:19)

(cid:19)

, 0

, 1

.

(3)

172

Here F ∈ [0, 1]H×W (and f ∈ [0, 1]) indicates the highest normalized frequency
up to which the image phase should be preserved at each location.

Phase Extraction. We apply DT-CWPT to both the source (green) and a Gaus-
sian noise sample (blue), yielding complex-valued high-frequency packets {Hl}L
l=1
as well as a real-valued low-frequency packet L. The high-frequency packets ad-
mit direct magnitude-phase decomposition. The low-frequency packet remains
real-valued and therefore does not inherently support wavelet-phase extraction
to serve as a structural anchor. Since L lacks an intrinsic phase decomposition,
we extract its global phase from its Fourier spectrum. For all complex packets
{Hl}L

l=1, we extract the localized wavelet phase directly:

F(L) = |F(L)| exp (j∠F(L)) ,

Hl = |Hl| exp (j∠Hl) .

(4)

Although we invoke the Fourier transform, it is applied only to L, whose support
is chosen to lie safely below the minimum cutoff frequency, avoiding any addi-
tional Fourier-domain low-pass filtering that could introduce structural artifacts.

173

174

175

176

177

178

179

180

181

182

183

184

185

186

NormalizeRadius Map Freq Map or =......=..............................Packet-wiseLocalPhaseInjection Diffusion Model.........=VAE-Enc.StructuredNoise...Downsample from  to VAE-Dec.ECCV 2026 Submission #6276

7

, f max
l

Packet-wise Local Phase Injection. For each high-frequency packet with support
[f min
], we choose to inject the source phase based on the downsampled
l
cutoff mask ˆF. This allows us to control the phase injection on a local level,
enabling applications such as instance-level style transfer (cf . Sec. 5.6). Using
the midpoint frequency of each packet as threshold, we compute a packet-specific
injection mask Ml, which determines whether the source phase is injected:

∠ ˆHl = Ml ⊙ ∠H(x)

l + (1 − Ml) ⊙ ∠H(ϵ)

l

.

(5)

Typically, very high-frequency packets are dominated by noise phase (blue), very
low-frequency packets are dominated by source phase (green), and intermediate
packets are mixed spatially depending on ˆF (green-blue gradient). After phase in-
jection, we reconstruct the final structured noise ˆϵ using the inverse DT-CWPT:
ˆϵ = DT-CWPT−1(cid:16)ˆL, { ˆHl}L

where

(cid:17)

l=1

(cid:16)

L(ϵ)(cid:17)(cid:12)
(cid:12)
(cid:12) exp

ˆL = F −1 (cid:16)(cid:12)
(cid:12)
(cid:12)F

L(x)(cid:17)(cid:17)(cid:17)
In practical implementations, we approximate DT-CWPT with a recursive DT-
CWT construction [8]. Details are provided in the Supplementary Material.

(cid:16)(cid:12)
(cid:12)H(ϵ)
(cid:12)

(cid:12)
(cid:12)
(cid:12) exp

, ˆHl =

j∠ ˆHl

j∠F

(6)

(cid:17)(cid:17)

(cid:16)

(cid:16)

(cid:16)

.

l

5 Experiments

We empirically evaluate the proposed ψ-PD framework on the task of high-
fidelity sim-to-real translation. To demonstrate generalization across modalities,
we evaluate our approach on both image and video translation, using FLUX-
dev [3] and WAN 2.2 14B [43] as the respective diffusion backbones. Following
NeuralRemaster [52], we train using open-domain datasets [2, 53], rather than
autonomous-driving-specific data. Unless otherwise specified, all experiments use
a single global scalar cutoff radius r (equivalently, a constant cutoff F). We only
use a spatially varying cutoff tensor (radius map R and its normalized cutoff
map F ∈ [0, 1]H×W ) in Sec. 5.6.

5.1 Datasets

SYNTHIA is a large-scale synthetic dataset of urban scenes [37]. We use the
SYNTHIA-RAND-CITYSCAPES subset, which contains 9,400 images rendered
under diverse environmental conditions, including varying weather, lighting, and
seasonal appearances. Each image is provided with pixel-aligned depth maps and
semantic annotations. This dataset is used to evaluate image translation quality.

CARLA is an open-source urban driving simulator that provides controllable
virtual environments for autonomous driving research [9]. Following the data col-
lection protocols of nuCarla [32] to prevent the occurrence of abruptly spawned
or teleported agents, we collect a dataset comprising 60 driving sequences ac-
quired at 10 Hz, each containing 109 frames, across 3 distinct towns. Ego-vehicle
trajectories are produced using the simulator’s default autopilot controller. This
benchmark is used to evaluate high-fidelity video translation quality.

187

188

189

190

191

192

193

194

195

196

197

198

199

200

201

202

203

204

205

206

207

208

209

210

211

212

213

214

215

216

217

218

219

220

221

222

223

224

187

188

189

190

191

192

193

194

195

196

197

198

199

200

201

202

203

204

205

206

207

208

209

210

211

212

213

214

215

216

217

218

219

220

221

222

223

224

8

ECCV 2026 Submission #6276

5.2 Evaluation Metrics

Sim-to-real translation requires balancing visual realism, structural faithfulness,
and semantic consistency. We therefore evaluate the proposed method along
complementary dimensions, using modality-specific metrics for image and video
translation.

Image Translation. For image translation, we evaluate visual quality, structural
alignment, and semantic consistency.

– Visual quality is measured using the Appearance Score (AS) following [52],
defined as AS = (x⊤tp)/(x⊤tn). Here, x denotes the CLIP [34] embedding
of the translated image, while tp and tn denote the CLIP embeddings of pos-
itive and negative textual prompts, respectively. Higher AS values indicate
stronger alignment with real photographic appearance.

– Structural alignment is assessed using Depth-SSIM following [52]. We
estimate monocular depth from the images using Depth Anything V2 [50]
and compute SSIM against the ground-truth simulator depth.

– Semantic consistency is evaluated using mean Intersection over Union
(mIoU), obtained by applying Segformer [48] to the images and comparing
predictions against the ground-truth simulator annotations.

Video Translation. For video translation, we additionally account for temporal
coherence and downstream utility.

– Visual quality is measured using Fréchet Inception Distance (FID) with
the validation set of nuScenes [6]. To mitigate layout mismatches between
simulated and real scenes, we adopt a semantics-aware sampling strategy
that matches semantically corresponding regions between domains before
computing distributional distances following [35]. VGG features of in total
176,400 pairs are matched and compared.

– Temporal consistency is evaluated using Motion Smoothness (MS) follow-
ing [18], which measures motion plausibility and temporal coherence across
consecutive frames.

– Downstream utility is assessed using the LightEMMA framework [33],
with Gemini-2.5-Flash as the core vision-language model. LightEMMA pre-
dicts a 3-second future trajectory conditioned on the current frame and the
preceding 3-second trajectory. We restrict evaluation to the central 49 frames
of each sequence, thereby ensuring complete availability of both past and fu-
ture trajectories for every evaluated frame. For each sequence, framewise
evaluation is conducted 25 times, i.e., at 5 Hz. We report Average Displace-
ment Error (ADE) and Final Displacement Error (FDE) with respect to the
simulator-provided ground-truth trajectories.

5.3 Quantitative Results

Image Translation. Tab. 1 summarizes the quantitative performance of the pro-
posed method against established baselines in the image translation task. As

225

226

227

228

229

230

231

232

233

234

235

236

237

238

239

240

241

242

243

244

245

246

247

248

249

250

251

252

253

254

255

256

257

258

259

260

261

262

263

264

265

225

226

227

228

229

230

231

232

233

234

235

236

237

238

239

240

241

242

243

244

245

246

247

248

249

250

251

252

253

254

255

256

257

258

259

260

261

262

263

264

265

Table 1: Quantitative evaluations for image translation. Bold indicates the
best performance, while underlined denotes the second best.

ECCV 2026 Submission #6276

9

Method

Input

SDEdit [25]
ControlNet-Tile [55]
QWen-Edit [47]
NeuralRemaster [52]
Cosmos-Transfer 2.5 [1]

Ours

Appearance Score ↑ Depth-SSIM ↑ mIoU ↑
38.72

0.9143

0.9791

1.0065
1.0203
1.0295
1.0319
1.0326

1.0387

0.8456
0.8534
0.8644
0.8635
0.8939

0.8913

21.79
24.67
28.33
27.84
33.35

33.80

266

267

268

269

270

271

272

273

274

275

276

277

278

279

280

281

282

283

284

285

286

287

288

289

290

291

292

293

294

295

expected, all evaluated generative methods successfully bridge the domain gap,
yielding Appearance Scores that strictly improve upon the synthetic input. The
proposed approach achieves the highest visual realism score (AS = 1.0387), in-
dicating a strong alignment with real-world photographic statistics.

Crucially, this improvement in appearance does not come at the severe ex-
pense of underlying structural integrity. While stochastic corruption methods
such as SDEdit [25] and lightweight conditioning mechanisms like ControlNet-
Tile [55] exhibit substantial drops in geometric and semantic faithfulness, the
proposed method maintains a robust structural anchor. QWen-Edit [47] generally
maintains the original semantics but does not reliably preserve the scene struc-
ture. Specifically, our approach yields a Depth-SSIM of 0.8913. While Cosmos-
Transfer 2.5 [1] achieves a marginally higher Depth-SSIM (0.8939), it does so at
the cost of visual realism (AS = 1.0326).

Furthermore, the proposed method demonstrates superior robustness against
semantic drift. It achieves the highest semantic consistency among all evalu-
ated translation methods (mIoU = 33.80), outperforming both global phase-
preserving NeuralRemaster (mIoU = 27.84) and conditioning-based Cosmos-
Transfer 2.5 (mIoU = 33.35). These results indicate that the localized fre-
quency control provided by the DT-CWPT domain allows for a more favor-
able consistency–realism trade-off, enabling significant appearance shifts in high-
frequency regions while safely preserving the salient low-frequency anchors nec-
essary for downstream perception tasks.

Video Translation. Tab. 2 presents the quantitative evaluation for the video
translation task. Extending sim-to-real translation to the temporal domain re-
quires balancing visual realism with temporal coherence, while crucially preserv-
ing the geometric and semantic cues necessary for downstream decision-making.
In terms of perceptual quality, the proposed ψ-PD framework achieves the
lowest FID (35.76), indicating the strongest distributional alignment with real-
world driving videos. Importantly, this enhancement in visual fidelity does not
compromise temporal stability; our method maintains a Motion Smoothness of

266

267

268

269

270

271

272

273

274

275

276

277

278

279

280

281

282

283

284

285

286

287

288

289

290

291

292

293

294

295

10

ECCV 2026 Submission #6276

Table 2: Quantitative evaluation on video translation. Relative changes of ADE
and FDE w.r.t. the input are reported in parentheses (%). Bold indicates the best
performance. NR: NeuralRemaster; CT: Cosmos-Transfer 2.5.

Method

FID ↓ MS ↑

(%)

Input

48.87 98.58

ADE(m) ↓

1s

0.52

2s

2.01

3s

4.48

Avg.

2.34

FDE ↓

(m)

5.22

NR [52] 37.64 98.30 0.53 (+2.2%) 2.08 (+3.5%) 4.63 (+3.2%) 2.41 (+3.3%) 5.39 (+3.2%)
2.11 (+5.3%) 4.77 (+6.3%) 2.47 (+5.5%) 5.55 (+6.3%)
CT [1]

47.20 98.81 0.52 (-0.1%)

Ours

35.76 98.61 0.49 (-6.7%) 1.90 (-5.5%) 4.25 (-5.3%) 2.21 (-5.5%) 4.95 (-5.2%)

296

297

298

299

300

301

302

303

304

305

306

307

308

309

310

311

312

313

314

315

316

317

318

319

320

321

322

323

324

325

326

327

98.61, comparable to the original simulator input (MS = 98.58) and outperform-
ing the global phase-preserving baseline, NeuralRemaster (MS = 98.30).

The most significant distinction between the evaluated methods emerges in
their downstream utility. While baselines succeed in improving visual realism
to varying degrees, they actively degrade the drivability of the sequences. Both
NeuralRemaster and Cosmos-Transfer 2.5 exhibit increased ADE and FDE met-
rics relative to the simulator input across all temporal horizons. For instance,
Cosmos-Transfer 2.5 increases the average ADE by 5.5% and FDE by 6.3%. This
performance drop suggests that global Fourier constraints and dense conditional
injection introduce subtle spatial warping or semantic drift that disrupts the
planner’s reasoning.

In contrast, our method is the only one that consistently outperforms the
input trajectory across all evaluated horizons (1s, 2s, and 3s). By applying local-
ized wavelet-based phase control, ψ-PD reduces the average ADE by 5.5% and
the FDE by 5.2% relative to the raw simulator videos. The method reliably syn-
thesizes realistic high-frequency domain variations while preserving the critical
low-frequency information required to support stable, long-horizon motion plan-
ning, including scene boundaries, lane geometries, relative agent configurations,
and traffic signal states.

5.4 Qualitative Results

Image Translation. Fig. 4 provides a qualitative comparison between the pro-
posed ψ-PD framework and existing baseline methods on the image translation
task. SDEdit and ControlNet-Tile struggle to preserve structural consistency,
often causing semantic errors such as translating pedestrians as traffic barriers.
QWen-Edit [47] tends to maintain the original semantics but does not preserve
the exact underlying structure. NeuralRemaster often introduces local geometric
warping that misaligns structural boundaries causing structural misregistration.
We argue this behavior arises because its global phase coupling mechanism in-
herits the non-local interference artifacts previously analyzed in Sec. 4.1. While
Cosmos-Transfer 2.5 preserves structure effectively, it suffers from semantic drift.
Critical scene elements, such as traffic barriers, are altered. In contrast, the pro-
posed method effectively mitigates these failure modes.

296

297

298

299

300

301

302

303

304

305

306

307

308

309

310

311

312

313

314

315

316

317

318

319

320

321

322

323

324

325

326

327

ECCV 2026 Submission #6276

11

t
u
p
n
I

]
5
2
[

t
i
d
E
D
S

]
5
5
[

e
l
i

T
-
t
e
N
l
o
r
t
n
o
C

]
7
4
[

t
i
d
E
-
n
e
W
Q

]
2
5
[

r
e
t
s
a
m
e
R
l
a
r
u
e
N

]
1
[

5
.
2

)
s
r
u
O
(
D
P
-
ψ

r
e
f
s
n
a
r
T
-
s
o
m
s
o
C

Fig. 4: Qualitative comparison of image translation. Red boxes indicate struc-
tural misalignment, orange boxes low visual realism, and yellow boxes loss of semantic
consistency.

328

329

330

Video Translation. Fig. 5 illustrates the direct impact of translation fidelity
on downstream motion planning across two representative driving scenarios.
Ground-truth future ego trajectories for a 3-second horizon are depicted in green,

328

329

330

12

ECCV 2026 Submission #6276

(a) Input
ADEavg: 2.74 m, FDE: 5.81 m

(b) Cosmos-Transfer 2.5 [1]
ADEavg: 3.09 m, FDE: 7.31 m

(c) Ours
ADEavg: 1.57 m, FDE: 3.19 m

(d) Input
ADEavg: 1.90 m, FDE: 5.31 m

(e) NeuralRemaster [52]
ADEavg: 3.47 m, FDE: 6.78 m

(f ) Ours
ADEavg: 0.44 m, FDE: 1.07 m

Fig. 5: Qualitative results on downstream utility of translated videos.
Ground-truth future ego trajectories for the next 3 s are drawn in green, and pre-
dicted trajectories are drawn in red.

331

332

333

334

335

336

337

338

339

340

341

342

343

344

345

346

347

348

349

350

351

352

353

354

355

while the predicted trajectories generated by the downstream planner are shown
in red.

In the first scenario (Figs. 5a to 5c), the original simulator input inherently
induces a baseline planning error (ADE = 2.74 m, FDE = 5.81 m) due to the
domain gap. When translated using Cosmos-Transfer 2.5 (Fig. 5b), we observe
semantic drift in the traffic-light state. The erroneous signal cue leads the VLM-
based planner to remain static, resulting in a severe trajectory deviation (ADE =
3.09 m, FDE = 7.31 m). A parallel trend is observed in the second scenario
(Figs. 5d to 5f). The NeuralRemaster baseline (Fig. 5e) exhibits semantic drift by
hallucinating an incorrect speed-limit sign. This causes the VLM-based planner
to select an overly conservative and misaligned trajectory, with substantially
increased error (ADE = 3.47 m, FDE = 6.78 m). Our approach succeeds in both
cases and consistently yields the smallest trajectory deviations. Further details
regarding the driving intent outputs derived from the VLM are provided in the
Supplementary Materials.

5.5 Ablation Study

Under the setting of sim-to-real image translation, we ablate the structure-
preservation mechanism in ψ-PD by comparing global Fourier-domain phase
preservation with localized DT-CWPT-based frequency control. This setting
highlights an inherent tension in translation: pushing for higher realism often
comes at the cost of lower consistency (geometric alignment and semantic faith-
fulness to the source). Keeping the diffusion backbone and training protocol
fixed, we sweep hyperparameters that modulate this consistency–realism bal-
ance; each setting corresponds to an operating point. We visualize the enve-
lope of achievable operating points as a trade-off frontier in Fig. 6: methods

331

332

333

334

335

336

337

338

339

340

341

342

343

344

345

346

347

348

349

350

351

352

353

354

355

M
I
S
S
-
h
t
p
e
D

0.92

0.9

0.88

0.86

0.84

0.82

0.8

0.78

0.98

1

1.02 1.04 1.06

Appearance Score

ECCV 2026 Submission #6276

13

U
o
I
m

40

35

30

25

20

15

10

0.98

1

1.02 1.04 1.06

Appearance Score

Input

NeuralRemaster [52]

Ours

Fig. 6: Ablation study on consistency–realism trade-offs. Compared to global
phase preservation, the proposed ψ-PD consistently yields a superior Pareto frontier,
pushing toward the top-right region (higher realism and higher consistency).

356

357

358

359

360

361

362

363

364

365

366

367

368

369

370

371

372

373

374

375

376

377

378

379

with a curve that lies further toward the top-right corner admit better oper-
ating points across the sweep. We observe that global Fourier filtering yields
a steep frontier—realism can be improved, but structural/semantic consistency
degrades rapidly. This observation indicates that global Fourier filtering is intrin-
sically non-local, such that adjustments intended to enhance realism inevitably
propagate throughout the domain and thereby contaminate geometrically criti-
cal regions. By contrast, DT-CWPT shifts the frontier outward, producing more
realistic translations at better geometric and semantic consistency, demonstrat-
ing the advantage of localized wavelet-based frequency control for sim-to-real
translation.

5.6

Instance-Level Style Transfer

In contrast to previous methods, like NeuralRemaster, ψ-PD enables users to
selectively remaster chosen regions while leaving the rest of the image largely in-
tact. For instance-level style transfer, we first transform the remastering-strength
map into a normalized cutoff tensor F ∈ [0, 1]H×W , where higher remaster-
ing strength corresponds to a lower cutoff value. The resulting field F(x, y)
is then used to modulate packet-wise phase injection at each spatial location.
Specifically, higher cutoff values preserve the source phase over a broader fre-
quency range, thereby enhancing structural anchoring and yielding cleaner ob-
ject boundaries, whereas lower cutoff values facilitate increased injection of noise
phase, enabling more aggressive remastering effects. As we demonstrate in Fig. 7,
this per-pixel cutoff can successfully remaster specific image regions, like in-
dividual semantic instances, without requiring the computational overhead of
conditioned approaches.

356

357

358

359

360

361

362

363

364

365

366

367

368

369

370

371

372

373

374

375

376

377

378

379

14

ECCV 2026 Submission #6276

t
u
p
n
I

p
a
m
g
n
i
r
e
t
s
a
m
e
R

l
e
v
e
l
-
e
c
n
a
t
s
n
I

l
l
u
F

Fig. 7: Instance-level Style Transfer. Brighter areas in the remastering map in-
dicate stronger remastering, while darker regions are kept closer to the input. We
additionally include the full-remastering variant in the last row for comparison.

380

381

382

383

384

385

386

387

388

389

390

391

392

393

394

6 Conclusion

We introduced Wavelet Phase Diffusion (ψ-PD), a phase-preserving diffusion
framework that replaces global Fourier-domain constraints with localized con-
trol in the DT-CWPT domain. By operating on localized wavelet packets, ψ-PD
reduces global spectral coupling and avoids the non-local artifacts that arise from
Fourier bases, while remaining model-agnostic and requiring no auxiliary con-
ditioning modules or architectural changes. This spatial locality enables strong
appearance translation while maintaining geometric and semantic consistency,
and it naturally supports instance-level style transfer without boundary leakage.
Across sim-to-real image translation on SYNTHIA, ψ-PD improves realism
and semantic consistency while maintaining strong structural alignment. On
driving video translation in CARLA, it preserves critical cues for downstream
decision making: VLM-based drivability evaluation shows that prior baselines
degrade planning performance after translation, whereas ψ-PD reduces ADE
and FDE relative to the simulator input by 5.5% and 5.2%, respectively.

380

381

382

383

384

385

386

387

388

389

390

391

392

393

394

ECCV 2026 Submission #6276

15

References

1. Ali, A., Bai, J., Bala, M., Balaji, Y., Blakeman, A., Cai, T., Cao, J., Cao, T., Cha,
E., Chao, Y.W., et al.: World simulation with video foundation models for physical
ai. arXiv:2511.00062 (2025) 1, 4, 9, 10, 11, 12

2. bghira: photo-concept-bucket. Hugging Face Datasets

(2026), https : / /
huggingface.co/datasets/bghira/photo-concept-bucket, accessed: 2026-03-04
7

3. Black Forest Labs: FLUX. https : / / github . com / black - forest - labs / flux

(2024), accessed: 2026-03-04 3, 7

4. Blattmann, A., Dockhorn, T., Kulal, S., Mendelevitch, D., Kilian, M., Lorenz, D.,
Levi, Y., English, Z., Voleti, V., Letts, A., et al.: Stable video diffusion: Scaling
latent video diffusion models to large datasets. arXiv:2311.15127 (2023) 3

5. Brooks, T., Holynski, A., Efros, A.A.: Instructpix2pix: Learning to follow image

editing instructions. In: CVPR (2023) 4

6. Caesar, H., Bankiti, V., Lang, A.H., Vora, S., Liong, V.E., Xu, Q., Krishnan, A.,
Pan, Y., Baldan, G., Beijbom, O.: nuScenes: A multimodal dataset for autonomous
driving. In: CVPR (2020) 8

7. Chen, W.T., Fang, H.Y., Hsieh, C.L., Tsai, C.C., Chen, I., Ding, J.J., Kuo, S.Y.,
et al.: All snow removed: Single image desnowing algorithm using hierarchical dual-
tree complex wavelet representation and contradict channel loss. In: ICCV (2021)
4

8. Cotter, F.: Uses of Complex Wavelets in Deep Convolutional Neural Networks.
Ph.D. thesis, Apollo - University of Cambridge Repository (2019). https://doi.
org/10.17863/CAM.53748, https://www.repository.cam.ac.uk/handle/1810/
306661 7

9. Dosovitskiy, A., Ros, G., Codevilla, F., Lopez, A., Koltun, V.: CARLA: An open

urban driving simulator. In: CoRL (2017) 7

10. Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y.,
Lorenz, D., Sauer, A., Boesel, F., et al.: Scaling rectified flow transformers for
high-resolution image synthesis. In: ICML (2024) 3

11. Friedrich, P., Durrer, A., Wolleb, J., Cattin, P.C.: cwdm: conditional wavelet dif-
fusion models for cross-modality 3d medical image synthesis. arXiv:2411.17203
(2024) 4

12. Friedrich, P., Wolleb, J., Bieder, F., Durrer, A., Cattin, P.C.: Wdm: 3d wavelet
diffusion models for high-resolution medical image synthesis. In: MICCAI workshop
on deep generative models (2024) 4

13. Gottlieb, D., Shu, C.W.: On the gibbs phenomenon and its resolution. SIAM review

(1997) 5

14. Guth, F., Coste, S., De Bortoli, V., Mallat, S.: Wavelet score-based generative

modeling. NeurIPS (2022) 4

15. Harris, F.J.: On the use of windows for harmonic analysis with the discrete fourier

transform. Proceedings of the IEEE (1978) 5

16. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. NeurIPS

(2020) 3

17. Huang, Y., Huang, J., Liu, J., Yan, M., Dong, Y., Lv, J., Chen, C., Chen, S.:
Wavedm: Wavelet-based diffusion models for image restoration. IEEE TMM (2024)
4

18. Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q.,
Chanpaisit, N., et al.: Vbench: Comprehensive benchmark suite for video genera-
tive models. In: CVPR (2024) 8

395

396

397

398
399

400

401
402

403
404

405

406
407

408
409

410

411
412

413

414

415
416

417

418

419
420

421
422

423

424
425

426

427
428

429

430
431

432
433

434
435

436
437

438
439

440

441
442

443

444

395

396

397

398
399

400

401
402

403
404

405

406
407

408
409

410

411
412

413

414

415
416

417

418

419
420

421
422

423

424
425

426

427
428

429

430
431

432
433

434
435

436
437

438
439

440

441
442

443

444

16

ECCV 2026 Submission #6276

19. Kingma, D., Salimans, T., Poole, B., Ho, J.: Variational diffusion models. NeurIPS

(2021) 3

20. Kingsbury, N.: Complex wavelets for shift invariant analysis and filtering of signals.

Applied and computational harmonic analysis (2001) 5

21. Korkmaz, C., Tekalp, A.M., Dogan, Z.: Training generative image super-resolution
models by wavelet-domain losses enables better control of artifacts. In: CVPR
(2024) 4

22. Lipman, Y., Chen, R.T., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for

generative modeling. In: ICLR (2023) 3

23. Liu, X., Gong, C., Liu, Q.: Flow straight and fast: Learning to generate and transfer

data with rectified flow. In: ICLR (2023) 3

24. Mallat, S.G.: A theory for multiresolution signal decomposition: the wavelet rep-

resentation. IEEE TPAMI (1989) 4

25. Meng, C., He, Y., Song, Y., Song, J., Wu, J., Zhu, J.Y., Ermon, S.: SDEdit: Guided
image synthesis and editing with stochastic differential equations. In: ICLR (2022)
4, 9, 11

26. Moser, B.B., Frolov, S., Raue, F., Palacio, S., Dengel, A.: Waving goodbye to low-
res: A diffusion-wavelet approach for image super-resolution. In: 2024 International
Joint Conference on Neural Networks (IJCNN) (2024) 4

27. Mou, C., Wang, X., Xie, L., Wu, Y., Zhang, J., Qi, Z., Shan, Y.: T2i-adapter:
Learning adapters to dig out more controllable ability for text-to-image diffusion
models. In: AAAI (2024) 3

28. Oppenheim, A.V., Lim, J.S.: The importance of phase in signals. Proceedings of

the IEEE (1981) 2, 4

29. Peebles, W., Xie, S.: Scalable diffusion models with transformers. In: ICCV (2023)

3

30. Peng, B., Wang, J., Zhang, Y., Li, W., Yang, M.C., Jia, J.: Controlnext: Powerful
and efficient control for image and video generation. arXiv:2408.06070 (2024) 3
31. Phung, H., Dao, Q., Tran, A.: Wavelet diffusion models are fast and scalable image

generators. In: CVPR (2023) 4

32. Qiao, Z., Cao, Z., Liu, H.X.: nuCarla: A nuscenes-style bird’s-eye view perception

dataset for carla simulation. arXiv:2511.13744 (2025) 7

33. Qiao, Z., Li, H., Cao, Z., Liu, H.X.: LightEMMA: Lightweight end-to-end multi-

modal model for autonomous driving. arXiv:2505.00284 (2025) 8

34. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry,
G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models
from natural language supervision. In: ICML (2021) 8

35. Richter, S.R., AlHaija, H.A., Koltun, V.: Enhancing photorealism enhancement.

IEEE TPAMI (2021) 8

36. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution

image synthesis with latent diffusion models. In: CVPR (2022) 3

37. Ros, G., Sellart, L., Materzynska, J., Vazquez, D., Lopez, A.M.: The synthia
dataset: A large collection of synthetic images for semantic segmentation of ur-
ban scenes. In: CVPR (2016) 7

38. Selesnick, I.W., et al.: On the dual-tree complex wavelet packet and m-band trans-

forms. IEEE Transactions on Signal Processing (2008) 5

39. Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., Ganguli, S.: Deep unsuper-

vised learning using nonequilibrium thermodynamics. In: ICML (2015) 3

40. Song, Y., Ermon, S.: Generative modeling by estimating gradients of the data

distribution. NeurIPS (2019) 3

445

446

447

448

449

450

451

452

453

454

455

456

457

458

459

460

461

462

463

464

465

466

467

468

469

470

471

472

473

474

475

476

477

478

479

480

481

482

483

484

485

486

487

488

489

490

491

492

493

494

445

446

447

448

449

450

451

452

453

454

455

456

457

458

459

460

461

462

463

464

465

466

467

468

469

470

471

472

473

474

475

476

477

478

479

480

481

482

483

484

485

486

487

488

489

490

491

492

493

494

ECCV 2026 Submission #6276

17

41. Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S., Poole, B.: Score-
based generative modeling through stochastic differential equations. In: ICLR
(2021) 3

42. Taubman, D.S., Marcellin, M.W., Rabbani, M.: Jpeg2000: Image compression fun-

damentals, standards and practice. Journal of Electronic Imaging (2002) 4

43. Team Wan, Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.W., Chen, D., Yu, F., Zhao,
H., Yang, J., Zeng, J., Wang, J., Zhang, J., Zhou, J., Wang, J., Chen, J., Zhu, K.,
Zhao, K., Yan, K., Huang, L., Feng, M., Zhang, N., Li, P., Wu, P., Chu, R., Feng,
R., Zhang, S., Sun, S., Fang, T., Wang, T., Gui, T., Weng, T., Shen, T., Lin, W.,
Wang, W., Wang, W., Zhou, W., Wang, W., Shen, W., Yu, W., Shi, X., Huang,
X., Xu, X., Kou, Y., Lv, Y., Li, Y., Liu, Y., Wang, Y., Zhang, Y., Huang, Y., Li,
Y., Wu, Y., Liu, Y., Pan, Y., Zheng, Y., Hong, Y., Shi, Y., Feng, Y., Jiang, Z.,
Han, Z., Wu, Z.F., Liu, Z.: Wan: Open and advanced large-scale video generative
models. arXiv:2503.20314 (2025) 3, 7

44. Tencent Hunyuan Foundation Model Team: Hunyuanvideo 1.5 technical report.

arXiv:2511.18870 (2025) 3

45. Tencent Hunyuan3D Team: Hunyuan3d 2.5: Towards high-fidelity 3d assets gen-

eration with ultimate details. arXiv:2506.16504 (2025) 3

46. Voleti, V., Yao, C.H., Boss, M., Letts, A., Pankratz, D., Tochilkin, D., Laforte, C.,
Rombach, R., Jampani, V.: Sv3d: Novel multi-view synthesis and 3d generation
from a single image using latent video diffusion. In: ECCV (2024) 3

47. Wu, C., Li, J., Zhou, J., Lin, J., Gao, K., Yan, K., ming Yin, S., Bai, S., Xu, X.,
Chen, Y., Chen, Y., Tang, Z., Zhang, Z., Wang, Z., Yang, A., Yu, B., Cheng, C.,
Liu, D., Li, D., Zhang, H., Meng, H., Wei, H., Ni, J., Chen, K., Cao, K., Peng, L.,
Qu, L., Wu, M., Wang, P., Yu, S., Wen, T., Feng, W., Xu, X., Wang, Y., Zhang, Y.,
Zhu, Y., Wu, Y., Cai, Y., Liu, Z.: Qwen-image technical report. arXiv:2508.02324
(2025) 3, 4, 9, 10, 11

48. Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J.M., Luo, P.: Segformer:
Simple and efficient design for semantic segmentation with transformers. In:
NeurIPS (2021) 8

49. Xie, Y., Jampani, V., Zhong, L., Sun, D., Jiang, H.: Omnicontrol: Control any joint

at any time for human motion generation. In: ICLR (2024) 3

50. Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J., Zhao, H.: Depth anything

v2. NeurIPS (2024) 8

51. Yu, Y., Zhan, F., Lu, S., Pan, J., Ma, F., Xie, X., Miao, C.: Wavefill: A wavelet-

based generation network for image inpainting. In: ICCV (2021) 4

52. Zeng, Y., Ochoa, C., Zhou, M., Patel, V.M., Guizilini, V., McAllister, R.:
structure-aligned generation.

Neuralremaster: Phase-preserving diffusion for
arXiv:2512.05106 (2025) 1, 2, 4, 7, 8, 9, 10, 11, 12, 13

53. zengxianyu: open-sora-pexels-subset. Hugging Face Datasets (2026), https://
huggingface.co/datasets/zengxianyu/open- sora- pexels- subset, accessed:
2026-03-04 7

54. Zhang, B., Cheng, Y., Yang, J., Wang, C., Zhao, F., Tang, Y., Chen, D., Guo,
B.: Gaussiancube: Structuring gaussian splatting using optimal transport for 3d
generative modeling. NeurIPS (2024) 3

55. Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to text-to-image

diffusion models. In: ICCV (2023) 3, 9, 11

56. Zhao, S., Chen, D., Chen, Y.C., Bao, J., Hao, S., Yuan, L., Wong, K.Y.K.: Uni-
controlnet: All-in-one control to text-to-image diffusion models. NeurIPS (2023)
3

495

496

497

498

499

500

501

502

503

504

505

506

507

508

509

510

511

512

513

514

515

516

517

518

519

520

521

522

523

524

525

526

527

528

529

530

531

532

533

534

535

536

537

538

539

540

541

542

543

544

495

496

497

498

499

500

501

502

503

504

505

506

507

508

509

510

511

512

513

514

515

516

517

518

519

520

521

522

523

524

525

526

527

528

529

530

531

532

533

534

535

536

537

538

539

540

541

542

543

544

