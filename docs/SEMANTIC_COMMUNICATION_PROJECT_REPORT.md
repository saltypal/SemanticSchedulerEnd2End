# Semantic Communication and Semantic Aware Scheduling for Disaster Image Transmission

**Project Progress Report**  
**Satya Srinivas Paladugu, Nandini Padavala, and Jahnvi**  
**1 October 2026**

**Abstract**—We present an integrated investigation of learned image communication and task aware transmission scheduling. The communication study evaluates Swin Transformer based joint source channel coding, DeepJSCC, nonlinear transform source channel coding, semantic guided generative reconstruction, and communication codebooks. The scheduling study assigns transmission order to complete encoded image representations from five simulated user equipments using image content, task relevance, delivery history, urgency, impact, and evidence quality. On 24 Kodak images at 10 dB, native SwinJSCC at rate 96 achieved a mean peak signal to noise ratio of 29.326 dB and structural similarity of 0.858 using 73,728 complex channel uses per image. A JPEG quality 20 digital reference achieved 29.511 dB and 0.832 using 243,499 complex channel uses under the same Stage 1A transport setup. A separate Kodak comparison evaluates DeepJSCC and NTSCC with their communication codebooks. The static scheduling study analyzes 12 disaster images under a prescribed triage order. The dynamic study uses 260 LADI v2 images, delivered evidence memory, and a discrete event model; it compares classical policies, image classifiers, a feature classifier, a pairwise ranker, semantic rules, and a deep Q network. Together these experiments establish a measured progression from image reconstruction to content sensitive ordering of whole latent transmissions.

**Index Terms**—Semantic communication, joint source channel coding, Swin Transformer, DeepJSCC, NTSCC, SQ-GAN, vector quantization, image scheduling, reinforcement learning, disaster imagery.

## I. Introduction

Conventional image transmission typically compresses a source into bits, protects those bits with a channel code, modulates them, and reconstructs the image after reception. The quality of the result depends on both source compression and the wireless channel. Learned joint source channel coding (JSCC) instead trains an encoder and decoder around the expected channel so that the representation itself is adapted to transmission [1], [2]. For a disaster response task, a second question follows: if several images compete for limited transmission opportunity, which image should be sent next?

Our project addresses these questions as connected but distinct research problems. First, we study how image encoders and decoders trade reconstruction fidelity against communication resources. Second, we study how a scheduler can rank the resulting whole latent transmissions according to the information they contribute to a disaster assessment. The scheduling decision concerns **which semantic data item is transmitted next**. User equipment identity supplies arrival and fairness context; it does not define an image's intrinsic importance.

The investigation combines a main SwinJSCC communication path with comparative DeepJSCC, NTSCC, and SQ-GAN studies. It then links image derived parameters to static and dynamic scheduler evaluations. The dynamic environment updates its knowledge state only after successful delivery, making the usefulness of a later image depend on what the receiver already knows. Our purpose is to examine that coupling experimentally while retaining clear definitions of reconstruction quality, channel resources, semantic priority, and scheduling utility.

## II. Communication Model and Research Questions

Let an RGB image be **x**, with dimensions H × W × 3 and values in [0, 1]. A learned encoder maps x, and where applicable a target rate and channel state, into a latent representation **z**. The simulated communication channel produces a received representation **ẑ**. A learned decoder reconstructs **x̂** from ẑ. The image study measures how closely x̂ reproduces x at a stated resource cost. The scheduling study treats each complete encoded image as one nonpreemptive job. A policy chooses a currently eligible job from a queue, and the receiver's successful deliveries affect subsequent semantic novelty.

For a native continuous symbol JSCC path, the number of **complex channel uses** is the resource of interest. For a digital path, a bitstream becomes a number of channel uses only after its framing, channel coding, and modulation are specified. The compressed file length, stored latent tensor size, index count, and radio symbols are therefore reported separately. This distinction is especially important for codebook methods, which can either quantize a latent before another analog channel or transmit protected digital indices.

The principal communication questions are the fidelity available at a fixed signal to noise ratio, the channel uses required for that fidelity, the behavior across rate and noise settings, and the visual effect of quantization. The scheduling questions are whether task aware methods place useful evidence early, how delivery history changes the value of related images, and how a learned sequential policy compares with deterministic and supervised alternatives.

## III. Encoder and Decoder Investigations

### A. SwinJSCC: Hierarchical Image Encoding and Reconstruction

SwinJSCC is the principal communication model in our integrated experiment. It uses hierarchical Swin Transformer blocks to convert image patches into a channel facing latent [2]. A regular window confines attention to a manageable local region; a shifted window in the next block crosses earlier window boundaries. Patch merging increases the receptive field while reducing spatial resolution. The decoder uses corresponding learned transformations to recover image structure from the received latent. This arrangement allows local edges and textures to interact with larger scene structures without constructing global attention over every full resolution pixel.

The evaluated architecture begins with patches of size two and windows of size eight. Its four encoder stages use channel dimensions 128, 192, 256, and 320, with block depths 2, 2, 6, and 2. The decoder mirrors the hierarchical progression. The selected adaptive model supports a rate grid of 32, 64, 96, 128, and 192 and an evaluated signal to noise ratio grid of 1, 4, 7, 10, and 13 dB. Its rate and signal to noise adaptation conditions the learned transformation on the available operating point. The principal Stage 1A comparison uses the adaptive model at rate 96 and 10 dB.

In the native transmission path, active real latent values are paired into in phase and quadrature components, normalized for channel power, and perturbed by additive white Gaussian noise (AWGN). The decoder receives the noisy representation together with the model's rate mask. The receiver output is compared with the original image. SwinJSCC is trained for reconstruction using mean squared error, which encourages pixel fidelity; it is not directly optimized for the later disaster priority rubric.

Training used 800 DIV2K training images and 100 validation images, with Kodak24 reserved for evaluation. A fixed operating point base model was trained first, followed by the rate and signal to noise adaptive variant initialized from compatible weights. The best recorded validation mean squared errors were **0.00270769** for the base model at epoch 194 and **0.00300691** for the adaptive model at epoch 131. These values summarize different training phases; the Kodak measurements below assess the selected adaptive model. Training used two NVIDIA T4 GPUs with data parallel execution.

For a Kodak image with H × W = 393,216 pixels, rate 96 corresponds to **73,728 complex channel uses**. The channel bandwidth ratio is

<div class="equation">CBR = N<sub>complex</sub> / (3HW) = 73,728 / [3(393,216)] = 0.0625. <span>(1)</span></div>

The same operating rate on a 1024 by 1024 image produces **196,608 complex uses** because the spatial dimensions change. In the larger image experiment, the cached latent has shape 1 × 4096 × 320; the stored full precision tensor occupies 5,242,880 bytes. That storage measure describes an in memory representation, whereas the active complex symbols describe native communication use.

### B. DeepJSCC: Convolutional Joint Source Channel Coding

DeepJSCC provides a compact convolutional reference for learned analog image communication [1]. A convolutional encoder progressively transforms the RGB input into real latent values. These values are paired, power normalized, and carried over AWGN; a convolutional decoder reconstructs the image. The evaluated model uses 19 latent channels. For a 128 by 128 block, the encoder produces a 19 × 29 × 29 tensor, corresponding to 15,979 real values and 7,990 complex uses after pairing.

Kodak images were processed through 128 by 128 blocks and assembled after decoding. The available native DeepJSCC checkpoint was configured for a 200 dB channel condition, while the shared comparison applies a 10 dB channel. Its measured 10 dB result therefore records this specific checkpoint and channel pairing. The study also evaluates a 256 entry codebook followed by digital channel protection. DeepJSCC supplies a valuable architectural contrast with the much larger hierarchical Swin model: a compact CNN can be encoded efficiently, while robustness depends strongly on matching training and operating channel conditions.

### C. NTSCC: Nonlinear Transform and Adaptive Channel Allocation

Nonlinear transform source channel coding (NTSCC) combines learned analysis and synthesis transforms with distribution modeling and spatially adaptive resource allocation [3]. An analysis transform produces latent (y). A hyperprior estimates latent statistics, and a content dependent rate map determines how many channel dimensions different latent patches receive. A JSCC mapping transmits the selected representation, and a synthesis transform reconstructs the image. This mechanism permits detailed regions to receive more channel resources than comparatively simple regions.

Our comparison includes a native NTSCC path and a four part, rate conditioned product quantization (PQ4) variant with 256 representatives per subcodebook. The native rate reported below counts its recorded complex latent channel symbols. Per patch rate selection information is accounted for as a distinct representation, because it is not included in that native CBR field. The codebook variant converts part of the communication problem into protected digital indices and therefore has a different resource profile from native NTSCC.

### D. SQ-GAN: Semantic Guided Generative Reconstruction

The SQ-GAN investigation broadens the project beyond pixel reconstruction alone [4]. Its architecture uses image information and semantic segmentation information as related sources. A semantic conditioned adaptive mask identifies informative regions, quantization compresses the selected representation, and a generative decoder reconstructs the image. SQ-GAN acts primarily at the source coding stage; its reported compression ratios should therefore be interpreted through its own transmission assumptions. Conceptually, a segmentation map can guide which structures deserve attention before communication resources are assigned.

The team presentation records nine SQ-GAN operating configurations formed by three image settings and three semantic settings. These are reported in Section V as configuration measurements from that experiment. They provide a rate, fidelity, structure, and perceptual quality view complementary to the Kodak SwinJSCC, DeepJSCC, and NTSCC benchmark. The visual example in the presentation also compares an original image with a corresponding reconstruction.

### E. Communication Codebooks

A communication codebook replaces a continuous latent vector with the index of a representative vector. If a codebook has (K=256) entries, each selected index has an eight bit representation before protocol framing and error correction. The encoder chooses the nearest representative; the decoder looks up that representative and continues reconstruction. Quantization reduces the description of the latent but adds distortion whenever the representative differs from the original vector.

We investigated **SwinJSCC with a 256 entry vector codebook**, **DeepJSCC with a 256 entry codebook**, and **NTSCC with a four part 256 entry product codebook**. The Swin work includes two distinct communication paths. One packs codebook indices into digital frames, applies a length 1024 information, length 2048 coded bit LDPC construction, and transmits with QPSK. The other quantizes and restores a dense continuous latent before the standard analog channel. The first is a protected digital index transmission; the second still consumes analog latent channel symbols. Keeping these experiments separate is necessary to interpret both quality and rate correctly.

## IV. Experimental Measures and Image Data

The main image reconstruction benchmark uses the 24 Kodak test images at native 768 by 512 or 512 by 768 resolution. The same source images are used across each reported Kodak operating point. The consolidated comparison contains 360 observations, representing 15 operating points measured on 24 images each. A separate Stage 1A investigation sweeps Swin rate, channel signal to noise ratio, JPEG quality, and the Swin digital codebook path. Calibration images for the codebooks are kept separate from Kodak test images; the NTSCC PQ4 calibration used CIFAR imagery.

Peak signal to noise ratio (PSNR) measures pixel error in decibels, with a larger value indicating lower mean squared error. Structural similarity (SSIM) compares local luminance, contrast, and structure, with one representing an exact match. LPIPS estimates learned perceptual distance, where smaller is preferable [5]. CBR measures complex channel uses divided by (3HW). Digital frame success means that the protected frame was received and passed the experiment's validity check. A failed frame is counted as a transport outcome, with no reconstruction quality imputed. Run time values are useful implementation measurements, while quality and channel resource comparisons carry the principal scientific conclusions.

## V. Encoder and Decoder Results

### A. SwinJSCC and Digital References at 10 dB

**TABLE I. Stage 1A Kodak24 Comparison at 10 dB**

| Method | Complex uses per image | Mean PSNR | Mean SSIM | Valid frames |
|:---|---:|---:|---:|:---|
| Native SwinJSCC, rate 96 | **73,728** | **29.326 dB** | **0.858** | 24 reconstructions |
| JPEG Q20 with LDPC and QPSK | 243,499 | 29.511 dB | 0.832 | 24 of 24 |
| SwinJSCC K256 indices with LDPC and QPSK | 13,312 | 20.729 dB | 0.549 | 24 of 24 |
| Raw RGB digital reference | 9,438,208 | Exact copy reference | Exact copy reference | Resource calculation |

At this operating point, native SwinJSCC used approximately **3.30 times fewer** channel symbols than the JPEG Q20 digital chain while achieving a mean PSNR within 0.185 dB and a mean SSIM higher by 0.026. Quantized Swin indices with digital transmission reduced channel use by approximately **5.54 times** relative to native Swin, accompanied by an 8.60 dB PSNR reduction. Thus, the codebook result quantifies a clear compression versus reconstruction tradeoff rather than suggesting that index size alone describes fidelity.

![Fig. 1. Kodak24 reconstruction fidelity against complex channel uses for the 10 dB Stage 1A comparison.](../results/Stage1A/four_way_comparison/02_quality_vs_channel_uses_10db.png)

**Fig. 1.** Reconstruction quality and communication resource use at 10 dB. The raw RGB entry is a calculated resource reference; the plotted codec quality measurements correspond to completed reconstructions.

### B. Noise Robustness and Rate Adaptation

At rate 96, mean SwinJSCC PSNR increases as the simulated channel becomes less noisy. The JPEG Q20 and quantized digital paths have different behavior: in the recorded Stage 1A experiment, their protected frames did not validate at 1 dB, whereas all 24 validated from 4 to 13 dB. Native JSCC provided a reconstruction at every evaluated signal to noise ratio.

**TABLE II. Native SwinJSCC at Rate 96 Across the Tested Channel Conditions**

| Signal to noise ratio | 1 dB | 4 dB | 7 dB | 10 dB | 13 dB |
|:---|---:|---:|---:|---:|---:|
| Mean PSNR | 27.853 dB | 28.622 dB | 29.074 dB | 29.326 dB | 29.458 dB |
| JPEG Q20 digital frames valid | 0 of 24 | 24 of 24 | 24 of 24 | 24 of 24 | 24 of 24 |
| Swin K256 digital frames valid | 0 of 24 | 24 of 24 | 24 of 24 | 24 of 24 | 24 of 24 |

![Fig. 2. Native SwinJSCC signal to noise response and the valid digital frame counts in Stage 1A.](../results/Stage1A/four_way_comparison/03_snr_robustness_and_frame_success.png)

**Fig. 2.** Reconstruction response across signal to noise ratio and digital frame validity under the stated channel coding configuration.

The 10 dB rate sweep shows substantial fidelity improvement from rate 32 to rate 96. The mean PSNR then remains near 29.33 dB at rates 128 and 192 for this trained model and test set.

**TABLE III. SwinJSCC Rate Sweep on Kodak24 at 10 dB**

| Rate setting | Complex uses | CBR | Mean PSNR with image standard deviation | Mean SSIM |
|---:|---:|---:|---:|---:|
| 32 | 24,576 | 0.02083 | 25.751 ± 2.448 dB | 0.728 |
| 64 | 49,152 | 0.04167 | 28.573 ± 2.555 dB | 0.836 |
| 96 | 73,728 | 0.06250 | **29.326 ± 2.610 dB** | **0.858** |
| 128 | 98,304 | 0.08333 | 29.326 ± 2.607 dB | 0.858 |
| 192 | 147,456 | 0.12500 | 29.325 ± 2.608 dB | 0.858 |

The deviations in Table III describe variation between the 24 images, rather than independent repetitions of channel noise. Rate 96 is a strong measured operating point in this sweep because a higher setting adds symbols without a material mean quality gain under the tested conditions.

### C. Comparative Kodak24 Codec Study

Table IV consolidates the image quality and resource figures recorded for the main codec families at 10 dB. A blank communication entry means that this operating point recorded a file or representation size without a specified channel mapping. In particular, the JPEG entry is file compression alone, unlike Table I's JPEG plus LDPC and QPSK chain. The Swin codebook entry in Table IV uses analog transmission after quantization, unlike Table I's digital index chain.

**TABLE IV. Consolidated Kodak24 Results at 10 dB, Means Over 24 Images**

| Codec and transport | Complex uses | Payload bits | PSNR | SSIM | LPIPS |
|:---|---:|---:|---:|---:|---:|
| Raw RGB24 reference | Not assigned | 9,437,184 | Exact copy | 1.000 | 0.000 |
| JPEG Q20 file | Not assigned | 243,016 | 29.511 | 0.832 | 0.172 |
| Native DeepJSCC | 191,748 | Not assigned | 16.435 | 0.237 | 0.958 |
| DeepJSCC K256 digital | 161,792 | 161,632 | 20.366 | 0.435 | 0.651 |
| Native NTSCC | **28,283** | Not assigned | **29.791** | 0.832 | 0.222 |
| NTSCC PQ4 K256 digital | 120,192 | 119,564 | 27.186 | 0.798 | 0.250 |
| Native SwinJSCC, rate 96 | 73,728 | Not assigned | 29.328 | **0.858** | **0.190** |
| Swin K256 followed by analog channel | 73,728 | 12,768 index diagnostic | 20.716 | 0.546 | 0.556 |

The native NTSCC point has the lowest recorded complex channel use among the native learned codecs and reaches 29.791 dB PSNR. Its adaptive rate map is a separate piece of information, so that channel use figure specifically describes its measured latent symbols. Native SwinJSCC reaches the highest mean SSIM among the listed learned native paths. DeepJSCC's 10 dB measurements reflect the available checkpoint trained for a substantially different, 200 dB condition. These distinctions preserve the value of the comparison while identifying the operating conditions represented by each number.

The consolidated run also recorded approximate mean encoder and decoder times of **82.5 and 174.8 ms** for native DeepJSCC, **6,692 and 3,498 ms** for native NTSCC, and **6,899 and 5,744 ms** for native SwinJSCC. The codebook variants recorded **235 and 2,660 ms** for DeepJSCC digital, **11,345 and 5,973 ms** for NTSCC PQ4 digital, and **4,146 and 3,630 ms** for Swin quantization followed by analog transmission. These are execution measurements from the recorded evaluation setting. They convey implementation cost alongside quality, without equating different codec architectures or timing methods solely through one elapsed time.

The recorded model sizes also show the architectural range in this study: approximately **167,212 parameters** for the compact DeepJSCC model, **31.36 million** for NTSCC, and **33.07 million** for SwinJSCC. These counts help explain why the communication methods differ substantially in inference complexity as well as reconstruction behavior.

Table V adds the complete 24 image average computational profile for each learned codec and its codebook variant in the recorded CPU evaluation. Total time includes the recorded per image encoding and decoding work. Peak RSS Δ is the increase in peak CPU resident memory during the measured process, rather than a measurement of GPU memory.

**TABLE V. Kodak24 Learned Codec Computation and Memory at 10 dB**

| Method | CBR | Total time per image | Model size | Peak RSS Δ | Status |
|:---|---:|---:|---:|---:|:---|
| DeepJSCC | 0.16255 | 257 ms | 0.64 MB | 1.17 MB | Measured |
| DeepJSCC codebook | 0.13715 | 2.89 s | 0.66 MB | 35.59 MB | Measured |
| NTSCC | **0.02398** | 10.19 s | 126.50 MB | 1,570.82 MB | Measured |
| NTSCC codebook | 0.10189 | 17.32 s | 131.64 MB | 1,569.36 MB | Measured |
| SwinJSCC | 0.06250 | 12.64 s | 312.92 MB | 21.17 MB | Measured |
| SwinJSCC codebook | 0.06250 | 7.78 s | 312.92 MB | 21.07 MB | Measured |

JPEG quality was also swept as a conventional source compression reference. These values describe compressed file size and reconstruction, not protected radio symbols.

**TABLE VI. JPEG File Quality Sweep on Kodak24**

| JPEG quality | Mean payload bits | Mean PSNR | Mean SSIM | Mean LPIPS |
|---:|---:|---:|---:|---:|
| 20 | 243,016 | 29.511 dB | 0.832 | 0.172 |
| 30 | 308,384 | 30.910 dB | 0.869 | 0.116 |
| 40 | 363,772 | 31.888 dB | 0.890 | 0.088 |
| 50 | 417,354 | 32.684 dB | 0.905 | 0.070 |
| 60 | 477,045 | 33.465 dB | 0.917 | 0.056 |
| 70 | 569,115 | 34.545 dB | 0.931 | 0.041 |
| 80 | 725,988 | 36.134 dB | 0.948 | 0.025 |
| 90 | 1,100,620 | 39.157 dB | 0.968 | 0.008 |

### D. SQ-GAN Configuration Study

The semantic guided generative investigation measured three image settings against three semantic settings. Table VII reports the values displayed in the team presentation. They are operating configuration measurements; no sample count is assigned beyond the information recorded with the presentation.

**TABLE VII. Reported SQ-GAN Operating Configurations**

| Image setting | Semantic setting | CBR | PSNR | SSIM | MS SSIM | LPIPS | Time s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.4 | 0.4 | 0.038940 | 25.270 | 0.7633 | 0.8621 | 0.3018 | 11.640 |
| 0.4 | 0.6 | 0.046799 | 25.649 | 0.7651 | 0.8646 | 0.3021 | 12.087 |
| 0.4 | 1.0 | 0.062439 | 25.673 | 0.7650 | 0.8647 | 0.3017 | 12.028 |
| 0.7 | 0.4 | 0.050690 | 25.571 | 0.7732 | 0.8764 | 0.2923 | 12.773 |
| 0.7 | 0.6 | 0.065848 | 26.036 | 0.7753 | 0.8792 | 0.2927 | 14.898 |
| 0.7 | 1.0 | 0.074188 | 26.058 | 0.7754 | 0.8795 | 0.2920 | 12.034 |
| 1.0 | 0.4 | 0.062439 | 25.757 | 0.7792 | 0.8821 | 0.2876 | 10.368 |
| 1.0 | 0.6 | 0.070297 | 26.278 | 0.7811 | 0.8848 | **0.2864** | 10.854 |
| 1.0 | 1.0 | 0.085938 | **26.316** | **0.7812** | **0.8851** | 0.2866 | 11.384 |

Increasing the image setting generally improves pixel and structural quality while increasing CBR. The 1.0 image and 1.0 semantic setting gives the highest displayed PSNR, SSIM, and multiscale SSIM; the 1.0 and 0.6 setting gives the lowest displayed LPIPS. These complementary measures illustrate why semantic and perceptual reconstruction studies benefit from more than a single fidelity number.

### E. Additional Transformer Communication Comparison

An exploratory WITT comparison was included in the presentation as a separate transformer based study [6]. At the reported Kodak24 10 dB point, WITT without adaptation at rate 96 used **73,728 complex symbols** and reached **22.84 dB PSNR**; the rate 96 WITT configuration used the same symbol count and reached **22.47 dB**. A WITT K32 codebook configuration was reported at **73,731 symbols** and **14.61 dB**. The companion JPEG Q20 point was reported at **88,306 symbols** and **29.14 dB** under that exploratory accounting. This experiment has its own transport and measurement configuration and is therefore presented separately from the Stage 1A and consolidated Kodak comparisons.

### F. Latent Production and Selected Receiver Reconstruction

The communication and scheduling parts were connected through a measured SwinJSCC payload pilot. Sixty curated disaster images were encoded at 1024 by 1024 with the selected rate 96 checkpoint. Every saved record specified a 1 × 4096 × 320 latent and **196,608 modeled complex channel uses**. The mean observed encoding latency was approximately **8.15 s per image** in that pilot. This was a one time image preparation measurement; scheduling episodes can reuse the cached latent and semantic record.

An actual selected image replayed through the checkpoint, AWGN channel at 10 dB, and decoder reached **27.442 dB PSNR** and **0.8533 SSIM**, with **196,608 complex uses** and a measured replay latency of **6,555.8 ms**. This result demonstrates the end to end reconstruction path for a selected whole latent. The larger 260 image scheduling study below used simulated fixed service units for its dynamic comparisons, so its schedule quality and this checkpoint replay quantify different aspects of the system.

<div class="paired-figure"><img src="../results/scheduler-codec-smoke/replay/codec-smoke_input.png" alt="Original disaster assessment image"><img src="../results/scheduler-codec-smoke/replay/codec-smoke_reconstruction.png" alt="SwinJSCC reconstruction at 10 dB"></div>

**Fig. 3.** A selected 1024 by 1024 input image and its SwinJSCC reconstruction after the simulated 10 dB channel. The corresponding measured quality is 27.442 dB PSNR and 0.8533 SSIM.

## VI. Semantic Parameters and Scheduling Methodology

### A. Disaster Task and Seven Parameters

The demonstration task is post disaster access and damage triage. We assign task specific priorities to disaster image categories while retaining the LADI v2 dataset's native multilabel annotations [7]. The 260 image catalogue spans 12 selection strata and is divided into **156 training, 52 validation, and 52 test images**. A stratum organizes the experiment; an image can still carry multiple original LADI labels. The priority order is a declared triage rubric for this task, not a universal ordering for all disaster responses.

Seven scheduling parameters connect the images and receiver context to the decision model.

**TABLE VIII. Semantic Scheduling Parameters**

| Parameter | Meaning | Experimental source |
|:---|:---|:---|
| P1 | Scene importance estimate | Image classifier output |
| P2 | Context relevance to the triage task | CLIP image and task text cosine similarity [8] |
| P3 | Novelty against successfully delivered evidence | Image embedding compared with receiver history |
| P4 | Scene prediction confidence | Classifier confidence |
| P5 | Task impact | Task priority rubric |
| P6 | Urgency | Task priority rubric |
| P7 | Evidence sufficiency | Experiment annotation or provided evidence assessment |

P1, P2, P3, and P4 are image or receiver context derived. P5, P6, and P7 use task supplied information in the present experiment. This design makes the full feature study an assisted semantic scheduling condition and permits image only comparisons through the image classifier policies. The 24 byte logical scheduling header carries P2, P3, P5, P6, and P7 together with a presence mask, history version, and integrity check. P1 and P4 are available in the simulator's scheduling information. The header is modeled as a protected control sidecar to the Swin latent payload; it is not counted as an extra analog latent symbol.

The receiver maintains a task scoped knowledge state. If eᵢ is a cached image embedding and Dₜ is the set of embeddings for successfully delivered images before decision t, decision time novelty is computed from the maximum similarity to delivered evidence:

<div class="equation">P3(i, t) = [1 − max<sub>eⱼ ∈ Dₜ</sub> cos(eᵢ, eⱼ)] / 2. <span>(2)</span></div>

At empty history, the policy uses a cold start novelty value with an availability indicator. Once a related image arrives successfully, the novelty of another related queued image can decrease. A failed transmission, an image expiring in the queue, or an overflow drop does not add evidence to the receiver's history. This is the mechanism that makes the dynamic task more than a fixed priority sorting problem.

### B. Five User Equipments and Dynamic Episodes

Five simulated user equipments create distinct temporal traffic patterns: a periodic drone, a burst camera, a responder device with variable gaps between arrivals, a sparse survey device, and a relay camera with short bursts. Their image assignments are mixed across priority strata, so a device name does not imply a criticality level. At each event time, the simulator completes any active service, records its outcome, updates delivered evidence after success, admits new arrivals, expires waiting jobs when applicable, and presents the currently eligible queue to the policy. Service is nonpreemptive: the selected complete latent job is served before the next choice.

The standard learned policy observation admits at most 25 candidate positions. The main DQN receives semantic values, availability masks, age, deadline slack, modeled service cost, and user equipment context. In the evaluated configuration, each candidate contributes 22 values and global queue context contributes nine values, giving a **559 dimensional input** and 25 candidate action outputs. A mask removes padded and ineligible actions. Each decision chooses **one next job**, rather than emitting an entire order at once.

### C. Static Ordering Study

The separate static notebook evaluates **12 distinct LADI v2 test images** in one fixed input order. All candidates are available for an ordering comparison, and the expected order follows a declared task triage rule. This isolates the question of how traditional orderings align with semantic priority before receiver history and live arrivals are introduced. A match is an item placed at its exact expected position. Kendall agreement and normalized discounted cumulative gain (nDCG) examine broader ranking quality.

**TABLE IX. Static Ordering Agreement on the Same 12 Images**

| Scheduler | Exact position matches | Mismatches | Kendall agreement | nDCG |
|:---|---:|---:|---:|---:|
| Declared semantic priority reference | 12 | 0 | 1.000 | 1.000 |
| Earliest deadline first | 5 | 7 | 0.818 | 0.816 |
| First in first out | 4 | 8 | 0.364 | 0.554 |
| Weighted shortest processing time | 1 | 11 | 0.152 | 0.493 |
| Round robin completion | 0 | 12 | −0.273 | 0.402 |
| Shortest processing time | 0 | 12 | −0.364 | 0.414 |

The semantic priority reference uses the same declared ordering rule as the expected order, so its exact match is an experiment check. The other policies pursue deadline, arrival, fairness, or transfer time objectives. Earliest deadline first was the closest conventional ordering on these 12 fixed images. This static study demonstrates metric behavior on one curated set and motivates the more realistic dynamic arrival experiment.

### D. Dynamic Policies and Learning Objectives

The dynamic comparison includes first in first out, shortest job first, earliest deadline first, user equipment round robin, aging, fixed class priority, and priority plus deadline scheduling. It also includes a transparent semantic greedy score, semantic weighted and lexicographic rules, a LADI image classifier policy, a CLIP based image priority classifier with and without a deadline tie rule, a feature ordinal classifier, a pairwise linear ranker, and a small deep Q network (DQN). Fixed class priority and the one step marginal utility policy are reference policies with access to rubric information unavailable to ordinary image only policies.

The image classifiers infer a priority category from image derived outputs and schedule according to the predicted importance. The feature ordinal classifier receives the normalized semantic parameter vector and availability indicators, predicts priority classes, and orders jobs by expected class. Its supervised objective is multinomial cross entropy with regularization and class balancing. Because P5 and P6 already encode aspects of the experiment's rubric, this model is best read as an **assisted feature condition**, not as an image only classifier. The pairwise ranker learns from pairs with different declared priorities; a logistic preference loss fits which observable feature vector should precede the other.

Semantic greedy directly scores relevance, novelty, impact, urgency, sufficiency, and modeled cost using a transparent multiplicative rule. It needs no training. Its design is useful as an interpretable comparator: a later image can lose novelty when a similar image has already been delivered. The semantic weighted and lexicographic variants supply two further deterministic ways to combine those same considerations.

The DQN estimates an action value for every eligible candidate. Its evaluated network has **559 inputs, hidden layers of 64 and 32 rectified units, and 25 Q values**. The agent observes a state, selects one job, receives a simulator reward after service, and learns from the transition to the next state. For a nonterminal transition, the target is the observed reward plus the discounted maximum next state Q value from the target network. Training minimizes the Huber temporal difference between this target and the selected action's estimated Q value [9]. The saved experiment trains three seeds with 300 episodes per seed and selects checkpoints using validation semantic utility. Their best recorded validation scores were **0.110220**, **0.114680**, and **0.112316** for seeds 42, 43, and 44, respectively.

The reward is generated by the scheduling environment. Successful timely delivery gains value from the image's task priority and evidence contribution. A first successfully delivered item from an evidence group contributes more than a redundant related item; better evidence quality can still add value. Waiting delay reduces the value of urgent items. Modeled service cost, newly expired jobs, and failed transport add penalties. Thus the learned Q values represent **expected future scheduling reward**, not a classifier's probability that an image belongs to a class. Position mismatches are a separate evaluation measure. This distinction explains why a policy can achieve strong utility while disagreeing with an exact class order.

## VII. Dynamic Scheduling Results

### A. Held Out Priority Prediction

On the 52 image test partition, the classifier policies achieved the following priority prediction results. Macro F1 averages F1 across the priority categories; balanced accuracy averages recall across them.

**TABLE X. Image Disjoint Test Priority Classification, 52 Images**

| Priority predictor | Macro F1 | Balanced accuracy | Information used |
|:---|---:|---:|:---|
| CLIP image priority classifier | 0.540 | 0.536 | Image embedding |
| LADI image classifier | 0.581 | 0.685 | Image derived class prediction |
| Feature ordinal classifier | **0.857** | **0.880** | Semantic vector including rubric supplied P5 and P6 |

These values measure priority prediction, not schedule quality. A scheduler must additionally respond to arrival time, deadlines, service cost, competing images, and delivered history.

### B. Fixed Streams From Unseen Images

Thirty fixed test streams were generated from the 52 held out images. Every policy faced the same offered scenarios. Semantic utility area under the curve (AUC) integrates accumulated task value over time, normalized by the common scenario window. The metric rewards delivering useful, sufficiently clear, and nonredundant evidence early. Priority pair match asks whether the policy put the higher rubric priority first when two different priority classes were comparable. Critical on time is the fraction of critical items served within their modeled deadline. These metrics answer different questions and should be read together.

**TABLE XI. Dynamic Comparison on 30 Fixed Image Disjoint Streams**

| Policy | Semantic utility AUC | Priority pair match | Critical on time |
|:---|---:|---:|---:|
| One step marginal utility reference | **0.4801** | Not reported | 76.8% |
| CLIP image priority plus deadline | 0.4586 | 70.6% | 79.0% |
| CLIP image priority | 0.4566 | 75.4% | 82.2% |
| DQN | **0.4453** | 55.3% | 71.5% |
| Fixed priority plus deadline | 0.4344 | 100.0% | 100.0% |
| LADI image classifier priority | 0.4335 | 83.5% | 82.0% |
| Fixed class priority | 0.4325 | 100.0% | 100.0% |
| Semantic greedy | 0.4314 | 96.8% | 97.8% |
| Semantic weighted | 0.4296 | 96.4% | 96.8% |
| Feature ordinal classifier | 0.4286 | **98.7%** | 99.0% |
| Pairwise linear ranker | 0.4285 | 97.9% | 97.8% |
| Semantic lexicographic | 0.4281 | 98.3% | 98.2% |
| User equipment round robin | 0.4125 | 49.9% | 70.5% |
| Earliest deadline first | 0.4096 | 49.3% | 60.3% |
| Aging | 0.4051 | Not separately reported | 72.8% |
| First in first out | 0.4051 | 49.4% | 72.8% |
| Shortest job first | 0.4051 | 49.4% | 72.8% |

The DQN's 0.4453 utility AUC exceeds the semantic greedy and feature classifier values in this stream study, while the CLIP image policies and one step reference produce higher AUC. The feature classifier reaches 98.7% **pairwise priority agreement** because its inputs include rubric supplied impact and urgency. That 98.7% is neither a percentage of DQN actions nor a measured gain in radio throughput. The DQN favors a different future reward tradeoff and therefore has lower exact priority pair agreement.

### C. Repeated Evidence and Bursty Queue Pressure

To exercise receiver history, another study repeatedly offered the 52 held out images through five user equipments. Each episode contained 260 offered jobs; five fixed episodes contributed 1,300 offered arrivals. The receiver history changed following successful deliveries in all five episodes. A 25 place queue, burst arrivals, deadlines, and nonpreemptive service forced policies to act under contention. This scenario uses the **same pretrained DQN**, with no result dependent retraining.

In Table XII, a **whole decision match** compares each selected job with the scenario's declared class and urgency based decision target. It is different from Table XI's pairwise priority agreement. The utility AUC still measures the accumulated value of delivered evidence. The one step marginal utility policy is a useful privileged comparator, not a proven optimum over every future decision.

**TABLE XII. Repeated Evidence and Bursty Overload, Five Episodes**

| Policy | Semantic utility AUC | Whole decision match | Critical on time |
|:---|---:|---:|---:|
| One step marginal utility reference | **0.12185** | 61.7% | 90.2% |
| DQN | **0.11592** | 31.7% | 60.8% |
| Semantic greedy | 0.11581 | 83.7% | 94.0% |
| Pairwise linear ranker | 0.11514 | 86.9% | 95.4% |
| Semantic lexicographic | 0.11510 | **94.6%** | 96.2% |
| Semantic weighted | 0.11488 | 82.8% | 93.2% |
| Feature ordinal classifier | 0.11469 | 90.4% | **97.2%** |
| CLIP image classifier | 0.11407 | 41.3% | 73.8% |
| LADI image classifier | 0.11392 | 59.5% | 76.6% |
| CLIP image priority plus deadline | 0.11388 | 38.7% | 72.4% |
| Fixed class priority plus deadline | 0.11371 | 79.6% | 99.8% |
| Fixed class priority | 0.11355 | 77.7% | 99.8% |
| User equipment round robin | 0.10082 | 26.7% | 52.2% |
| Earliest deadline first | 0.09969 | 26.6% | 31.6% |
| First in first out | 0.09830 | 26.5% | 52.2% |
| Shortest job first | 0.09830 | 26.5% | 52.2% |
| Aging | 0.09830 | 26.5% | 52.2% |

Under this particular repeated evidence workload, DQN had the highest utility AUC among the evaluated ordinary policies, at **0.11592**, compared with **0.11581** for semantic greedy. The difference is small, approximately **0.00011** AUC over five episodes. Semantic lexicographic and the feature classifier achieved much stronger declared decision agreement; the feature classifier also delivered the highest fraction of critical jobs before deadline among ordinary policies. This is a useful result rather than a contradiction: reward based utility, exact class decision match, and deadline completion emphasize different scheduling behavior.

![Fig. 4. Comparison of semantic utility, decision agreement, and critical delivery in the repeated evidence workload.](../results/scheduler-policy-local/combined_dynamic_overload_5ep/combined_dynamic_comparison.png)

**Fig. 4.** Policy comparison in the repeated evidence and bursty arrival experiment. The DQN's measured utility advantage over semantic greedy is narrow; semantic rule and classifier policies are stronger on declared decision agreement.

## VIII. Discussion and Next Research Stage

The communication and scheduling results reveal complementary routes to semantic efficiency. SwinJSCC uses a learned channel facing representation and achieves attractive reconstruction quality per complex symbol at its measured Kodak operating point. NTSCC demonstrates a different mechanism, adapting communication rate spatially. SQ-GAN adds a semantic guided generative perspective. Codebook experiments expose how aggressively reducing the latent description changes both fidelity and transport design. These findings motivate quality evaluation at the level of the actual downstream disaster task, in addition to PSNR, SSIM, and LPIPS.

The scheduling studies show why priority and utility must be distinguished. A static priority reference can exactly reproduce a declared target order. In a stream, however, jobs arrive at different times, deadlines and queue pressure change, and a successfully delivered view can make a related view redundant. The DQN learns from these delayed effects through temporal difference training; classifiers learn priority categories from supervised examples; semantic rules evaluate current features directly. Table XII demonstrates a workload where these approaches rank differently under utility, class decision match, and critical deadline service. No single measure should substitute for the others.

The next experimental stage is to evaluate downstream damage or access assessment on images **after** reconstruction, so that channel induced visual loss can be related to task performance. Codec comparison would benefit from matched channel training conditions, complete side information accounting, and repeated noise realizations. Scheduler comparison would benefit from a larger family of held out arrival traces, uncertainty intervals, measured service cost for the full image catalogue, and training objectives tuned explicitly to the final deployment criterion. In a network implementation, the compact digital scheduling information would be conveyed through a robust control mechanism while the SwinJSCC latent uses its specified channel representation. These steps extend the present scientific foundation toward system level semantic communication.

## IX. Conclusion

We have advanced a complete research trajectory from learned image representation to dynamic semantic scheduling. The encoder and decoder study establishes measured rate, channel robustness, reconstruction quality, and codebook tradeoffs across SwinJSCC, DeepJSCC, NTSCC, SQ-GAN, and an exploratory WITT comparison. The scheduling study adds seven task aware parameters, receiver delivery history, five user equipment streams, static ranking references, supervised priority predictors, and a DQN that selects the next whole latent job. The resulting evaluations identify where semantic information improves ordering and where different definitions of scheduling success lead to different policy rankings. Our complementary technical emphases span SwinJSCC and dynamic semantic scheduling, DeepJSCC and NTSCC with static scheduling analysis, and SQ-GAN, respectively.

## References

[1] E. Bourtsoulatze, D. B. Kurka, and D. Gündüz, “Deep Joint Source-Channel Coding for Wireless Image Transmission,” *IEEE Transactions on Cognitive Communications and Networking*, 2019. [Online]. Available: https://arxiv.org/abs/1809.01733

[2] K. Yang *et al.*, “SwinJSCC: Taming Swin Transformer for Deep Joint Source-Channel Coding,” 2023. [Online]. Available: https://arxiv.org/abs/2308.09361

[3] J. Dai *et al.*, “Nonlinear Transform Source-Channel Coding for Semantic Communications,” 2021. [Online]. Available: https://arxiv.org/abs/2112.10961

[4] F. Pezone, S. Barbarossa, and G. Caire, “SQ-GAN: Semantic Image Communications Using Masked Vector Quantization,” 2025. [Online]. Available: https://arxiv.org/abs/2502.09520

[5] R. Zhang *et al.*, “The Unreasonable Effectiveness of Deep Features as a Perceptual Metric,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition*, 2018. [Online]. Available: https://arxiv.org/abs/1801.03924

[6] K. Yang *et al.*, “WITT: A Wireless Image Transmission Transformer for Semantic Communications,” 2022. [Online]. Available: https://arxiv.org/abs/2211.00937

[7] S. Scheele, K. Picchione, and J. Liu, “LADI v2: Multi-label Dataset and Classifiers for Low-Altitude Disaster Imagery,” 2024. [Online]. Available: https://arxiv.org/abs/2406.02780

[8] A. Radford *et al.*, “Learning Transferable Visual Models From Natural Language Supervision,” in *Proc. International Conference on Machine Learning*, 2021. [Online]. Available: https://arxiv.org/abs/2103.00020

[9] V. Mnih *et al.*, “Human-level control through deep reinforcement learning,” *Nature*, vol. 518, pp. 529–533, 2015. [Online]. Available: https://www.nature.com/articles/nature14236
