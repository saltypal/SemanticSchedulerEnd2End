# Semantic-Aware Decoder Scheduling at the O-RAN Edge: Stateless Semantic Headers, Receiver-Side Novelty, and Constrained Learning for One-Shot Generative Image Communication

*Living draft. IEEE journal structure in single-column Markdown. Target: IEEE TCCN / TWC / IoT-J.*
*Methods synced to FINAL_PLAN_V3 v3.1 (2026-11-10). Status: **pseudo-results**. Every number in [brackets] is a placeholder until the experiment exists.
Cross-references: `docs/reviews/history/FINAL_PLAN_V2.md`, `docs/reviews/history/PLAN_V2_EXPLAINED.md`.*

Authors: [names], [affiliations]

---

## Abstract
Generative semantic codecs such as SQ-GAN compress an image into a few hundred bytes, but they move
the cost from the channel to the receiver. Decoding is slow, and during a disaster surge an edge
decoder pool cannot keep up with arriving payloads. We study **which received image should be
decoded next, and which should never be decoded**, when every UE transmits once, keeps no history,
and attaches only a fixed 64-byte header. We make four contributions:

1. **A stateless semantic header.** It carries a calibrated urgency posterior, mission relevance, a
   64-bit semantic hash, and a codec-aware *Semantic Token Summary* (STS). The STS is derived for free
   from SQ-GAN's own semantic mask and predicts how much of the important content will be real rather
   than hallucinated after decoding.
2. **Receiver-side novelty.** The scheduler detects cross-UE redundancy with locality-sensitive
   hashing, so no UE state is needed.
3. **A timely-semantic-value objective with a Critical-miss constraint.** It is solved by a
   drift-plus-penalty scheduler (SV-DPP) and by constrained PPO, including a variant trained from VLM
   feedback (RLAIF).
4. **Evaluation against human labels.** We evaluate on CrisisMMD human damage annotations mixed with
   everyday imagery, using event-disjoint splits and about 20 classical, 5G-derived, index-based and
   learned schedulers.

SV-DPP raises Critical Timely Recall from [x]% (strict priority) to [y]% at load ρ = [1.2], and reduces
redundant decodes by [z]%.

**Index Terms:** semantic communication, goal-oriented scheduling, O-RAN, edge inference, generative
image coding, Lyapunov optimization, constrained reinforcement learning, RLAIF, disaster response.

---

## I. Introduction
**Motivation.** 6G envisions semantic and goal-oriented communication [1]–[4]. Generative codecs
reach extreme compression by letting the receiver *synthesize* detail [5]–[8], [37]. This creates a new
bottleneck: **receiver compute**. In a disaster, thousands of phones, drones and vehicles upload
images within minutes [9]. A generative decoder at the edge cannot decode them all in time, and
not all of them matter. Selfies, memes and retweeted duplicates compete with images of collapsed
buildings and injured people.

**Gap.** Existing semantic resource allocation schedules *radio* resources and assumes importance is
known [10]–[13], [38]–[40]. Queueing-aware SemCom controls the semantic rate of one stream [14].
Edge-inference serving schedules DNN requests by deadline [15]–[17], but it is blind to image content.
Disaster-imagery works classify urgency [9], [18] without connecting it to the communication system.
To our knowledge, no work schedules **generative semantic decoding** with:

- a compact header that travels with the payload;
- no per-UE state;
- an objective that requires the *decoded* image to remain task-correct.

**Contributions.**
1. System model and problem: decoder-side semantic scheduling at the O-RAN edge, posed as a constrained
   MDP (Sec. III).
2. A 64-byte stateless semantic header, and codec-aware STS / predicted decode utility û derived from
   SQ-GAN's masking (Sec. IV).
3. Receiver-side semantic novelty via SimHash LSH (Sec. V).
4. SV-DPP, a renewal-frame completion-aware Lyapunov scheduler with per-tier debt queues (extending
   [60], [14]), and SCOPE, one constrained, teacher-guided training framework applied to PPO, DQN,
   learning to rank and RLAIF (composed from [20]-[24], [63]-[68]) (Sec. VI).
5. A human-labelled, event-disjoint benchmark with trace-driven simulation, validated by a live
   end-to-end run (Sec. VII–VIII).

*We adapt DPP and PPO-Lagrangian. The novelty is the problem, the header, the receiver-side novelty and
the evaluation.*

---

## II. Related Work
**A. Semantic and goal-oriented communication.**
- Foundations and surveys: [1]–[4], [41].
- DeepJSCC [5] and Transformer / SwinJSCC codecs [6].
- Generative SemCom [7], [37], [42], including GAN-based and packet-erasure-robust variants [43], [44].
- SQ-GAN [8]: semantic-conditioned masked vector quantisation with runtime masking fractions.
- User- and query-oriented semantic coding with large vision-language models [45], [46].
- These works optimise the codec. We take the codec as given and schedule its decoding.

**B. Semantic resource allocation.**
- Importance-aware RA for task-oriented SemCom [10]; multi-view semantic RA for edge AI [11].
- Semantic spectral efficiency [12]; QoE-driven OFDMA SemCom [38].
- Content-preference feedback [39]; goal-oriented utility optimisation [40].
- All of these allocate radio resources and assume importance is known at the scheduler.

**C. Timeliness.**
- Age of Information [25]; semantic and version AoI [26], [27]; Age of Incorrect Information [28].
- Time-utility functions in real-time scheduling [29].
- Our deadline curve g(·) follows the TUF view.

**D. Queueing and index policies.**
- Generalised cμ rule [30]; Whittle index for convex delay costs with abandonment [31].
- Lyapunov drift-plus-penalty [19]; its SemCom use with a virtual semantic-error queue [14].
- Wireless QoS schedulers PF / M-LWDF / EXP [32], [33]; we adapt them to a decoder queue.

**E. Edge inference serving.**
- Deadline-aware multi-DNN GPU serving [15]; priority- and interference-aware serving [16]; RL for
  hard/soft real-time inference [17]; pricing priority under excess demand [34].
- Progressive edge inference under deadlines [47].

**F. Redundancy and deduplication.**
- LSH [35], SimHash [36]; CLIP-based semantic deduplication [48].
- Redundancy-aware uplink for overlapping fields of view [49].

**G. Disaster imagery and VLMs.**
- CrisisMMD [9], MEDIC [18], VLM disaster benchmarks [50], [51].
- O-RAN architecture [52].

**H. Learning to schedule.**
- LambdaMART [53], PPO [54], constrained RL [20]–[22], RLAIF [23], [24].
- Human-in-the-loop RL for SemCom latency [55].

---

## III. System Model and Problem Formulation
**Fig. 1 [PLACEHOLDER]: system architecture.**
- UEs run SegFormer [56] → SQ-GAN encoder + SAMM → CLIP [57] → 64-B header.
- One-shot transmission over an internet-like pipe.
- O-Cloud edge site runs: header parser → shared queue → scheduler → c SQ-GAN decoder workers →
  responder application.
- The near-RT RIC xApp manages the policy (weights, ε, mission registry) over A1/E2 [52].

**A. Traffic.**
- U UEs, each with a mission profile π ∈ Π (Appendix A).
- Everyday images arrive as Poisson(λ_e).
- Disaster images arrive in their **real tweet chronology**, time-compressed by a factor κ.
- Each image is sent once as `[h_i][payload_i]`. An optional loss probability p_loss ∈ [0, 0.05] is
  included as sensitivity.

**B. Receiver queue.**
- A single shared queue with c identical decoder workers; decode time S_i is measured on GPU per codec.
- The scheduler observes only headers {h_i}. It sees neither UE identity nor payload content.
- Actions at each worker-free epoch: decode j, defer, or purge a set P.

**C. Value.** An image i finished at time t yields

  v_i(t) = w(k_i) · u_i · g_{k_i}(t − τ_i) · (1 − dup_i),

where:
- k_i = true tier (Critical / High / Medium / Low);
- u_i ∈ {0, 1} = decoded image still task-correct;
- g_k = deadline curve: 1 up to D_k, linear decay to 0 at 2D_k;
- τ_i = capture time;
- dup_i = 1 if a near-duplicate was already delivered.

**D. Problem (CMDP).**

  max  lim (1/T) E Σ v_i   s.t.  lim (#Critical deadline misses)/(#Critical arrivals) ≤ ε.

Only header predictions (p_i, û_i, n_i) are causal. True k_i and u_i are used for evaluation and for
the RL training reward on training events only.

**Table I [PLACEHOLDER]: notation.**

---

## IV. Stateless Semantic Header and Codec-Aware Descriptors
**A. Header layout (64 B).** Table II gives every field with its offset, size and justification
(`PLAN_V2_EXPLAINED.md` §9).
- Fields: version/flags, message id, capture time, 24-bit payload length, codec config, 16-bit mission
  id, SPI posterior, mission relevance, 64-bit semantic hash, STS (14 B incl. û), CRC-16 over the
  header, and 20 B reserved.
- Bit-depths are chosen by a decision-flip-rate test (Fig. 7).

**B. Semantic Priority Indicator (SPI).** CLIP ViT-B/32 embedding → class-balanced multinomial
logistic regression → temperature scaling, trained on human-derived tiers (Sec. VII-A) **including
everyday images**.

**C. Mission relevance.** r_i = cos(CLIP(x_i), CLIP(text_π)), with the mission text pre-shared via the
registry.

**D. Scene descriptors: from detectors to the codec's own mask.**
1. YOLO / COCO scene block: average 34.45 B, dense maximum 307 B.
2. COCO removal: 8.16 B; full header 110.45 → 84.16 B.
3. Open-vocabulary detection (Grounding-DINO): flexible, but a second network on the UE.
4. **STS.**
   - SQ-GAN tiles the image into 16×32 tokens and SAMM keeps the top m fraction [8].
   - Majority-vote the 20-class map onto the grid, then compute over the **kept** tokens:
     - a 7-superclass histogram;
     - retention ratios for person / vehicle / building / road;
     - the centroid and spread of kept objects.
   - Kept tokens are transmitted; dropped tokens are synthesized. STS therefore measures how much of
     the task-relevant content is *real*.

**Algorithm 1 [PLACEHOLDER]: STS extraction** (pseudo-code in the explainer §6).

**E. Predicted decode utility û.** A logistic model over (STS, SPI, m), trained offline on measured
u, estimates P(decoded image task-correct).

**Table III [PLACEHOLDER]:** descriptor comparison (bytes, UE ms, ρ(û, u), CTR@D) for {none, YOLO,
Grounding-DINO, STS, learned 8-B}.

---

## V. Receiver-Side Semantic Novelty
**A. SimHash.**
- b_j = 𝟙[r_jᵀ e > 0], j = 1..64, with R shared.
- P(b_j ≠ b'_j) = θ/π [36], so the Hamming distance follows Binomial(64, θ/π).

**B. Novelty and dedup.**
- The receiver keeps a ring of decoded hashes plus the hashes of queued items.
- n_i = min Hamming distance / 64.
- Purge if d ≤ τ_dup; partial credit if τ_dup < d ≤ τ_sim.
- Among queued duplicates, keep the copy with the highest SPI, then the highest û.

**C. Calibration.** τ_dup and τ_sim are set from CrisisMMD repost pairs on the validation event.

**Fig. 4 [PLACEHOLDER]:** Hamming distributions for duplicates vs. distinct images, and ROC vs. hash
length (32 / 64 / 128 / 256 bits).

---

## VI. Scheduling Policies
**A. Baselines.** Table IV [PLACEHOLDER] defines about 20 policies:
- Classical queueing: FIFO, LIFO, Random, RR over missions, DRR/WFQ.
- Real-time: EDF, LLF.
- Priority: strict priority (SPI-first), priority + aging.
- 5G-derived (adapted): PF over missions, M-LWDF, EXP.
- Index theory: generalised cμ, Whittle-type.
- Learning to rank: LambdaMART, RankNet.
- RL: DQN, PPO.
- Oracle.

**B. SV-DPP (v3.1; see FINAL_PLAN_V3 Sec. 6).** SV-DPP extends completion-aware Lyapunov scheduling
[60] and semantic DPP [14] to decoder-side scheduling with noisy headers.

*Debt queues.* One debt queue per tier, Z_k <- max(0, Z_k + eta_k A_k - T_k), where A_k and T_k are
soft counts of predicted arrivals and of predicted on-time decodes. The target is on timeliness, and
fidelity enters through value. A necessary feasibility condition is sum_k eta_k lambda_k E[S_B]/(Bc) < 1.

*Score.* A differential score
s_i = V * dE[v_i] + sum_k Z_k p_ik dG_i, where dG_i is the change in on-time probability from decoding
now instead of deferring. It is computed from the empirical latency CDF, with a first-pass estimate of
the wait.

*Batching and dedup.* Batching follows a renewal-frame ratio rule [61]. Deduplication is soft,
using a perceptual key and a semantic key, with a delivered-only memory.

*Claims.* Debt stability is verified empirically. In the degenerate case SV-DPP reduces to
expected-weight priority. Proposition [PLACEHOLDER]: its relation to generalised c-mu/h [62] and to
Whittle with abandonment [31] for c = 1.

**Algorithm 2 [PLACEHOLDER]: SV-DPP.**

**C. SCOPE: a training framework for every learner.** The objective is
min_theta max_lambda -J_v + sum_k lambda_k (eta_k - J_k) + beta(n) L_imit(theta; pi_SV-DPP).
- Constraint costs enter only through returns.
- Imitation queries the analytic teacher at the states the learner actually visits [63]-[66].
- beta(n) decays with live experience [67].

Instances:
- SCOPE-PPO: set-attention actor;
- SCOPE-DQN: DQfD-style margin loss [68];
- SCOPE-LTR: listwise distillation.

A proof-of-value experiment compares SCOPE with the teacher, BC-only, DAgger and plain RL in regimes
where the teacher is misspecified.

**D. SCOPE-AIF (RLAIF) [23], [24].**
- (i) Reward-only: a Bradley-Terry reward model fitted on order-swapped Qwen2.5-VL pairwise
  preferences plus pointwise log-probabilities.
- (ii) A fully AI-feedback pipeline: SPI and teacher built from Qwen labels.

Both are evaluated on human labels.

**E. Fidelity judge.** u_i = 1[judge(recon) = judge(orig)], with a DINOv2 probe as the judge
(independent of the CLIP-based SPI). A human audit covers 300 reconstructions.

---

## VII. Experimental Setup
**A. Data and labels.**
- CrisisMMD v2: 7 events with human annotations [9]. Tier mapping (Table V):
  - severe → Critical, mild → High, little_or_none & informative → Medium;
  - injured / dead / missing people → Critical (override);
  - not informative → Low.
- Everyday images: Places365 [58] minus damage-lookalike classes, plus people photos [59] → Low.
- Event-disjoint splits: train {Harvey, Irma, California wildfires, Iraq–Iran earthquake}, validation
  {Sri Lanka floods}, test {Maria, Mexico earthquake}.
- VLM labels are used only for κ agreement and RLAIF.

**B. Codec profiling.** SQ-GAN at m ∈ {0.25, 0.55}, plus JPEG/BPG controls. u is measured by an
independent CLIP-L/14 receiver probe. LPIPS is also reported.

**C. Simulator.**
- Trace-driven and event-driven, with c ∈ {1, 2, 4} and ρ ∈ {0.6 … 1.5}.
- ≥ 20k arrivals per run, 10% warm-up, 10 realisations.
- Built on measured stage timings.
- Live end-to-end validation run on [500] images.

**D. Metrics.**
- Headline: CTR@D (Critical Timely Recall).
- Normalised timely value; Critical-correct per decoder-second.
- Per-tier deadline rate and p95 sojourn; Low starvation; redundant decodes; decision µs.
- Bootstrap CIs over images and events.

---

## VIII. Results [ALL PLACEHOLDERS]
- **Fig. 5:** CTR@D vs. load for all policies.
- **Fig. 6:** normalised value vs. load; Pareto front of CTR@D vs. Low starvation.
- **Table VI:** main comparison at ρ = 1.2, c = 2.
- **Table VII:** header ablation (SPI, +relevance, +novelty, +STS, +û, YOLO-instead, oracle).
- **Fig. 7:** decision-flip rate vs. bit-depth.
- **Fig. 8:** SC-PPO vs. SC-PPO-AIF vs. SV-DPP learning curves; RLAIF gap.
- **Table VIII:** SPI with vs. without everyday images (false-Critical rate on selfies).
- **Table IX:** trace-driven vs. live run agreement.
- **Fig. 9:** sensitivity to weights and deadlines (×0.5, ×2) and to loss probability.

---

## IX. Discussion and Limitations
- STS inherits Cityscapes classes (no water/fire/rubble); an ADE20K-merged retraining is in progress.
- SQ-GAN is out of distribution on disaster photos.
- Tweet timing is a proxy for field traces.
- The tier mapping is a design choice (a sensitivity analysis is provided).
- DPP guarantees hold on time-averages, not per image.
- The VLM judge has order bias.

## X. Conclusion
[PLACEHOLDER]

---

## References
*Verify every arXiv ID before submission. 26xx IDs come from the alphaXiv index and are summarised
from their abstracts.*

[1] D. Gündüz et al., "Beyond transmitting bits: Context, semantics, and task-oriented communications," IEEE JSAC, vol. 41, no. 1, 2023.
[2] E. C. Strinati and S. Barbarossa, "6G networks: Beyond Shannon towards semantic and goal-oriented communications," Computer Networks, 2021.
[3] E. Uysal et al., "Semantic communications in networked systems," IEEE Network, 2022.
[4] H. Xie, Z. Qin, G. Y. Li, B.-H. Juang, "Deep learning enabled semantic communication systems," IEEE TSP, 2021.
[5] E. Bourtsoulatze, D. B. Kurka, D. Gündüz, "Deep joint source-channel coding for wireless image transmission," IEEE TCCN, 2019.
[6] K. Yang et al., "SwinJSCC: Taming Swin Transformer for deep joint source-channel coding," IEEE TCCN, 2024.
[7] E. Erdemir, T.-Y. Tung, P. L. Dragotti, D. Gündüz, "Generative joint source-channel coding for semantic image transmission," IEEE JSAC, 2023.
[8] F. Pezone, S. Barbarossa, G. Caire, "SQ-GAN: Semantic image communications using masked vector quantization," IEEE TCCN (arXiv:2502.09520).
[9] F. Alam, F. Ofli, M. Imran, "CrisisMMD: Multimodal Twitter datasets from natural disasters," ICWSM, 2018.
[10] "Importance-aware resource allocation for collaborative task-oriented semantic communication," arXiv:2606.29052.
[11] "Multi-view semantic-aware resource allocation for edge AI," arXiv:2610.34532.
[12] L. Yan, Z. Qin, R. Zhang, Y. Li, G. Y. Li, "Resource allocation for text semantic communications," IEEE WCL, 2022.
[13] "Semantic-aware sub-band allocation for terahertz communications," arXiv:2608.30984.
[14] Y. E. Sagduyu, T. Erpek, "When semantic communication meets queueing: Cross-layer latency and task fidelity optimization," arXiv:2605.05514.
[15] "EdgeServing: Deadline-aware multi-DNN serving at the edge," arXiv:2605.05527.
[16] "Strait: Perceiving priority and interference in ML inference serving," arXiv:2604.28175.
[17] "Hard and soft real-time DNN inference scheduling on heterogeneous multi-accelerator systems via RL," arXiv:2610.29503.
[18] F. Alam et al., "MEDIC: A multi-task learning dataset for disaster image classification," arXiv:2108.12828.
[19] M. J. Neely, Stochastic Network Optimization with Application to Communication and Queueing Systems, Morgan & Claypool, 2010.
[20] A. Ray, J. Achiam, D. Amodei, "Benchmarking safe exploration in deep reinforcement learning," OpenAI, 2019.
[21] C. Tessler, D. J. Mankowitz, S. Mannor, "Reward constrained policy optimization," ICLR, 2019.
[22] A. Stooke, J. Achiam, P. Abbeel, "Responsive safety in RL by PID Lagrangian methods," ICML, 2020.
[23] Y. Bai et al., "Constitutional AI: Harmlessness from AI feedback," arXiv:2212.08073, 2022.
[24] H. Lee et al., "RLAIF: Scaling reinforcement learning from human feedback with AI feedback," arXiv:2309.00267, 2023.
[25] S. Kaul, R. Yates, M. Gruteser, "Real-time status: How often should one update?" IEEE INFOCOM, 2012.
[26] "Minimizing quantized semantic age of information in foundation-model-based semantic communications," arXiv:2606.31303.
[27] "Diffusion model-based RL for version age of information scheduling," arXiv:2601.18069.
[28] "Minimizing age of incorrect information in pull-based remote estimation of a Markov source," arXiv:2610.22265.
[29] C. D. Locke, "Best-effort decision making for real-time scheduling," PhD thesis, CMU, 1986.
[30] J. A. van Mieghem, "Dynamic scheduling with convex delay costs: The generalized cμ rule," Annals of Applied Probability, 1995.
[31] "Whittle index approach to multi-server scheduling with convex delay costs and impatient customers," arXiv:2609.19792.
[32] M. Andrews et al., "Providing quality of service over a shared wireless link," IEEE Commun. Mag., 2001.
[33] S. Shakkottai, A. Stolyar, "Scheduling for multiple flows sharing a time-varying channel: The exponential rule," AMS Translations, 2002.
[34] "Inference auctions," arXiv:2609.40070.
[35] P. Indyk, R. Motwani, "Approximate nearest neighbors: Towards removing the curse of dimensionality," STOC, 1998.
[36] M. S. Charikar, "Similarity estimation techniques from rounding algorithms," STOC, 2002.
[37] "Towards a joint task-oriented and generative semantic communication framework for 6G networks," arXiv:2606.31426.
[38] "QoE-driven hybrid resource allocation for multi-user OFDMA semantic communications," arXiv:2610.32077.
[39] "Optimization of collaborative semantic communication network performance with channel and content preference feedback," arXiv:2607.25011.
[40] "Reference-dependent utility optimization for goal-oriented communication systems," arXiv:2611.12480.
[41] "Foundation models for generalizable semantic and goal-oriented communication," arXiv:2609.07853.
[42] "Diffusion-aided bandwidth-efficient semantic communication with adaptive requests," arXiv:2510.26442.
[43] "GAN-based semantic communication for image transmission in IoV," arXiv:2608.27989.
[44] "Lightweight generative image semantic communication over packet erasure channels," arXiv:2609.06989.
[45] "Personalized digital semantic communication for image transmission with vision-language models," arXiv:2608.14260.
[46] "Generalized user-oriented image semantic coding empowered by large vision-language model," arXiv:2509.08913.
[47] "Split-gate pooled-evidence stochastic-rollout scheduling for timely progressive edge inference," arXiv:2608.08338.
[48] A. Abbas et al., "SemDeDup: Data-efficient learning at web-scale through semantic deduplication," arXiv:2303.09540, 2023.
[49] "Exploiting overlapping fields of view for redundancy-aware uplink transmission in vehicular 6G," arXiv:2606.31715.
[50] "DisasterVL: Evidence-based visual reasoning dataset and model for disaster scene understanding," arXiv:2611.12144.
[51] "DisasterInsight: A building-centric benchmark for evaluating VLMs in disaster response," arXiv:2601.18493.
[52] M. Polese et al., "Understanding O-RAN: Architecture, interfaces, algorithms, security, and research challenges," IEEE COMST, 2023.
[53] C. J. C. Burges, "From RankNet to LambdaRank to LambdaMART: An overview," Microsoft Research TR, 2010.
[54] J. Schulman et al., "Proximal policy optimization algorithms," arXiv:1707.06347, 2017.
[55] "Latency-aware human-in-the-loop reinforcement learning for semantic communications," arXiv:2602.15640.
[56] E. Xie et al., "SegFormer: Simple and efficient design for semantic segmentation with transformers," NeurIPS, 2021.
[57] A. Radford et al., "Learning transferable visual models from natural language supervision," ICML, 2021.
[58] B. Zhou et al., "Places: A 10 million image database for scene recognition," IEEE TPAMI, 2018.
[59] A. Kuznetsova et al., "The Open Images dataset V4," IJCV, 2020.

[60] B. Zhao, Q. Sun, N. Li, C.-L. I, "Lyapunov-based completion-aware scheduling for PDU set-based real-time XR traffic," arXiv:2608.05537.
[61] M. J. Neely, "Dynamic optimization and learning for renewal systems," IEEE TAC, 2013.
[62] Z. Long, N. Shimkin, H. Zhang, J. Zhang, "Dynamic scheduling of multiclass many-server queues with abandonment: The generalized c-mu/h rule," Operations Research, 2020.
[63] S. Ross, G. Gordon, J. A. Bagnell, "A reduction of imitation learning and structured prediction to no-regret online learning (DAgger)," AISTATS, 2011.
[64] S. Ross, J. A. Bagnell, "Reinforcement and imitation learning via interactive no-regret learning (AggreVaTe)," arXiv:1406.5979, 2014.
[65] W. Sun, J. A. Bagnell, B. Boots, "Truncated horizon policy search: Combining RL and imitation learning (THOR)," ICLR, 2018.
[66] C.-A. Cheng et al., "Fast policy learning through imitation and reinforcement (LOKI)," UAI, 2018.
[67] V. N. Vitale et al., "From prior-guided heuristics to deployable agents (MGA-RL)," arXiv:2609.03590.
[68] T. Hester et al., "Deep Q-learning from demonstrations," AAAI, 2018.
[69] J. Cao et al., "EdgeServing: Deadline-aware multi-DNN serving at the edge," arXiv:2605.05527 (cf. [15]).
[70] S. Aalto, "Whittle index approach to multi-server scheduling with convex delay costs and impatient customers," arXiv:2609.19792 (cf. [31]).
[71] E. Pizzi et al., "A self-supervised descriptor for image copy detection (SSCD)," CVPR, 2022.
[72] "Benchmarking pretrained vision embeddings for near- and duplicate detection in medical images," arXiv:2312.07273.
[73] "On the de-duplication of LAION-2B," arXiv:2303.12733.
[74] "Adversarial collision attacks on image hashing functions," arXiv:2011.09473.
[75] S. Gul et al., "PDU set-based QoS handling in 3GPP: Release 18 overview and future directions," IEEE Commun. Stand. Mag., 2025.
[76] FEMA, Preliminary Damage Assessment Guide, 2021; S. Benson et al., "Disaster triage: START, then SAVE," Prehosp. Disaster Med., 1996.
[77] M. Imran et al., "Processing social media messages in mass emergency: A survey," ACM CSUR, 2015.
[78] M. Oquab et al., "DINOv2: Learning robust visual features without supervision," TMLR, 2024.
[79] "TempoNet: Slack-quantized transformer-guided reinforcement scheduler," arXiv:2602.18109.

---

## Appendix A: Mission profile registry
See `docs/reviews/MISSION_PROFILES_DRAFT.md`: 26 draft profiles across UAV, V2X/ADAS, public safety,
CCTV/IoT, industrial, consumer/XR and NTN, mapped to 3GPP TS 22.125 / 22.186 / 22.261 / 22.263 /
22.104 / 22.179 and ITU-R M.2160. **Some clause references are unverified.** The registry maps each
16-bit mission id to its paragraph.

## Appendix B: Header field justification and bit-depth study
Table and flip-rate methodology (`PLAN_V2_EXPLAINED.md` §9).

## Appendix C: Tier mapping and sensitivity
Full rule table, the alternative mapping, per-event tier counts [PLACEHOLDER], and Cohen's κ vs. Qwen
[PLACEHOLDER].

## Appendix D: Hyperparameters
SV-DPP (V, ε), SC-PPO (network, learning rate, clip, β schedule, η), RLAIF (number of pairs,
reward-model loss) [PLACEHOLDER].

## Appendix E: Prior design iterations (negative results)
- Old circular waiting-penalty benchmark: FIFO 5.822, SPI-first 3.374, LambdaMART 3.218, PPO 4.594,
  DDQN 4.710.
- YOLO / Grounding-DINO trials.
- VLM-reviewer gating attempts.

These are reported for transparency.
