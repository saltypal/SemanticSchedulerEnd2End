# Design: how every scheduler uses STS, and DeLTA-Rank (proposed learned scheduler)

Status: PROPOSAL for review (2026-11-13). Nothing below is implemented except part A step 1 (W-STS, in progress).

## A. STS in SV-DPP and the analytic policies (decision D19)

Problem: in plan v3, STS reached SV-DPP only through the fidelity predictor u_hat; fidelity was dropped (D5), so
STS had no effect on any analytic policy (learners did read it, but STS is empty until the SQ-GAN v1 pass is merged).

Design: the receiver turns (SPI, STS) into one tier belief before the scheduler sees the job.
1. **Receiver fusion model** `q = Fuse(SPI, STS)`: multinomial logistic regression on [log SPI (4), STS (13)],
   class-balanced, trained on the train split, temperature-calibrated on val. Header-only (both fields are sent).
2. **Every policy** (SV-DPP, EDF variants, Whittle, cmu, D-over, ...) reads `q` where it read SPI. For SV-DPP this
   changes E[v_i] = sum_k q_ik w_k g_k(.) zeta and the debt arrivals A_k = sum_i q_ik. Nothing else in v3 sec. 6 changes.
3. **Measured, not assumed:** the header ablation strips STS (fusion off -> belief = SPI). Reported on val:
   macro-F1 / Critical recall of SPI vs Fuse(SPI, STS), and CTR@D of every policy with and without STS.

## B. SemCover: a semantic-native learned scheduler (distinct method; replaces the DeLTA-Rank proposal)

Naming: SemCover is its own method (architecture + objective). SCOPE stays the training *recipe* (imitation +
constrained RL) used for all learners; SemCover is trained with that recipe but is not a SCOPE variant.

### B.0 Philosophy
The edge does not schedule *jobs*, it schedules *meaning*. The value of decoding an image is not a property of the
image alone: it is (1) how urgent its meaning is for the current mission, times (2) how much NEW meaning it adds
given what the edge has already decoded and what else is in the batch. Two photos of the same collapsed bridge are
worth less together than one bridge and one flooded hospital. So the scheduler works in the semantic space the header
already exposes, and picks the batch that best COVERS urgent meaning.

### B.1 The semantic space the receiver can see (header-only, 64 B)
- Semantic key (SimHash of CLIP): Hamming distance d is an angle estimate, cos(theta) ~ cos(pi d / 64). So the receiver
  has an approximate CLIP geometry between ANY two images (queued or already delivered) without decoding.
- STS (13): what the codec kept, per scene class (7-class kept-token histogram, retention, centroid/spread): the
  scene composition.
- SPI (4) fused with STS (part A): tier belief. Relevance to the mission (CLIP image vs mission text) and mission id.
- Perceptual key (pHash): near-copy detector.

### B.2 Model
- **Semantic urgency (quality) q_i** = learned function of [fused tier belief, STS, relevance, mission embedding,
  slack/age]: a set-attention encoder over the queue (existing SetPolicy) so urgency is judged in context
  (e.g. the 10th fire photo is less urgent if 9 are already queued).
- **Semantic similarity kernel S_ij** = learned mix of cos(pi d_sem/64), STS-histogram similarity and pHash
  near-copy indicator (positive semidefinite by construction: a weighted sum of PSD kernels).
- **Selection = conditional determinantal point process (DPP) over the queue**: L = diag(q) S diag(q). The next
  batch B is the greedy MAP of det(L_B) CONDITIONED on the receiver memory of recently delivered images (Schur
  complement), with exact incremental-Cholesky greedy MAP (Chen et al., NeurIPS 2018). This trades urgency against
  semantic redundancy, both inside the batch and against what was already decoded. Duplicates are handled
  softly by the geometry (near-zero marginal gain) instead of hard thresholds; Critical corroboration (zeta_C = 0.25)
  is kept by a tier-dependent floor on the kernel for Critical-Critical pairs.
- **Deadline feasibility mask**: jobs that cannot finish before their hard deadline (latency CDF) are excluded, and
  q_i is scaled by the on-time probability G_i(now) (same quantity SV-DPP uses).
- Note on naming: "DPP" here is determinantal point process, not drift-plus-penalty; the paper must spell it out.

### B.3 Learning (SCOPE recipe, semantic objective)
1. **Semantic coverage gain** (training labels only): marginal timely value of a batch = sum over its images of
   w_k g_k(finish) zeta (true tier, true duplicate group), i.e. the simulator's own value. The DPP log-likelihood of
   high-value batches (from simulator rollouts of SV-DPP / genie, train split) trains q and the kernel weights
   (supervised stage, minutes).
2. **Fusion head**: cross-entropy of the fused tier belief vs true tier (decision-focused, shares the encoder).
3. **Constrained RL fine-tune**: sample batches from the k-DPP policy (its log-probability is tractable), reward
   r = value - sum_k lambda_k miss_k with per-tier timeliness targets eta_k (existing dual updates), annealed
   distillation toward SV-DPP early (SCOPE).
Train split only; tuned on val; 5 seeds.

### B.4 What is new (claimed narrowly)
- Scheduling a decoder queue as conditional DPP selection in a header-only semantic space (SimHash angle + scene
  composition), conditioned on receiver memory: urgency x semantic novelty, not urgency alone.
- Learned semantic kernel and urgency from header fields, trained with the SCOPE recipe (DPP likelihood + constrained RL).
- Soft, geometry-based redundancy handling replacing threshold dedup, with a corroboration floor for Critical.
Ablations: kernel off (S = I, reduces to an urgency ranker), memory conditioning off, STS off, each loss term off.
Extra metric beside CTR@D: distinct semantic clusters (true duplicate groups) of Critical images delivered on time.


### B.7 Additions from recent work (2021-2026)
1. **Semantic freshness (Age of Semantic Importance).** Chen & Gong (arXiv 2403.07386, 2024) define AoSI: freshness and
   semantic loss together, minimised with DQN over scheduling. SemCover adopts it in the memory: the receiver memory
   is a *time-decayed* set, so a scene delivered long ago no longer suppresses a new similar image
   (memory weight exp(-age/tau_k) per tier; equivalent to the sliding-window DPP of Chen et al. 2018).
2. **Structure-guided learning.** Monotonicity-driven DRL for semantic-aware scheduling (arXiv 2305.13706, 2023)
   reports ~30% over benchmarks by building known structure into the learner. SemCover constrains the urgency
   network to be monotone non-decreasing in the Critical/High belief and in deadline pressure (monotone MLP),
   which cuts sample needs and gives an interpretable policy.
3. **Constrained MDP with Lagrangian duality** for semantic-aware scheduling (arXiv 2609.07563) matches our
   tier-timeliness duals; cited as the formal frame for the RL stage.
4. **Coverage with a guarantee.** Facility-location coverage combined with query relevance is monotone submodular, so
   greedy selection is within (1 - 1/e) of the best batch (used in SAVER, arXiv 2605.20713, and MASCOT, arXiv
   2608.12532). SemCover offers two coverage objectives: log-det (DPP, best diversity, not monotone) and
   mission-relevance facility location (provable (1-1/e) per decision). The paper reports both; the guarantee is the
   theory point.
5. **Learning-based edge serving baseline.** BCEdge (arXiv 2305.01519; IEEE journal version) uses maximum-entropy RL
   for adaptive batching under SLOs. It is the closest learning-based edge-serving baseline, so we add a BCEdge-style
   SAC batching baseline next to Clipper / Clockwork / Shepherd / EdgeServing.
6. **Semantic redundancy at the edge.** Tri-Select (arXiv 2512.16469, 2025) filters redundant crowdsensing images
   (disaster response as a use case) and notes centralised filtering struggles at the edge. Positioned as related work:
   SemCover does redundancy-aware selection header-only, with no decoding.
7. **Quality-diversity in RL.** DPP-based RL post-training for quality and diversity (DQO, 2025) and differentiable DPP
   learning (Wasserstein DPP, arXiv 2011.09712) support training SemCover's k-DPP policy by policy gradient, with the
   Wasserstein relaxation as a fallback if the gradient is noisy.
8. **Survey frame.** Luo, Delfani, Salimnejad & Pappas (arXiv 2512.12758, 2025) trace AoI -> semantics of information ->
   goal-oriented communication; SemCover sits in the goal-oriented, semantics-of-information line.
All above are preprints or proceedings found 2026-11-13; publication status not verified.

### B.5 Prior work
DPPs for ML (Kulesza & Taskar 2012); fast greedy MAP DPP for diverse recommendation (Chen, Zhang & Zhou, NeurIPS 2018,
arXiv 1709.05135, verified); learning DPP kernels / k-DPPs (Kulesza & Taskar 2011); SimHash as random-hyperplane LSH of
cosine (Charikar 2002); semantics-empowered communication / value of information (Kountouris & Pappas 2021); our
SCOPE (sec. 7 of plan v3). Entries other than Chen et al. are from memory and must be checked before citing.

### B.6 Cost and risk
Reuses SetPolicy, dual updates, train_cli; new pieces: kernel module, conditional greedy MAP (~60 lines), k-DPP
log-probability for the RL stage. About 3-4 h for one worker plus tests. Risk: today is the deadline. Fallback: if it
does not beat SCOPE-LTR / SV-DPP on val, report it as a semantic-diversity variant with its ablations; SV-DPP stays
the main proposal.
