# Related work and additional scheduler requirements

This extends the project's existing research roadmap. The focus is emergency image observations from multiple UEs, semantic importance, response deadlines, and receiver knowledge. These are proposed changes, not implemented capabilities or verified performance gains.

## 1. Closely related references

### Decision-aware multi-UAV semantic scheduling

**Training-Free Decision-Aware Semantic Scheduling for Agentic Multi-UAV Missions**, F. Rezazadeh, H. Chergui, L. Liu and M. Debbah, 2026.

Source: https://github.com/frezazadeh/agentic-uav-semantic-scheduling . Reviewed source commit: `c70e2ec471271c645a143da34b4413c12b101bc0`. The repository identifies an associated manuscript, but provides no publisher/arXiv identifier in its citation; publication venue/status was not established.

This is the closest implementation found in this search. Its UAV search/rescue simulation estimates how a candidate message changes teammates' decisions, selects messages and transmission rates jointly, uses common and private beliefs, and adds freshness and shared-uplink resource accounting. The code contains value-based scheduling, work-conserving fill, optional fusion gains, expiry and fragmentation variants. Its final variant does not simply enable every option: the development notes explicitly discuss removing multi-slot messages after an ablation. Those reported outcomes were not rerun here.

Inspect `gosc/voi.py`, `goscuav/gosc_plus.py`, `goscuav/sota.py`, `experiments/check_scheduler.py`, `experiments/signalling_check.py` and `experiments/run_confirm.py` at the reviewed commit. Especially useful are separation of an oracle teammate model from operational estimates, comparisons with freshness/incorrectness baselines, explicit signalling accounting, held-out confirmatory episodes and heuristic-versus-exhaustive optimization checks.

**Implication:** receiver belief, freshness and marginal decision value already appear in close related work. Calling our contribution “KB + semantic scheduling” would be insufficient. Investigate differences around independently labeled real images, calibrated semantic errors, explicit operational deadlines, useful reception quality, representation choice, and full latency/overhead. A complete literature review must still establish whether any proposed combination is new.

### Push/pull communication for anomalies

**A Combined Push-Pull Access Framework for Digital Twin Alignment and Anomaly Reporting**: https://arxiv.org/abs/2508.21516 . Author code: https://github.com/signetlabdei/push-pull-anomaly-tracking . The author README states presentation at IEEE INFOCOM ASoI 2026, a workshop, rather than the main conference.

The author abstract distinguishes urgent source-originated anomaly pushes from receiver-requested updates to maintain an accurate digital twin. The code supports resource allocation between them and AoII-based evaluation.

**Borrow:** immediately push compact emergency alerts; let the receiver pull a fuller image, another viewpoint or a confirmation when it would resolve an important uncertainty. Charge the return request and resulting transmission. This changes knowledge acquisition, not only queue order.

The same repository describes a separate journal submission, **Push-Pull Medium Access for Digital Twin Alignment and Low-Latency Anomaly Reporting**. Its arXiv link is incomplete, so this review does not provide a verified identifier or claim acceptance for that second manuscript.

### Incorrect receiver information and multi-user links

**The Age of Incorrect Information for Multi-User Link Scheduling Over Fading Channels**, J. Li et al. Author-lab bibliography lists ICC 2025 and a 2026 IEEE Transactions on Communications version: https://ieeexplore.ieee.org/document/11475769 . Bibliographic source inspected: https://github.com/the-thinking-reed/5311lab/blob/main/_pages/about.md . This is a citation-level lead, not a full-text algorithm review.

For the established general AoII idea, the reviewed GOSC source cites Maatouk, Kriouile, Assaad and Ephremides, **The Age of Incorrect Information: A New Performance Metric for Status Updates**, IEEE/ACM Transactions on Networking, 2020. Its AoII-greedy implementation is an adapted benchmark, not evidence of universal optimality.

**Borrow:** evaluate how long the receiver remains materially wrong about an event, weighted by the seriousness of the error. An old but still correct observation and a fresh but incorrect observation should have different outcomes. True mismatch is evaluator information; the deployed policy must estimate it from available evidence and uncertainty.

### Image communication with feedback

**DeepJSCC-f: Deep Joint Source-Channel Coding of Images with Feedback**, David Burth Kurka and Deniz Gündüz, IEEE JSAIT, 2020: https://arxiv.org/abs/1911.11174 . Author implementation: https://github.com/ipc-lab/deepJSCC-feedback . The title and publication are confirmed by the author README; its full paper was not retrieved here.

**Borrow as a design direction:** feedback-assisted image transfer and an explicit distinction between initial receipt and subsequent refinement. Test whether incremental evidence improves the receiver task before the deadline. Raw channel feedback, a packet acknowledgement, and a statement that the event was understood are different signals and have different costs.

### Decoder-side information

**Distributed Deep Joint Source-Channel Coding with Decoder-Only Side Information**, Selim F. Yilmaz, Ezgi Ozyilkan, Deniz Gündüz and Elza Erkip, ICMLCN 2024: https://arxiv.org/abs/2310.04311 . Author implementation/abstract: https://github.com/ipc-lab/deepjscc-wz . DOI: `10.1109/ICMLCN59089.2024.10625214`.

The author abstract and implementation incorporate correlated information present only at the receiver into decoding. Their code uses stereo image datasets, so successful use of arbitrary emergency history is not established by these results.

**Borrow:** distinguish a KB used for deciding what to send from side information that a compatible decoder can actually exploit. For correlated observations, separately test decision-only knowledge and knowledge-assisted decoding. Version/check receiver state; test missing, stale and mismatched references.

### Adaptive image transmission rate

**Deep Joint Source-Channel Coding for Wireless Image Transmission with Adaptive Rate Control**, Mingyu Yang and Hun-Seok Kim: https://arxiv.org/abs/2110.04456 . Author implementation: https://github.com/mingyuyng/Dynamic_JSCC . The arXiv version is 2021; the collected code index lists ICASSP 2022.

**Borrow:** choose how much communication resource to spend on an image rather than treating every image as one fixed-cost job. In our task, choose the least costly validated representation that satisfies a receiver task requirement and response deadline. Train/calibrate on development data and count all control and model costs.

### Knowledge-based incremental retransmission

**Adaptive Bit Rate Control in Semantic Communication with Incremental Knowledge-based HARQ**, Q. Zhou et al., IEEE Open Journal of the Communications Society, 2022: https://ieeexplore.ieee.org/document/9827978 . Author implementation: https://github.com/ZJUNICE/Semantic-Communication-IK-HARQ . The author README's arXiv identifiers are internally inconsistent; use the publisher identifier and resolve metadata before producing a final manuscript bibliography.

The code concerns text communication. **Borrow as an image-system hypothesis:** when another transmission is needed, request complementary missing evidence or refinement rather than always retransmitting an identical full payload. An image adaptation requires its own implementation and validation.

## 2. Additional requirements, ordered by impact

| Requirement | Why it matters | Acceptance evidence |
|---|---|---|
| Protect emergency criticality and deadlines | Small/easy routine jobs must not consume the opportunity needed by a feasible critical job | Matched adversarial queue cases; class-specific misses and slack |
| Compact alert before full image | A critical full image may be too large or slow | Alert contains usable event evidence and arrives sooner; false alerts and all overhead measured |
| Receiver belief and event identity | Historical similarity cannot establish current receiver understanding | Repeats, escalation, corroboration, resolution and failed delivery produce different state updates |
| Incorrectness-aware urgency | Freshness alone can spend bandwidth updating already correct knowledge | Ground-truth receiver mismatch and harm-duration outcomes, plus causal predicted-state inputs |
| Expected useful delivery | Transmission selection is not proof of task success | Calibrated deadline/delivery estimates and actual reconstructed-image task outcomes |
| Representation/rate selection | Sometimes a crop, thumbnail or alert is adequate | Same action choices across competing policies; correct task results under matched resources |
| Active evidence requests | Knowledge should influence which observation is acquired next | Compare passive KB versus costed receiver pulls for missing or ambiguous evidence |
| Corroboration with dependence handling | Multiple views can increase confidence or merely repeat evidence | Do not double count correlated frames; test independent and dependent views |
| Event-aware expiry and service completion | Backlogs otherwise deliver obsolete data | Drop/replace superseded records; handle large jobs through a feasible smaller action or costed multi-slot service |
| Work-conserving safe fallback | A poor utility model can leave useful capacity idle | Fill genuinely spare budget without delaying protected critical work; test flat/missing scores |
| Complete communication and compute accounting | Rich descriptors/KB sync can erase apparent gains | Measure source extraction, headers, requests/grants, payload, ACKs, retries, KB sync, receiver compute |
| Bounded online execution and recovery | A clever policy must still act within its response window | p99 decision latency, queue-growth limits, static fallback, delayed/out-of-order feedback and restart tests |

An emergency alert is an independently validated representation, not just a high priority number. It may include event type, source/time/location where available, uncertainty and a small visual cue. Ground those fields in evidence. Poor confidence in a possible severe hazard should invoke the declared cautious policy rather than automatic demotion.

Knowledge state needs a stable event ID, calibrated belief/severity, capture and receipt timestamps, provenance, task/model version, unresolved information needs, expiry and delivery state. Keep source observations and useful receiver knowledge separate. Update receiver knowledge from actual receipt and receiver feedback, with a conservative state estimate while feedback is delayed. A similarity match or a missing ACK is not an event resolution.

Re-evaluate priority on time changes, acknowledgements, task changes and new evidence. Bounded event-triggered re-scoring and incremental data structures can avoid repeatedly rerunning a neural model on the entire queue. The policy consumes typed snapshots; ML, transport and state mutation remain in their existing owning modules.

## 3. A stronger scheduling formulation to investigate

For each feasible representation r of observation i, estimate:

`gain_i,r(t) = P(usable receipt by deadline | information available at t, r) * E(operational harm reduced | usable receipt, receiver state, r)`.

These terms are estimates; independently annotated receiver outcomes evaluate their accuracy. Avoid giving policy code unseen ground truth or future channel realizations. Correlated/complementary evidence can make joint gain non-additive, so compare a greedy approximation against an exact small-instance solver.

Use protected criticality and urgency ordering around this estimated gain. A single global gain-per-bit ratio can favor cheap low-stakes updates and violate the user's emergency priority goal. Specify whether urgency precedes value within a class, or whether the policy uses validated urgency bands. Expose tradeoffs rather than hiding them in arbitrary weights.

The first version chooses from alert/thumbnail/full-image representations using a fixed rubric and fixed receiver state. A subsequent version adjusts estimated gain using delivered event knowledge, then introduces active pull requests. Keep a fixed-codec track so scheduler benefit remains identifiable; add a codec-by-policy matrix later.

## 4. Benchmark additions

Extend Timely Semantic Event Delivery with:

1. **Same event, different truth:** unchanged repeat; worsening hazard; hazard resolved; lookalike background; two corroborating sources; misleading evidence; transmission failure.
2. **Three information conditions:** oracle semantics for diagnosis; predicted semantics on held-out events; delayed/noisy receiver knowledge. Clearly label oracle inputs.
3. **Prediction robustness:** asymmetric critical false negatives, false alerts, calibration drift, service-time error and unexpected channels.
4. **Low-budget cases:** emergency image exceeds one slot; only an alert fits; new critical event appears during an ongoing routine transfer; all sources become urgent together.
5. **Knowledge failures:** lost/delayed ACKs, stale references, incorrect event association, expired KB records, task changes, store outage and restart.
6. **Matched total budgets:** uplink and downlink signalling, retransmissions, descriptors/headers, source/receiver computation and knowledge maintenance. Report cached-trace diagnostics separately from full runtime.
7. **Expanded baselines:** global EDF, least slack, semantic priority plus remaining service, weighted AoI/Whittle under stated assumptions, AoII-greedy, value-only and push/pull variants alongside existing FIFO/RR/class policies. Adapt all to identical feasible representations when isolating policy choice.
8. **Independent outcomes:** critical-event recall by deadline at a frozen false-alert operating point; duration/severity of receiver error; useful-reception p95/p99; starvation/drop reasons; resource and energy use. PSNR alone is insufficient.
9. **Component ablations:** no KB, stale KB, no feedback, no active pull, fixed representation, no uncertainty handling, no expiry/fill, and alternative fragmentation. More mechanisms need not improve performance.
10. **Confirmatory testing:** freeze thresholds/configurations and splits before held-out episodes; publish paired effect sizes and episode-clustered confidence intervals; perform a pilot for sample size and report losses. Compare small-instance optimizer gaps and online decision-time limits.

Whittle/AoII adaptations have model assumptions and should not be presented as exact reproductions or optimal policies outside those assumptions. Simulated seed repetitions do not replace independent real event data. Static images with invented chronological order cannot establish naturally evolving scene performance; distinguish controlled replay from recorded event sequences.

## 5. What “best” should mean and next implementation sequence

Set the target to high critical-event recall before deadline with bounded false alerts, bounded latency and resource use, and robust behavior under overload and uncertainty. Establish the practical operating envelope and report where other policies win. The evidence can support superiority on defined tasks and conditions, not universal superiority.

Proceed in this order:

1. **Static integration:** event manifests, validated criticality/deadlines, measured JPEG costs, the semantic/urgency policy, global EDF/least slack and the cited semantic-plus-remaining-work baseline. Keep real receiver task evaluation.
2. **Delivery-aware knowledge:** chronological event state, usable-reception feedback, uncertain/stale-state handling, independent receiver-error metrics and corroboration cases.
3. **Information control:** compact emergency alert, costed active requests, and representation selection. Ablate each independently and retain only demonstrated gains.
4. **Communication/optimization study:** matched digital/JSCC comparisons, adaptive rates, exact small-instance bounds, held-out confirmatory tests and complete overhead.
5. **Deployment validation:** quantitative response targets, authenticated source policy, queue/recovery/fallback behavior and target-hardware measurements.

A defensible candidate research question is: under scarce multi-UE communication, do deadline-protected policies that choose observations and representations using calibrated receiver-event uncertainty improve on-time critical-event recognition after full signalling and computation costs? Novelty remains to be established against the above work and a wider literature review.

## Review scope

This is a targeted related-work search, not an exhaustive systematic review. Author repository READMEs, selected GOSC implementation files and experiment scripts, and the identified lab bibliography were inspected. No cited experiment was reproduced. arXiv/publisher full-text access remains unavailable under the current egress policy, so bibliographic and repository evidence are explicitly separated from full-paper claims. No numerical performance advantage is asserted for this project.
