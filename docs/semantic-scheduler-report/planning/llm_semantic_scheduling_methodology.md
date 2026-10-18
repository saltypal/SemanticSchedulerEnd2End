# Inspiration from Semantic Scheduling for LLM Inference

## Identified paper and inspected sources

- Paper: **Semantic Scheduling for LLM Inference**, Wenyue Hua et al., 2025, arXiv:2506.12204.
- Landing page: https://arxiv.org/abs/2506.12204
- PDF: https://arxiv.org/pdf/2506.12204
- Author repository: https://github.com/Wenyueh/latency_optimization_with_priority_constraints
- Reviewed source commit: `6d232be8107c64e0733fd0a15fd9e2da22e326e4`.
- A separate replication README supplies the arXiv citation: https://github.com/augustoafleal/unisinos-semantic-scheduling#refer%C3%AAncia . It is a secondary source, not the authors' implementation.

The paper title is confirmed by the author repository README. This review inspected the authors' code, README and experiment scripts. Direct requests to arXiv were denied by the environment's egress proxy, so the full PDF, proofs and numerical results were not reviewed. Links identify the paper but do not constitute a downloaded local PDF. Do not attribute unverified performance figures or claim the complete paper was read.

## What the accessible author implementation does

The README motivates semantic request priority using critical, time-sensitive applications such as medical emergencies. It assigns five priority levels (0 highest through 4 lowest), combines priority with estimated remaining computation, and updates remaining work during continuous service.

`main/queue_management.py`, particularly `TwoDimensionalPriorityQueue.push/pop`, orders requests by semantic priority and predicted remaining work. The conceptual key is `(predicted_priority, predicted_remaining_service)`, with cache loading included where the serving loop computes queue keys. This is class priority with shorter remaining work inside a class. It is not explicit earliest-deadline scheduling.

The serving loop accounts for prefill and decoding stages and the effect of batching. `main/memory_management.py` uses a reversed priority/work heap to select cache eviction candidates and accounts for save/load/recomputation costs. KV cache is computational attention state; it is not a knowledge base of real-world events received by an application.

The README and source expose FCFS, SJF and priority-only comparisons. Author experiment scripts vary traffic gaps and bursts, priority-prediction error, output-length prediction error, prediction latency and batching, and hardware/model timing configurations. `main/utils.py` adds predictor waiting time to recorded waiting time and summarizes by true priority class.

The inspected public simulation generates priorities and lengths randomly and uses oracle-plus-injected-error predictors. This supports controlled sensitivity experiments. It does not by itself establish semantic classification accuracy on independent real emergency data. Do not generalize this observation to every experiment in the unavailable PDF.

## Mapping to SemanticSchedulerEnd2End

| Author implementation concept | Proposed project adaptation |
|---|---|
| Prompt semantic priority | Calibrated image/event criticality under an operational rubric |
| Remaining generation work | Remaining extraction/encoding/transmission/decoding/receiver processing |
| Continuous request arrivals | Chronological multi-UE image/event arrivals |
| Prefill/decode batching constraints | Measured extraction, codec and communication service-stage constraints |
| Priority-aware cache/resource management | Explicit GPU/buffer allocation and bounded preemption; account for retained or recomputed work |
| Priority predictor error | Critical-event false negatives, false positives, calibration and domain shift |
| Length predictor error | Payload/service-time/link-capacity estimation error |
| Predictor latency | Semantic extraction, descriptor upload and KB lookup latency |
| Per-priority waiting time | Per-criticality on-time useful reception, critical misses and latency tails |

There is no encoder/decoder recommendation for wireless image communication in this implementation. Its LLM prefill/decode stages do not validate JPEG, SwinJSCC, DeepJSCC or NTSCC. Codec selection still needs independent resource-matched receiver-task experiments.

## Recommended baseline and proposed extensions

Add a labeled literature-inspired baseline that sorts by `(semantic_class, estimated_remaining_service_time, arrival, job_id)`. Update remaining work after service. Define preemption and resumption at real packet/service boundaries and include their cost. The current project's single-slot complete-job abstraction cannot represent the whole method directly.

Retain this baseline separately from the proposed deadline-aware policy. Short remaining service is an efficiency signal, while urgency is remaining deadline slack:

`slack_i(t) = deadline_i - t - estimated_remaining_service_i(t)`.

The first proposed policy can use `(criticality, urgency_band, -fixed_task_value, exact_slack, deterministic_ties)` under feasibility constraints. Estimate/classify criticality with independent annotations and validation rather than assigning it from filenames. Protect uncertain possible emergencies through a declared domain policy.

The dynamic policy adds causal receiver state and estimated marginal task gain. A critical repeated observation may add little after useful receipt, while escalation or corroboration may add much more. Sender observation history and receiver useful knowledge are different stores. Use ACK/task feedback available at decision time; failed delivery must not create a false belief that an emergency is already known.

These are proposed extensions. The paper's use of continuous scheduling and changing remaining work is not evidence that it evaluates receiver event knowledge, delivery acknowledgements or operational deadlines.

## Experimental design to borrow and strengthen

1. **Separate sources of gain.** FCFS, SJF, semantic-priority-only, semantic-priority-plus-remaining-work, global EDF, least slack, the static semantic/urgency rule, and the dynamic KB policy. Compare common resource models and matched information access.
2. **Vary arrivals and congestion.** Low load, saturated load, bursts, different UE counts and heterogeneity. Save the realized trace rather than assuming the same seed automatically creates identical exogenous traces for different algorithms.
3. **Inject prediction errors.** Sweep semantic classification and service estimates independently. Include realistic critical false negatives and measured confusion matrices, not only symmetric/random class perturbations.
4. **Charge inference overhead.** Include semantic processing, batching delay, descriptor transport, KB retrieval, receiver evaluation and feedback. Report zero-overhead oracle diagnostics separately.
5. **Test constrained state and memory.** Account for inference batching, buffer pressure, preemption, recomputation and saved state. Verify whether these are actual costs on the target runtime or simulation assumptions.
6. **Report by criticality.** Critical-event recall by deadline, false alerts, event-specific task quality, p95/p99 useful-reception latency, drops/starvation, and total resource overhead. Global average waiting time cannot establish the emergency objective.
7. **Keep real-data and controlled tests distinct.** Oracle metadata isolates policy behavior. Predicted metadata on independent event splits establishes operational semantic performance. Dynamic event sequences establish causal KB benefit.
8. **Use independent episodes for uncertainty.** Paired effect sizes and confidence intervals clustered by event/episode. Keep duplicate frames out of train/test leakage and avoid counting the same event repeatedly as new success.

The strongest transferable lesson is to evaluate the scheduler under incorrect and expensive predictions. Semantic priority only helps when its errors and compute/communication costs are compatible with the response window.

## Research positioning and next concrete action

Semantic-class-plus-shortest-work scheduling should be cited and reproduced as a baseline. Its transplantation to image jobs alone would be a weak novelty claim. Investigate an acknowledgement-aware, deadline-constrained event-information policy whose independently measured outcome is useful receiver knowledge, including redundancy, escalation, uncertainty and complete system overhead. Novelty still requires a broader literature review.

For the next implementation milestone, extend the job/service contract with remaining work and introduce the literature-inspired baseline plus global EDF/least slack. Connect real image manifests and measured JPEG costs. Add error and overhead sweeps to the previously proposed benchmark. Then add receiver knowledge as a separately measurable component. Reproduction of the authors' simulator is a separate task; the reviewed checkout imports `config` but does not track a corresponding `config.py`, so it was not executed here.
