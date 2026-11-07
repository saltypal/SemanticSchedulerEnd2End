# Ground truth for multi-UE semantic image sequence scheduling

25 October 2026. Research design; no new scheduler, episode campaign, image
annotation, VLM judging or dataset access is implemented or executed here.
Parameter extraction/validation is a parallel workstream and remains unchanged.
The accompanying ue_profile_drafts.json contains descriptive draft archetypes,
not deployed sources, priority labels or a validated workload.

## Recommendation

Use an immutable, annotation-backed event/observation ledger and an explicit
mission utility contract as the primary reference. Score the outcome of each
schedule, not its visual resemblance to chronological arrivals or one reference
permutation. Add calibrated/blinded VLM preference review as a secondary study,
and use existing human importance annotations where their task matches. Generated
simulation facts are a controlled mechanism study; real annotated sequences test
transfer. No new human reviewers are assumed available.

This design allows objective statements about selected tasks without claiming a
universal ground-truth priority order for unrelated user missions. If the only
reference is a model judgment, call the result model-assessed preference or
pseudo-label agreement, not independently verified operational utility.

## Literature methods and implications

### Outcome-defined semantic scheduling

Chen et al., Semantic-aware Transmission Scheduling: a Monotonicity-driven Deep
Reinforcement Learning Approach (2023): original full text inspected, system model
and reward/cost sections. Their example uses known remote-estimation dynamics and
an AoI-dependent estimation-MSE cost. There is no human master order of sensor
updates. A policy is evaluated by the declared application's cost. The wireless
transmission/control setting is different from a receiver's image decode queue.
Do not import their AoI penalty as a universal visual importance score.
https://arxiv.org/abs/2305.13706

### Fact/subtopic relevance, novelty and diversity

TREC 2011 Web Track: original evaluation report inspected. Assessors label
documents against query subtopics; diversity metrics such as alpha-nDCG/ERR-IA
reward covering aspects while avoiding repetition. The reference is a relevance
matrix, not one compulsory permutation. Pooled/incomplete judgments have limits.
Our fact-by-image matrix is an adaptation of that evaluation principle. Dynamic
arrivals, deadlines, decoder cost and reconstructed evidence need added definitions.
https://trec.nist.gov/pubs/trec20/papers/WEB.OVERVIEW.pdf
Foundational full-text introduction:
https://cormack.uwaterloo.ca/novelty.pdf

### Existing human temporal importance references

SumMe (Creating Summaries from User Videos, 2014): original methods/evaluation
inspected by bounded research delegate. Multiple human summary subsets are
available per source video; they do not prescribe a decoder service order or
universal emergency priority. TVSum (2015) provides existing shot importance
annotations. Source chronology, task, split and multiple-reference protocol must
be preserved. New reviewers are not required to reuse published annotations.
https://projet.liris.cnrs.fr/imagine/pub/proceedings/ECCV-2014/papers/8695/86950505.pdf
https://github.com/yalesong/tvsum

Otani et al., Rethinking the Evaluation of Video Summaries (CVPR 2019): original
paper inspected. The paper shows that random selection and segmentation choices
can obtain deceptively competitive conventional F-scores. Keep random/uniform
controls, and do not use a summary-overlap number as sufficient evidence of useful
semantic ordering. Summaries are subsets, not generally reordering policies.
https://openaccess.thecvf.com/content_CVPR_2019/papers/Otani_Rethinking_the_Evaluation_of_Video_Summaries_CVPR_2019_paper.pdf

### VLM/LLM scoring with independent evaluation

Lee et al., Video Summarization with Large Language Models (CVPR 2025): original
full text inspected, model and sections 4.1-4.3. A multimodal model captions frames,
an LLM scores local temporal context and an attention model refines importance.
Evaluation still uses SumMe/TVSum human references and rank correlations; they do
not establish correctness by asking the same model to approve its own ordering.
Full-video context in that offline summarization architecture cannot be given to
our causal runtime scheduler.
https://arxiv.org/abs/2504.11199

Keith and Jin, Meaningful Moments: A Corpus of VLM-Derived Temporal Importance
Labels for Video Action Recognition (October 2026 preprint, inspected v2): original
full text inspected, sampling protocol, downstream evaluation, section 7 limits.
The labels are explicitly VLM pseudo-labels. A frozen recognizer and independently
provided action class evaluate importance-led versus inverted/uniform temporal
sampling under matching budgets. The work reports no human temporal-importance
reference and conditions scoring on the known action label; it supports evidence
verification/sampling, not unknown-action discovery or universal image priority.
The design principle is especially relevant: validate usefulness against a fixed
external task answer, not solely judge self-agreement. Results are not transferred
to our dataset, device, image decoder or scheduler.
https://arxiv.org/abs/2610.16009

### Simulator references and real tracks

CATER: original paper/official label-generation code inspected by delegate.
Scene metadata specify instances, per-frame locations and action intervals.
Labels derive from simulator state. World truth is not automatically visible
evidence: an occluded object must not earn visible-discovery credit solely because
the simulator knows its location. Toy rendered scenes establish controlled
temporal reasoning only, not real disaster relevance.
https://github.com/rohitgirdhar/CATER

Habitat (2019): original task/evaluation sections inspected by delegate. Episodes
have scene/start/goal metadata and outcome measures such as task success/SPL;
they do not require matching one action sequence. This is an outcome-evaluation
precedent, not a proposal to replace our exogenous streams with navigation.
https://arxiv.org/html/1904.01201

BDD100K (CVPR 2020): original paper task descriptions inspected. The tracking
subsets provide temporal object correspondence in driving sequences. Supports
same/new tracked-object evidence and scene conditions actually annotated, not
automatic crash/medical-distress labels or cross-camera physical identity.
https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_BDD100K_A_Diverse_Driving_Dataset_for_Heterogeneous_Multitask_Learning_CVPR_2020_paper.pdf

### Judge reliability

MLLM-as-a-Judge (2024): original full text inspected, evaluation tasks, human
references, bias/consistency sections. It compares scoring, pair comparison and
batch ranking against human annotations, and finds stronger pair comparison
alignment plus position/length/self-preference and inconsistent judgments. It
does not validate arbitrary scheduler collages or every newer Qwen model.
https://arxiv.org/html/2402.04788v2
ACL 2026 judge study inspected for the earlier workstream:
https://aclanthology.org/2026.acl-long.703/

These sources answer different evidence questions. There is no inspected paper
that establishes the entire proposed heterogeneous-UE/contact-sheet benchmark as
already validated. Search was targeted to decision value, not an exhaustive survey.
Some publisher/project links failed; accessible author/official/arXiv full texts
were used. No dataset sample access or local model inference is established.

## The three references we must keep distinct

1. **Factual reference:** published/constructed annotations of what is present,
   where, when and which entity/condition it refers to. Unknown stays unknown.
2. **Utility contract:** application/operator definition of which facts matter,
   their timeliness and how unrelated missions share service. This is a declared
   policy choice, not a fact inferred by a classifier or a universal ethical scale.
3. **Policy prediction:** a parameter-only model's score/order/next action. Evaluate
   consequences against 1 and 2. Multiple actions/orders may tie.

Ground-truth scalar parameters or a perfect full reference permutation are not
necessary. We need labels/outcomes adequate for the claim. Published image-level
incident labels alone cannot certify that two photos depict the same physical
event. Curated exact copies and xBD crops can support controlled correspondence;
natural multi-view claims require real supporting labels or measured world state.

## Multi-UE missions and fair arbitration

Separate physical platform (phone/car/drone) from application (photo sharing,
inspection, monitoring). The platform name must not determine service priority.
The 12 draft archetypes can instantiate 10/20/50 device instances. They do not
create extra independent visual evidence or validated urgency labels.

Primary route: mixed sources/use cases serving one declared controller goal per
episode. This preserves the implemented one-profile queue's boundary while testing
sender diversity. Images not answering that controller mission may be irrelevant,
but this does not mean personal messaging is universally low value.

Separate heterogeneous-mission extension: all UEs have their own missions, with a
predeclared arbitration contract. One defensible control is equal weight per
active mission in macro normalized task utility. Equal utility weights do not
guarantee equal jobs/decoder time or prevent starvation. A true entitlement needs
a separate explicit resource/service constraint. Any emergency
tier/deadline guarantees must be explicitly operator specified. No VLM invents a
chat-versus-rescue value tradeoff. Report per-mission outcomes and starvation, not
only an aggregate. Duplicating a source should not silently increase its mission's
utility weight. A phone app can contain incident evidence too.

The current DecoderQueue validates one task/profile per queue. Mixed-profile
arbitration and natural-language missions are research extensions, not existing
runtime functionality. Do not mix profiles by bypassing validation.

## Data-backed episode record

Each episode has a fixed source manifest and immutable evaluator ledger containing:

- Scenario/seed, source video/event/site, parent and duplicate groups.
- UE instance, profile/mission version, task and arbitration contract.
- Frame image hash, source frame/time, capture time and complete-message ready time.
- Object/track/event-condition facts supported in that observation, with label
  source, valid time interval, visible/occluded/unknown status and uncertainties.
- Allowed age/deadline when explicitly supplied; state transition facts separately
  from repeated observations. Recording a newly damaged state can add evidence.
- Opaque payload identity, common received-noise artifact and service-cost basis.
- Transmitter KB state/provenance, and immutable header values/masks returned by
  the independent parameter pipeline.

Use real sequences/annotated tracks where possible. Videos supply sampled,
chronologically ordered image jobs, each independently image-encoded; inter-frame
video-codec dependencies are outside this experiment. Random unrelated images
form a controlled synthetic stream, not proof of natural camera chronology.

Cache image-independent vision outputs once per source/version. Compute
mission/history-dependent parameters using each episode's causal sender history;
do not reuse one cached novelty value after changing that history. In this replay,
the sender commits each offered observation immediately after its header has been
serialized, regardless of later transport/decoder success. The KB contains prior
offered observations, not acknowledged receiver knowledge. Source offer order and
warm-start history are fixed and independent of receiver policy; unused captured
frames do not silently enter the KB. ACK-conditioned sender adaptation would be
a distinct feedback experiment requiring causal, potentially policy-dependent
header regeneration. Sender KBs query before committing and do not use future
frames. All policies receive the
same pre-extracted arrivals/headers and independent received payloads. Receiver
success history evolves according to each policy, with no automatic knowledge of
facts preserved in a reconstruction.

Repeated reports across UEs are separate from sender-local novelty. Add verified
overlap only: the same source frame intentionally replayed by two UEs is an
explicit control, not a natural multi-camera capture. Use trusted identities only
if actually available at runtime; evaluator-only track/fact IDs stay hidden.

## How to generate episode diversity

Cover independently varied factors, rather than unrestricted permutations:

- UE counts 10/20/50, separately controlled offered load and decoder capacity.
- Mostly routine traffic, rare events, burst incidents and all-relevant queues.
- Low/high local repetition, local history cold/warm starts, and separately varied
  verified cross-source overlap.
- Same view repeated, viewpoint/lighting change with same facts, and complementary
  view with annotated additional evidence. Where correspondence is unavailable,
  mark the cell unsupported rather than let Qwen invent it.
- Short visible events, condition changes and delayed arrivals/deadlines where valid.
- Missing/incorrect header fields, genuine decoder failures and noisy reconstructions.
- Held-out videos/events/sites/domains and, separately, scheduler-unseen missions.

Keep release times causal. The scheduler may select only ready complete messages.
Keep task descriptions and service rules fixed before sampling outcomes. Group
source clips/events/parents/near duplicates before sampling frames or episodes;
thousands of episode seeds do not create thousands of independent real-world events.
Do not tune workload prevalence to force a learned model win.

Start with tiny supported fixtures and one bounded feasibility cohort, then run a
predeclared cached-feature campaign. Choose scale from runtime, effect size and
independent-group variability. Billions of rearrangements are not required to
establish a useful mechanism and can amplify one flawed generator. No arbitrary
episode count is a power calculation. Final evaluation groups remain locked.

## Metrics and references

Before episode execution fix an arrival cutoff T_arrival and observation horizon
T_obs >= T_arrival. Only complete-message arrivals up to the declared cutoff
enter the offered workload; no policy-dependent admission exclusions are allowed.
The scored time integral ends at T_obs. Jobs unfinished then receive no completed
evidence credit; optional diagnostic draining after the horizon cannot change the
score. Tail/guard duration is declared from workload/service assumptions before
policy comparison, never padded after seeing a policy's completion time. Report
late offered opportunities and infeasible deadlines separately. Raw area rewards
earlier opportunities by design; compare policies on identical episodes and report
discovery delay relative to each fact's first supported ready arrival. Report an
appropriate known-fact reference/feasibility ceiling where available; a larger-case
greedy reference is not a proven ceiling or guaranteed normalized maximum.

Static ready batches retain A_K and coverage@K as a regression/control arm. Use
an exact known-fact sequence reference only for bounded static cases (current
evaluator N<=12). Larger known-fact greedy is an ideal-information reference, not
proof of optimality. A dynamic full-future oracle is a clairvoyant bound and must
not be described as a causal policy baseline.

For dynamic streams use outcomes versus completion time, not only image count.
Let F_m be supported offered facts for mission m during the episode. Define C_m(t)
as the fraction first exposed successfully by t within the declared allowed age/
deadline. Write T = T_obs. For an equal-weight-mission-utility study, a candidate summary is

    J = (1/T) integral_0^T mean_over_defined_missions C_m(t) dt.

Weights can be 1 within a factual mission; any different severity/value choices
must be separately justified/predeclared. Fixed offered denominators are shared
across policies; report feasibility limits and undefined/zero-utility missions.
This measures timely discoveries. For current-state tracking, stale beliefs must
instead be evaluated against a time-varying state-accuracy/error reference; static
union coverage does not establish accurate current situational awareness.

Report independently:

- Timely distinct-fact recall/coverage and discovery delay from the first complete
  ready observation supporting the fact. Do not invent physical event onset from
  an image's timestamp.
- Redundancy, missed short events, deadline misses and per-mission starvation.
- Downstream question/classification/detection accuracy against published answers,
  using one frozen receiver task model and the same decoded-prefix budgets.
- Service/queue/policy/extraction costs and completion failures; source-evidence
  potential versus post-codec task outcomes.
- Optional alpha-nDCG/ERR-IA for a clearly defined static subtopic-labelled arm;
  Kendall/Spearman only where legitimate human importance references exist.

Source facts cannot silently become reconstructed facts. A frozen task model's
answers evaluated against independent answers support a claim about that measured
pipeline's task accuracy, not a guarantee that all visual facts are human-verified.
If reconstruction fact retention has only VLM assessments, say model-assessed.

All policies share source arrivals, headers, payload/noise, decoder costs and
permitted observations. Offline predecoding/cache-based simulation is labelled
replay, not repeated live inference. If optimizing J with RL, rewards must match
the actual piecewise time integral; using final coverage or wrong duration
accounting changes the objective. Current A_K reward rule does not automatically
transfer to variable-duration continuous-time episodes.

## VLM review and contact sheets

Contact sheets are useful audit displays. Arrival chronology is not a correct
priority order, and image-to-image collage similarity is not a schedule metric.
An improvement intentionally changes order. Large sheets can erase small damage,
text and relational evidence, and single layouts confound visual order/position
preferences with decision quality.

Recommended review unit: a short window of a few ready alternatives and the
permitted past, or two anonymized policy prefixes with matched time/frame budget.
Provide mission/service contract, capture/ready/completion times, neutral IDs and
zoomable original frames where needed. Offline factual annotation may examine
the full source, but a claim about a decision at time t cannot reward selecting
frames not ready at t. Exclude policy names, predicted priority scores and biased
explanations. Allow A/B/tie/insufficient-evidence and preserve disagreements.

Qwen outputs structured evidence judgments before scalar preference:

    episode_id, decision/window_id, mission_id, reviewer_version,
    evidence_supported, evidence_contradicted, unknowns,
    repeated_facts, additional_supported_facts,
    timing_violations, preferred_prefix_or_tie, rationale_refs.

Separate source annotations from pseudo-annotations, and freeze the evidence
catalog before scoring policies. Store original outputs, prompts, model/processor
versions and input hashes. Evaluate judge agreement against published facts or
existing human references, plus controlled duplicate/new-fact cases. Swap left/
right policy placement, image layout/thumbnail resolution and neutral UE IDs;
measure flip/abstention rates. These tests diagnose a judge, not certify all tasks.

Use a fixed, representative audit sample and all predeclared failure categories,
not only favorable comparisons. Additional review of worst cases is separately
labelled diagnostics. Model consensus is not an independent human/world reference.
Teacher/judge overlap and frozen-producer training exposure must be disclosed.

If using VLM-derived temporal importance labels for training, separately test
downstream task outcome under importance-led, inverted, shuffled and uniform
selections with equal budgets, following the mechanism in Meaningful Moments.
Do not give the runtime actor a true action/class label that is supposed to be
unknown. Exact labels, logical queries, clock validity and serialization are
handled by code. Luna can execute validation, prepare audit sheets and analyse
errors; model opinions are not oracle labels.

## Artifacts and immediate implementation boundary

Proposed episode artifacts: profiles.json, missions.json, source_manifest.jsonl,
evaluator_facts.jsonl (restricted), arrivals.jsonl, parameters.jsonl, decisions.jsonl,
completions.jsonl, task_outcomes.jsonl, judge_reviews.jsonl and run manifest.
Do not put evaluator_facts or raw images in policy observations. Persist episode
generation source/configuration, caches and their keys, splits, seeds, errors and
judge provenance. Avoid copying model outputs between distinct runs without hashes.

First implementation gate: one label-backed mission, 2-3 UE instances and a tiny
ordered sequence with repeat/new-fact/condition-change cases. Validate annotations,
causality, history, tie handling and outcome arithmetic before scale. Then widen
to supported real sequences and 10/20/50 UEs with the same generator/interface.
Heterogeneous-mission arbitration is a separate protocol choice, not an implicit
effect of profile paragraphs. Parameter-extractor improvements run independently
against a stable interface. No billions-of-episodes/VLM campaign is authorized by
this research note, and no such campaign has been launched.

The minimum truthful study can use native factual missions plus existing human
summary references and a declared simulated environment. It cannot establish
arbitrary real rescue urgency or natural-view identity without adequate evidence.
