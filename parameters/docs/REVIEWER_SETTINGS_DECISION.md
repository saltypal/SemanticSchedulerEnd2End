# Reviewer settings: measured choice and corrective experiment

## Completed follow-up and recommendation

Run g completed all72calls in478.72internal seconds. All109checksum-listed
artifacts verify. Main recomputation matched all6saved metric rows exactly;
per-case raw files, modality receipts and fresh-group isolation also checked.
The logging repair produced no closed-file errors.

Preferred **pilot candidate**, with full qualification withheld:

| Setting | Value |
|---|---|
| Model | Qwen3-VL-4B-Instruct, pinned revision below |
| Prompt | candidate_provenance_explicit_v4 |
| Output budget | 192tokens |
| Image resize | Aspect-preserving contain, longest side512 |
| Effective pixel cap | 262144 |
| Generation | Greedy,do_sample=false |
| Model execution | FP16,SDPA,cuda:1 |
| Assessed claims | Mission scope |

Loadable settings and qualification status:
`config/reviewer_pilot_recommendation.json`. This records a diagnostic candidate;
it does not change global defaults or promote an unqualified evaluator.

| Fresh24case result | Original v2 | Explicit v4 |
|---|---:|---:|
| Valid | 24/24 | 24/24 |
| Joint-correct verdict pairs | 19/24 | 20/24 |
| Consistency balanced accuracy | .9375 | 1.0000 |
| Sufficiency balanced accuracy | .84375 | .87500 |
| Contradicted descriptions rejected | 7/8 | 8/8 |
| Incomplete descriptions rejected | 4/8 | 4/8 |
| Full engineering gate | fail | fail |
| Review-cell wall time,including persistence | 132.57s | 141.24s |

Paired cases:19both correct,4both wrong,1v4-only correct,0v2-only correct.
This is one improved case across8groups, not demonstrated general superiority.
V4 exposed diagnostics were5/6image and6/6text-only; both used6/6valid responses,
while v2 used5/6valid in each mode. Text-only is not a vision qualification result.

### What the remaining failures mean

All4remaining v4errors are building-absent images with candidate claims containing
only `building.present:false`. It declares sufficiency despite the missing
`building.affected_or_greater` key. The strict explicit-key gate is therefore
failed and its scores remain unchanged. No threshold or answer was revised.

These cases expose a construct distinction. A declared ontology in which
`affected_building implies building_present` could allow a decoder to recover
the negative condition from a negative presence claim. That is representational
recovery, and is not proof that the predicted presence claim is factually correct.
Partial annotations, detector non-detection and unknown values cannot automatically
establish actual absence. The raw replies are compatible with logical inference,
reference copying or visual substitution; they do not prove which mechanism acted.

Before another GPU grid, specify separately:literal key completeness checked by
code;logical closure under a versioned true/false/unknown decoder;and correctness
against supported native facts. Preserve raw VLM scores separately. Include
building-present/condition-missing controls (a genuinely unresolved condition),
incorrect-presence cases,unknown labels and inconsistent claim pairs. Changing
this contract requires a separately labeled prospective experiment, not a
retroactive claim that g passed. Use the VLM for image-dependent assessment where
it contributes information. No additional GPU run was launched.

Evidence:`results/ops/20261025_reviewer_followup/verified_g/verification.json`;
runtime artifacts and109file integrity audit are mapped in KAGGLE_RUNS.md.

## Archived pre-follow-up decision

Grid f completed all192reviews over8settings on8paired image groups. None passes
the predeclared qualification gate. There is no validated best configuration.

## Best existing diagnostic reference

Qwen3-VL-4B-Instruct, revision ebb281ec70b05090aa6165b016eac8ec08e71b17;
FP16,SDPA,GPU1,do_sample=false; v2 prompt,192output tokens,512-side contain,
262144effective pixel cap,mission claim scope. This gives18/24 jointly correct
valid verdict pairs and8/8contradiction rejection. The higher-resolution v2
gives17/24jointly correct and7/8contradiction rejection, though its sufficiency
balanced accuracy is better (.8125 versus .78125) and it has one fewer invalid
response. Both omit3/8required omission rejections. Thus no universal winner or
qualified extraction evaluator follows. Low-resolution reference case-inference
sum124.76s versus143.60s at higher resolution; excludes loading/setup.

Both token limits give identical aggregate results. All6invalid responses have
empty explanations and33generated tokens, with hit_token_limit=false. Increasing
max_new_tokens cannot explain or repair this observed failure. Concise v3 is
rejected:its booleans accept all8contradicted candidates, sometimes while its
reason explicitly identifies the contradiction.

## Root causes supported by evidence

1. Prompt/parser mismatch:v2 asks for a reason naming missing/conflicting fields
   even on successful complete cases, while the parser requires a nonempty reason.
   Versioned v4 explicitly specifies a success sentence. Historical v2 is unchanged.
2. Source attribution:v2 sometimes claims an omitted condition was explicitly
   supplied. Raw example95158f1ccbcaea5f-control-0-incomplete.txt invents the missing
   affected_or_greater key. New instructions reiterate that another false field
   does not populate the missing key. This improvement remains a hypothesis.
3. V3 derived-booleans failure:its reason can identify conflict while consistency
   remains true. Additional output tokens/resolution did not repair it.

The control expectations agree with exact supplied keys and declared protocol;
no swapped candidate/reference, inverted native fact or expected-answer leak was
found. However strict explicit-key completeness is a narrower construct than
semantic sufficiency:logical implications and compressed representations require
their own documented decoding rules. Do not equate this control score with
general visual understanding or universal parameter sufficiency.

## Smallest useful next experiment

Run72calls on the same pinned model and fixed decoding/image settings:

- Six exposed development cases (one absent-building and one affected-building
  parent, each complete/incomplete/contradicted), both prompts, image and text-only
  modes:24calls. Text-only is explicitly diagnostic and cannot validate vision.
- Eight new paired groups,24cases, both prompts with images:48calls. Exclude all
  56prior c/e/fcontrol groups. Freeze prompts before opening those results.

Keep existing 95%valid,80%balanced-accuracy,75%each-class and75%omission gates.
No parser relaxation, response rewriting, extracted-truth injection or automatic
selection. Report joint-correct cases, both endpoint matrices, invalid reasons,
per-case timing and paired failures. Fresh groups become development-exposed
after inspection. Do not rerun until a concrete case-level mechanism warrants it.

Budget:server1200s/internal900s,two assigned T4s,reviewerGPU1,GPU0idle,maximum
0.667aggregateGPUhours. No additional image download or model family expansion.

## External configuration check

The pinned official model generation_config uses sampling by default:
do_sample=true,temperature=.7,top_p=.8,top_k20. Our explicit greedy override is
deliberate for repeatability; it is not a model-loading bug. These defaults do not
establish the best decoding for this exact control protocol. This corrective
experiment keeps greedy decoding fixed to isolate the prompt change.

Source:https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/ebb281ec70b05090aa6165b016eac8ec08e71b17/generation_config.json

## Current local evidence

Full suite68tests passed in9.39s after follow-up implementation. Actual CPU
tokenizer/processor preflight then passed visual and explicitly text-only input
construction, with no model weights or GPU inference. Visual pixel cap applies
to actual grid areas; text-only inputs contain no pixel_values. Runtime inference
and actual fresh-control outcomes remain unmeasured until the notebook executes.
