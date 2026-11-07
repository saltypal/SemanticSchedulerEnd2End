# UE streamer and rate study

The existing decoder Environment, policies and feature registry remain shared. Evaluation/ue_scenarios.py adds mixed, mission-conditioned and manifest-supported incident stream modes. The legacy make_episode defaults remain the training default. Causal relevance/novelty construction is shared through received_features; sender KB updates follow each offered capture, independently of receiver decisions.

## Rates

Let lambda_u be each active UE's offered images per time unit and d the mean service time. Decoder capacity is1/d, and offered load rho=sum(lambda_u)*d. Fixed-total tests hold rho constant while UE count changes. Fixed-per-UE tests hold lambda_u constant and allow rho to rise with population. Joins/pauses reduce achieved arrivals; always report realized rate/load as well as requested rho. A middle-window burst factor4 is normalized to preserve expected whole-window traffic before availability effects. These are synthetic time units unless service time is calibrated; do not label them images/second or infer physical vehicle speed.

The run includes12/50/100 UEs, rho0.5/0.9/1.2, per-UE rates0.01/0.02 per unit, burst factors1/4, one seed, two image-pool modes and four reference policies.60 fixed traces yield240 paired policy rows. No model fitting, GPU run or final-test evaluation occurs. One seed is a pilot, not statistical superiority.

## Content and chronology

Each UE receives one of15 mission templates. Mixed mode draws from the whole pool. Mission mode draws from declared mission pools with probability0.8 plus background traffic. Without a curated manifest, the rate driver builds CLIP upper-quartile mission pools and labels them model proxies. This is not independent mission-relevance ground truth.

Incident mode uses one declared shared incident per episode and requires supported sequence/evidence/condition annotations. Routine traffic precedes the incident; additional views and changed conditions advance under one shared timeline. Background observations continue during the incident. Exact-repeat checks never replace a new condition with an obsolete image. Source scenes, identities and annotations remain evaluator/scenario-only metadata; actor snapshots contain the same permitted numeric evidence as before. Geographic trajectories are not fabricated for unrelated photos.

Real incident mode is unavailable when no supported manifest exists. A positive fixture test is not a claim that real camera-view correspondence was obtained.

## Manifest

JSONL rows require image_id, kind (routine/distractor/additional_view/condition_change), missions (registered profile IDs) and provenance. Related views/changes additionally require incident_id, sequence_id, sequence_order and evidence_id. Condition changes require condition. Every selected mission must have sufficient routine and incident-stage support. Split groups first; the training CLI filters annotations to the training or validation cohort before constructing streams. Never build final labels from runtime predictions.

Example controlled fixture (not real data):

```json
{"image_id":"example","kind":"condition_change","missions":["damage_drone"],"provenance":"declared synthetic fixture","incident_id":"demo","sequence_id":"demo-seq","sequence_order":2,"evidence_id":"building-state-2","condition":"changed"}
```

## Training integration

train_real_suite.py accepts --stream-mode mixed|mission|incident, --scenario-manifest PATH, --stream-duration and --stream-burst-factor. It uses the same Episode interface and existing model/loss implementations. Non-mixed semantic modes require a valid mission/incident manifest; missing support fails explicitly. The legacy default is unchanged. The real-data rate study can be launched with ue_rate_study.py --archive ... --vector-extension ... --mission-vectors ... --output UNIQUE_DIRECTORY; optional --annotations enables supported incident mode.

## Outputs and validation

rate-pilot-20261101/ contains source/input manifest, exact offered trace metadata, per-policy metrics, CSV and execution logs. Metrics include intended/realized load, peak and time-average queue length, restricted waiting including unselected reports, unselected fraction, per-tier delay/miss counts, synthetic-time delay cost and selection milliseconds. Queue waiting is measured until dispatch; decoder service remains nonpreemptive. The tail window is explicit and reported; do not confuse capped waiting with an uncensored latency average. Identical episodes are reused across policies.

Regression results are stored separately under runs/regressions-20261101/. No natural temporal semantic or learned-policy superiority claim follows from the pilot.
