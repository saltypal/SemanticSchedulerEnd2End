# How published work obtains image priority labels

26 October 2026. Focus: direct image urgency, contextual semantic importance and the provenance of supervision. This is a bounded primary-source review, not an exhaustive survey. No classifier training or image-archive download was performed.

## Correction to the earlier recommendation

TREC-IS's modality limitation is real, but the earlier recommendation was too narrow: direct image-urgency research exists, as does semantic-communication work using VLM-generated importance with human validation. The SPI idea remains a research direction. TREC-IS cannot be promoted into a universal image-SPI dataset merely because its message labels have four levels.

The four-level prediction architecture previously proposed remains a candidate. Its label source must be established before training. This review supersedes the presumption that weak TREC message labels are our preferred image supervision.

## Direct precedents and what they actually did

### 1. DisasterNet — HICSS 2020

[Author's full paper](https://www.dhirajmurthy.com/wp-content/uploads/2019/11/DisasterNet-Pre-Print.pdf), DOI 10.24251/HICSS.2020.071, sections 3–5.

The authors sampled 1,128 Hurricane Harvey images and human-coded urgency using a literature-informed rubric: 0 spam/unrelated, 1 not urgent, 2 somewhat urgent, 3 moderately urgent, 4 highly urgent. VGG-16 features fed MLP classifiers. Urgency validation accuracy was 61.95%; Table 1 gives macro-F1 0.5847, while section 5.3 says 0.5841. The study used an 80/20 stratified random split within one hurricane. Its listed urgency-class counts sum to 1,127; the one-image discrepancy is unexplained. A public download of the original image/urgency labels was not verified.

**Our inference:** this is a direct precedent for learning ordinal urgency from images. It supports constructing explicit image-level labels, not copying tweet priority. Its single-event evaluation does not establish generalization. A consistent rename of its four non-spam levels would preserve ordering, but the meaning would remain hurricane-specific; its highest level cannot silently become universal Critical.

### 2. The related 2019 and 2023 studies

[Robertson et al., 2019, Progress in Disaster Science](https://doi.org/10.1016/j.pdisas.2019.100030), [author manuscript](https://www.dhirajmurthy.com/wp-content/uploads/2019/05/Robertson-et-al-Deep-Learning_PIDS_inpress.pdf).

This earlier study compared human-coded Hurricane Harvey imagery with a VGG-16/MLP pipeline for urgency and time period. It combines qualitative human interpretation with learned visual predictions. Treat it and DisasterNet as related work from one research group/event, not independent multi-domain replications.

[Johnson et al., 2023, Journal of Management Information Systems](https://doi.org/10.1080/07421222.2023.2172778).

The publisher's abstract explicitly reports human-annotated hurricane-image urgency, relevance, time period, damage and relief classification. Full methods were access-restricted in this check. Exact annotation/reliability details and dataset access are unverified; no numerical result is imported from a secondary summary.

### 3. SA-OOSC — 2025 preprint; 2026 ICNC version located

[Full paper and appendices](https://arxiv.org/html/2509.07436), [ICNC paper](https://www.conf-icnc.org/2026/papers/p928-zhang.pdf), [released annotations](https://github.com/xyfyyds/Semantic-Communication-Cityscapes).

YOLO detects driving-scene objects. GPT-4V receives the image, detections and task instructions and assigns low/medium/high object importance; background is separate. The resulting supervision trains semantic coding. For validation, three experienced drivers annotate objects in 100 Cityscapes images. Appendix I reports 92.57% overall agreement with its human benchmark and 86.10% for High. The paper's prompt includes explicit proximity and traffic-element rules. The repository names GPT-4o-preview, differing from the paper's GPT-4V terminology.

**Our inference:** VLM-assisted importance annotation and learning from it have a concrete SemCom precedent. Its evidence is about objects/patches in driving. Converting this to one urgency label per whole image is a new modelling decision needing evaluation. Reported agreement does not transfer to Qwen or our missions. The prompt's rules are policy assumptions, not universal laws of importance.

**Actual access check, not just a paper claim:** public GitHub tree at commit `ebb4d7239c9f4b8eca3258120b57e73ea281c3ee` was not truncated. Each of three human-annotation directories contains 100 JSON files; automated-importance and detector directories contain 100 each. Five matching JSON files for one image were retrieved and parsed (1,760 bytes total). They contain three human judgements, one automated judgement and seven detected objects. Judgements disagree on one object's level. This verifies sample structure/access, not full-dataset accuracy. No images were downloaded. Receipt: `results/phase0_trecis/literature_20261026/SA_OOSC_ACCESS_PROBE.json`.

### 4. VERI — 2025 preprint / WACV 2026

[Paper](https://arxiv.org/html/2505.15367), [official repository](https://github.com/Dasol-Choi/VERI-Emergency), [dataset](https://huggingface.co/datasets/Dasool/VERI-Emergency).

The study builds 100 synthetic danger/safe image pairs using scenario definitions, generation and iterative human verification; it adds 25 real-image pairs. Labels are binary risk, not ordinal priority. It reports substantial false alarms by VLMs. This supplies hard safe-versus-emergency controls, including visually similar counterexamples. It cannot independently validate Medium versus High versus Critical. Synthetic and real subsets must be reported separately.

**Our inference:** use it to test overreaction by a proposed SPI teacher/classifier, not as a fabricated four-level training set. The article has an internal count inconsistency in its creation paragraph; retain its headline 200-synthetic/50-real description without presenting a new count audit.

### 5. YOLO-based semantic communication — 2023

[Full paper](https://arxiv.org/html/2306.14138).

For an orchard digital-twin task, the authors derive priority from detector confidence and use it for power allocation. They do not obtain independent human urgency labels. This is a declared operational proxy for that task.

**Our inference:** it demonstrates that some SemCom papers define a score and evaluate a downstream objective rather than train a universal priority classifier. We should not adopt detector confidence as emergency urgency: an uncertain, partly occluded person may still require attention.

### 6. STEM — Autonomous Robots, 2026

[Full paper, section 3.2](https://doi.org/10.1007/s10514-026-10247-6).

The method creates an offline priority mapping from task-context object names using BERT embeddings, cosine similarity to the target and integer scaling. It evaluates target-search behaviour. These priorities estimate usefulness for finding the target, not human-labelled urgency.

**Our inference:** task-relative relevance can produce an operational priority signal without VLM image annotation, but it should remain our relevance component rather than become evidence of a four-level urgency classifier.

### 7. Semantic Feature Scheduling and Rate Control — TWC, 2024

[Publisher abstract](https://ieeexplore.ieee.org/document/10729742/), DOI 10.1109/TWC.2024.3480352.

The published abstract describes feature importance through contribution to task inference and schedules features/rates accordingly. Full technical derivation was not inspected; no specific importance formula or result is claimed here.

**Our inference:** task contribution is another meaning of semantic importance. Feature value for classification, whole-image response urgency and decode ordering are related research problems but require different labels/objectives.

## The three evidenced labelling approaches

| Approach | Training target comes from | Independent evaluation comes from | Appropriate claim |
|---|---|---|---|
| Human visual urgency | An explicit image annotation rubric and human judgements | Held-out human-labelled images/groups | Predicting the defined image urgency in tested domains |
| VLM-assisted supervision | Image + task + instructions produce pseudo-labels | Existing or newly collected human labels in the same target space | Distilling task-specific semantic importance with measured teacher agreement |
| Operational score | Declared class preferences, relevance or measured task contribution | Downstream task outcomes under controlled resource limits | Improving the specified task objective |

A VLM is not automatically invalid as a labeller. It is also not automatically a valid evaluator of its own labels. Another VLM's agreement is an additional model check, not a replacement for independent reference labels.

## Concrete route for our project

1. Preserve the minimal runtime outputs: SPI, relevance, causal UE novelty and trusted capture metadata. The image is the scheduled unit; object scores are optional internal evidence only.
2. Give **direct image-urgency supervision** precedence over TREC's message-level proxy. DisasterNet's corpus/rubric access is the highest-value remaining access question. No public label download was verified in this review, and contacting authors requires explicit authorization.
3. Use the released SA-OOSC human annotations to validate an importance-labelling mechanism in its native driving/object setting without requiring new human reviewers. Preserve native 1/2/3 labels. This is a supporting domain experiment, not ground truth for four-class whole-image urgency.
4. Use VERI's published human-verified binary distinctions as an independent overreaction diagnostic. Do not manufacture four ordinal grades from safe/danger labels.
5. If we extend training using VLM-produced whole-image SPI labels, declare the image, mission, annotation rubric, version, label provenance and unavailable cases. Train a compact student if worthwhile. Validate native visual concepts with published annotations and validate ordinal SPI only where ordinal reference labels exist. No teacher-label self-evaluation claim.
6. TREC-IS remains useful for message-priority comparisons and modality-gap analysis. MEDIC/CrisisMMD remain useful for their native visual tasks. Neither is silently relabelled into image urgency.

The original four-class classifier design can be implemented once the ordinal target is supported by actual image labels. Broader screenshot/personal-photo/industrial urgency would require matching labels or remain an explicitly model-assisted extension. The existence of valid domain-specific methods does not establish an already available universal SPI dataset.

## Decision and immediate action

The SPI direction is retained. The earlier dataset recommendation and confidence were premature. Replace the TREC-first assumption with a direct visual-label/access check, and use published human annotations as validation wherever their target matches. A VLM-assisted path is justified by SA-OOSC, with its validation requirement and domain limitations carried over explicitly.

Completed in this review: primary-paper method checks, reported-outcome/limitation extraction, publisher-versus-full-text coverage distinctions, and a pinned metadata-only access probe. No paid API, cloud run, model download, training or image-archive download occurred. No superiority or full reproduction result is claimed.
