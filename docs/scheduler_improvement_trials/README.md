# Scheduler improvement trials

## What we are testing

The receiver has one decoder. When several images are ready, the next image should ideally tell the receiver something it still needs to know. A second picture of a road the receiver already understands might be less useful than a picture of a different road. That depends on whether the scheduler can recognize the repeat before decoding and whether the reconstruction actually supports a correct decision.

The main comparison changes one thing: whether the scheduling score uses completed receiver observations. Both policies must receive the same header, service estimate, duplicate gate and age guard. The unit of useful evidence is a verified road-segment condition within a defined time window. A true condition earns credit once and only after a correct reader output within its time budget.

## Day 1 — Development qualification, 7 November 2026

The branch starts from scheduler commit `43aa658b7e91f5ab574f3aba920454add918c085`. The original teammate checkout contained local changes, so the trials use a separate checkout. The protected scheduler test is closed. The 30-minute checkpoint is 15:00 IST.

### Reader and header-only ceiling

The first probe uses the existing 12-image class-stratified LADI sample. The sample was already used for development. It has native image-level `roads_damage` labels, which describe visible road damage. They do not provide a road-segment identifier, obstruction adjudication or an arrival trace. The frozen MIT Lincoln Laboratory reference reader uses its published road-damage logit. Its model card identifies the training split as the training source.

The probe compares three paths under the same 0.5 decision threshold:

1. Source pixels → frozen reader.
2. Source pixels → frozen reader → one-byte quantized score → receiver decision.
3. Source pixels → repository JPEG Q55 → actual bytes → reconstructed pixels → frozen reader.

The prototype packs the quantized score and a 64-bit difference hash into nine bytes. Those bytes exclude message identity and transport overhead. The hash describes coarse appearance. It is computed before dispatch and has no access to reference labels. Its availability alone does not validate repeat linkage. Header-only scoring pays for the full sender inference used to produce its score. JPEG is a byte-codec control while the pinned SQ01 bundle is being located. JPEG measurements cannot substitute for SQ-GAN measurements.

The first attempt stopped because the installed Pillow did not expose the method used to read the downsampled pixels. The corrected probe uses the grayscale buffer directly. The run receipt stores source, model, and manifest hashes and separate timings for sender scoring and header production, encode, decode, and receiver inference.

### Result from the initial 12 images

All three paths called all 12 images negative. Each matched 11 native negatives and missed the only native positive. The positive's source score was 0.00213 and JPEG-reconstruction score was 0.00250. Both are far below the fixed 0.5 threshold. The header rounded the source score to one quantization step. The result shows that the initial reader setting failed its only positive example. Identical decisions here do not establish that a header-only system solves the task.

The median local CPU time was 1,623.7 ms for source image preparation, scoring and header creation, 47.8 ms for JPEG encoding, 61.9 ms for JPEG decoding and 1,505.9 ms for the receiver reader. The median payload was 236,698 bytes. All 12 local images lacked capture timestamps and GPS in the inspected EXIF fields. The first batch therefore cannot qualify chronology or road linkage.

The follow-up uses a separate, label-balanced public LADI validation batch and keeps the test split closed. Sampling is conditional on a visible road and is for development diagnostics. No population precision or field prevalence will be inferred from it.
### Input qualification

- LADI supplies native image-level road-damage labels and can support a bounded reader study. The local sample cannot establish same-road identity.
- RescueNet supplies blocked and clear road labels from Hurricane Michael. One disaster cannot support a multi-event claim by itself.
- The CRASAR-U-DROIDs road benchmark supplies aligned road annotations over ten disasters. Its orthomosaics and aligned labels are a candidate source for matched road views. Orthomosaic order is not a receiver arrival trace. Repeated views need independent linkage and must remain separate from runtime observations. Access to file listings establishes availability of the listing only. It does not qualify the dataset or any model for this trial.

Sources: [LADI dataset](https://huggingface.co/datasets/MITLL/LADI-v2-dataset), [frozen reference reader](https://huggingface.co/MITLL/LADI-v2-classifier-small-reference), [RescueNet](https://doi.org/10.1038/s41597-023-02799-4), [CRASAR road benchmark](https://ojs.aaai.org/index.php/AAAI/article/view/41248), [CRASAR dataset](https://huggingface.co/datasets/CRASAR/CRASAR-U-DROIDs/tree/4c2732a7ab9658517ca8caa69bfee9e1f87ca532).

## Day 2 — Freeze before opening a holdout

Pending qualification. The frozen manifest needs per-event roles, source and duplicate groups, evaluator-only road identities, decision budgets, a practically worthwhile gain, and false-report and tail-wait guardrails. Record reader and codec hashes before inference. A physical arrival trace and a chronological capture replay are different data sources. When only capture chronology is available, report results as controlled replay and label the timing assumptions.

## Day 3 — Matched development replay

Pending qualification. Use the same offered jobs and outcomes for FIFO, the incumbent priority rule, the static duplicate-aware policy and the receiver-state policy. Admit only complete arrivals. Update state only from receiver outputs, including erroneous reports. Failed, late and unfinished jobs earn zero evidence credit. Test that the policy interface cannot access road truth, future arrivals, or future outcomes. Measure sender overhead separately from the receiver bottleneck.

## Day 4 — Holdout

Closed. Once frozen, evaluate it once. Pair differences by event, report leave-one-event-out sensitivity and keep actual repeats separate from injected repeats. Replayed seeds share evidence and do not add independent events.

## Day 5 — Decision

Pending. The claim requires a useful gain over the matched duplicate-aware control after overhead and without violating the false-report and starvation guardrails. If the header-only path already supplies equivalent task outcomes, choose a task that needs visual evidence or stop the decode-scheduling claim. A Q1 submission decision requires evidence beyond a small development probe.


