# Original-algorithm-first comparison and Qwen correction

## Decision

Ordinary neural RankNet is the main learned comparison candidate again, not an
asserted winner. Linear RankNet and LambdaMART are essential strong alternatives.
CAHR means coverage-aware header ranker: the same small scoring network with a
coverage-swap-weighted objective. It is retained as an experimental arm, not an
established novel algorithm or mandatory deployment choice. Existing results,
validation selections, negative outcomes and source snapshots are unchanged.

## Your original table: actual implementation status

| Method | Status |
|---|---|
| FIFO | Implemented and evaluated, no fitting |
| Round robin | Implemented and evaluated, no fitting |
| SJF | Not part of the measured equal-cost study; actual decoder costs needed |
| EDF | No defensible application deadlines supplied; not claimed evaluated |
| Fixed semantic formula | Implemented, novelty coefficient selected on C |
| Priority classifier | Trained and evaluated |
| Pointwise regression | Trained and evaluated |
| Linear pairwise RankNet | Trained and evaluated |
| Ordinary neural RankNet | Trained and evaluated; default terminal demonstration |
| LambdaMART | Trained and evaluated |
| Listwise neural ranker | Deferred, as in the original table |
| Contextual bandit | Deferred, as in the original table |

The eight supported original approaches now have a single terminal comparison:
`python scripts/compare_original_schedulers.py`.
This uses existing trained artifacts, not another test-driven fitting campaign.
The tiny demonstration uses one UE; round robin consequently matches FIFO.
The full benchmark used four UEs. Neither demo ordering nor JSON validity proves
semantic correctness. Ordinary RankNet also put the repeat first in the displayed
example; restoring its name does not solve the underlying information limitation.

## Why it looked like everything failed

- Correct header parsing, queueing, checkpoint loading, fitting and held-out
  measurement work. Negative comparative results do not make that code fail.
- The experiment does not prove that a learned model beats an already strong
  native importance predictor on a small, sparse-label development set.
- A static per-item scorer has no access to pending-image pairwise relationships.
  Sender novelty describes prior sender observations, not which queued report
  already delivered a fact. Header-only information limits remain for all rankers.
- Qwen's old prompt mixed absence of damage with unassessability, and the old
  pilot checked format without establishing annotation usefulness. That was an
  inadequate acceptance gate. Prompt ambiguity is a hypothesis, not confirmed
  sole cause of the all-abstention outcome.

## Why 32 and 445

32 was an execution pilot size for Qwen, not an empirical sample-size justification
for semantic annotation. Its failure should have stopped expansion/quality claims.
There were 1,934 extracted unique images overall. The official validation set had
890 images: A=445 fit scheduler, B=178 calibrate extractor claims, C=267 select
models. The held-out test had 1,044 retained unique images. The frozen LADI reference
extractor was trained on the official training split; this study avoided treating
those same images as clean evaluation data.

445 is sufficient for a bounded software/learning pilot, not a validated sample-size
guarantee. Many image pairs or queues do not create independent disasters. A larger
campaign can use the official training images for scheduler fitting too: that is
not inherently forbidden. However, because the frozen extractor already trained
on them, its in-sample predictions may be easier than deployment predictions.
An out-of-fold extractor setup is a stronger way to control that mismatch. The
445-image choice was a conservative prototype scope decision, not a fundamental
limit of this approach. Additional data require a fresh external held-out set.
Reusing the inspected old test
for iterative improvement would contaminate the next comparison.

## Qwen repair experiment

A separate private Kaggle pilot, `hmmmkk/disaster-qwen-repair-pilot`, was submitted.
It runs the pinned Ollama/Qwen producer on a synthetic red-square/blue-circle visual
transport control and up to four role-A images from one validation shard. The control
is not disaster data. Each receives an independent plain-language image description
and a revised schema-constrained assessment. The revised rubric explicitly separates
visible absence of damage from insufficient visual detail. Image size is bounded at
1024 pixels, and all raw responses, hashes and errors are saved. No test images,
scheduler training, Gemini keys or paid calls are used.

This combined diagnostic can reveal whether useful vision/assessment works; it
does not isolate which changed factor caused improvement. Assessability is not
accuracy. Concrete claims still require image/native-label review. If it fails,
the primary scheduling benchmark remains on native LADI labels and calibrated
supervised image predictions, without pretending the VLM provided annotations.
Gemini is an optional later reviewer, not a way to create independent gold labels.
No particular $30 credit entitlement or available API balance has been verified.

### Measured repair-pilot outcome

Version 1 completed and its raw outputs, logs and exact source were downloaded to
`results/disaster_decoder_v3/20261023_qwen_repair_v1/`. Diagnostic inference took
135.645 seconds (not including setup/downloads). The visual control description
correctly named the red square and blue circle. Of four role-A images, two were
assessed for each task; the other two had no native building/road targets. Thus
universal abstention was removed in this small diagnostic. However, the one native
building-damage-positive image received evidence_grade=0, even though the prose
description mentioned damaged structures. All four road/flood critical labels
were negative, so this pilot cannot establish sensitivity on those tasks.

Decision: Qwen is NOT accepted as a priority-label generator. Use native labels
for primary training/evaluation and calibrated supervised extractor probabilities
for runtime features. Keep Qwen outputs as inspectable auxiliary proposals only.
The improved response pattern does not identify the sole failure cause because
prompt, resolution, schema and output budget changed together. No paid API calls
were needed. `semantic_audit.json` records comparisons and downloaded-file hashes.

## Relevant additional algorithms, not extra implementation commitments

- Ranking SVM: a regularized margin-based pairwise comparator, useful for small
  tabular feature sets. [Joachims, KDD 2002](https://www.cs.cornell.edu/~tj/publications/joachims_02c.pdf).
- ListNet: a listwise alternative to pairwise training; test after securing
  meaningful ranking targets. [Cao et al., 2007](https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach/).
- MMR: explicitly trades relevance against similarity to previously selected
  items. Standard MMR needs pairwise similarity not present in our eight scalar
  receiver fields. It belongs in an explicitly information-advantaged diagnostic,
  not a secretly richer matched baseline. [Carbonell and Goldstein, SIGIR 1998](https://www.cs.cmu.edu/afs/cs/Web/People/jgc/publication/MMR_DiversityBased_Reranking_SIGIR_1998.pdf).
- RankNet and LambdaMART remain established starting points, not inventions of
  this project. [Burges et al., ICML 2005](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/),
  [Burges, 2010 overview](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/).

First finish validating inputs and the original matched comparisons. Adding every
possible learner now would increase the search space without repairing label validity.
