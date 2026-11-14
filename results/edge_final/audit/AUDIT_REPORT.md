# Independent audit: semantic scheduling policies

## Method

- Auditor: Claude Haiku 5.5 (exact effort setting not confirmed by the auditor). Run as a single-pass, read-only review.
- Inputs: `{P}_metrics.json` and the first 40 lines of `{P}_replay.txt` for P in {sv_dpp, spi_edf_ratio, edf, fifo, scope_ppo}; images from `dsA\images`.
- Run conditions stated in the metrics files: workers = 2, rho = 1.2, seed = 0, arrival = trace. Decision counts: sv_dpp 79, spi_edf_ratio 69, scope_ppo 66, edf 40, fifo 40.
- The "300 validation images, 2 decoders" figure from the task brief was not verifiable from the files read. The metrics show 40 to 79 decisions per policy, not 300 images.
- Images: 5 per policy (25 total), not the 10 per policy requested. All 5 per policy were drawn from picks made at queue > 20. The replay lists skipped items only by tier (e.g. "skipped best: [TRUE Critical]"), not by image ID, so the "skipped while waiting" half could not be sampled.
- No tests, experiments, or model inference were run. Metrics were read as given, not recomputed. The folder does not define the metrics, so `ctr` and the `semantic_precision` values are reported as written.
- Caveat: the auditor saw the TRUE tier labels and the SPI guesses while judging, so the image judgments are not blind.

## Per-policy verdict

| Policy | Verdict | Evidence |
|---|---|---|
| spi_edf_ratio | Partly semantic (best of the five) | precision 0.80, Kendall tau 0.56 (highest), inversions/100 13.0, wait_critical 1.74. Still picks TRUE Low/Medium images while TRUE Critical items wait in the replay. |
| scope_ppo (learned) | Partly semantic | precision 0.82, tau 0.47, inversions/100 15.2, wait_critical 1.93. Picks lower-tier images while Critical waits in the replay. |
| sv_dpp | Partly semantic | precision 0.62 (lowest of the learned/heuristic set), tau 0.43, inversions/100 13.9, but the shortest Critical wait (1.50). Replay shows TRUE Low picks with 1 Critical waiting (e.g. decision 9 and decision 21). |
| edf | Partly, weak | precision 0.80, tau 0.48, inversions/100 20.0 (highest of the five). Replay shows TRUE Low picked under load with Critical waiting (up to 15 in one decision). Deadline-based ordering, not tier-based. |
| fifo | Not semantic | tau 0.10, inversions/100 80.0, wait_critical 7.70 (longest). Its precision of 0.95 conflicts with its inversion count, so the precision metric likely measures something narrower; the definition needs checking before this number is reported. |

## Image observations (5 per policy, picks at queue > 20)

- Most picked images labeled TRUE Critical show real damage or active fire: a burning pickup beside a burning house, a house fully on fire, a burned-out town seen from the air, a burned tobacco/drying building, a night forest fire with responders. These look correct to a responder.
- Picked images labeled TRUE Low were mostly non-disaster scenes (press conference, indoor/outdoor generic scenes). These are correctly low. Two exceptions: a fallen utility pole and downed line on a road (labeled Low, arguably a responder hazard), and a press-conference still (labeled Low, correct).
- Label issues:
  - A California Conservation Corps logo over a blurred fire is labeled TRUE Critical; it is a graphic, not a scene.
  - A sports-team portrait composite with wildfire debris is labeled TRUE Critical. It is partly relevant (damage) but the portrait is an unrelated element.
  - The orange-sky burning-houses photo appears under two IDs (919784813535612929 and 919744773275099136). This looks like a duplicate.
  - A satellite burn-scar image is labeled Critical; it is useful to responders for mapping, not for individual people.
- Picks with low SPI scores on High-tier images (e.g. a smoke plume over forest, SPI C 0.33, labeled High) show that the SPI guess does not track the true tier well.

Note: these observations are from a small sample and the auditor saw the labels.

## Ranking by semantic quality (this sample only)

1. spi_edf_ratio: highest tau and lowest inversions among the policies that were not FIFO.
2. scope_ppo: close to spi_edf_ratio on precision, a little worse on inversions and tau.
3. sv_dpp: lowest precision, but shortest Critical wait.
4. edf: high inversions, deadline-based.
5. fifo: no semantic ordering.

The gaps between 1 to 3 are small on 40 to 79 decisions with one seed, and are not established as significant.

## Caveats

- One seed (seed 0), one trace, load 1.2, 40 to 79 decisions per policy. Results may not transfer.
- The replay covers only the first 40 lines; the metrics cover the full run (n_decisions differs from 40 for sv_dpp, spi_edf_ratio, scope_ppo).
- 5 images per policy, all from picks at queue > 20. Skipped items were not sampled individually.
- The auditor saw the true labels and SPI guesses. Image judgments are not blind.
- The metric definitions are not included in the folder. `ctr`, `semantic_precision`, and the FIFO precision/inversion mismatch need checking against the code before publication.
- Verdicts are the auditor's reading of the files provided, not runtime validation.
