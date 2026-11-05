# Local YOLO-World / Grounding DINO experiment

Real cached images; descriptive outputs only. No box ground truth, mAP or urgency accuracy.
Each image ran in a separate process: load and first inference are separate, not warm steady-state timings.
Same four explicit nouns: person, car, building, bridge. This is a local vocabulary variant, not the runtime16-category default.

| Model | Image ID prefix | Flip | Status | Instances | Categories | Largest area | Union area | First extraction incl. image I/O (s) |
|---|---|---|---|---:|---:|---:|---:|---:|
| grounding_dino | 067728640b | False | ok | 2 | 1 | 0.0001504132499715063 | 0.0002914306730478937 | 48.93072730000131 |
| grounding_dino | 1203c91f2d | False | ok | 3 | 2 | 0.16011549543577197 | 0.16011549543577197 | 41.91673610010184 |
| grounding_dino | 01adbd5612 | False | ok | 1 | 1 | 0.014707657713679637 | 0.014707657713679637 | 28.51453019992914 |
| grounding_dino | fa0a011bd6 | False | ok | 8 | 1 | 0.004788368719337837 | 0.015812243829032707 | 46.34129310003482 |
| grounding_dino | 8be286a38e | False | ok | 20 | 2 | 0.20606548022672022 | 0.6778014174500606 | 39.79591970006004 |
| grounding_dino | c40233fcdf | False | ok | 1 | 1 | 0.4322045205056078 | 0.4322045205056078 | 27.707433699979447 |
| grounding_dino | 1203c91f2d | True | ok | 4 | 2 | 0.16254748130658062 | 0.16329187238727194 | 46.275270999991335 |
| yolo_world | 067728640b | False | ok | 0 | 0 | None | 0.0 | 0.557151300017722 |
| yolo_world | 1203c91f2d | False | ok | 0 | 0 | None | 0.0 | 0.5033104000613093 |
| yolo_world | 01adbd5612 | False | ok | 0 | 0 | None | 0.0 | 0.522141500026919 |
| yolo_world | fa0a011bd6 | False | ok | 6 | 1 | 0.005603682565087091 | 0.01451961859421409 | 0.5138384999008849 |
| yolo_world | 8be286a38e | False | ok | 3 | 1 | 0.08055461541970033 | 0.08727160208537368 | 0.5439871000126004 |
| yolo_world | c40233fcdf | False | ok | 0 | 0 | None | 0.0 | 0.437180700013414 |
| yolo_world | 1203c91f2d | True | ok | 0 | 0 | None | 0.0 | 0.501654899911955 |

Mission views are computed from the same boxes, without another inference. They measure the extent of selected detected nouns, not mission urgency.
Grounding DINO phrases may combine nouns; ambiguous phrases are not silently mapped into one target category.
A missed detection cannot establish absence. Scores and thresholds differ by producer and are not comparable probabilities.
Native preprocessing budgets differ (YOLO640 versus DINO native defaults); this is not a matched-FLOP comparison.
Flip consistency is a sensitivity diagnostic, not correctness. Different counts alone cannot select the better detector.


## Main-thread visual observations (AI diagnostic, not reference annotations)

- Image5 is a map/infographic. DINO's retained `person` box covers43.22%of the
frame; visual inspection does not support that person detection. YOLO returned
zero queried objects. Neither output says whether the infographic's message
requires attention.
- Crowdimage4: DINO returns20boxes withnestedpersonhead/bodyregions; these are
not20verifieddistinctpersons. YOLO returns3personregions although other people
are visibly present. StandardIoUNMS doesnot guarantee oneboxperentity.
- Bridgeimage1: DINO returnsabridgeregion; YOLOreturnsnoqueriedobjects atthese
settings. Thisisusefullocalizationevidence, not proofbridgefailure/submersion.
- Bridgeflip: DINOcount3->4 andunioncoverage.160115->.163292; YOLO0->0. Consistent
emptyoutputisnotcorrectness, andcoverageclosenessdoesnotverifyobjectidentity.

## Provisional practical interpretation

KeepYOLOaslow-costbaseline; useDINOasexplicitoptionalofflinequery/localization
producer. Neither is universal spatial truth. Do not promote DINO merelybecause
it hasmoreboxes. Query/profileapplicability, missing/no-retained semantics and
counterexampletests must remain visible. GroundingDINOisnot a SPIverification
oracle; mapcase demonstrates how object-area-to-urgency assumptions can fail.
No runtimeparameterdefinition, notebook9 orproductiondefaults were changed by
these localexperimentalresults. Sol High reviewed the outputs; see HIGH_REVIEW.md for findings and boundaries.


Timing clarification: first_inference_seconds in the raw files includes image loading/conversion, optional flip, detector inference and postprocessing. It excludes separately measured model loading and overlay rendering. These are first-extraction measurements, not pure forward latency.

Inherited prior SPI labels in source manifests are evaluator-only historical sampling metadata; detectors receive only images and the fixed four nouns.


Setup note: the initial Ultralytics worker logged a library settings initialization/reset warning in the user-profile configuration location. A preexisting settings snapshot was not captured. The helper now sets YOLO_CONFIG_DIR to the experiment for future runs; the executed pre-change helper is preserved in local_detector_setup_20261028. This post-run path-isolation change was syntax-checked only and did not generate the measurements above.
