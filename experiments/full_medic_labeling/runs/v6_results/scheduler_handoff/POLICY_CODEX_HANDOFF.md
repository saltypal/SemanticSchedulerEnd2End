# Parameter handoff for Policy Design Codex
Canonical repository: D:/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End
Use experiments/full_medic_labeling/runs/v6_results/scheduler_handoff/image_records.jsonl for1000source-qualified image/YOLOrecords. offline_references.jsonl contains teacherSPI and nativeannotations separately; these are not runtime studentpredictions. README.md documents missingfields and limitations. ../ANALYSIS.md has comparison outcomes. Nested version6 comparison/final_test_and_head_profiles.json and frozen_pair_selections.json contain48pair results and selections.

Selected on development: SigLIP2 google/siglip2-base-patch16-224 visionencoder + classweighted RBF-SVM head. Use exact encoder revision/preprocessing from downloaded comparison/features/siglip2_b16/identity.json; do not substitute CLIP vectors. Pipeline uses EXIF RGB aspect-preserving224pad grey127 and native normalization then L2normalizedglobalvector. Head expects embedding, not rawimage. Shared loading/encode code: experiments/spi_classifier_repair/medic_campaign.py.

Remote completedweights version6 ONLY: /kaggle/working/experiments/full_medic_labeling/runs/main_20261029T154606Z/comparison/class_weighted/siglip2_b16/rbf_svm.joblib. Remote encoderidentity: sameprefix comparison/features/siglip2_b16/identity.json. Localweightretrieval pending at handoffcreation; do not assume present. Version-specificSDK download required; latestalias may refer to activev7.

Test macroF1 .5388, accuracy .8485, Medium recall0. All48pairMediumrecall0. Not readyfor4classproduction. Use developmentexperiment only. Teacherlabels cannot be both runtimeinput and independent scoringtruth. Need actualpredictions from model, fold-isolated/OOF where required. Actualv6records.json split/group manifest must be joined by imageID; source native split differs from actualfit split.

Relevance/novelty were intentionally skipped: generate from declared missionparagraph and causalperUEhistory. Capturemetadata unavailable; simulatedmetadata must be identified. No implicitSPI-from-detectorrule. Detectorrecords contain boxes/counts/categories/largestboxfraction; no physicaldistance or urgency inference. Enforce same imagegroups across parameter and scheduler stages, preserve finalholdout and avoid queuesplit leakage.

Active4000campaign version7 sameKaggleidentity is RUNNING; do notstopit. It reuses1000labels, labels3000more then runs fourencoders/allheads whencomplete. Do not wait for it to build the scheduler's data/interface pipeline.
