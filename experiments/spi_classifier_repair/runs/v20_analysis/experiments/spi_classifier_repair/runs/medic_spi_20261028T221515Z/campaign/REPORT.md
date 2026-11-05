# MEDIC parameter study

Selected: clip_b32 + random_forest
Development macro-F1: 0.5921
Runtime promotion: False

Reference is Qwen pseudo-label agreement, not human urgency accuracy. Prior detector-study exposure is disclosed.
See comparison/coverage.json, release/development.json, release/locked_test.json, release/profiles.json and release/readiness.json.
Profiles distinguish CPU/GPU, partial supported FLOPs, memory scope and cold/setup costs. Unknown FLOPs are not zero.
Optional encoder fine-tuning/CORN were not included; this run compares frozen representations and learned heads.
A failed diagnostic/readiness gate prevents bulk final parameter caching. No scheduler/codec results are claimed.