# Version6 compact results audit
All188supported balance/pair configurations completed (48original,44weighted,48oversampled,48undersampled). Class-weighted kNN unsupported.48unique encoder/head pairs evaluated after development-selected balancing.

Development-selected winner: siglip2_b16 + rbf_svm, class_weighted; development macroF1 0.6854. Test macroF1 0.5388, accuracy 0.8485, recall Low/Medium/High/Critical [0.8202247191011236, 0.0, 0.8888888888888888, 0.75]. Test support [89, 6, 198, 4].

All48 pairs have zero Medium test recall. Balancing did not solve minority generalization. Label quality, very low independent Medium support and rubric separability remain unresolved. Not ready for fourclass promotion. Targets are VLM pseudo-labels, not independently verified urgency.

Highest test macroF1 observed was SigLIP2 gatedABMIL .6217; selecting it now would be test-informed. It is descriptive, not the pretest-selected winner. Test critical support is only4; one example changes recall25points. Exact/tweet grouping does not establish event/near-duplicate independence. Retuning against this test requires a new final holdout.

Fullpipeline profiles saved remotely per logs. Local cost_scope.json contains stale limitations predating fullpipeline execution; actual profiling files must supersede those statements where supported. Retrieval/profiler-coverage audit pending. Do not call unavailable or partial FLOPs exact.
