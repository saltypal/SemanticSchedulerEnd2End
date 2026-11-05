# WSL mini cache integration checkpoint

Execution: Ubuntu-26.04, user Satya. Canonical repository remains on /mnt/d; no second project copy.

Verified archive checksum and all6000 raw-image hashes;5952 unique raw images. All6000 scene records passed current93-class pure-contract validation and exact received-wire/record parity. Exported received_features.npz has6000x387 rows/features plus image IDs for joining outside model scoring. Group and exact-hash split crossings:0. Near duplicates remain unaudited. Source mix:1000 Open Images,1000 Places365,4000 CrisisMMD.

2069 pool observations share raw hashes with SPI-head training records. The whole pool is development material; existing source split names do not establish a fresh independent scheduler test. Reference records supply0 independently available urgency labels. Mission relevance and causal novelty are0 available; their values AND masks remain zero. No fabricated parameter values or SPI-as-ground-truth training.

Based on cached nonzero received scene values, proposed full93 sparse encoding costs mean34.4513,p9545,max73 scene bytes with a preconfigured profile, compared with369 self-describing bytes. These are calculated sizes, not a tested compact serializer, and exclude SPI/scalars/routing/profile provisioning/payload. This supports testing lossless sparse transport before dropping fields. Detector accuracy is still unmeasured.

No image extraction, model/teacher inference, training, codec, dependency install, new download or Kaggle operation occurred. The existing WSL semantic-kaggle environment has NumPy but no torch/sklearn; a proper training runtime and qualified queue/history preference records are outstanding. Current archive has image-level predictions rather than those contextual preferences. Existing legacy coverage-policy code is not relabeled as the frozen D/P temporal system.

Artifacts: manifest.json,execution.json,summary.json,feature_registry.json,received_features.npz,provenance.jsonl,run.log,artifact_manifest.json and operational capture logs. The v1 launch wrapper pre-created the unique output directory, triggering intentional rejection before work; preserved under mini-wsl-ingress-20261030-v1. Diagnosed v2 succeeded.

Reproduce with the existing WSL interpreter and a NEW output path:

```bash
/home/Satya/.venvs/semantic-kaggle/bin/python -u experiments/stream_scheduler/mini_cache_preflight.py --archive experiments/mini_dataset/runs/mini_spi_20261030_v1/downloads/Mini_dataset_6000.tar --output experiments/stream_scheduler/runs/<new-run-id>
```

Next component: receiver/cache ingress and exact D evaluator, followed by the fixed contextual preference records and temporal scorer. All subsequent source edits, runs and package setup execute through WSL as explicitly requested. Preserve the external SPI head and full cache.
