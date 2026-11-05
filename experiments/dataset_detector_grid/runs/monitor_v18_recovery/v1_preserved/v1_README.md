# MEDIC 300 development input (v18)

Private Kaggle input containing the exact 300 MEDIC images and original v18 records. `images.zip` extracts to `images/<sha256>.img`; the next notebook must verify each archive member against `input_manifest.json`, then safely extract to its run-scoped `input_images/` directory. `records.json` is the exact byte-preserved v18 acquisition record file; do not rewrite source image paths. Use the manifest's relative paths after extraction.

Source: Hugging Face `QCRI/MEDIC`, revision `bc2e6f3c99f511ce89055cad5a661c69e43ec6de`, train split; sampled by seed17 shuffled100-row pages. The source card declares CC BY-NC-SA 4.0. This managed dataset remains private.

`input_manifest.json` contains the original record hash, archive hash, image count, and every image path/hash/size. Native MEDIC labels remain in records. The v18 run did not emit an additional native_features schema.
