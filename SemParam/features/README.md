# Parameter extraction

- `clip.py`: lazy image/text embeddings.
- `observation.py`, `extractors.py`: observations and independent estimators.
- `scheduler.py`: earlier P1-P7 feature representation.
- `latent.py`: whole-latent task relevance and confirmed receiver novelty.
- `ladi_prediction.py`: explicit cached classifier predictions.

Receiver history lives in `SemParam/history`, serialization in `SemParam/headers`,
and semantic codebooks in `SemParam/Codebooks`. Evaluation labels never become
inferred evidence implicitly.
