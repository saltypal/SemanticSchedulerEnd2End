"""Bounded image-only extraction; native labels stay in a separate source manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--allow-model-download', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.limit <= 10 or (args.output.exists() and any(args.output.iterdir())):
        parser.error('preflight limit must be 1..10 and output must be new/empty')
    args.output.mkdir(parents=True, exist_ok=True)
    from Archived.owners_before_simplification_20261027.SemParam.decoder_extractors import FrozenImageExtractor
    from Archived.owners_before_simplification_20261027.SemParam.decoder_context_profiles import STUDY_PROFILES, KNOWN_PROFILES, TRANSFER_PROFILES
    from Scheduler.training.decoder_v3_study import write_json
    from Evaluation.decoder_datasets import sha256
    import sys
    selected_profiles = {key: STUDY_PROFILES[key] for key in KNOWN_PROFILES+TRANSFER_PROFILES}
    extractor = FrozenImageExtractor(args.device, local_files_only=not args.allow_model_download,
                                     profiles=selected_profiles)
    started = time.perf_counter()
    try:
        rows = json.loads(args.records.read_text(encoding='utf-8'))[:args.limit]
        if not rows:
            raise ValueError('no source images')
        write_json(args.output/'provenance.json', dict(interpreter=sys.executable, python=sys.version,
                                                      manifest_sha256=sha256(args.records), requested_device=args.device,
                                                      local_models_only=not args.allow_model_download,
                                                      output_scope='frozen_image_predictions_not_truth',
                                                      profiles=list(selected_profiles)))
        from PIL import Image
        count = 0
        with (args.output/'features.jsonl').open('x', encoding='utf-8') as handle:
            for row in rows:
                print(f'Extract image-only evidence {count+1}/{len(rows)} id={row["image_id"]}', flush=True)
                path = Path(row['image_path'])
                if sha256(path) != row['image_id']:
                    raise ValueError('source image checksum differs from adapter identity')
                with Image.open(path) as image:
                    evidence = extractor.extract(image, row['image_id'])
                # No source labels, split roles or native facts enter ImageEvidence.
                handle.write(json.dumps(evidence.to_dict(), allow_nan=False)+'\n')
                handle.flush()
                count += 1
                write_json(args.output/'progress.json', dict(count=count, statuses=evidence.statuses,
                                                            elapsed_seconds=time.perf_counter()-started))
        write_json(args.output/'status.json', dict(status='complete', count=count, device=extractor.device,
                                                  elapsed_seconds=time.perf_counter()-started))
    except Exception as error:
        write_json(args.output/'failure.json', dict(type=type(error).__name__, message=str(error),
                                                   traceback=traceback.format_exc()))
        raise
    finally:
        extractor.close()


if __name__ == '__main__':
    main()
