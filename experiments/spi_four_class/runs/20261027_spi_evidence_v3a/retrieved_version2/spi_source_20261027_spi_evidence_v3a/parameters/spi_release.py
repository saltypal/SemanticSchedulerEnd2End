"""Build a gated four-class development release without overwriting history."""
import argparse
import hashlib
import json
from pathlib import Path
from .artifacts import write_json
from .spi import SPIClassifier
from .spi_labels import LEVELS


def candidate_release(model_path, validation_path, current_manifest, output, root=None):
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    model_path, output = Path(model_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('release output exists; preserve previous candidate')
    if not model_path.is_relative_to(root):
        raise ValueError('release checkpoint must be persisted inside the canonical project')
    classifier = SPIClassifier.load(model_path)
    if classifier.levels != LEVELS:
        raise ValueError('four actual trained classes required; historical three-class model rejected')
    report = json.loads(Path(validation_path).read_text(encoding='utf-8'))
    per_class = report['student']['per_class']
    checks = {'export_parity': report['export_max_absolute_difference'] <= 1e-4,
        'all_class_support': all(per_class[k]['support'] >= 5 for k in LEVELS),
        'per_class_recall': all(per_class[k]['recall'] is not None and per_class[k]['recall'] >= .70 for k in LEVELS)}
    manifest = json.loads(Path(current_manifest).read_text(encoding='utf-8'))
    manifest['spi'] = {**manifest['spi'], 'model_path': str(model_path.relative_to(root)),
        'model_sha256': classifier.artifact['sha256'],
        'file_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'supported_levels': list(LEVELS), 'unsupported_levels': [],
        'model_status': 'four_class_development_candidate' if all(checks.values()) else 'four_class_validation_failed',
        'score_calibration': classifier.artifact.get('score_calibration', 'uncalibrated'),
        'four_class_status': 'trained four-output artifact; independent urgency validation absent',
        'validation_reference': 'VLM teacher agreement, not independent human urgency truth'}
    manifest.update(status=manifest['spi']['model_status'], release_checks=checks,
        runtime_promotion_allowed=all(checks.values()), independent_urgency_validated=False)
    # Produce a candidate first; caller must preserve the previous selected
    # manifest before an explicitly justified installation.
    write_json(output, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--current-manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(candidate_release(args.model, args.validation, args.current_manifest, args.output), indent=2))


if __name__ == '__main__':
    main()
