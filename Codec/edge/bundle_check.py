"""Verify an SQ-GAN bundle folder without loading torch weights.

Repeats the checks of Codec/integration/sqgan_payload.py:38-45 (algorithm sqgan, pipeline_completed,
every listed file present with matching size and sha256, no path escaping the folder) and adds the
components the loader later needs: last.pt, the ten assets/checkpoint/*.ckpt (sqgan_payload.py:48),
portable_config.json with teacher_model/teacher_revision (sqgan_payload.py:90). ``scan_last_pt_keys``
greps the pickle inside last.pt for state-dict key prefixes (heuristic, no torch).
``--print-codec-yaml`` prints values to paste into config/edge/codec.yaml.
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

from Codec.integration.sqgan_payload import COMPONENTS, sha256


def scan_last_pt_keys(last_pt):
    """Return the required key prefixes found as raw strings in last.pt's data.pkl."""
    wanted = [b"core." + c.encode() + b"." for c in COMPONENTS] + [b"teacher."]
    with zipfile.ZipFile(last_pt) as z:
        names = [n for n in z.namelist() if n.endswith("data.pkl")]
        if not names:
            raise ValueError("last.pt is not a torch zip archive")
        blob = z.read(names[0])
    return {w.decode(): (w in blob) for w in wanted}


def verify_bundle(folder, deep=False):
    """Return ``{"ok": bool, "errors": [...], "manifest": dict|None}``. Never raises on bad content."""
    folder = Path(folder).resolve()
    errors = []
    try:
        manifest = json.loads((folder / "bundle.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"ok": False, "errors": ["bundle.json unreadable: %s" % exc], "manifest": None}
    if manifest.get("algorithm") != "sqgan" or not manifest.get("pipeline_completed"):
        errors.append("completed SQ-GAN bundle required (algorithm sqgan, pipeline_completed true)")
    listed = set()
    for entry in manifest.get("files", []):
        path = (folder / entry["path"]).resolve()
        listed.add(entry["path"])
        if not path.is_relative_to(folder):
            errors.append("path escapes bundle: " + entry["path"])
        elif not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            errors.append("size/sha256 mismatch: " + entry["path"])
    required = ["last.pt", "portable_config.json"] + ["assets/checkpoint/%s.ckpt" % c for c in COMPONENTS]
    errors += ["component missing from bundle.json files: " + r for r in required if r not in listed]
    cfg_path = folder / "portable_config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        errors += ["portable_config.json lacks " + k for k in ("teacher_model", "teacher_revision") if not cfg.get(k)]
    if deep and (folder / "last.pt").is_file():
        try:
            errors += ["last.pt has no keys starting " + k for k, ok in scan_last_pt_keys(folder / "last.pt").items() if not ok]
        except (zipfile.BadZipFile, ValueError) as exc:
            errors.append("last.pt scan failed: %s" % exc)
    return {"ok": not errors, "errors": errors, "manifest": manifest}


def codec_yaml_values(folder, kaggle_dataset="satyapaladuguplus/scheduler-sqgan-v1", bundle_subdir="", sts_profile=2):
    """Values for ``codecs.v1`` in config/edge/codec.yaml (last_pt_sha256 is the loader's checkpoint_sha256)."""
    folder = Path(folder).resolve()
    cfg = json.loads((folder / "portable_config.json").read_text(encoding="utf-8"))
    return {"kaggle_dataset": kaggle_dataset, "bundle_subdir": bundle_subdir,
            "last_pt_sha256": sha256(folder / "last.pt"),
            "image_fraction": float(cfg.get("image_fraction", 0.25)),
            "semantic_fraction": float(cfg.get("semantic_fraction", 0.25)), "sts_profile": sts_profile}


def format_codec_yaml(values):
    lines = ["  v1:"] + ['    %s: %s' % (k, json.dumps(v)) for k, v in values.items()]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle")
    ap.add_argument("--deep", action="store_true", help="also grep last.pt for state-dict key prefixes")
    ap.add_argument("--print-codec-yaml", action="store_true")
    ap.add_argument("--kaggle-dataset", default="satyapaladuguplus/scheduler-sqgan-v1")
    ap.add_argument("--bundle-subdir", default="")
    a = ap.parse_args(argv)
    result = verify_bundle(a.bundle, a.deep)
    for e in result["errors"]:
        print("FAIL:", e, file=sys.stderr)
    if not result["ok"]:
        return 1
    print("bundle OK:", len(result["manifest"]["files"]), "files verified")
    if a.print_codec_yaml:
        print(format_codec_yaml(codec_yaml_values(a.bundle, a.kaggle_dataset, a.bundle_subdir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
