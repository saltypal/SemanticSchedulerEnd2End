"""Paper-ready reports for the edge study. Inputs are only read; outputs go into ``--out``.

Subcommands (``python -m Evaluation.edge.report <subcommand> ...``):

* ``codec-quality``: SQ-GAN v1 pass cache (``<c2>/shards/shard_XXXX``) joined to the simtable on
  ``image_id``. Per m level: LPIPS, recon mIoU and payload bytes (mean/median/p10/p90), overall and by
  split, source and tier; failed-image counts; the D6 gate verdict for the val split (provisional
  thresholds, CLI-settable). Writes ``codec_quality.csv`` and ``codec_quality.json``.
* ``results``: run_sweep result JSON file(s) or folder(s) -> ``main_table.csv``, ``main_table.tex``,
  ``ctr_vs_rho.png`` (one panel per workers value) and ``ctr_vs_workers.png`` (one panel per rho).
* ``header-proofs``: tidy CSV from ``Evaluation.edge.header_proofs`` -> ``ablation_table.csv``,
  ``ablation_table.tex``, ``ablation.png`` (cumulative none -> full, and leave-one-out).

Only numpy, json, csv and matplotlib (Agg) are used. Policy order and optional display names come from
the groups in ``config/edge/policies.yaml`` (``--policies-yaml`` overrides the repo file); without it the
registry names are used in first-seen order.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

NAN = float("nan")
LPIPS_MAX = 0.5          # provisional D6 threshold: a val mean LPIPS above this fails
MIOU_MIN = 0.5           # provisional D6 threshold: a val mean recon mIoU below this fails
QUALITY = ("payload_bytes", "lpips", "miou")
STAT_KEYS = ("n", "mean", "median", "p10", "p90")
CODEC_FIELDS = (["m", "group", "value", "n_images", "n_failed", "n_no_metric"]
                + ["%s_%s" % (q, s) for q in QUALITY for s in STAT_KEYS])
ABLATION_ORDER = ("cum:none", "cum:spi", "cum:spi+relevance", "cum:spi+relevance+keys", "cum:full",
                  "drop:spi", "drop:relevance", "drop:keys", "drop:sts", "equal_bytes")
ABLATION_METRICS = ("ctr_ungated", "value", "value_per_gpu_s", "low_starvation", "redundant_per_100")
FULL_VARIANT = "cum:full"


# ----------------------------------------------------------------------------- shared helpers
def _f(x):
    return NAN if x is None else float(x)


def _csv_val(v):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    return v


def _clean(x):
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def _write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _csv_val(r.get(k)) for k in fields})


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(obj), f, indent=1)


def _tex(s):
    """Escape text for LaTeX (policy names contain underscores)."""
    out = str(s)
    for ch, rep in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                    ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                    ("^", r"\textasciicircum{}")):
        out = out.replace(ch, rep)
    return out


def _tri_and(verdicts):
    """False if any verdict is False, None if any is None (and none False), else True; None for empty."""
    if not verdicts:
        return None
    if any(v is False for v in verdicts):
        return False
    if any(v is None for v in verdicts):
        return None
    return True


def _m_value(tag):
    return float(str(tag)[1:])


# ----------------------------------------------------------------------------- codec-quality
def read_jsonl(path):
    path = Path(path)
    if path.is_dir():
        path = path / "simtable_v1.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _level_tags(shard):
    """``[(tag, m)]`` from the ``offsets_m<m>.npy`` files of a shard, sorted by m (sqgan_ok column order)."""
    tags = [p.name[len("offsets_"):-len(".npy")] for p in shard.glob("offsets_m*.npy")]
    return sorted(((t, _m_value(t)) for t in tags), key=lambda x: x[1])


def _optional_array(shard, name):
    path = shard / name
    return np.load(path).astype(float).reshape(-1) if path.is_file() else None


def read_cache(c2_dirs, notes):
    """Per (image, m level) records from the shards of each feature-pass cache folder.

    Payload bytes of image i at level m are ``offsets_m[i+1] - offsets_m[i]``, counted only where
    ``sqgan_ok[i, level]`` is set and the payload is non-empty. Returns ``(records, avail, n_shards)``
    where ``avail[tag] = {"lpips": bool, "miou": bool}`` tells which quality files exist.
    """
    shards = []
    for root in c2_dirs:
        shards += sorted((Path(root) / "shards").glob("shard_*"))
    if not shards:
        raise FileNotFoundError("no shard_* folders under %s" % (list(c2_dirs),))
    records, avail, used = [], {}, 0
    for shard in shards:
        if not (shard / "uids.json").is_file() or not (shard / "sqgan_ok.npy").is_file():
            notes.append("skipped %s: uids.json or sqgan_ok.npy missing" % shard.name)
            continue
        uids = json.loads((shard / "uids.json").read_text(encoding="utf-8"))
        n = len(uids)
        ok = np.load(shard / "sqgan_ok.npy").reshape(n, -1).astype(bool)
        used += 1
        for j, (tag, _m) in enumerate(_level_tags(shard)):
            offs = np.load(shard / ("offsets_%s.npy" % tag)).reshape(-1)
            if offs.size != n + 1:
                notes.append("skipped %s at %s: offsets length %d != %d + 1" % (shard.name, tag, offs.size, n))
                continue
            flags = ok[:, j] if j < ok.shape[1] else np.zeros(n, dtype=bool)
            lp = _optional_array(shard, "lpips_%s.npy" % tag)
            mi = _optional_array(shard, "miou_%s.npy" % tag)
            slot = avail.setdefault(tag, {"lpips": False, "miou": False})
            slot["lpips"] |= lp is not None
            slot["miou"] |= mi is not None
            for i, uid in enumerate(uids):
                nbytes = int(offs[i + 1] - offs[i])
                encoded = bool(flags[i]) and nbytes > 0
                records.append(dict(
                    m=tag, image_id=str(uid), encoded=encoded,
                    payload_bytes=float(nbytes) if encoded else NAN,
                    lpips=float(lp[i]) if (encoded and lp is not None) else NAN,
                    miou=float(mi[i]) if (encoded and mi is not None) else NAN))
    for tag, flags in sorted(avail.items(), key=lambda kv: _m_value(kv[0])):
        for q, present in flags.items():
            if not present:
                notes.append("%s_%s.npy not found in any shard: %s skipped at %s" % (q, tag, q, tag))
    return records, avail, used


def join_table(records, table_rows):
    """Attach split/source/tier from the table by image_id. Returns (joined, counts)."""
    by_id = {str(r["image_id"]): r for r in table_rows}
    cache_ids = {r["image_id"] for r in records}
    joined = []
    for rec in records:
        row = by_id.get(rec["image_id"])
        if row is not None:
            joined.append(dict(rec, split=row.get("split"), source=row.get("source"), tier=row.get("tier")))
    counts = dict(n_cache_images=len(cache_ids),
                  n_joined_images=len(cache_ids & set(by_id)),
                  n_cache_not_in_table=len(cache_ids - set(by_id)),
                  n_table_not_in_cache=len(set(by_id) - cache_ids))
    return joined, counts


def _stats(vals):
    v = np.array([x for x in vals if math.isfinite(x)], dtype=float)
    if v.size == 0:
        return dict(n=0, mean=NAN, median=NAN, p10=NAN, p90=NAN)
    return dict(n=int(v.size), mean=float(v.mean()), median=float(np.median(v)),
                p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)))


def summarise(records, avail):
    """One row per (m level, group): overall, then split, source and tier values."""
    rows = []
    for m in sorted({r["m"] for r in records}, key=_m_value):
        at_m = [r for r in records if r["m"] == m]
        checks = [q for q in ("lpips", "miou") if avail.get(m, {}).get(q)]
        groups = [("overall", "all", at_m)]
        for field in ("split", "source", "tier"):
            for val in sorted({str(r[field]) for r in at_m}):
                groups.append((field, val, [r for r in at_m if str(r[field]) == val]))
        for name, val, recs in groups:
            row = dict(m=m, group=name, value=val, n_images=len(recs),
                       n_failed=sum(1 for r in recs if not r["encoded"]),
                       n_no_metric=sum(1 for r in recs if r["encoded"]
                                       and any(not math.isfinite(r[q]) for q in checks)))
            for q in QUALITY:
                s = _stats([r[q] for r in recs])
                row.update({"%s_%s" % (q, k): s[k] for k in STAT_KEYS})
            rows.append(row)
    return rows


def d6_gate(rows, split="val", lpips_max=LPIPS_MAX, miou_min=MIOU_MIN):
    """D6 verdict per m level from the split's mean recon mIoU and mean LPIPS.

    A level is ``usable`` False when mean mIoU < ``miou_min`` or mean LPIPS > ``lpips_max``; None when a
    mean is undefined (metric missing) and nothing fails. The top-level ``usable`` is the conservative
    AND over the levels. Thresholds are provisional.
    """
    per_m = {}
    for m in sorted({r["m"] for r in rows}, key=_m_value):
        r = next((x for x in rows if x["m"] == m and x["group"] == "split" and str(x["value"]) == split), None)
        miou = _f(r["miou_mean"]) if r else NAN
        lp = _f(r["lpips_mean"]) if r else NAN
        miou_ok = None if not math.isfinite(miou) else bool(miou >= miou_min)
        lp_ok = None if not math.isfinite(lp) else bool(lp <= lpips_max)
        per_m[m] = dict(n_images=int(r["n_images"]) if r else 0, miou_mean=miou, lpips_mean=lp,
                        miou_ok=miou_ok, lpips_ok=lp_ok, usable=_tri_and([miou_ok, lp_ok]))
    return dict(split=split, provisional=True,
                thresholds=dict(miou_min=miou_min, lpips_max=lpips_max),
                rule="usable false if mean mIoU < miou_min or mean LPIPS > lpips_max, per m level",
                usable=_tri_and([v["usable"] for v in per_m.values()]), per_m=per_m)


def codec_quality(c2_dirs, table_path, out_dir, gate_split="val", lpips_max=LPIPS_MAX, miou_min=MIOU_MIN):
    """Join the SQ-GAN v1 cache to the table and write codec_quality.csv/.json. Returns the result dict."""
    notes = []
    c2_dirs = [c2_dirs] if isinstance(c2_dirs, (str, Path)) else list(c2_dirs)
    records, avail, n_shards = read_cache(c2_dirs, notes)
    joined, counts = join_table(records, read_jsonl(table_path))
    rows = summarise(joined, avail)
    gate = d6_gate(rows, gate_split, lpips_max, miou_min)
    result = dict(inputs=dict(c2=[str(c) for c in c2_dirs], table=str(table_path), n_shards=n_shards,
                              **counts),
                  levels=sorted(avail, key=_m_value), rows=rows, gate=gate, notes=notes)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "codec_quality.csv", rows, CODEC_FIELDS)
    _write_json(out / "codec_quality.json", result)
    return result


# ----------------------------------------------------------------------------- results
def load_cells(paths):
    """Cells from run_sweep result JSON files (or folders holding them), tagged with split and scenario set."""
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files += [f for f in sorted(p.glob("*.json")) if _has_cells(f)]
        else:
            files.append(p)
    cells = []
    for f in files:
        doc = json.loads(Path(f).read_text(encoding="utf-8"))
        if "cells" not in doc:
            raise ValueError("%s has no 'cells': not a run_sweep result" % f)
        cfg = doc.get("config") or {}
        split, scen = cfg.get("split"), tuple(cfg.get("scenario_names") or [])
        for c in doc["cells"]:
            cells.append(dict(c, split=split, scenario_set=scen, source_file=str(f)))
    return cells


def _has_cells(path):
    try:
        return "cells" in json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False


def select_cells(cells, split=None, scenario_set=None):
    """Filter to one split and one scenario set; raise when the choice is ambiguous or a cell repeats."""
    if split is not None:
        cells = [c for c in cells if c["split"] == split]
    elif len({c["split"] for c in cells}) > 1:
        raise ValueError("results cover several splits %s; pass --split" % sorted(map(str, {c["split"] for c in cells})))
    if scenario_set is not None:
        cells = [c for c in cells if ",".join(c["scenario_set"]) == scenario_set]
    elif len({c["scenario_set"] for c in cells}) > 1:
        raise ValueError("several scenario sets %s; pass --scenario-set" % sorted(
            ",".join(s) for s in {c["scenario_set"] for c in cells}))
    if not cells:
        raise ValueError("no cells left after filtering")
    seen = {}
    for c in cells:
        key = (c["policy"], int(c["workers"]), float(c["rho"]))
        if key in seen:
            raise ValueError("duplicate cell %s in %s" % (key, c["source_file"]))
        seen[key] = c
    return cells


def policy_order(policies_yaml=None):
    """(ordered policy names, display-name map) from config/edge/policies.yaml; empty when unavailable."""
    try:
        if policies_yaml:
            import yaml
            with open(policies_yaml, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            from Evaluation.edge.edge_config import load_config
            cfg = load_config("policies")
    except (FileNotFoundError, ImportError, OSError):
        return [], {}
    cfg = cfg or {}
    order = []
    for key, members in cfg.items():
        if key == "display_names" or not isinstance(members, list):
            continue
        for name in members:
            if isinstance(name, str) and name not in order:
                order.append(name)
    display = dict(cfg.get("display_names") or {})
    return order, display


def _ci(cell, key="ctr_ungated"):
    lo, hi = (cell.get("ci95") or {}).get(key) or (None, None)
    return _f(lo), _f(hi)


def _main_table_csv(path, by_key, policies, display, combos):
    fields = ["policy", "display_name"]
    for w, r in combos:
        fields += ["w%d_rho%g_%s" % (w, r, s) for s in ("ctr_mean", "ci_lo", "ci_hi")]
    rows = []
    for p in policies:
        row = dict(policy=p, display_name=display.get(p, p))
        for w, r in combos:
            c = by_key.get((p, w, r))
            mean, (lo, hi) = (_f(c["ctr_ungated"]), _ci(c)) if c else (NAN, (NAN, NAN))
            row.update({"w%d_rho%g_ctr_mean" % (w, r): mean, "w%d_rho%g_ci_lo" % (w, r): lo,
                        "w%d_rho%g_ci_hi" % (w, r): hi})
        rows.append(row)
    _write_csv(path, rows, fields)


def _main_table_tex(path, by_key, policies, display, combos):
    workers = sorted({w for w, _ in combos})
    groups = [(w, [r for ww, r in combos if ww == w]) for w in workers]
    ncol = len(combos)
    lines = ["% CTR@D mean [95% bootstrap CI]; best mean per column in bold. Requires booktabs.",
             "\\begin{tabular}{l" + "c" * ncol + "}", "\\toprule"]
    head1, cmids, pos = [""], [], 2
    for w, rs in groups:
        head1.append("\\multicolumn{%d}{c}{$W=%d$}" % (len(rs), w))
        cmids.append("\\cmidrule(lr){%d-%d}" % (pos, pos + len(rs) - 1))
        pos += len(rs)
    lines.append(" & ".join(head1) + " \\\\")
    lines.append(" ".join(cmids))
    head2 = ["Policy"] + ["$\\rho=%g$" % r for _, rs in groups for r in rs]
    lines += [" & ".join(head2) + " \\\\", "\\midrule"]
    best = {}
    for col in combos:
        vals = [_f(by_key[(p, col[0], col[1])]["ctr_ungated"]) for p in policies if (p, col[0], col[1]) in by_key]
        vals = [v for v in vals if math.isfinite(v)]
        best[col] = max(vals) if vals else None
    for p in policies:
        cells = [_tex(display.get(p, p))]
        for col in combos:
            c = by_key.get((p, col[0], col[1]))
            if c is None or not math.isfinite(_f(c["ctr_ungated"])):
                cells.append("--")
                continue
            lo, hi = _ci(c)
            text = "%.3f [%.3f, %.3f]" % (_f(c["ctr_ungated"]), lo, hi)
            cells.append("\\textbf{%s}" % text if best[col] is not None and _f(c["ctr_ungated"]) == best[col] else text)
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ctr_figure(path, by_key, policies, display, panel):
    """One line per policy. ``panel`` = 'workers' (x = rho, one panel per workers) or 'rho' (x = workers)."""
    if panel == "workers":
        panels = sorted({k[1] for k in by_key})
        xs = sorted({k[2] for k in by_key})
        get = lambda p, pv, xv: by_key.get((p, pv, xv))  # noqa: E731
        xlabel, title = r"$\rho$", lambda pv: "workers = %d" % pv
    else:
        panels = sorted({k[2] for k in by_key})
        xs = sorted({k[1] for k in by_key})
        get = lambda p, pv, xv: by_key.get((p, xv, pv))  # noqa: E731
        xlabel, title = "workers", lambda pv: r"$\rho$ = %g" % pv
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.6), sharey=True, squeeze=False)
    handles = {}
    for ax, pv in zip(axes[0], panels):
        for p in policies:
            pts = [(xv, get(p, pv, xv)) for xv in xs]
            pts = [(xv, c) for xv, c in pts if c is not None]
            if not pts:
                continue
            x = np.array([xv for xv, _ in pts], dtype=float)
            y = np.array([_f(c["ctr_ungated"]) for _, c in pts])
            lo = np.array([_ci(c)[0] for _, c in pts])
            hi = np.array([_ci(c)[1] for _, c in pts])
            (line,) = ax.plot(x, y, marker="o", label=display.get(p, p))
            ax.fill_between(x, lo, hi, alpha=0.15, color=line.get_color())
            handles.setdefault(p, line)
        ax.set_title(title(pv), fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_xticks(xs)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("CTR@D")
    if handles:
        order = [p for p in policies if p in handles]
        fig.legend([handles[p] for p in order], [display.get(p, p) for p in order],
                   loc="center right", fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def results(run_paths, out_dir, split=None, scenario_set=None, policies_yaml=None):
    """run_sweep JSON -> main table (csv, tex) and CTR figures. Returns the written file paths."""
    cells = select_cells(load_cells(run_paths), split, scenario_set)
    order, display = policy_order(policies_yaml)
    policies = sorted({c["policy"] for c in cells},
                      key=lambda p: (order.index(p) if p in order else len(order), p))
    by_key = {(c["policy"], int(c["workers"]), float(c["rho"])): c for c in cells}
    combos = sorted({(w, r) for _, w, r in by_key})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = [out / "main_table.csv", out / "main_table.tex", out / "ctr_vs_rho.png",
               out / "ctr_vs_workers.png"]
    _main_table_csv(written[0], by_key, policies, display, combos)
    _main_table_tex(written[1], by_key, policies, display, combos)
    _ctr_figure(written[2], by_key, policies, display, "workers")
    _ctr_figure(written[3], by_key, policies, display, "rho")
    return [str(p) for p in written]


# ----------------------------------------------------------------------------- header-proofs
def _kind(variant):
    if variant.startswith("cum:"):
        return "cumulative"
    if variant.startswith("drop:"):
        return "leave-one-out"
    return "equal-bytes"


def _variant_rank(v):
    return (ABLATION_ORDER.index(v) if v in ABLATION_ORDER else len(ABLATION_ORDER), v)


def read_header_csv(path):
    """``{(policy, workers, rho, variant): {metric: value}}`` from the ablation / equal_bytes rows."""
    cells = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["experiment"] not in ("ablation", "equal_bytes"):
                continue
            key = (r["policy"], int(r["workers"]), float(r["rho"]), r["variant"])
            val = float(r["value"]) if r["value"] not in ("", None) else NAN
            cells.setdefault(key, {})[r["metric"]] = val
    return cells


def ablation_rows(cells):
    combos = sorted({(p, w, r) for p, w, r, _ in cells})
    rows = []
    for p, w, r in combos:
        full = cells.get((p, w, r, FULL_VARIANT), {}).get("ctr_ungated", NAN)
        variants = sorted({v for pp, ww, rr, v in cells if (pp, ww, rr) == (p, w, r)}, key=_variant_rank)
        for v in variants:
            m = cells[(p, w, r, v)]
            row = dict(kind=_kind(v), variant=v, policy=p, workers=w, rho=r,
                       ctr_delta_vs_full=m.get("ctr_ungated", NAN) - full)
            row.update({k: m.get(k, NAN) for k in ABLATION_METRICS})
            rows.append(row)
    return rows


def _ablation_tex(path, rows):
    combos = sorted({(r["policy"], r["workers"], r["rho"]) for r in rows})
    variants = sorted({r["variant"] for r in rows}, key=_variant_rank)
    by = {(r["policy"], r["workers"], r["rho"], r["variant"]): r for r in rows}
    lines = ["% CTR@D mean (change vs. full header in parentheses). Requires booktabs.",
             "\\begin{tabular}{l" + "c" * len(combos) + "}", "\\toprule",
             " & ".join(["Variant"] + ["%s, $W=%d$, $\\rho=%g$" % (_tex(p), w, r) for p, w, r in combos]) + " \\\\",
             "\\midrule"]
    last_kind = None
    for v in variants:
        kind = _kind(v)
        if last_kind is not None and kind != last_kind:
            lines.append("\\midrule")
        last_kind = kind
        cells = [_tex(v)]
        for p, w, r in combos:
            c = by.get((p, w, r, v))
            if c is None or not math.isfinite(c["ctr_ungated"]):
                cells.append("--")
            else:
                cells.append("%.3f (%+.3f)" % (c["ctr_ungated"], c["ctr_delta_vs_full"])
                             if math.isfinite(c["ctr_delta_vs_full"]) else "%.3f" % c["ctr_ungated"])
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ablation_figure(path, rows):
    combos = sorted({(r["policy"], r["workers"], r["rho"]) for r in rows})
    fig, axes = plt.subplots(len(combos), 2, figsize=(11, 3.4 * len(combos)), squeeze=False)
    for (p, w, r), ax_row in zip(combos, axes):
        mine = [x for x in rows if (x["policy"], x["workers"], x["rho"]) == (p, w, r)]
        cum = [x for x in mine if x["kind"] == "cumulative"]
        ax = ax_row[0]
        names = [x["variant"].replace("cum:", "") for x in cum]
        ax.bar(range(len(cum)), [x["ctr_ungated"] for x in cum], color="#4C72B0")
        ax.set_xticks(range(len(cum)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("CTR@D")
        ax.set_title("%s, W=%d, rho=%g: cumulative none -> full" % (p, w, r), fontsize=9)
        ax = ax_row[1]
        drop = [x for x in mine if x["kind"] in ("leave-one-out", "equal-bytes")]
        ax.bar(range(len(drop)), [x["ctr_delta_vs_full"] for x in drop],
               color=["#C44E52" if x["kind"] == "equal-bytes" else "#55A868" for x in drop])
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(drop)))
        ax.set_xticklabels([x["variant"].replace("drop:", "-") for x in drop], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("change in CTR@D vs full")
        ax.set_title("leave-one-out (green) and equal-bytes (red)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def header_proofs(csv_path, out_dir):
    """Ablation tables and chart from the tidy header_proofs CSV. Returns the written file paths."""
    cells = read_header_csv(csv_path)
    if not cells:
        raise ValueError("no ablation / equal_bytes rows in %s" % csv_path)
    rows = ablation_rows(cells)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = [out / "ablation_table.csv", out / "ablation_table.tex", out / "ablation.png"]
    _write_csv(written[0], rows, ["kind", "variant", "policy", "workers", "rho"]
               + list(ABLATION_METRICS) + ["ctr_delta_vs_full"])
    _ablation_tex(written[1], rows)
    _ablation_figure(written[2], rows)
    return [str(p) for p in written]


# ----------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m Evaluation.edge.report",
                                 description="Codec quality, policy result tables and header ablations.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("codec-quality", help="SQ-GAN v1 cache joined to the simtable, with the D6 gate")
    p.add_argument("--c2", nargs="+", required=True, help="feature-pass cache folder(s)")
    p.add_argument("--table", required=True, help="simtable_v1.jsonl (or its folder)")
    p.add_argument("--out", required=True)
    p.add_argument("--gate-split", default="val")
    p.add_argument("--lpips-max", type=float, default=LPIPS_MAX, help="provisional D6 threshold")
    p.add_argument("--miou-min", type=float, default=MIOU_MIN, help="provisional D6 threshold")

    p = sub.add_parser("results", help="run_sweep JSON -> main table and CTR figures")
    p.add_argument("--runs", nargs="+", required=True, help="run_sweep result JSON file(s) or folder(s)")
    p.add_argument("--out", required=True)
    p.add_argument("--split", help="required when the runs cover several splits")
    p.add_argument("--scenario-set", help="comma-joined scenario names; required when several sets exist")
    p.add_argument("--policies-yaml", help="override config/edge/policies.yaml")

    p = sub.add_parser("header-proofs", help="header_proofs.csv -> ablation table and chart")
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)

    a = ap.parse_args(argv)
    if a.cmd == "codec-quality":
        res = codec_quality(a.c2, a.table, a.out, a.gate_split, a.lpips_max, a.miou_min)
        gate = res["gate"]
        print(json.dumps(dict(usable=gate["usable"], split=gate["split"], levels=res["levels"],
                              notes=res["notes"], out=str(Path(a.out) / "codec_quality.csv")), indent=1))
        return res
    if a.cmd == "results":
        written = results(a.runs, a.out, a.split, a.scenario_set, a.policies_yaml)
    else:
        written = header_proofs(a.csv, a.out)
    print("\n".join(written))
    return written


if __name__ == "__main__":
    main()
