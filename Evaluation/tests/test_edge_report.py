"""Edge report (codec-quality with the D6 gate, results tables/figures, header-proof ablations) on synthetic inputs."""
import csv
import json
import math

import numpy as np
import pytest

from Evaluation.edge import report

PNG_MAGIC = b"\x89PNG"


# ----------------------------------------------------------------------------- codec-quality fixtures
def _write_shard(root, name, uids, ok, offsets, lpips=None, miou=None):
    shard = root / "shards" / name
    shard.mkdir(parents=True)
    (shard / "uids.json").write_text(json.dumps(uids), encoding="utf-8")
    np.save(shard / "sqgan_ok.npy", np.array(ok, dtype=bool))
    for tag, arr in offsets.items():
        np.save(shard / ("offsets_%s.npy" % tag), np.array(arr, dtype=np.int64))
    for kind, d in (("lpips", lpips or {}), ("miou", miou or {})):
        for tag, arr in d.items():
            np.save(shard / ("%s_%s.npy" % (kind, tag)), np.array(arr, dtype=np.float32))
    return shard


def _cache(tmp_path):
    """Two shards, levels m0.25 and m0.55. LPIPS/mIoU exist only for m0.25 (m0.55 missing -> skipped).

    payload bytes m0.25: a=10 b=15 c=15 d=12 ; m0.55: a=20 (b failed, empty payload) c=30 (d failed).
    """
    c2 = tmp_path / "c2"
    _write_shard(c2, "shard_0000", ["a", "b", "c"], [[1, 1], [1, 0], [1, 1]],
                 offsets={"m0.25": [0, 10, 25, 40], "m0.55": [0, 20, 20, 50]},
                 lpips={"m0.25": [0.1, 0.3, np.nan]}, miou={"m0.25": [0.8, 0.6, 0.4]})
    _write_shard(c2, "shard_0001", ["d"], [[1, 0]],
                 offsets={"m0.25": [0, 12], "m0.55": [0, 0]},
                 lpips={"m0.25": [0.5]}, miou={"m0.25": [0.9]})
    return c2


def _table(tmp_path):
    rows = [dict(image_id="a", split="val", source="crisismmd", tier=0),
            dict(image_id="b", split="val", source="coco", tier=1),
            dict(image_id="c", split="test", source="crisismmd", tier=0),
            dict(image_id="d", split="val", source="coco", tier=1),
            dict(image_id="e", split="val", source="crisismmd", tier=2)]   # not in the cache
    path = tmp_path / "simtable_v1.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _row(res, m, group, value=None):
    return next(r for r in res["rows"] if r["m"] == m and r["group"] == group
                and (value is None or r["value"] == value))


def test_payload_bytes_come_from_offsets_and_failures_are_counted(tmp_path):
    res = report.codec_quality([_cache(tmp_path)], _table(tmp_path), tmp_path / "out")
    o25 = _row(res, "m0.25", "overall")
    assert o25["n_images"] == 4 and o25["n_failed"] == 0
    assert o25["payload_bytes_mean"] == pytest.approx(np.mean([10, 15, 15, 12]))
    assert o25["payload_bytes_median"] == pytest.approx(np.median([10, 15, 15, 12]))
    assert o25["payload_bytes_p10"] == pytest.approx(np.percentile([10, 15, 15, 12], 10))
    assert o25["lpips_n"] == 3 and o25["lpips_mean"] == pytest.approx((0.1 + 0.3 + 0.5) / 3)
    assert o25["miou_mean"] == pytest.approx((0.8 + 0.6 + 0.4 + 0.9) / 4)
    o55 = _row(res, "m0.55", "overall")
    assert o55["n_failed"] == 2 and o55["n_images"] == 4
    assert o55["payload_bytes_mean"] == pytest.approx(25.0)      # only a=20 and c=30 encoded
    assert math.isnan(o55["lpips_mean"]) and o55["lpips_n"] == 0


def test_join_breakdowns_and_notes(tmp_path):
    res = report.codec_quality([_cache(tmp_path)], _table(tmp_path), tmp_path / "out")
    assert res["inputs"]["n_cache_images"] == 4 and res["inputs"]["n_joined_images"] == 4
    assert res["inputs"]["n_table_not_in_cache"] == 1 and res["inputs"]["n_cache_not_in_table"] == 0
    assert _row(res, "m0.25", "split", "test")["n_images"] == 1
    assert _row(res, "m0.25", "source", "coco")["n_images"] == 2
    assert _row(res, "m0.25", "tier", "1")["miou_mean"] == pytest.approx((0.6 + 0.9) / 2)
    assert any("lpips_m0.55.npy" in n for n in res["notes"])


def test_csv_and_json_written(tmp_path):
    out = tmp_path / "out"
    report.codec_quality([_cache(tmp_path)], _table(tmp_path), out)
    with open(out / "codec_quality.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {r["m"] for r in rows} == {"m0.25", "m0.55"}
    doc = json.loads((out / "codec_quality.json").read_text(encoding="utf-8"))
    assert doc["levels"] == ["m0.25", "m0.55"] and doc["gate"]["split"] == "val"


def test_d6_gate_passes_and_is_undecided_without_metrics(tmp_path):
    res = report.codec_quality([_cache(tmp_path)], _table(tmp_path), tmp_path / "out")
    gate = res["gate"]
    # m0.25 val: mean mIoU (0.8+0.6+0.9)/3 = 0.767 >= 0.5 and mean LPIPS (0.1+0.3+0.5)/3 = 0.3 <= 0.5
    assert gate["per_m"]["m0.25"]["miou_mean"] == pytest.approx((0.8 + 0.6 + 0.9) / 3)
    assert gate["per_m"]["m0.25"]["usable"] is True
    # m0.55 has no quality files: the verdict is undecided, so the overall flag is None, not True
    assert gate["per_m"]["m0.55"]["usable"] is None and gate["usable"] is None
    assert gate["provisional"] is True and gate["thresholds"] == dict(miou_min=0.5, lpips_max=0.5)


def test_d6_gate_thresholds_flag_usable_false(tmp_path):
    c2, table = _cache(tmp_path), _table(tmp_path)
    low_miou = report.codec_quality([c2], table, tmp_path / "o1", miou_min=0.8)
    assert low_miou["gate"]["per_m"]["m0.25"]["miou_ok"] is False
    assert low_miou["gate"]["per_m"]["m0.25"]["usable"] is False and low_miou["gate"]["usable"] is False
    high_lpips = report.codec_quality([c2], table, tmp_path / "o2", lpips_max=0.2)
    assert high_lpips["gate"]["per_m"]["m0.25"]["lpips_ok"] is False
    assert high_lpips["gate"]["usable"] is False


def test_d6_gate_pure_logic():
    rows = [dict(m="m0.25", group="split", value="val", n_images=3, miou_mean=0.7, lpips_mean=0.2),
            dict(m="m0.55", group="split", value="val", n_images=3, miou_mean=0.49, lpips_mean=0.2)]
    gate = report.d6_gate(rows, "val")
    assert gate["per_m"]["m0.25"]["usable"] is True and gate["per_m"]["m0.55"]["usable"] is False
    assert gate["usable"] is False
    assert report.d6_gate(rows, "test")["usable"] is None     # no test rows: undecided


def test_codec_cli_subcommand(tmp_path, capsys):
    report.main(["codec-quality", "--c2", str(_cache(tmp_path)), "--table", str(_table(tmp_path)),
                 "--out", str(tmp_path / "cli"), "--miou-min", "0.5"])
    assert (tmp_path / "cli" / "codec_quality.csv").is_file()
    assert "usable" in capsys.readouterr().out


# ----------------------------------------------------------------------------- results fixtures
def _cell(policy, workers, rho, ctr, ci=None):
    return dict(policy=policy, kwargs={}, workers=workers, rho=rho, realisations=3, n_measured=100,
                ctr_ungated=ctr, value=1.0, value_per_gpu_s=0.5, low_starvation=0.1, redundant_per_100=2.0,
                ci95={"ctr_ungated": ci if ci is not None else [ctr - 0.05, ctr + 0.05],
                      "value_per_image": [0.9, 1.1]}, kappa=None, scenario={})


def _sweep_doc(split="val", scen=()):
    cells = []
    for w in (1, 2):
        for rho in (0.8, 1.0):
            base = 0.5 + 0.1 * w - 0.2 * rho
            cells.append(_cell("fifo", w, rho, base))
            cells.append(_cell("sv_dpp", w, rho, base + 0.2, ci=[None, None] if (w, rho) == (2, 1.0) else None))
    return dict(config=dict(split=split, policies=["fifo", "sv_dpp"], workers=[1, 2], rhos=[0.8, 1.0],
                            scenario_names=list(scen)), git_sha="x", cells=cells)


def _write_json(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _policies_yaml(tmp_path):
    path = tmp_path / "policies.yaml"
    path.write_text("main_table: [sv_dpp, fifo]\nextended: [edf]\n", encoding="utf-8")
    return path


def test_results_main_table_csv_order_and_values(tmp_path):
    run = _write_json(tmp_path / "runs" / "sweep.json", _sweep_doc())
    out = tmp_path / "out"
    written = report.results([run], out, policies_yaml=_policies_yaml(tmp_path))
    assert sorted(p.replace("\\", "/").rsplit("/", 1)[-1] for p in written) == sorted([
        "main_table.csv", "main_table.tex", "ctr_vs_rho.png", "ctr_vs_workers.png"])
    with open(out / "main_table.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["policy"] for r in rows] == ["sv_dpp", "fifo"]          # group order from policies.yaml
    assert float(rows[1]["w2_rho1_ctr_mean"]) == pytest.approx(0.5 + 0.2 - 0.2)
    assert float(rows[0]["w1_rho0.8_ctr_mean"]) == pytest.approx(0.5 + 0.1 - 0.16 + 0.2)
    assert float(rows[0]["w1_rho0.8_ci_lo"]) == pytest.approx(0.5 + 0.1 - 0.16 + 0.2 - 0.05)
    assert rows[0]["w2_rho1_ci_lo"] == ""                              # None CI stays empty, not 'nan'


def test_results_tex_and_figures(tmp_path):
    run = _write_json(tmp_path / "runs" / "sweep.json", _sweep_doc())
    out = tmp_path / "out"
    report.results([run], out, policies_yaml=_policies_yaml(tmp_path))
    tex = (out / "main_table.tex").read_text(encoding="utf-8")
    assert "\\begin{tabular}" in tex and "\\textbf{" in tex and "sv\\_dpp" in tex
    assert "$W=2$" in tex and "$\\rho=0.8$" in tex
    for name in ("ctr_vs_rho.png", "ctr_vs_workers.png"):
        assert (out / name).read_bytes()[:4] == PNG_MAGIC


def test_results_folder_input_and_split_guard(tmp_path):
    _write_json(tmp_path / "runs" / "val.json", _sweep_doc("val"))
    _write_json(tmp_path / "runs" / "test.json", _sweep_doc("test"))
    with pytest.raises(ValueError, match="several splits"):
        report.results([tmp_path / "runs"], tmp_path / "o1")
    report.main(["results", "--runs", str(tmp_path / "runs"), "--out", str(tmp_path / "o2"), "--split", "test",
                 "--policies-yaml", str(_policies_yaml(tmp_path))])
    assert (tmp_path / "o2" / "main_table.csv").is_file()


def test_results_rejects_duplicate_cells_and_non_sweep_json(tmp_path):
    doc = _sweep_doc()
    doc["cells"].append(dict(doc["cells"][0]))
    run = _write_json(tmp_path / "dup.json", doc)
    with pytest.raises(ValueError, match="duplicate cell"):
        report.results([run], tmp_path / "o")
    bad = _write_json(tmp_path / "bad.json", {"ablation": []})
    with pytest.raises(ValueError, match="no 'cells'"):
        report.load_cells([bad])


# ----------------------------------------------------------------------------- header-proofs fixtures
VARIANT_CTR = {"cum:none": 0.40, "cum:spi": 0.50, "cum:spi+relevance": 0.55, "cum:spi+relevance+keys": 0.58,
               "cum:full": 0.60, "drop:spi": 0.57, "drop:relevance": 0.59, "drop:keys": 0.50,
               "drop:sts": 0.60, "equal_bytes": 0.45}


def _header_csv(tmp_path):
    path = tmp_path / "header_proofs.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(("experiment", "variant", "policy", "workers", "rho", "metric", "value"))
        for variant, ctr in VARIANT_CTR.items():
            exp = "equal_bytes" if variant == "equal_bytes" else "ablation"
            w.writerow((exp, variant, "sv_dpp", 2, 1.0, "ctr_ungated", ctr))
            w.writerow((exp, variant, "sv_dpp", 2, 1.0, "value", 2 * ctr))
        w.writerow(("dedup", "thr=0.5", "receiver", 2, 1.0, "precision", 0.9))
        w.writerow(("overhead", "64/(64+payload)", "", "", "", "mean", 0.12))
    return path


def test_header_proofs_table_deltas_and_kinds(tmp_path):
    out = tmp_path / "out"
    report.header_proofs(str(_header_csv(tmp_path)), out)
    with open(out / "ablation_table.csv", newline="", encoding="utf-8") as f:
        rows = {r["variant"]: r for r in csv.DictReader(f)}
    assert len(rows) == len(VARIANT_CTR)                              # dedup / overhead rows are ignored
    assert rows["cum:none"]["kind"] == "cumulative" and rows["drop:keys"]["kind"] == "leave-one-out"
    assert rows["equal_bytes"]["kind"] == "equal-bytes"
    assert float(rows["cum:full"]["ctr_delta_vs_full"]) == pytest.approx(0.0)
    assert float(rows["cum:none"]["ctr_delta_vs_full"]) == pytest.approx(0.40 - 0.60)
    assert float(rows["drop:keys"]["ctr_delta_vs_full"]) == pytest.approx(0.50 - 0.60)
    assert float(rows["drop:sts"]["ctr_delta_vs_full"]) == pytest.approx(0.0)
    assert float(rows["equal_bytes"]["value"]) == pytest.approx(0.90)


def test_header_proofs_tex_and_chart(tmp_path):
    out = tmp_path / "out"
    written = report.header_proofs(str(_header_csv(tmp_path)), out)
    assert len(written) == 3
    tex = (out / "ablation_table.tex").read_text(encoding="utf-8")
    assert "\\begin{tabular}" in tex
    assert "0.600 (+0.000)" in tex
    assert (out / "ablation.png").read_bytes()[:4] == PNG_MAGIC


def test_header_proofs_cli_and_empty_guard(tmp_path):
    report.main(["header-proofs", "--csv", str(_header_csv(tmp_path)), "--out", str(tmp_path / "cli")])
    assert (tmp_path / "cli" / "ablation.png").is_file()
    empty = tmp_path / "empty.csv"
    empty.write_text("experiment,variant,policy,workers,rho,metric,value\n"
                     "overhead,64/(64+payload),,,,mean,0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no ablation"):
        report.header_proofs(str(empty), tmp_path / "e")
