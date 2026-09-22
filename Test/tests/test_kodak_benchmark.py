import csv

import pytest

from Evaluation.kodak_benchmark import CSV_COLUMNS, append_rows_csv, dataset_files, summarize_rows


def complete_row(run_id="run-1", image_id="kodim01"):
    row = {column: None for column in CSV_COLUMNS}
    row.update(
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "method": "dummy",
            "operating_point": "default",
            "image_id": image_id,
            "psnr_db": 30.0,
            "ssim": 0.9,
            "total_time_ms": 10.0,
        }
    )
    return row


def test_append_rows_is_idempotent_and_keeps_schema(tmp_path):
    output = tmp_path / "benchmark.csv"
    assert append_rows_csv(output, [complete_row()]) == 1
    assert append_rows_csv(output, [complete_row()]) == 0
    assert append_rows_csv(output, [complete_row(image_id="kodim02")]) == 1

    with output.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames) == CSV_COLUMNS
        assert [row["image_id"] for row in reader] == ["kodim01", "kodim02"]


def test_dataset_guard_requires_all_24_named_images(tmp_path):
    for index in range(1, 25):
        (tmp_path / f"kodim{index:02d}.png").touch()
    assert len(dataset_files(tmp_path)) == 24
    (tmp_path / "kodim17.png").unlink()
    with pytest.raises(FileNotFoundError, match="kodim17.png"):
        dataset_files(tmp_path)


def test_summary_counts_all_images_and_averages_metrics():
    rows = [complete_row(image_id="kodim01"), complete_row(image_id="kodim02")]
    rows[0]["psnr_db"] = 20.0
    rows[1]["psnr_db"] = 30.0
    summary = summarize_rows(rows)
    assert summary[0]["images"] == 2
    assert summary[0]["psnr_db_mean"] == 25.0
