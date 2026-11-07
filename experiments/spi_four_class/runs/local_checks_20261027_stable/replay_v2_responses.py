from pathlib import Path
import hashlib
import json
import sys

sys.path.insert(0, str(Path.cwd()))
from parameters.spi_protocol import parse_evidence

response_dir = Path(
    "experiments/spi_four_class/runs/20261027_spi_evidence_v3a/"
    "retrieved_version2/experiments/spi_four_class/runs/"
    "20261027_spi_evidence_v3a/campaign/raw/qwen35_9b"
)
records = []
for path in sorted(response_dir.glob("visual_transport_*.txt")):
    raw = path.read_text(encoding="utf-8")
    item = {
        "response_file": path.name,
        "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    try:
        parsed, normalized_wrapper = parse_evidence(raw)
        item.update(
            status="parsed",
            wrapper_normalized=normalized_wrapper,
            claims={key: parsed[key] for key in (
                "action_needed", "substantial_consequence", "immediate_threat"
            )},
            level=parsed["level"],
            available=parsed["available"],
            reason=parsed["reason"],
        )
    except Exception as error:
        item.update(
            status="invalid",
            error_type=type(error).__name__,
            error=str(error),
        )
    records.append(item)

counts = {}
for item in records:
    key = item.get("level", "invalid")
    counts[key] = counts.get(key, 0) + 1
summary = {
    "response_count": len(records),
    "parsed_count": sum(item["status"] == "parsed" for item in records),
    "invalid_count": sum(item["status"] == "invalid" for item in records),
    "normalized_wrapper_count": sum(item.get("wrapper_normalized", False) for item in records),
    "level_counts": counts,
    "records": records,
    "semantic_repair_performed": False,
}
output = Path(
    "experiments/spi_four_class/runs/20261027_spi_evidence_v3a/"
    "retrieved_version2/replay_after_parserfix.json"
)
output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps({key: summary[key] for key in (
    "response_count", "parsed_count", "invalid_count",
    "normalized_wrapper_count", "level_counts", "semantic_repair_performed"
)}, indent=2))
