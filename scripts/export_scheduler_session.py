"""Export one explicitly selected local JSONL session to a private JSON snapshot."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def export(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    data = source.read_bytes()
    lines = data.splitlines(keepends=True)
    skipped_partial_tail = False
    records = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines)-1 or line.endswith(b'\n'):
                raise
            skipped_partial_tail = True
    if not records or records[0].get('type') != 'session_meta':
        raise ValueError('expected native Codex session metadata')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as handle:
        json.dump(dict(format='codex_session_snapshot_v1',
            exported_utc=datetime.now(timezone.utc).isoformat(),
            source=str(source), source_snapshot_sha256=hashlib.sha256(data).hexdigest(),
            skipped_incomplete_tail=skipped_partial_tail,
            privacy_notice='Private conversation/tool output. Review before sharing. Not a native sidebar import.',
            session_id=records[0]['payload']['id'], records=records), handle,
            ensure_ascii=False)
    reread = json.loads(destination.read_text(encoding='utf-8'))
    assert len(reread['records']) == len(records)
    print(json.dumps(dict(path=str(destination), records=len(records),
        bytes=destination.stat().st_size, sha256=hashlib.sha256(destination.read_bytes()).hexdigest())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    export(args.source, args.output)
