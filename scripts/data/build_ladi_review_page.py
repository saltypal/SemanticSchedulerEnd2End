"""Build an offline human-review form for the 30 proposed LADI evidence pairs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import html
import json
from pathlib import Path

from Evaluation.data.ladi_manifest import MANIFEST_FIELDS, STRATA, validate_ladi60_manifest


def build_review_page(manifest_path: str | Path, output_path: str | Path) -> Path:
    source = Path(manifest_path).resolve()
    target = Path(output_path).resolve()
    rows = validate_ladi60_manifest(source, require_reviewed=False)
    if target.parent != source.parent:
        raise ValueError("Review HTML must sit beside the manifest and images")
    groups = defaultdict(list)
    for row in rows:
        groups[row["evidence_group_id"]].append(row)
    ordered_groups = sorted(groups, key=lambda key: (
        STRATA.index(groups[key][0]["selection_stratum"]), key))
    cards = []
    for group in ordered_groups:
        images = []
        for row in sorted(groups[group], key=lambda item: item["image_id"]):
            relative_path = Path(row["image_path"]).relative_to(source.parent)
            image_id = html.escape(row["image_id"])
            images.append(f'''
              <article class="image">
                <img src="{html.escape(relative_path.as_posix())}" alt="{image_id}">
                <strong>{image_id}</strong>
                <label>Evidence sufficiency
                  <select data-sufficiency="{image_id}">
                    <option value="">Choose</option><option value="1">1 — usable</option>
                    <option value="0">0 — insufficient</option>
                  </select>
                </label>
              </article>''')
        cards.append(f'''
          <section class="group" data-group="{html.escape(group)}">
            <div class="group-heading"><h2>{html.escape(group)}</h2>
            <span>{html.escape(groups[group][0]["selection_stratum"])}</span></div>
            <div class="pair">{''.join(images)}</div>
            <label class="confirm"><input type="checkbox" data-confirm="{html.escape(group)}">
              I reviewed both images and confirm they represent related evidence.</label>
            <label class="note">Review note or override rationale
              <input type="text" data-note="{html.escape(group)}" placeholder="Optional"></label>
          </section>''')
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    columns = json.dumps(MANIFEST_FIELDS)
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LADI-60 evidence review</title>
<style>
  :root {{font:16px/1.45 system-ui,sans-serif;color:#13243a;background:#f3f7fb}}
  body {{max-width:1160px;margin:0 auto;padding:24px}}
  h1 {{font-size:2rem;margin-bottom:.2rem}} p {{max-width:850px}}
  .bar {{position:sticky;top:0;z-index:3;background:#123456;color:white;padding:14px 18px;
         border-radius:12px;display:flex;align-items:center;gap:16px;flex-wrap:wrap;box-shadow:0 8px 24px #16344a33}}
  .bar input {{padding:8px;border-radius:6px;border:0}} button {{padding:10px 15px;border:0;border-radius:8px;
          background:#18d6bb;color:#082f39;font-weight:700;cursor:pointer}}
  .group {{background:white;border:1px solid #d7e5ed;border-radius:14px;margin:20px 0;padding:18px;
          box-shadow:0 4px 18px #16344a0a}}
  .group-heading {{display:flex;justify-content:space-between;align-items:center;gap:12px}}
  .group-heading h2 {{font-size:1.1rem;margin:0 0 10px}} .group-heading span {{color:#4d6b7c}}
  .pair {{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}
  .image {{display:grid;gap:7px}} .image img {{width:100%;height:260px;object-fit:contain;background:#eaf0f4}}
  label {{display:grid;gap:5px}} select,.note input {{font:inherit;padding:8px;border:1px solid #adc4d2;border-radius:7px}}
  .confirm {{display:flex;align-items:center;margin-top:16px;font-weight:650}}
  .note {{margin-top:12px}} #status {{font-weight:700}}
  @media(max-width:650px) {{body {{padding:12px}} .pair {{grid-template-columns:1fr}} .image img {{height:210px}}}}
</style></head><body>
<h1>LADI-60 evidence review</h1>
<p>These 30 pairs were nominated by image similarity, not by verified event identity.
Confirm a pair only if receiving its first image would materially reduce the new
task evidence in its second image. A shared LADI label or similar scenery alone
is insufficient. Set P7 sufficiency for both views. If a pair is wrong, edit
its evidence-group assignment in the draft CSV, regenerate this page, and review again.</p>
<div class="bar"><label>Reviewer name <input id="reviewer" autocomplete="name" placeholder="Your real name"></label>
<span id="status">0 / 30 pairs confirmed</span><button id="export">Export reviewed CSV</button></div>
{''.join(cards)}
<script id="manifest" type="application/json">{data}</script>
<script>
const rows = JSON.parse(document.getElementById('manifest').textContent);
const columns = {columns};
const groups = [...document.querySelectorAll('[data-group]')];
const status = document.getElementById('status');
document.querySelectorAll('[data-confirm]').forEach(box => box.addEventListener('change', () => {{
  status.textContent = `${{document.querySelectorAll('[data-confirm]:checked').length}} / ${{groups.length}} pairs confirmed`;
}}));
function csvCell(value) {{ return '"' + String(value ?? '').replaceAll('"','""') + '"'; }}
document.getElementById('export').addEventListener('click', () => {{
  const reviewer = document.getElementById('reviewer').value.trim();
  if (!reviewer) {{ alert('Enter the real reviewer name.'); return; }}
  for (const group of groups) {{
    const id = group.dataset.group;
    if (!group.querySelector('[data-confirm]').checked) {{ alert('Review or regroup ' + id); return; }}
    if ([...group.querySelectorAll('[data-sufficiency]')].some(input => input.value === '')) {{
      alert('Set both P7 values in ' + id); return;
    }}
  }}
  const byImage = new Map([...document.querySelectorAll('[data-sufficiency]')]
      .map(input => [input.dataset.sufficiency, input.value]));
  const notes = new Map(groups.map(group => [group.dataset.group,
      group.querySelector('[data-note]').value.trim()]));
  for (const row of rows) {{
    row.image_path = 'images/' + row.image_id + '.png';
    row.evidence_sufficiency = byImage.get(row.image_id);
    row.annotation_source = 'human_reviewed_task_rubric';
    row.reviewer = reviewer;
    row.rationale = notes.get(row.evidence_group_id) || row.rationale || '';
    row.review_status = 'reviewed';
  }}
  const csv = [columns.map(csvCell).join(','), ...rows.map(row =>
      columns.map(name => csvCell(row[name])).join(','))].join('\r\n') + '\r\n';
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([csv], {{type:'text/csv;charset=utf-8'}}));
  link.download = 'ladi60_semantic_scheduler_v1_reviewed.csv';
  link.click(); URL.revokeObjectURL(link.href);
}});
</script></body></html>'''
    target.write_text(page, encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build_review_page(args.manifest, args.output))


if __name__ == "__main__":
    main()
