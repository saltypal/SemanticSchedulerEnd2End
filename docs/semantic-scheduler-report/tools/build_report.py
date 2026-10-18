"""Build the report's evidence-derived figures and standalone HTML.

Run with the repository's minimal scheduler venv. No application source is edited.
"""
from pathlib import Path
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess

ROOT = Path('/workspace/reports/semantic-scheduler')
REPO = Path('/workspace/SemanticSchedulerEnd2End')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))
os.environ.setdefault('MPLBACKEND', 'Agg')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                     'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False})
demo = json.loads((ROOT / 'execution/demo.json').read_text())
run_dir = Path(demo['output'])
traces = json.loads((run_dir / 'traces.json').read_text())
jobs = json.loads((run_dir / 'jobs.json').read_text())
assert len(jobs) == 6 and len(traces) == 4
assert json.loads((run_dir / 'status.json').read_text())['status'] == 'completed'
assert (run_dir / 'policy_comparison.png').stat().st_size > 1000
expected = {'fifo': [1, 2, 4, 5, 6], 'round_robin': [1, 2, 4, 5, 6],
            'fixed_priority': [2, 3, 5, 4, 1, 6],
            'priority_edf': [3, 2, 5, 4, 1, 6]}
by_id = {j['job_id']: j for j in jobs}
colors = {0: '#f8c8c4', 1: '#ffe1a5', 2: '#cce3f5'}
labels = {'fifo': 'FIFO', 'round_robin': 'Round robin',
          'fixed_priority': 'Fixed priority', 'priority_edf': 'Priority + EDF'}
fig, ax = plt.subplots(figsize=(10.8, 4.5))
fig.subplots_adjust(left=.17,right=.985,bottom=.24,top=.76)
for row, trace in enumerate(traces):
    seq = [int(k.rsplit('-', 1)[1]) for d in trace['decisions'] for k in d['job_ids']]
    assert seq == expected[trace['policy']]
    assert trace['metrics']['completed'] == (5 if row < 2 else 6)
    assert trace['metrics']['mean_wait_ms'] == (8 if row < 2 else 15)
    for decision in trace['decisions']:
        for key in decision['job_ids']:
            job = by_id[key]
            ax.barh(row, 10, left=decision['now_ms'], height=.64,
                    color=colors[job['priority_class']], edgecolor='#ffffff', linewidth=2)
            ax.text(decision['now_ms'] + 5, row, key[-4:],
                    ha='center', va='center', fontsize=11, color='#18283b')
    if row < 2:
        ax.plot(20, row + .39, marker='v', color='#b73832', markersize=6)
ax.set_yticks(range(4), [labels[t['policy']] for t in traces])
ax.invert_yaxis()
ax.set_xlim(0, 65)
ax.set_xticks(range(0, 61, 10))
ax.set_xlabel('Simulated time (milliseconds); each block is one ten-ms slot')
ax.grid(axis='x', alpha=.15)
ax.set_axisbelow(True)
fig.suptitle('Actual six-job replay: which picture card goes when?', x=.02,y=.985,
             ha='left',fontweight='bold',fontsize=15)
fig.legend(handles=[Patch(facecolor=c, label=f'Class {k}') for k, c in colors.items()],
          loc='upper left',bbox_to_anchor=(.16,.91),ncol=3,frameon=False,fontsize=10)
fig.text(.02, .025, 'The red marker at 20 ms identifies job 0003 expiry under FIFO and round robin.\n'
                    'Synthetic demands, one job per slot; no radio decoding or semantic inference occurs in this replay.',
         fontsize=9, color='#445267')
fig.savefig(ROOT / 'assets/scheduling-timeline.svg', bbox_inches='tight')
fig.savefig(ROOT / 'assets/scheduling-timeline.png', bbox_inches='tight', dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.6), layout='constrained')
ages = np.linspace(0, 60, 241)
axes[0].plot(ages, np.exp(-ages/10), color='#287566', lw=2.5)
axes[0].scatter([0, 5, 10, 20], np.exp(-np.array([0, 5, 10, 20])/10),
                color='#287566', s=25)
axes[0].set(title='Freshness: an older picture gets a fading sticker',
            xlabel='Seconds since capture', ylabel='Freshness score', ylim=(-.04, 1.05))
remaining = np.linspace(0, 90, 241)
axes[1].plot(remaining, np.clip(1-remaining/60, 0, 1), color='#b5483f', lw=2.5)
axes[1].set(title='Urgency: less time left means more pressure',
            xlabel='Seconds remaining before deadline', ylabel='Urgency score', ylim=(-.04, 1.05))
for ax in axes:
    ax.grid(alpha=.15)
    ax.set_yticks([0, .25, .5, .75, 1])
fig.savefig(ROOT / 'assets/parameter-curves.svg', bbox_inches='tight')
fig.savefig(ROOT / 'assets/parameter-curves.png', bbox_inches='tight', dpi=180)
plt.close(fig)

text = (ROOT / 'complete-report.md').read_text()
text = text.replace('{{TIMELINE}}',
    '![A fresh replay of all four policies. Block labels are job numbers; colors identify supplied priority classes.](assets/scheduling-timeline.svg)')
curve_anchor = 'These arithmetic examples do not claim fresh CLIP inference.'
if '](assets/parameter-curves.svg)' not in text:
    text = text.replace(curve_anchor,
        '![Shapes of the configured freshness and urgency formulas. These are calculated illustrations, not measured emergency outcomes.](assets/parameter-curves.svg)\n\n' + curve_anchor)
(ROOT / 'complete-report.md').write_text(text)
planning = ROOT / 'planning'
planning.mkdir(exist_ok=True)
for path in Path('/workspace/plans').glob('*.md'):
    shutil.copy2(path, planning / path.name)

subprocess.run(['pandoc', 'complete-report.md', '--from=markdown', '--to=html5',
                '--standalone', '--toc', '--toc-depth=1', '--embed-resources',
                '--css=assets/report.css', '--metadata=pagetitle:The Picture Messenger — Complete Scheduler Report',
                '--output=complete-report.html'], cwd=ROOT, check=True)
html_path = ROOT / 'complete-report.html'
html = html_path.read_text()
for word, cls in [('EXISTS', 'exists'), ('CHECKED NOW', 'checked'),
                  ('RECORDED EARLIER', 'recorded'), ('PROPOSED', 'proposed')]:
    html = html.replace(f'<strong>{word}</strong>', f'<strong class="status {cls}">{word}</strong>')
html_path.write_text(html)

summary = {
    'reviewed_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
    'focused_pytest': {'passed': 23, 'skipped': 1},
    'import_isolation_pytest': {'passed': 2},
    'external_worker_pytest': {'failed': 2, 'observed_cause': 'PyTorch absent in selected worker interpreter'},
    'fresh_demo_output': str(run_dir),
    'fixture_orders_verified_against_report': expected,
    'report_word_count': len(text.split()),
    'report_chapter_count': len(re.findall(r'^# \d+\.', text, flags=re.M)),
    'report_sha256': hashlib.sha256(text.encode()).hexdigest(),
    'trained_model_inference_run_for_report': False,
    'application_source_changed': False,
}
(ROOT / 'execution/report-evidence-summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
