"""Build a readable Kaggle/local notebook from canonical shared source."""
from pathlib import Path
import hashlib
import json
import subprocess
import nbformat as nb

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'experiments/scene_array_cache'
selection = ROOT / 'experiments/full_medic_labeling/runs/crisismmd4000_selection/selected_1000.json'
rows = json.loads(selection.read_text(encoding='utf8'))
hashes = {}
checkpoint = ROOT / 'experiments/full_medic_labeling/runs/crisismmd4000/checkpoint_audit/labels.jsonl'
for row in map(json.loads, checkpoint.read_text(encoding='utf8').splitlines()):
    hashes[row['item_id']] = row['image_sha256']
cohort = [(r['image_id'], r['image_path'], r['split'], hashes.get(r['image_id'])) for r in rows]
assert len(cohort) == 4000 and len({r[0] for r in cohort}) == 4000
sources = ['parameters/scene_vocabulary.py', 'parameters/scene_arrays.py', 'experiments/scene_array_cache/cache.py']
source_receipt = {p: hashlib.sha256((ROOT/p).read_text(encoding='utf8').encode('utf8')).hexdigest() for p in sources}
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
cells = []
md = lambda text: cells.append(nb.v4.new_markdown_cell(text))
code = lambda text: cells.append(nb.v4.new_code_cell(text))
md('''# Scene-array caching â€” frozen implementation v1

**Deliverable:** a source-qualified scene block for every one of the existing 4,000 CrisisMMD images.
This notebook does not relabel SPI, train a scheduler, or interrupt the current SPI notebook.

```
Pinned image â†’ EXIF/RGB â†’ YOLO-World with 93 explicit prompts
             â†’ per-class counts/areas/masks â†’ 369-byte scene block
             â†’ atomic SQLite cache â†’ JSONL + 374-feature scheduler arrays
```

The vocabulary and method are frozen for this implementation. New domain accuracy and scheduling benefit
remain unmeasured. `fixture` mode validates software contracts only; `real` mode performs actual detection.
Do not interpret successful caching, larger counts, or detector confidence as urgency validation.
''')
md('''## 0. Environment and execution identity

Canonical repository: `SemanticSchedulerEnd2End`. Local interpreter: its `.venv`.
Kaggle uses the managed runtime, not the laptop environment. Keep managed torch/CUDA unchanged.
Readable source snapshots below are generated from the canonical modules; no Base64 or hidden source archive.

Default execution is real on Kaggle, fixture locally. Set `SCENE_ARRAY_MODE=real` explicitly for local
inference. Local image downloads remain disabled. All generated files stay in this experiment's run directory.
''')
code('''print('ENVIRONMENT START', flush=True)
import os, sys, json, time, subprocess, hashlib, importlib.metadata
from pathlib import Path
IS_KAGGLE = Path('/kaggle/working').exists()
MODE = os.environ.get('SCENE_ARRAY_MODE', 'real' if IS_KAGGLE else 'fixture')
if MODE not in ('real', 'fixture'): raise ValueError('MODE must be real or fixture')
if IS_KAGGLE:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-deps', '-q',
        'ultralytics==8.3.237', 'git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
        'ftfy', 'regex', 'tqdm', 'psutil==7.2.2'])
    PROJECT = Path('/kaggle/working')
else:
    candidates = [Path.cwd(), *Path.cwd().parents]
    PROJECT = next((p for p in candidates if (p/'parameters/scene_arrays.py').is_file()), None)
    if PROJECT is None: raise RuntimeError('run locally from canonical repository or a child directory')
RUN_ID = os.environ.get('SCENE_ARRAY_RUN_ID', time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
if Path(RUN_ID).name != RUN_ID: raise ValueError('run ID must be one path component')
RUN = Path(os.environ.get('SCENE_ARRAY_RESUME_DIR', str(PROJECT/'experiments/scene_array_cache/runs'/RUN_ID)))
RUN.mkdir(parents=True, exist_ok=True)
BOOTSTRAP = RUN/'source_snapshot'
(BOOTSTRAP/'parameters').mkdir(parents=True, exist_ok=True)
(BOOTSTRAP/'experiments/scene_array_cache').mkdir(parents=True, exist_ok=True)
for package in ('parameters', 'experiments', 'experiments/scene_array_cache'):
    (BOOTSTRAP/package/'__init__.py').write_text('# Generated runtime package; canonical source is in the repository.\\n')
os.chdir(BOOTSTRAP)
sys.path.insert(0, str(BOOTSTRAP))
print('Actual interpreter', sys.executable, sys.version, 'mode', MODE, 'run', RUN, flush=True)
print('ENVIRONMENT END', flush=True)
''')
md('''## 1. Readable shared implementation

These cells materialize a verified runtime snapshot under this run directory. They never overwrite
the working repository. The canonical source hashes are checked immediately afterward.
''')
for relative in sources:
    code('%%writefile ' + relative + '\n' + (ROOT/relative).read_text(encoding='utf8'))
code('''from experiments.scene_array_cache.cache import (Detector, SETTINGS, WEIGHTS_SHA256, stage,
    write_json, file_hash, code_hash, environment, cache_all, report, profile, get_image, read_image, extract)
from parameters.scene_arrays import (CLASSES, OBJECTS, REGIONS, REGISTRY, REGISTRY_SHA256,
    WIRE_BYTES, N_CLASSES, FEATURE_COUNT, SceneArray, summarize, unavailable, digest)
SOURCE_RECEIPT = ''' + repr(source_receipt) + '''
SOURCE_COMMIT = ''' + repr(commit) + '''
with stage(RUN, '01_source_environment'):
    for relative, expected in SOURCE_RECEIPT.items():
        if code_hash(BOOTSTRAP/relative) != expected: raise ValueError('source snapshot hash mismatch')
    write_json(RUN/'source.json', {'commit': SOURCE_COMMIT, 'files': SOURCE_RECEIPT,
        'scope': 'selected source UTF8/LF from dirty checkout; commit alone does not identify these edits'})
    write_json(RUN/'environment.json', environment())
    print(environment(), flush=True)
''')
md('''## 2. Frozen vocabulary, representation and settings

85 object categories have count + area. Eight region categories have area only; their count masks
are false. Areas are rectangles, not segmentation or distance. Default COCO classes are replaced
explicitly. Zero detections is different from detector failure. Outside-vocabulary queries are errors,
not zero-valued observations. A saturated result can undercount and carries its cap flag.

Counts use uint16; areas use uint8 with rounding. Full vocabulary and producer SHA256 values,
status, and masks produce an exact **369-byte standalone scene block**. This is not the old 35-byte
header and does not include SPI, relevance, novelty or routing metadata. Float and wire-derived features
are exported separately so the scheduler can train on the representation it will receive.
''')
code('''with stage(RUN, '02_contract'):
    print('SETTINGS', SETTINGS, flush=True)
    for entry in REGISTRY['classes']: print(entry, flush=True)
    write_json(RUN/'registry.json', REGISTRY)
    write_json(RUN/'configuration.json', {'mode': MODE, 'settings': SETTINGS,
        'registry_sha256': REGISTRY_SHA256, 'scene_block_bytes': WIRE_BYTES,
        'feature_layout': ['log1p(counts)[93]', 'areas[93]', 'count masks[93]',
                           'area masks[93]', 'detector_ok', 'saturated'],
        'detector_accuracy_status': 'not measured for this expanded vocabulary'})
''')
md('''## 3. Exact existing 4,000-image cohort

This table contains identities and source paths only. Native labels, SPI labels, mission paragraphs
and future observations do not enter detection. The source dataset revision is pinned. Existing raw
image hashes are checked; new hashes are measured and saved, not claimed known in advance.
Data are fetched only on Kaggle or read from supplied local image paths. No full archive is downloaded.
''')
code('COHORT_ROWS = ' + repr(cohort) + '''
ROWS = [{'image_id': i, 'image_path': path, 'split': split,
         'image_sha256': sha, 'group': 'tweet:'+i.rsplit('_',1)[0]}
        for i, path, split, sha in COHORT_ROWS]
if not IS_KAGGLE and MODE == 'real':
    IMAGE_ROOT = Path(os.environ.get('SCENE_ARRAY_IMAGE_ROOT', str(PROJECT)))
    for row in ROWS:
        row['source_image_path'] = row['image_path']
        row['image_path'] = str((IMAGE_ROOT/row['image_path']).resolve())
with stage(RUN, '03_inputs'):
    assert len(ROWS) == 4000 and len({r['image_id'] for r in ROWS}) == 4000
    write_json(RUN/'input_manifest.json', ROWS)
    print('INPUTS', len(ROWS), 'known image hashes', sum(bool(r['image_sha256']) for r in ROWS), flush=True)
''')
md('''## 4. Numerical contract checks and isolated fixture mode

Fixtures test software behavior, never detector accuracy. They do not enter the real image cohort.
Checks include count/area semantics, quantization, failure versus empty output, and a rejected registry mismatch.
''')
code('''class FixtureDetector:
    signature = {'fixture': True, 'registry_sha256': REGISTRY_SHA256}
    producer_sha256 = digest(signature)
    device = 'cpu'
    def sync(self): pass
    def predict(self, image):
        return [{'label': 'person', 'box': [0, 0, 8, 8], 'score': .9},
                {'label': 'water', 'box': [0, 16, 32, 32], 'score': .8}]

with stage(RUN, '04_numerical_checks'):
    test_hash = '1'*64
    array, audit = summarize([{'label': 'person', 'box': [0,0,20,20], 'score': .9},
                              {'label': 'water', 'box': [0,50,100,100], 'score': .8}],
                             100, 100, test_hash)
    assert array.counts[0] == 1 and array.largest_area[0] == .04
    assert array.count_available[CLASSES.index('water')] is False and array.largest_area[CLASSES.index('water')] == .5
    decoded = SceneArray.deserialize(array.serialize(), test_hash)
    assert len(array.serialize()) == 369 and len(decoded.features(test_hash)) == FEATURE_COUNT
    assert max(abs(a-b) for a,b in zip(array.largest_area, decoded.largest_area)) <= .5/255
    empty, _ = summarize([], 100, 100, test_hash)
    assert empty.detector_ok and empty.count_available[0] and empty.counts[0] == 0
    assert not unavailable(test_hash).count_available[0]
    if MODE == 'fixture':
        from PIL import Image
        folder = RUN/'fixture_inputs'; folder.mkdir(exist_ok=True)
        ROWS = []
        for number in range(12):
            path = folder/(str(number)+'.png')
            Image.new('RGB', (32,32), (number*10, 20, 30)).save(path)
            ROWS.append({'image_id': 'fixture_'+str(number), 'image_path': str(path),
                         'image_sha256': file_hash(path), 'split': 'train'})
    print('Numerical checks passed; workload', len(ROWS), MODE, flush=True)
    write_json(RUN/'effective_input_manifest.json', ROWS)
''')
md('''## 5. Model identity and real-image preflight

One YOLO-World small checkpoint, FP32, one image at a time. Class text encoding is a one-time setup
cost and its actual feature hash is part of the producer identity. Retain the managed CUDA runtime.
Warmup calls are reported separately and excluded from the final per-image cache measurements.
''')
code('''with stage(RUN, '05_detector_preflight'):
    if MODE == 'fixture':
        DETECTOR = FixtureDetector(); WEIGHTS = None
        print('Fixture provider: no model inference or detector accuracy validation', flush=True)
    else:
        import torch
        DEVICE = os.environ.get('SCENE_ARRAY_DEVICE', 'cuda:0' if IS_KAGGLE else 'cpu')
        if IS_KAGGLE and not torch.cuda.is_available(): raise RuntimeError('Kaggle GPU unavailable')
        if IS_KAGGLE:
            from ultralytics.utils.downloads import attempt_download_asset
            WEIGHTS = Path(attempt_download_asset(str(RUN/'yolov8s-worldv2.pt')))
        else:
            WEIGHTS = Path(os.environ.get('SCENE_ARRAY_WEIGHTS', str(PROJECT/'models/yolov8s-worldv2.pt')))
        DETECTOR = Detector(WEIGHTS, DEVICE)
        write_json(RUN/'producer.json', {**DETECTOR.signature,
            'producer_sha256': DETECTOR.producer_sha256,
            'load_seconds': DETECTOR.load_seconds,
            'vocabulary_setup_seconds': DETECTOR.vocabulary_setup_seconds})
        real_checks = []
        for row in [r for r in ROWS if r['split']=='train'][:10]:
            path, sha = get_image(row, RUN/'images', IS_KAGGLE)
            result, _ = extract(path, row['image_id'], sha, DETECTOR)
            real_checks.append({'image_id': row['image_id'], 'status': result['status'],
                                'reason': result['failure_reason']})
        write_json(RUN/'preflight.json', real_checks)
        if len(real_checks) != 10 or any(r['status'] != 'ok' for r in real_checks):
            raise RuntimeError('real-image preflight failed; inspect saved evidence')
        first = next(r for r in ROWS if r['split']=='train')
        path, _ = get_image(first, RUN/'images', IS_KAGGLE)
        image = read_image(path)
        for _ in range(10): DETECTOR.predict(image)
        DETECTOR.sync()
        print('Real preflight and 10 warmups completed', flush=True)
''')
md('''## 6. Cache every image, commit each record, export every 25

Use `SCENE_ARRAY_RESUME_DIR` to resume the same run. Source, input list, vocabulary, weights,
text features and detector configuration must match exactly. Failed rows stay explicit; success is
not manufactured by dropping them. The notebook reports partial coverage and failures separately.
SQLite commits survive ordinary process failure; durable Kaggle output retrieval must still be verified.
''')
code('''with stage(RUN, '06_cache'):
    CACHE_PROGRESS = cache_all(ROWS, DETECTOR, RUN, allow_download=IS_KAGGLE, retry_failed=True,
                              seconds_limit=int(os.environ.get('SCENE_ARRAY_SECONDS', '14400')))
    print('CACHE RESULT', CACHE_PROGRESS, flush=True)
    if not CACHE_PROGRESS['coverage_complete']:
        raise RuntimeError('bounded session checkpointed; resume required before declaring completion')
''')
md('''## 7. Separate CPU/GPU cost passes

Release the bulk producer before profiling. For each backend: fresh model object and vocabulary setup,
ten warmups, ten fixed development images, three CPU calls/image or ten GPU calls/image. Synchronize GPU
work. Measure memory separately. Throughput is sequential batch-one throughput, not a batch-ten claim.
Supported forward FLOPs use two FLOPs/MAC; uncounted operators remain listed and totals stay **partial**.
No training or VLM runs occur in this scene-only notebook. Network/download time is separate from inference.
''')
code('''with stage(RUN, '07_cost_profiles'):
    import gc
    del DETECTOR
    gc.collect()
    if MODE == 'real':
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        devices = ('cpu', 'cuda:0') if torch.cuda.is_available() else ('cpu',)
        profile(ROWS, WEIGHTS, RUN, devices)
        if 'cuda:0' not in devices:
            write_json(RUN/'profiles/cuda_0.json', {'status': 'unavailable', 'reason': 'no GPU assigned'})
    else:
        write_json(RUN/'profiles/fixture.json', {'status': 'not executed',
            'reason': 'fixture contract checks are not model latency, FLOPs or memory measurements'})
''')
md('''## 8. Coverage, failures, quantization and scheduler handoff

The NPZ has ID-aligned float and quantized-wire feature matrices. Use wire features for deployment-matched
training and preserve the vocabulary/producer contracts. This notebook exports neither SPI labels nor
mission/history values; join those separately by verified identity and enforce your grouped splits.
''')
code('''with stage(RUN, '08_results'):
    SUMMARY = report(RUN)
    print(json.dumps(SUMMARY, indent=2), flush=True)
    import numpy as np
    with np.load(RUN/'scheduler_arrays.npz', allow_pickle=False) as data:
        assert data['float_features'].shape == (len(ROWS), FEATURE_COUNT)
        assert len(set(data['image_ids'].tolist())) == len(ROWS)
        wire_error = float(np.abs(data['float_features'][:,N_CLASSES:2*N_CLASSES] - data['wire_features'][:,N_CLASSES:2*N_CLASSES]).max())
        assert wire_error <= .5/255 + 1e-7
        print('Actual maximum area quantization error', wire_error, flush=True)
    write_json(RUN/'validation.json', {'mode': MODE, 'records': len(ROWS),
        'software_contract_checks': 'passed', 'area_quantization_max_error': wire_error,
        'real_detector_accuracy': 'not measured', 'scheduler_advantage': 'not evaluated',
        'cache_progress': CACHE_PROGRESS})
''')
md('''## 9. Inspect detections and save diagnostics

These plots expose category coverage, coordinate/count failures and example detections. They are not
accuracy plots. Region counts remain masked and are never displayed as numbers of floods or fires.
''')
code('''with stage(RUN, '09_diagnostics'):
    import matplotlib.pyplot as plt
    from PIL import ImageDraw
    (RUN/'figures').mkdir(exist_ok=True)
    counts = SUMMARY['per_class_detected_images']
    fig, ax = plt.subplots(figsize=(12,4))
    ax.bar(range(len(CLASSES)), [counts[c] for c in CLASSES])
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=70, ha='right')
    ax.set_ylabel('Images with retained detection')
    ax.set_title('Detector coverage by configured class â€” accuracy not established')
    fig.tight_layout(); fig.savefig(RUN/'figures/class_coverage.png', dpi=140); plt.show()
    cached = [json.loads(s) for s in (RUN/'scene_records.jsonl').read_text().splitlines()]
    lookup = {r['image_id']: r for r in ROWS}
    for number, record in enumerate([r for r in cached if r['status']=='ok'][:4]):
        source = lookup[record['image_id']]
        path, _ = get_image(source, RUN/'images', False)
        image = read_image(path); draw = ImageDraw.Draw(image)
        for detected in record['audit']['detections']:
            draw.rectangle(detected['box'], outline='red', width=2)
            draw.text(tuple(detected['box'][:2]), detected['label'], fill='red')
        image.thumbnail((700,700)); image.save(RUN/'figures'/('example_'+str(number)+'.png'))
        display(image)
    print('Figures saved', RUN/'figures', flush=True)
''')
md('''## 10. Final artifact inventory and acceptance

Retrieve `scene_cache.sqlite`, JSONL, NPZ, registry, source/environment identities, profiles, figures and
validation receipts. Avoid pulling the `images/` cache or model weights onto the laptop by default.
Every record must be inspected through its success/mask fields. Full coverage with failures is incomplete
extraction, even when the notebook itself reached the end. The old 35-byte header is not modified here.
''')
code('''with stage(RUN, '10_finalize'):
    report(RUN)
    write_json(RUN/'ACCEPTANCE.json', {'mode': MODE, 'coverage_complete': CACHE_PROGRESS['coverage_complete'],
        'all_images_successful': CACHE_PROGRESS['all_successful'], 'status': 'passed' if CACHE_PROGRESS['all_successful'] else 'incomplete',
        'claims': 'software and cache status only; semantic detection accuracy and ranking benefit unmeasured'})
    print('OUTPUT', RUN, 'scene bytes', WIRE_BYTES, 'features', FEATURE_COUNT, flush=True)
    if not CACHE_PROGRESS['all_successful']:
        raise RuntimeError('extraction has explicit failed records; inspect failures.json')
report(RUN)  # Final inventory includes the acceptance and completed stage receipts.
''')
md('## Queued scheduler class-group handoff\nAll93classes retained. Class groups are diagnostics, never fixed importance weights. Later group ablations must remove counts/areas/bothmasks together and refit on identical supervision, splits and traces. See class_groups.json. Tree is not configured in the current dictionary; screenshots and relations remain unsupported.')
code("CLASS_GROUPS = {'registry_sha256': '37181cef81ba6f9c777ab05fefe182a50dedaeda8454c7ad9216ff17e1eb91c0', 'groups': {'people': [0], 'transport': [1, 2, 3, 4, 5, 6, 7, 8], 'infrastructure': [80, 81, 82, 9, 83, 11, 85, 10, 12], 'environment': [86, 87, 88, 89, 90, 91, 92], 'animals': [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 84], 'indoor_devices': [56, 57, 60, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 58], 'other_everyday': [13, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 73, 74, 75, 76, 77, 78, 79]}, 'names': {'people': ['person'], 'transport': ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat'], 'infrastructure': ['building', 'bridge', 'utility pole', 'traffic light', 'traffic sign', 'stop sign', 'road', 'fire hydrant', 'parking meter'], 'environment': ['water', 'debris', 'smoke', 'flame', 'snow', 'ground', 'grass'], 'animals': ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal'], 'indoor_devices': ['chair', 'couch', 'dining table', 'bed', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'potted plant'], 'other_everyday': ['bench', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']}, 'role': 'diagnostics, not priority weights', 'ablation': 'remove count/area/bothmask columns together, refit same scorer, same splits/traces/tuning budget', 'limitations': ['generic animal overlaps specific animals', 'traffic sign overlaps stop sign', 'tree not configured: do not infer zero', 'screenshots/conditions unsupported by object counts']}\nwith stage(RUN, '11_scheduler_handoff'):\n    assert CLASS_GROUPS['registry_sha256'] == REGISTRY_SHA256\n    flat=[i for group in CLASS_GROUPS['groups'].values() for i in group]\n    assert len(flat)==N_CLASSES and set(flat)==set(range(N_CLASSES))\n    write_json(RUN/'class_groups.json', CLASS_GROUPS)\n    print('Class-group handoff exported',flush=True)\nreport(RUN)")
notebook = nb.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.13'},
    'source_commit': commit, 'source_files': source_receipt})
nb.validate(notebook)
HERE.mkdir(exist_ok=True)
target = HERE/'Scene_Array_Caching.ipynb'
nb.write(notebook, target)
assert target.stat().st_size < 1000000
(HERE/'source_manifest.json').write_text(json.dumps({
    'commit': commit, 'files': source_receipt,
    'notebook_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
    'cohort_count': len(cohort), 'known_raw_image_hashes': len(hashes)}, indent=2), encoding='utf8')
print(target, 'cells', len(cells), 'bytes', target.stat().st_size)
