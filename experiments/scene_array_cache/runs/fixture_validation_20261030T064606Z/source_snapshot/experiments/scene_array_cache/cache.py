"""Bounded scene-array cache; evaluator labels and SPI targets never enter detection."""
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import sqlite3
import sys
import time
import urllib.request

from parameters.scene_arrays import (CLASSES, OBJECTS, REGISTRY, REGISTRY_SHA256,
    WIRE_BYTES, SceneArray, summarize, unavailable, digest)

SETTINGS = {'conf': .25, 'iou': .5, 'imgsz': 640, 'max_det': 300,
            'agnostic_nms': False, 'half': False, 'augment': False, 'verbose': False}
DATA_REVISION = '10a5626ba112ab9c50369c473a2062bcf205d708'
WEIGHTS_SHA256 = '9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792'


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def code_hash(path):
    """Canonical UTF8/LF source identity across Windows and Kaggle writers."""
    return hashlib.sha256(Path(path).read_text(encoding='utf8').encode('utf8')).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    with tmp.open('w', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


@contextmanager
def stage(run, name):
    """Persist stdout/stderr alongside normal notebook output, including failures."""
    from contextlib import redirect_stdout, redirect_stderr
    run = Path(run)
    (run / 'logs').mkdir(parents=True, exist_ok=True)
    class Tee:
        def __init__(self, parent, log): self.parent, self.log = parent, log
        def write(self, text):
            self.parent.write(text); self.log.write(text); self.log.flush()
            return len(text)
        def flush(self): self.parent.flush(); self.log.flush()
        def isatty(self): return False
    start = time.monotonic()
    status = 'complete'
    with (run / 'logs' / (name + '.log')).open('a', encoding='utf8') as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            print('START', name, 'output', run, flush=True)
            try:
                yield
            except BaseException:
                import traceback
                status = 'failed'
                traceback.print_exc()
                raise
            finally:
                receipt = {'stage': name, 'status': status, 'seconds': time.monotonic() - start}
                write_json(run / 'logs' / (name + '.json'), receipt)
                print('END', receipt, flush=True)


def environment():
    values = {'python': sys.version, 'executable': sys.executable,
              'host': platform.node(), 'platform': platform.platform()}
    for name in ('torch', 'ultralytics', 'numpy', 'Pillow', 'clip', 'psutil'):
        try: values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: values[name] = 'not installed'
    try:
        import torch
        values['cuda_available'] = torch.cuda.is_available()
        values['gpus'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        values['torch_threads'] = torch.get_num_threads()
    except ImportError:
        values['cuda_available'] = False
    return values


class Detector:
    def __init__(self, weights, device):
        import torch
        from ultralytics import YOLOWorld
        start = time.perf_counter()
        weights = Path(weights)
        if file_hash(weights) != WEIGHTS_SHA256:
            raise ValueError('unapproved YOLO-World weights')
        self.device = device
        self.model = YOLOWorld(str(weights)).to(device)
        self.load_seconds = time.perf_counter() - start
        start = time.perf_counter()
        self.model.set_classes(list(CLASSES))
        self.sync()
        self.vocabulary_setup_seconds = time.perf_counter() - start
        features = self.model.model.txt_feats.detach().cpu().contiguous()
        self.signature = {'schema': 'scene_arrays_v1', 'registry_sha256': REGISTRY_SHA256,
            'weights_sha256': WEIGHTS_SHA256, 'settings': SETTINGS,
            'text_features_sha256': hashlib.sha256(features.numpy().tobytes()).hexdigest(),
            'ultralytics': importlib.metadata.version('ultralytics'),
            'torch': torch.__version__, 'device': device, 'dtype': 'float32',
            'input_convention': 'EXIF-upright RGB PIL, Ultralytics native letterbox',
            'postprocessing_source_sha256': code_hash(Path(__file__).parents[2] / 'parameters/scene_arrays.py'),
            'runner_source_sha256': code_hash(__file__)}
        self.producer_sha256 = digest(self.signature)
        # Text features are cached; the text encoder is not a per-image cost.
        self.model.model.clip_model = None
        import gc
        gc.collect()
        if device.startswith('cuda'): torch.cuda.empty_cache()
        self.parameters = sum(p.numel() for p in self.model.model.parameters())

    def sync(self):
        if self.device.startswith('cuda'):
            import torch
            torch.cuda.synchronize(self.device)

    def predict(self, image):
        result = self.model.predict(image, device=self.device, **SETTINGS)[0]
        names = [result.names[i] for i in range(len(result.names))]
        if tuple(names) != CLASSES:
            raise ValueError('detector vocabulary order differs from scheduler registry')
        rows = [{'label': result.names[int(i)], 'box': b, 'score': float(s)}
                for b, i, s in zip(result.boxes.xyxy.detach().cpu().tolist(),
                                   result.boxes.cls.detach().cpu().tolist(),
                                   result.boxes.conf.detach().cpu().tolist())]
        return rows


def get_image(row, directory, allow_download=False):
    """Bound network retries and check expected raw-image hashes when supplied."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = Path(row['image_path'])
    if source.is_file():
        path = source
    else:
        path = directory / (hashlib.sha256(row['image_id'].encode()).hexdigest() + '.image')
        if not path.exists():
            if not allow_download:
                raise FileNotFoundError('image missing; local bulk downloading is disabled')
            relative = row['image_path']
            if not relative.startswith('data_image/') or '..' in relative.split('/'):
                raise ValueError('not a pinned CrisisMMD image path')
            url = 'https://huggingface.co/datasets/QCRI/CrisisMMD/resolve/' + DATA_REVISION + '/' + relative
            for attempt in range(5):
                try:
                    suffix = '?download=true&attempt=' + str(attempt)
                    with urllib.request.urlopen(url + suffix, timeout=60) as response:
                        raw = response.read(20 * 1024 * 1024 + 1)
                    if len(raw) > 20 * 1024 * 1024:
                        raise ValueError('image exceeds declared 20MiB cap')
                    part = path.with_suffix('.partial')
                    part.write_bytes(raw)
                    part.replace(path)
                    break
                except Exception:
                    if attempt == 4: raise
                    time.sleep(min(16, 2 ** attempt))
    actual = file_hash(path)
    expected = row.get('image_sha256')
    if expected and expected != actual:
        raise ValueError('raw image SHA256 mismatch: ' + row['image_id'])
    return path, actual


def read_image(path):
    from PIL import Image, ImageOps
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert('RGB')


def extract(path, image_id, sha, detector):
    """Executed loading, predict, array and wire costs; cache reads use another path."""
    total = time.perf_counter()
    times = {}
    try:
        begin = time.perf_counter()
        image = read_image(path)
        times['image_load_exif_rgb_ms'] = (time.perf_counter() - begin) * 1000
        detector.sync(); begin = time.perf_counter()
        detections = detector.predict(image)
        detector.sync()
        times['detector_preprocess_forward_postprocess_transfer_ms'] = (time.perf_counter() - begin) * 1000
        begin = time.perf_counter()
        arrays, audit = summarize(detections, image.width, image.height,
                                  detector.producer_sha256, SETTINGS['max_det'])
        times['array_calculation_ms'] = (time.perf_counter() - begin) * 1000
        begin = time.perf_counter()
        wire = arrays.serialize()
        decoded = SceneArray.deserialize(wire, detector.producer_sha256)
        error = max(abs(a-b) for a, b in zip(arrays.largest_area, decoded.largest_area))
        if error > .5 / 255 + 1e-12: raise ValueError('area quantization bound violated')
        times['serialization_validation_ms'] = (time.perf_counter() - begin) * 1000
        status, reason = 'ok', None
    except Exception as error:
        arrays = unavailable(detector.producer_sha256)
        audit = {}
        wire = arrays.serialize()
        status, reason = 'failed', type(error).__name__ + ': ' + str(error)
    times['total_extraction_ms'] = (time.perf_counter() - total) * 1000
    return {'image_id': image_id, 'image_sha256': sha, 'scene': arrays.record(),
            'audit': audit, 'status': status, 'failure_reason': reason,
            'timing': times, 'measurement': 'fresh inference', 'wire_bytes': len(wire),
            'wire_hex': wire.hex()}, wire


def cache_all(rows, detector, run, allow_download=False, seconds_limit=14400, retry_failed=True):
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    ids = [r['image_id'] for r in rows]
    if len(set(ids)) != len(ids): raise ValueError('duplicate input IDs')
    identity = {'producer': detector.signature,
                'inputs': [{'image_id': r['image_id'], 'image_path': r['image_path'],
                            'image_sha256': r.get('image_sha256'), 'split': r.get('split'),
                            'group': r.get('group')} for r in rows]}
    expected = digest(identity)
    db = sqlite3.connect(run / 'scene_cache.sqlite')
    db.executescript('CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);'
                     'CREATE TABLE IF NOT EXISTS outputs(id TEXT PRIMARY KEY,record TEXT,wire BLOB,checksum TEXT);'
                     'CREATE TABLE IF NOT EXISTS previous_attempts(id TEXT,record TEXT,wire BLOB,checksum TEXT);')
    prior = db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
    if prior and prior[0] != expected:
        db.close(); raise ValueError('cache source, model, vocabulary or configuration mismatch')
    with db: db.execute('INSERT OR IGNORE INTO meta VALUES(?,?)', ('identity', expected))
    write_json(run / 'cache_identity.json', identity)
    write_json(run / 'registry.json', REGISTRY)
    write_json(run / 'environment.json', environment())
    started = time.monotonic()
    fresh = hits = 0
    try:
        for number, row in enumerate(rows):
            if time.monotonic() - started > seconds_limit:
                print('SESSION BUDGET: preserving checkpoint', flush=True); break
            prior = db.execute('SELECT record,wire,checksum FROM outputs WHERE id=?', (row['image_id'],)).fetchone()
            if prior:
                if hashlib.sha256(prior[0].encode() + prior[1]).hexdigest() != prior[2]:
                    raise ValueError('stored row integrity failure')
                SceneArray.deserialize(prior[1], detector.producer_sha256)
                old_record = json.loads(prior[0])
                if old_record['status'] == 'ok':
                    # Check measured content identity too, including initially unhashed local inputs.
                    verify = {**row, 'image_sha256': old_record['image_sha256']}
                    get_image(verify, run / 'images', allow_download)
                    hits += 1
                    continue
                if not retry_failed:
                    hits += 1
                    continue
                with db:
                    db.execute('INSERT INTO previous_attempts VALUES(?,?,?,?)', (row['image_id'], *prior))
            download_begin = time.perf_counter()
            try:
                path, sha = get_image(row, run / 'images', allow_download)
                fetch_ms = (time.perf_counter() - download_begin) * 1000
                record, wire = extract(path, row['image_id'], sha, detector)
                record['fetch_verify_ms'] = fetch_ms
            except Exception as error:
                array = unavailable(detector.producer_sha256); wire = array.serialize()
                record = {'image_id': row['image_id'], 'image_sha256': row.get('image_sha256'),
                          'scene': array.record(), 'status': 'failed',
                          'failure_reason': type(error).__name__ + ': ' + str(error),
                          'timing': {}, 'measurement': 'no inference: input failed',
                          'wire_bytes': len(wire), 'wire_hex': wire.hex()}
            # Routing/provenance fields are carried, never fed to scene extraction.
            record['source_group'] = row.get('group', 'tweet:' + row['image_id'].rsplit('_', 1)[0])
            record['source_split'] = row.get('split')
            text = json.dumps(record, sort_keys=True, allow_nan=False)
            checksum = hashlib.sha256(text.encode() + wire).hexdigest()
            with db: db.execute('INSERT OR REPLACE INTO outputs VALUES(?,?,?,?)', (row['image_id'], text, wire, checksum))
            fresh += 1
            if fresh % 25 == 0:
                print('SCENE CACHE', number+1, '/', len(rows), 'fresh', fresh, 'cache_hits', hits, flush=True)
                export(db, run, len(rows))
        return export(db, run, len(rows))
    finally:
        try:
            if sys.exc_info()[0] is None:
                export(db, run, len(rows))
            else:
                try:
                    export(db, run, len(rows))
                except Exception as error:
                    print('Partial export failed; preserving original error and SQLite:', str(error), flush=True)
        finally:
            db.close()


def export(db, run, expected_count):
    """One row per image, atomic flat JSONL and scheduler-ready numeric matrix."""
    import numpy as np
    run = Path(run)
    features, wire_features, ids, successful = [], [], [], []
    parts = run / 'scene_records.jsonl.partial'
    failures = []
    with parts.open('w', encoding='utf8') as stream:
        for i, text, wire, checksum in db.execute('SELECT id,record,wire,checksum FROM outputs ORDER BY id'):
            if hashlib.sha256(text.encode() + wire).hexdigest() != checksum:
                raise ValueError('cannot export corrupt cache row')
            record = json.loads(text); array = SceneArray.from_record(record['scene'])
            decoded = SceneArray.deserialize(wire, array.producer_sha256)
            stream.write(text + '\n')
            ids.append(i); successful.append(record['status'] == 'ok')
            features.append(array.features(array.producer_sha256))
            wire_features.append(decoded.features(array.producer_sha256))
            if record['status'] != 'ok': failures.append({'image_id': i, 'reason': record['failure_reason']})
        stream.flush(); os.fsync(stream.fileno())
    parts.replace(run / 'scene_records.jsonl')
    with (run / 'scheduler_arrays.npz.partial').open('wb') as stream:
        np.savez_compressed(stream, image_ids=np.array(ids),
                            float_features=np.asarray(features, dtype=np.float32).reshape(-1, 130),
                            wire_features=np.asarray(wire_features, dtype=np.float32).reshape(-1, 130),
                            successful=np.array(successful, dtype=bool), registry_sha256=REGISTRY_SHA256,
                            producer_sha256=(json.loads(db.execute('SELECT record FROM outputs LIMIT 1').fetchone()[0])['scene']['producer_sha256'] if ids else ''),
                            feature_layout='log1p_counts32,areas32,count_masks32,area_masks32,ok,saturated')
    (run / 'scheduler_arrays.npz.partial').replace(run / 'scheduler_arrays.npz')
    progress = {'stored': len(ids), 'expected': expected_count, 'successful': sum(successful),
                'failed': len(failures), 'coverage_complete': len(ids) == expected_count,
                'all_successful': len(ids) == expected_count and not failures,
                'wire_bytes_per_scene_block': WIRE_BYTES,
                'accuracy_status': 'not measured; no reference boxes supplied'}
    write_json(run / 'progress.json', progress)
    write_json(run / 'failures.json', failures)
    return progress


def profile(rows, weights, run, devices=('cpu', 'cuda:0')):
    """Ten fixed development images, 10 warmups, 30 CPU/100 GPU fresh calls."""
    import gc
    import numpy as np
    import psutil
    import torch
    run = Path(run)
    fixed = [r for r in rows if r.get('split') == 'train'][:10]
    if len(fixed) != 10: raise ValueError('ten declared development images required')
    paths = [get_image(r, run / 'images', False) for r in fixed]
    for device in devices:
        print('PROFILE', device, '10 fixed development images', flush=True)
        detector = Detector(weights, device)
        warm_path, warm_sha = paths[0]
        for _ in range(10):
            result, _ = extract(warm_path, fixed[0]['image_id'], warm_sha, detector)
            if result['status'] != 'ok': raise ValueError('profiling warmup failed: ' + result['failure_reason'])
        measured = []
        repeats = 3 if device == 'cpu' else 10
        for row, (path, sha) in zip(fixed, paths):
            for _ in range(repeats):
                result, _ = extract(path, row['image_id'], sha, detector)
                if result['status'] != 'ok': raise ValueError('profile invocation failed')
                measured.append({'image_id': row['image_id'], **result['timing']})
        detector.sync(); start = time.perf_counter()
        for row, (path, sha) in zip(fixed, paths):
            result, _ = extract(path, row['image_id'], sha, detector)
            if result['status'] != 'ok': raise ValueError('throughput pass failed')
        detector.sync(); elapsed = time.perf_counter() - start
        if device.startswith('cuda'): torch.cuda.reset_peak_memory_stats(device)
        result, _ = extract(warm_path, fixed[0]['image_id'], warm_sha, detector)
        if result['status'] != 'ok': raise ValueError('memory pass failed')
        detector.sync()
        memory = {'rss_bytes': psutil.Process().memory_info().rss,
                  'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None,
                  'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.startswith('cuda') else None,
                  'scope': 'absolute process memory, separate pass from timing'}
        try:
            image = read_image(warm_path)
            with torch.inference_mode(), torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as trace:
                detector.predict(image)
            events = trace.key_averages()
            flops = {'status': 'partial', 'supported_flops': sum(e.flops for e in events),
                     'uncounted_operators': [e.key for e in events if not e.flops],
                     'convention': 'two FLOPs per MAC', 'vocabulary_size': 32,
                     'image_size': 640, 'batch_size': 1, 'dtype': 'float32',
                     'scope': 'detector invocation including supported preprocess/postprocess operators; not full pipeline arithmetic'}
        except Exception as error:
            flops = {'status': 'failed', 'supported_flops': None, 'reason': str(error)}
        summary = {key: {'p50_ms': float(np.median([r[key] for r in measured])),
                         'p95_ms': float(np.percentile([r[key] for r in measured], 95))}
                   for key in measured[0] if key.endswith('_ms')}
        report = {'device': device, 'host': platform.node(), 'environment': environment(),
                  'producer': detector.signature, 'samples': measured, 'timing_summary': summary,
                  'warmup_calls': 10, 'measured_calls': len(measured),
                  'batch_one_sequential_throughput': 10 / elapsed,
                  'throughput_scope': '10 sequential batch-one full extractions; not batched inference',
                  'memory': memory, 'flops': flops, 'parameters': detector.parameters,
                  'weights_bytes': Path(weights).stat().st_size,
                  'model_object_load_seconds': detector.load_seconds,
                  'vocabulary_setup_seconds': detector.vocabulary_setup_seconds,
                  'load_scope': 'fresh object; OS/model download caches may be warm',
                  'scene_block_bytes': WIRE_BYTES}
        write_json(run / 'profiles' / (device.replace(':', '_') + '.json'), report)
        del detector; gc.collect()
        if device.startswith('cuda'): torch.cuda.empty_cache()


def report(run):
    """Cost/availability summary; never claims detector accuracy or ranking benefit."""
    import numpy as np
    run = Path(run)
    records = [json.loads(line) for line in (run / 'scene_records.jsonl').read_text().splitlines()]
    records = [r for r in records if r['status'] == 'ok']
    if not records: raise ValueError('no successful records for summary')
    arrays = [SceneArray.from_record(r['scene']) for r in records]
    write_json(run / 'summary.json', {
        'successful': len(records), 'saturated': sum(a.saturated for a in arrays),
        'all_zero_detections': sum(not any(a.largest_area) for a in arrays),
        'per_class_detected_images': {c: sum(a.largest_area[i] > 0 for a in arrays) for i, c in enumerate(CLASSES)},
        'area_zeroed_by_quantization_per_class': {
            c: sum(a.largest_area[i] > 0 and round(a.largest_area[i] * 255) == 0 for a in arrays)
            for i, c in enumerate(CLASSES)},
        'actual_fresh_extraction_p50_ms': float(np.median([r['timing']['total_extraction_ms'] for r in records])),
        'actual_fresh_extraction_p95_ms': float(np.percentile([r['timing']['total_extraction_ms'] for r in records], 95)),
        'area_quantization_max_error_bound': .5 / 255,
        'retained_evidence': 'labels/boxes/scores in audit; arrays contain counts and largest rectangles',
        'inference_scope': 'detector-dependent scene composition; not urgency, depth or complete scene semantics',
        'supervision_used': False, 'detector_accuracy_measured': False,
        'scheduler_benefit_measured': False})
    artifacts = [{'path': str(p.relative_to(run)), 'bytes': p.stat().st_size, 'sha256': file_hash(p)}
                 for p in sorted(run.rglob('*')) if p.is_file() and 'images' not in p.relative_to(run).parts
                 and not p.name.endswith(('.partial', '.sqlite-journal')) and p.name != 'artifact_manifest.json']
    write_json(run / 'artifact_manifest.json', artifacts)
    return json.loads((run / 'summary.json').read_text())
