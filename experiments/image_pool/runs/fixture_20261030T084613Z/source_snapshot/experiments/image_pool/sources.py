"""Small cloud-only acquisitions. Unavailable sources remain explicit, never bulk-downloaded."""
import csv
import hashlib
import io
import json
import random
import time
import urllib.request
import zipfile
from pathlib import Path

from experiments.scene_array_cache.cache import file_hash, write_json


class Budget:
    """Persist bytes actually transferred, including failed/partial requests."""
    def __init__(self, root, limit, cloud):
        self.root, self.limit, self.cloud = Path(root), int(limit), cloud
        self.root.mkdir(parents=True, exist_ok=True)
        self.receipt = self.root/'download_budget.json'
        self.used = json.loads(self.receipt.read_text())['used_bytes'] if self.receipt.exists() else 0

    def fetch(self, url, target, cap):
        target = Path(target)
        if target.is_file():
            if target.stat().st_size > cap: raise ValueError('cached file exceeds cap')
            return target
        if not self.cloud: raise RuntimeError('network image/metadata acquisition is cloud-only')
        if not url.startswith('https://'): raise ValueError('HTTPS input required')
        target.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            partial = target.with_suffix('.partial')
            written = 0
            try:
                with urllib.request.urlopen(url, timeout=60) as response, partial.open('wb') as out:
                    length = int(response.headers.get('Content-Length', '0'))
                    if length > cap or length > self.limit-self.used:
                        raise ValueError('content length exceeds declared download budget')
                    while True:
                        chunk = response.read(65536)
                        if not chunk: break
                        self.used += len(chunk); written += len(chunk)
                        if written > cap or self.used > self.limit:
                            raise ValueError('stream exceeded download cap')
                        out.write(chunk)
                partial.replace(target)
                return target
            except Exception:
                if attempt == 2: raise
                if self.used >= self.limit: raise RuntimeError('download budget exhausted')
                time.sleep(2**attempt)
            finally:
                write_json(self.receipt, {'used_bytes': self.used, 'limit_bytes': self.limit})


def stable_sample(rows, count, seed):
    """Stable manifest shuffle; no importance labels used for selection."""
    rows = sorted(rows, key=lambda r: hashlib.sha256((str(seed)+':'+r['source_id']).encode()).hexdigest())
    return rows[:count]


def coco(root, budget, seed):
    # Small official image-info archive, not the full image/annotation archives.
    url = 'https://images.cocodataset.org/annotations/image_info_test2017.zip'
    index = budget.fetch(url, root/'metadata/coco_image_info.zip', 64*1024*1024)
    with zipfile.ZipFile(index) as archive:
        info = archive.getinfo('annotations/image_info_test2017.json')
        if info.file_size > 64*1024*1024: raise ValueError('COCO metadata expansion too large')
        metadata = json.loads(archive.read(info))
    licenses = {r['id']: r for r in metadata.get('licenses', [])}
    rows = []
    for r in metadata['images']:
        rows.append({'dataset': 'coco', 'source_id': str(r['id']),
            'url': r.get('coco_url') or 'https://images.cocodataset.org/test2017/'+r['file_name'],
            'source_split': 'test2017', 'source_revision': file_hash(index),
            'parent_id': str(r['id']), 'source_page': 'https://cocodataset.org/',
            'license': licenses.get(r.get('license')), 'native_labels_available': False})
    return stable_sample(rows, 1000, seed)


def open_images(root, budget, seed):
    url = 'https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv'
    index = budget.fetch(url, root/'metadata/open_images_validation.csv', 64*1024*1024)
    rows = []
    with index.open(encoding='utf8') as stream:
        for r in csv.DictReader(stream):
            i = r['ImageID']
            rows.append({'dataset': 'open_images', 'source_id': i,
                'url': 'https://open-images-dataset.s3.amazonaws.com/validation/'+i+'.jpg',
                'source_split': 'validation', 'source_revision': file_hash(index) if not rows else rows[0]['source_revision'],
                'parent_id': i, 'license': r.get('License'), 'author': r.get('Author'),
                'source_page': 'https://storage.googleapis.com/openimages/web/download_v7.html',
                'native_labels_available': False})
    return stable_sample(rows, 1000, seed)


def crisismmd(manifest, seed):
    rows = []
    for r in manifest:
        relative = r['image_path']
        rows.append({'dataset': 'crisismmd', 'source_id': r['image_id'],
            'url': 'https://huggingface.co/datasets/QCRI/CrisisMMD/resolve/10a5626ba112ab9c50369c473a2062bcf205d708/'+relative,
            'source_split': r['split'], 'source_revision': '10a5626ba112ab9c50369c473a2062bcf205d708',
            'parent_id': 'tweet:'+str(r.get('tweet_id', r['image_id'].rsplit('_',1)[0])),
            'source_page': 'https://crisisnlp.qcri.org/crisismmd.html',
            'image_sha256': r.get('image_sha256'), 'parameter_training_exposed': True,
            'native_labels_available': False})
    return stable_sample(rows, 1000, seed)


def places(root, budget, seed):
    """Read bounded official TAR prefix, never download/retain the full archive."""
    import tarfile
    if not budget.cloud: raise RuntimeError('Places acquisition is cloud-only')
    url = 'https://data.csail.mit.edu/places/places365/val_256.tar'
    limit = min(64*1024*1024, budget.limit-budget.used)
    rows = []
    request = urllib.request.Request(url, headers={'Range': 'bytes=0-'+str(limit-1)})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206: raise ValueError('server did not honor bounded Places Range request')
        etag = response.headers.get('ETag')
        class Reader:
            used = 0
            def read(self, size=-1):
                size = min(65536 if size<0 else size, limit-self.used)
                if size<=0: return b''
                chunk = response.read(size)
                self.used += len(chunk); budget.used += len(chunk)
                return chunk
        reader = Reader()
        try:
            archive = tarfile.open(fileobj=reader, mode='r|')
            for member in archive:
                if not member.isfile() or not member.name.lower().endswith(('.jpg','.jpeg','.png')): continue
                if member.size > 20*1024*1024: continue
                # Sample from a bounded prefix. Do not claim whole-dataset uniformity.
                if int(hashlib.sha256((str(seed)+member.name).encode()).hexdigest()[:8],16)%4: continue
                stream = archive.extractfile(member)
                raw = stream.read(member.size)
                if len(raw) != member.size: break
                target = root/'images/places365'/(hashlib.sha256(member.name.encode()).hexdigest()+'.image')
                target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(raw)
                rows.append({'dataset':'places365', 'source_id':member.name, 'image_path':str(target),
                    'parent_id':member.name, 'source_split':'validation', 'source_revision':etag or 'official-val256',
                    'source_page':'https://github.com/CSAILVision/places365', 'native_labels_available':False,
                    'sampling_scope':'deterministic subsample of first64MiB TAR prefix; not uniform full dataset'})
                if len(rows)>=200: break
        except (tarfile.ReadError, EOFError):
            pass  # Expected if byte-limited prefix ends inside a later TAR member.
        finally:
            write_json(budget.receipt, {'used_bytes':budget.used,'limit_bytes':budget.limit})
    return rows


def managed_rows(path, source, spec, seed):
    """Accessible per-file manifest, not an instruction to download an archive.

    Required: source_id, image_path, source_revision, source_split, source_page.
    Temporal: clip_id AND frame_index; optional observed timestamp is provenance.
    Native annotations stay in the input manifest; not copied into runtime rows.
    """
    if path is None or not Path(path).is_file(): return []
    path = Path(path)
    all_rows = []
    for text in path.read_text(encoding='utf8').splitlines():
        r = json.loads(text)
        for key in ('source_id', 'image_path', 'source_revision', 'source_split', 'source_page'):
            if not r.get(key): raise ValueError(source+' manifest missing '+key)
        r['image_path'] = str((path.parent/r['image_path']).resolve())
        if not Path(r['image_path']).is_file(): raise ValueError('managed input missing selected file')
        if spec['role'] == 'temporal':
            if not r.get('clip_id') or type(r.get('frame_index')) is not int:
                raise ValueError('temporal input requires actual clip/frame identity')
            r['parent_id'] = 'clip:'+str(r['clip_id'])
        else:
            r['parent_id'] = str(r.get('parent_id', r['source_id']))
        r['dataset'] = source
        if source in ('imagenet', 'xbd') and not r.get('access_confirmed'):
            raise ValueError('licensed input requires access_confirmed metadata')
        all_rows.append(r)
    if spec['role'] == 'temporal':
        clips = sorted({r['clip_id'] for r in all_rows},
                       key=lambda x: hashlib.sha256((str(seed)+str(x)).encode()).hexdigest())[:spec['clips']]
        import numpy as np
        selected = []
        for clip in clips:
            frames = sorted((r for r in all_rows if r['clip_id']==clip), key=lambda r:r['frame_index'])
            if len({r['frame_index'] for r in frames}) != len(frames): raise ValueError('duplicate clip/frame identity')
            positions = np.linspace(0, len(frames)-1, min(len(frames), spec['frames_per_clip']), dtype=int)
            selected.extend(frames[i] for i in positions)
        return selected
    return stable_sample(all_rows, spec['cap']*3, seed)


def build_pool(config, root, crisis_manifest, managed=None, cloud=False):
    """Bounded acquisition only. Failures and unsupported sources remain in report."""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    managed = managed or {}
    budget = Budget(root, config['new_download_cap_bytes'], cloud)
    image_rows, references, status = [], [], {}
    for name, spec in config['sources'].items():
        print('POOL SOURCE START', name, 'cap', spec['cap'], flush=True)
        errors = []
        try:
            if name == 'coco': candidates = coco(root, budget, config['seed'])
            elif name == 'open_images': candidates = open_images(root, budget, config['seed'])
            elif name == 'crisismmd': candidates = crisismmd(crisis_manifest, config['seed'])
            elif name == 'places365' and not managed.get(name): candidates = places(root, budget, config['seed'])
            else: candidates = managed_rows(managed.get(name), name, spec, config['seed'])
            if not candidates:
                status[name] = {'status': 'input_required', 'selected': 0, 'cap': spec['cap'],
                                'reason': 'no accessible per-image/clip manifest; no full archive downloaded'}
                continue
            selected = 0
            for r in candidates:
                if selected >= spec['cap']: break
                if budget.used >= budget.limit: break
                try:
                    rid = name+':'+r['source_id']
                    if 'url' in r:
                        path = budget.fetch(r['url'], root/'images'/name/(hashlib.sha256(rid.encode()).hexdigest()+'.image'), config['per_image_cap_bytes'])
                    else: path = Path(r['image_path'])
                    sha = file_hash(path)
                    if r.get('image_sha256') and sha != r['image_sha256']: raise ValueError('expected image checksum mismatch')
                    from PIL import Image
                    with Image.open(path) as image: image.verify()
                    image_rows.append({'image_id': rid, 'image_path': str(path), 'image_sha256': sha,
                        'dataset': name, 'source_id': r['source_id'], 'source_revision': r['source_revision'],
                        'source_split': r['source_split'], 'parent_id': name+':'+r['parent_id'],
                        'clip_id': r.get('clip_id'), 'frame_index': r.get('frame_index'),
                        'observed_capture_timestamp': r.get('capture_timestamp'),
                        'role': spec['role'], 'source_page': r['source_page'],
                        'parameter_training_exposed': r.get('parameter_training_exposed', False),
                        'license': r.get('license'), 'native_label_scope': 'see separate references'})
                    references.append({'image_id': rid, 'native_annotations': r.get('native_annotations'),
                                       'native_labels_available': r.get('native_labels_available', False),
                                       'source_page': r['source_page'], 'urgency_truth_available': False})
                    selected += 1
                    if selected % 25 == 0: print('POOL', name, selected, '/', spec['cap'], flush=True)
                except Exception as error:
                    errors.append({'source_id': r['source_id'], 'reason': type(error).__name__+': '+str(error)})
            status[name] = {'status': 'complete' if selected==spec['cap'] else 'partial',
                            'selected': selected, 'cap': spec['cap'], 'failures': errors}
        except Exception as error:
            status[name] = {'status': 'unavailable', 'selected': 0, 'cap': spec['cap'], 'reason': str(error)}
        finally:
            write_json(root/'source_status.json', status)
            write_json(root/'input_records.json', image_rows)
            write_json(root/'reference_records.json', references)
            print('POOL SOURCE END', name, status.get(name), flush=True)
    return image_rows, references, status


def group_pool(rows, seed=29):
    """Conservative connected groups: clip/parent plus exact raw bytes across sources."""
    ids = [r['image_id'] for r in rows]
    if len(ids) != len(set(ids)): raise ValueError('duplicate source image identity')
    parent = {i:i for i in ids}
    def root(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    seen = {}
    for r in rows:
        for k in ('sha:'+r['image_sha256'], 'parent:'+r['parent_id']):
            if k in seen: parent[root(r['image_id'])] = root(seen[k])
            else: seen[k] = r['image_id']
    components = {}
    for r in rows: components.setdefault(root(r['image_id']), []).append(r)
    for members in components.values():
        group = 'group:'+hashlib.sha256('|'.join(sorted(r['image_id'] for r in members)).encode()).hexdigest()
        heldout = any(r['role']=='scheduler_unseen_task' for r in members)
        exposed = any(r['parameter_training_exposed'] for r in members)
        bucket = int(hashlib.sha256((str(seed)+group).encode()).hexdigest()[:8],16)%10
        split = 'transfer' if heldout else ('train' if exposed or bucket<7 else 'validation' if bucket<9 else 'test')
        for r in members:
            r.update(group=group, split=split,
                     near_duplicate_status='not audited; exact raw bytes and parent/clip groups enforced',
                     vision_pretraining_exposure='unknown; split is not proof vision-model unseen')
    return rows
