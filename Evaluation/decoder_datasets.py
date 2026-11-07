"""Published-label adapters and controlled polygon correspondence, no VLM truth."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re

TASKS = ('building_presence', 'building_damage_binary', 'structural_flooding_transfer')
XBD_DAMAGE = {'no-damage': 0, 'minor-damage': 1, 'major-damage': 1, 'destroyed': 1}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_path(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('source paths must be relative to the declared dataset root')
    path = (root/relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'missing or unsafe dataset file: {relative}')
    return path


def native_labels(labels):
    """LADI booleans only; missing and contradictory nesting stay unavailable."""
    def boolean(name):
        value = labels.get(name)
        if value is None:
            return None
        if value not in (False, True, 0, 1) or isinstance(value, str):
            raise ValueError('native labels must be boolean, never inferred from names')
        return int(value)
    present = boolean('buildings_any')
    damage = boolean('buildings_affected_or_greater')
    flood = boolean('flooding_structures')
    if present == 0 and damage == 1:
        return dict.fromkeys(TASKS), 'conflicting_building_labels'
    return dict(zip(TASKS, (present, damage, flood))), None


def ladi_cache_records(cache):
    from Scheduler.training.decoder_v3_study import load_cache, verify_extraction_cache
    raw = load_cache(cache)
    verify_extraction_cache(cache, raw)
    # Historic test is exposed development too. This adapter does NOT call it a fresh test.
    output, exclusions = [], []
    for row in raw:
        labels, reason = native_labels(row['labels'])
        if reason:
            exclusions.append(dict(image_id=row['image_id'], reason=reason))
            continue
        output.append(dict(schema='published_image_record_v1', source='ladi_cache',
                           image_id=row['image_id'], parent_id=row['image_id'], group=row['group'],
                           role='development_exposed', original_role=row['role'],
                           event=None, site=None, labels=labels,
                           label_definition='native_LADI_binary_not_emergency_priority',
                           evidence=row['evidence'].to_dict()))
    return output, exclusions


def ladi_manifest_records(manifest, limit=20):
    """Curated CSV adapter: consume native_labels_json, never prototype priority."""
    manifest = Path(manifest).resolve()
    rows, exclusions = [], []
    with manifest.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not {'image_path', 'native_labels_json', 'source_sha256'} <= set(reader.fieldnames or ()):
            raise ValueError('unsupported native LADI manifest schema')
        for item in reader:
            if len(rows)+len(exclusions) >= limit:
                break
            image = safe_path(manifest.parent, item['image_path'].replace('\\', '/'))
            identity = sha256(image)
            if identity != item['source_sha256'].lower():
                raise ValueError('source image does not match native manifest hash')
            labels, reason = native_labels(json.loads(item['native_labels_json']))
            if reason:
                exclusions.append(dict(image_id=identity, reason=reason))
                continue
            rows.append(dict(schema='published_image_record_v1', source='ladi_curated_native', image_id=identity,
                             image_path=str(image), parent_id=identity,
                             group=item.get('evidence_group_id') or identity, labels=labels,
                             role='development_exposed', original_role=item.get('split'),
                             upstream_split=item.get('source_split'), revision=item.get('dataset_revision'),
                             event=None, site=None, label_definition='native_LADI_binary',
                             excluded_prototype_columns=['base_priority', 'task_impact', 'urgency', 'evidence_sufficiency']))
    return rows, exclusions


def medic_records(root, tsv, limit=20):
    rows, exclusions = [], []
    with Path(tsv).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        required = {'image_path', 'informative', 'damage_severity'}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError('unsupported MEDIC annotation schema')
        for row in reader:
            if len(rows)+len(exclusions) >= limit:
                break
            try:
                path = safe_path(root, row['image_path'].replace('\\', '/'))
            except ValueError as error:
                exclusions.append(dict(path=row['image_path'], reason=str(error)))
                continue
            rows.append(dict(schema='published_image_record_v1', source='medic', image_id=sha256(path),
                             image_path=str(path), parent_id=sha256(path), group=sha256(path),
                             event=row.get('event_name') or None, event_verified=False,
                             labels=dict.fromkeys(TASKS),
                             native_stress_labels={k: row.get(k) for k in
                                                   ('informative', 'damage_severity', 'humanitarian')},
                             label_definition='general_damage_not_building_specific'))
    return rows, exclusions


def mask_records(root, manifest, class_map, source, limit=20):
    """Explicit pairs/class map: never infer masks, splits or conditions from filenames.

    Manifest JSON list: image_path, mask_path, optional event/site/role,
    and exhaustive=true only where complete source annotation supports negatives.
    Class-map keys are integer codes, values explicit normalized source class names.
    """
    import numpy as np
    from PIL import Image
    if source not in ('rescuenet', 'floodnet'):
        raise ValueError('unsupported mask source')
    names = {int(key): str(value) for key, value in class_map.items()}
    supported = {'background', 'void', 'water', 'vehicle', 'tree', 'pool', 'road_clear', 'road_blocked',
                 'road_flooded', 'road_nonflooded', 'building_no_damage', 'building_minor_damage',
                 'building_major_damage', 'building_destroyed', 'building_flooded', 'building_nonflooded',
                 'other'}
    if not names or not set(names.values()) <= supported:
        raise ValueError('unsupported explicit native mask class mapping')
    rows, exclusions = [], []
    for item in json.loads(Path(manifest).read_text(encoding='utf-8'))[:limit]:
        image = safe_path(root, item['image_path'])
        mask = safe_path(root, item['mask_path'])
        with Image.open(image) as source_image, Image.open(mask) as source_mask:
            if source_image.size != source_mask.size:
                raise ValueError('image and annotation geometry differ')
            values = np.asarray(source_mask)
        if values.ndim != 2 or not set(map(int, np.unique(values))) <= set(names):
            raise ValueError('mask must contain known scalar class IDs')
        present_classes = {names[int(value)] for value in np.unique(values)}
        exhaustive = item.get('exhaustive') is True and 'void' not in present_classes
        building = any(name.startswith('building_') for name in present_classes)
        damaged = bool(present_classes & {'building_minor_damage', 'building_major_damage', 'building_destroyed'})
        flooded = 'building_flooded' in present_classes
        known_damage_schema = source == 'rescuenet'
        labels = dict(building_presence=int(building) if building or exhaustive else None,
                      building_damage_binary=(int(damaged) if known_damage_schema and (damaged or exhaustive) else None),
                      structural_flooding_transfer=(int(flooded) if source == 'floodnet' and (flooded or exhaustive) else None))
        identity = sha256(image)
        group = item.get('site') or item.get('parent_id') or identity
        rows.append(dict(schema='published_image_record_v1', source=source, image_id=identity,
                         image_path=str(image), annotation_path=str(mask), annotation_sha256=sha256(mask),
                         parent_id=item.get('parent_id', identity), group=f'{source}:{group}',
                         event=item.get('event'), site=item.get('site'), role=item.get('role', 'unassigned'),
                         labels=labels, label_definition=f'{source}_native_mask_v1'))
    return rows, exclusions


def polygon_area(points):
    return abs(sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(points, points[1:]+points[:1])))/2


def parse_polygon(wkt):
    match = re.fullmatch(r'\s*POLYGON\s*\(\(\s*([^()]+)\s*\)\)\s*', wkt, re.IGNORECASE)
    if not match:
        raise ValueError('only simple single-ring POLYGON annotations are supported')
    points = []
    for pair in match.group(1).split(','):
        values = tuple(map(float, pair.split()))
        if len(values) != 2 or not all(math.isfinite(value) for value in values):
            raise ValueError('invalid polygon coordinates')
        points.append(values)
    if points and points[-1] == points[0]:
        points.pop()
    if len(points) < 3 or polygon_area(points) <= 0:
        raise ValueError('degenerate polygon')
    return points


def clip_polygon(points, rectangle):
    """Sutherland-Hodgman clipping against an axis-aligned crop."""
    points = list(points)
    x0, y0, x1, y1 = rectangle
    if not x0 < x1 or not y0 < y1:
        raise ValueError('invalid crop rectangle')
    for axis, boundary, sign in ((0, x0, 1), (0, x1, -1), (1, y0, 1), (1, y1, -1)):
        result = []
        if not points:
            break
        previous = points[-1]
        for current in points:
            pin = sign*(previous[axis]-boundary) >= 0
            cin = sign*(current[axis]-boundary) >= 0
            if pin != cin:
                ratio = (boundary-previous[axis])/(current[axis]-previous[axis])
                result.append(tuple(previous[d]+ratio*(current[d]-previous[d]) for d in (0, 1)))
            if cin:
                result.append(current)
            previous = current
        points = result
    return points


def overlap_windows(width, height):
    if width < 4 or height < 4:
        raise ValueError('controlled window source must be at least 4x4')
    windows = [(0, 0, width, height)]
    for y in (0, height//4, height//2):
        for x in (0, width//4, width//2):
            windows.append((x, y, x+width//2, y+height//2))
    return tuple(dict.fromkeys(windows))


def xbd_records(root, annotation_paths, limit=20):
    from PIL import Image
    rows, exclusions = [], []
    for relative in list(annotation_paths)[:limit]:
        path = safe_path(root, relative)
        annotation = json.loads(path.read_text(encoding='utf-8'))
        image_name = annotation.get('metadata', {}).get('img_name')
        if not image_name or 'post_disaster' not in image_name:
            exclusions.append(dict(path=str(path), reason='post-disaster metadata required'))
            continue
        images = [p for p in (path.parent.parent/'images'/image_name, Path(root)/'images'/image_name) if p.is_file()]
        if len(set(p.resolve() for p in images)) != 1:
            raise ValueError('xBD image metadata must resolve to exactly one source image')
        image = images[0].resolve()
        if not image.is_relative_to(Path(root).resolve()):
            raise ValueError('unsafe xBD image location')
        with Image.open(image) as opened:
            width, height = opened.size
        identity = sha256(image)
        objects = []
        seen = set()
        for feature in annotation.get('features', {}).get('xy', []):
            properties = feature.get('properties', {})
            uid, condition = properties.get('uid'), properties.get('subtype')
            if not uid or uid in seen:
                raise ValueError('missing/duplicate xBD object UID within source image')
            seen.add(uid)
            if condition not in XBD_DAMAGE:
                exclusions.append(dict(path=str(path), uid=uid, reason='unsupported/unclassified damage'))
                continue
            try:
                polygon = parse_polygon(feature['wkt'])
            except (ValueError, KeyError) as error:
                exclusions.append(dict(path=str(path), uid=uid, reason=str(error)))
                continue
            objects.append(dict(fact_id=f'xbd:{identity}:{uid}:{condition}', uid=uid,
                                condition=condition, damaged=XBD_DAMAGE[condition], polygon=polygon,
                                critical=condition in ('major-damage', 'destroyed')))
        rows.append(dict(schema='published_object_record_v1', source='xbd', image_id=identity,
                         parent_id=identity, group=f'xbd:{identity}', image_path=str(image),
                         annotation_path=str(path), annotation_sha256=sha256(path),
                         event=annotation.get('metadata', {}).get('disaster'), width=width, height=height,
                         objects=objects, label_definition='xbd_native_postevent_buildings'))
    return rows, exclusions


def controlled_windows(record, task):
    if task not in ('building_presence', 'building_damage_binary'):
        raise ValueError('unsupported controlled-overlap task')
    output = []
    for index, window in enumerate(overlap_windows(record['width'], record['height'])):
        visible = []
        for obj in record['objects']:
            if task == 'building_damage_binary' and not obj['damaged']:
                continue
            area = polygon_area(obj['polygon'])
            clipped = clip_polygon(obj['polygon'], window)
            fraction = polygon_area(clipped)/area if len(clipped) >= 3 else 0.
            if fraction >= .95 - 1e-12:
                visible.append(obj['fact_id'])
        # Empty windows are excluded rather than asserted complete native negatives.
        if visible:
            output.append(dict(image_id=f"{record['image_id']}:window{index}", parent_id=record['parent_id'],
                               group=record['group'], source='xbd_controlled_windows', task=task,
                               window=window, source_image=record['image_path'], facts=visible,
                               correspondence='controlled_source_polygon_not_natural_views'))
    return output


def assert_group_isolation(records, role_key='role'):
    """Exact image/parent/group/event-site components must not cross split roles."""
    ownership = {}
    for row in records:
        keys = [('image', row['image_id']), ('parent', row.get('parent_id')), ('group', row.get('group'))]
        if row.get('site'):
            keys.append(('site', row['site']))
        if row.get('event') and row.get('event_verified'):
            keys.append(('event', row['event']))
        for kind, value in keys:
            if value is None:
                continue
            key = (kind, value)
            if key in ownership and ownership[key] != row[role_key]:
                raise ValueError(f'{kind} component crosses split roles')
            ownership[key] = row[role_key]


def connected_groups(records, near_duplicate_pairs=(), event_held_out=False):
    """Union source derivatives and supplied duplicate links BEFORE assignment.

    Near-duplicate links are explicit image hashes produced by a separate audit;
    no similarity cluster is promoted into a physical asset correspondence.
    """
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        parent[find(b)] = find(a)
    keys, identities = {}, {}
    for i, row in enumerate(records):
        values = [('image', row['image_id']), ('parent', row.get('parent_id')), ('group', row.get('group'))]
        if row.get('site'):
            values.append(('site', row['site']))
        if event_held_out:
            if row.get('event') is None or row.get('event_verified') is not True:
                raise ValueError('event-held-out grouping needs verified event provenance for every record')
            values.append(('event', row['event']))
        for key in values:
            if key[1] is None:
                continue
            if key in keys:
                union(i, keys[key])
            keys[key] = i
        identities.setdefault(row['image_id'], []).append(i)
    for a, b in near_duplicate_pairs:
        if a not in identities or b not in identities:
            raise ValueError('duplicate link refers to unknown source identity')
        union(identities[a][0], identities[b][0])
    members = {}
    for i, row in enumerate(records):
        members.setdefault(find(i), []).append(row['image_id'])
    group_ids = {root: hashlib.sha256('\n'.join(sorted(values)).encode()).hexdigest()
                 for root, values in members.items()}
    return [{**row, 'group': group_ids[find(i)]} for i, row in enumerate(records)]


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=('ladi_manifest', 'medic', 'rescuenet', 'floodnet', 'xbd'), required=True)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--class-map', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=20)
    args = parser.parse_args()
    if args.limit < 1 or (args.output.exists() and any(args.output.iterdir())):
        parser.error('positive limit and a new/empty output directory required')
    args.output.mkdir(parents=True, exist_ok=True)
    from Scheduler.training.decoder_v3_study import write_json
    import traceback
    print(f'Start native annotation audit source={args.source} limit={args.limit}', flush=True)
    try:
        if args.source == 'ladi_manifest':
            records, excluded = ladi_manifest_records(args.manifest, args.limit)
        elif args.source == 'medic':
            if args.root is None:
                raise ValueError('MEDIC dataset root required')
            records, excluded = medic_records(args.root, args.manifest, args.limit)
        elif args.source == 'xbd':
            if args.root is None:
                raise ValueError('xBD dataset root required')
            annotation_paths = json.loads(args.manifest.read_text(encoding='utf-8'))
            records, excluded = xbd_records(args.root, annotation_paths, args.limit)
        else:
            if args.root is None or args.class_map is None:
                raise ValueError('mask dataset root and explicit class-map JSON required')
            records, excluded = mask_records(args.root, args.manifest,
                                             json.loads(args.class_map.read_text()), args.source, args.limit)
        grouped = connected_groups(records)
        write_json(args.output/'records.json', grouped)
        write_json(args.output/'exclusions.json', excluded)
        report = dict(status='complete', source=args.source, count=len(records), excluded=len(excluded),
                      manifest_sha256=sha256(args.manifest), native_labels_only=True,
                      independently_reviewed_by_project=False)
        write_json(args.output/'audit.json', report)
        print(json.dumps(report), flush=True)
    except Exception as error:
        write_json(args.output/'failure.json', dict(type=type(error).__name__, message=str(error),
                                                   traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
