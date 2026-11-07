"""Versioned detector evidence arrays. Geometry is not urgency or physical depth."""
from dataclasses import dataclass
import hashlib
import json
import math
import struct


SCHEMA = 'scene_arrays_v1'
# IDs are positions, never reordered. Regions have no defensible instance count.
OBJECTS = ('person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'boat',
           'airplane', 'train', 'building', 'bridge', 'utility pole',
           'traffic light', 'traffic sign', 'tree', 'animal', 'chair', 'couch',
           'table', 'bed', 'television', 'laptop', 'mobile phone', 'refrigerator')
REGIONS = ('road', 'water', 'debris', 'smoke', 'flame', 'snow', 'ground', 'grass')
CLASSES = OBJECTS + REGIONS
REGISTRY = {'schema': SCHEMA, 'classes': [
    {'id': i, 'name': label, 'prompt': label,
     'count_applicable': label in OBJECTS} for i, label in enumerate(CLASSES)],
    'area': 'largest clipped xyxy rectangle divided by oriented frame area',
    'empty_detection_area': 0.0, 'counts': 'retained boxes, not verified assets'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


REGISTRY_SHA256 = digest(REGISTRY)
PREFIX = struct.Struct('<BB32s32s')
WIRE_BYTES = PREFIX.size + 32 * 2 + 32 + 4 + 4  # 170 bytes; scene block only


@dataclass(frozen=True)
class SceneArray:
    counts: tuple
    largest_area: tuple
    count_available: tuple
    area_available: tuple
    detector_ok: bool
    saturated: bool
    producer_sha256: str
    registry_sha256: str = REGISTRY_SHA256

    def __post_init__(self):
        for v in (self.counts, self.largest_area, self.count_available, self.area_available):
            if len(v) != 32:
                raise ValueError('exactly 32 ordered slots required')
        if self.registry_sha256 != REGISTRY_SHA256:
            raise ValueError('incompatible vocabulary/schema')
        if len(bytes.fromhex(self.producer_sha256)) != 32:
            raise ValueError('full producer SHA256 required')
        if type(self.detector_ok) is not bool or type(self.saturated) is not bool:
            raise ValueError('boolean status flags required')
        for i, (n, a, cm, am) in enumerate(zip(self.counts, self.largest_area,
                                              self.count_available, self.area_available)):
            if type(n) is not int or not 0 <= n <= 65535:
                raise ValueError('uint16 detection counts required')
            if not isinstance(a, (int, float)) or not math.isfinite(a) or not 0 <= a <= 1:
                raise ValueError('finite normalized area required')
            if type(cm) is not bool or type(am) is not bool:
                raise ValueError('boolean masks required')
            if not self.detector_ok and (cm or am or n or a):
                raise ValueError('failed detector cannot expose available measurements')
            if cm and CLASSES[i] not in OBJECTS:
                raise ValueError('region instance counts are inapplicable')
            if not cm and n != 0 or not am and a != 0:
                raise ValueError('unavailable values must have canonical zero storage')

    def record(self):
        return {'schema': SCHEMA, 'registry_sha256': self.registry_sha256,
                'producer_sha256': self.producer_sha256,
                'counts': list(self.counts), 'largest_area': list(self.largest_area),
                'count_available': list(self.count_available),
                'area_available': list(self.area_available),
                'detector_ok': self.detector_ok, 'saturated': self.saturated}

    @classmethod
    def from_record(cls, row):
        if row['schema'] != SCHEMA:
            raise ValueError('incompatible scene schema')
        return cls(*(tuple(row[k]) for k in ('counts', 'largest_area',
                                            'count_available', 'area_available')),
                   row['detector_ok'], row['saturated'], row['producer_sha256'],
                   row['registry_sha256'])

    def features(self, expected_producer_sha256):
        if expected_producer_sha256 != self.producer_sha256:
            raise ValueError('scheduler and extractor producer contracts differ')
        return ([math.log1p(n) for n in self.counts] + list(self.largest_area)
                + list(map(float, self.count_available))
                + list(map(float, self.area_available))
                + [float(self.detector_ok), float(self.saturated)])

    def serialize(self):
        def mask(values):
            return sum(int(v) << i for i, v in enumerate(values))
        flags = int(self.detector_ok) | (int(self.saturated) << 1)
        return (PREFIX.pack(1, flags, bytes.fromhex(self.registry_sha256),
                            bytes.fromhex(self.producer_sha256))
                + struct.pack('<32H', *self.counts)
                + bytes(round(v * 255) for v in self.largest_area)
                + struct.pack('<II', mask(self.count_available), mask(self.area_available)))

    @classmethod
    def deserialize(cls, data, expected_producer_sha256):
        if len(data) != WIRE_BYTES:
            raise ValueError('scene block must be exactly 170 bytes')
        version, flags, registry, producer = PREFIX.unpack_from(data)
        if version != 1 or flags & ~3 or registry.hex() != REGISTRY_SHA256:
            raise ValueError('invalid version, flags or vocabulary')
        if producer.hex() != expected_producer_sha256:
            raise ValueError('incompatible detector configuration/weights')
        counts = struct.unpack_from('<32H', data, PREFIX.size)
        start = PREFIX.size + 64
        area = tuple(v / 255 for v in data[start:start + 32])
        cm, am = struct.unpack_from('<II', data, start + 32)
        return cls(counts, area, tuple(bool(cm & (1 << i)) for i in range(32)),
                   tuple(bool(am & (1 << i)) for i in range(32)),
                   bool(flags & 1), bool(flags & 2), producer.hex())


def summarize(detections, width, height, producer_sha256, max_detections=300):
    """One box list -> arrays plus audit. No category importance weights."""
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError('positive oriented image dimensions required')
    if type(max_detections) is not int or max_detections <= 0:
        raise ValueError('positive detection cap required')
    counts = [0] * 32
    areas = [0.0] * 32
    rejected = []
    kept = []
    for number, row in enumerate(detections):
        # A wrong dictionary is an integration failure, not a discarded object.
        label = row.get('label')
        if label not in CLASSES:
            raise ValueError('class outside configured vocabulary: ' + str(label))
        i = CLASSES.index(label)
        try:
            box = tuple(float(x) for x in row['box'])
            if len(box) != 4 or not all(math.isfinite(x) for x in box):
                raise ValueError('invalid box')
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                raise ValueError('reversed/empty box')
            x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
            y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
            if x2 <= x1 or y2 <= y1:
                raise ValueError('box outside image')
            score = float(row['score'])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('invalid detector score')
        except (ValueError, KeyError, TypeError) as error:
            rejected.append({'index': number, 'reason': str(error)})
            continue
        if label in OBJECTS:
            counts[i] += 1
        areas[i] = max(areas[i], (x2 - x1) * (y2 - y1) / (width * height))
        kept.append({'class_id': i, 'label': label, 'box': [x1, y1, x2, y2], 'score': score})
    result = SceneArray(tuple(counts), tuple(areas), tuple(c in OBJECTS for c in CLASSES),
                        (True,) * 32, True, len(detections) >= max_detections,
                        producer_sha256)
    return result, {'detections': kept, 'rejected_boxes': rejected,
                    'width': width, 'height': height}


def unavailable(producer_sha256):
    return SceneArray((0,) * 32, (0.0,) * 32, (False,) * 32, (False,) * 32,
                      False, False, producer_sha256)


def class_slot(label):
    """Outside vocabulary is distinct from an available slot containing zero."""
    if label not in CLASSES:
        raise ValueError('outside_vocabulary: ' + str(label))
    return CLASSES.index(label)
