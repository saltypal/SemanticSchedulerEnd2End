"""Numerical and cache contracts; fixtures do not validate detector accuracy."""
import json
import sqlite3
from pathlib import Path
import pytest
from PIL import Image
from parameters.scene_arrays import (CLASSES, OBJECTS, WIRE_BYTES, FEATURE_COUNT, SceneArray,
    summarize, unavailable, class_slot)
from experiments.scene_array_cache.cache import cache_all, file_hash

PRODUCER = '1' * 64


def test_count_area_mask_and_wire():
    rows = [{'label': 'person', 'box': [0, 0, 10, 20], 'score': .9},
            {'label': 'person', 'box': [0, 0, 20, 20], 'score': .8},
            {'label': 'water', 'box': [0, 50, 100, 100], 'score': .7}]
    result, audit = summarize(rows, 100, 100, PRODUCER)
    assert result.counts[class_slot('person')] == 2
    assert result.largest_area[class_slot('person')] == .04
    assert result.count_available[class_slot('water')] is False
    assert result.counts[class_slot('water')] == 0
    assert result.largest_area[class_slot('water')] == .5
    assert len(result.features(PRODUCER)) == FEATURE_COUNT
    wire = result.serialize()
    assert len(wire) == WIRE_BYTES == 369
    decoded = SceneArray.deserialize(wire, PRODUCER)
    assert decoded.counts == result.counts
    assert max(abs(x-y) for x, y in zip(decoded.largest_area, result.largest_area)) <= .5/255


def test_missingness_empty_failure_and_outside_vocabulary():
    empty, _ = summarize([], 100, 100, PRODUCER)
    failed = unavailable(PRODUCER)
    assert empty.count_available[0] and empty.counts[0] == 0
    assert empty.area_available[0] and empty.largest_area[0] == 0
    assert not failed.count_available[0] and not failed.area_available[0]
    with pytest.raises(ValueError, match='outside_vocabulary'): class_slot('smartphone screenshot')
    with pytest.raises(ValueError, match='outside configured'):
        summarize([{'label': 'unknown'}], 100, 100, PRODUCER)
    with pytest.raises(ValueError, match='incompatible'):
        SceneArray.deserialize(empty.serialize(), '2'*64)


def test_box_clipping_invalid_and_saturation():
    r, audit = summarize([{'label': 'person', 'box': [-10, -20, 50, 50], 'score': .7},
                          {'label': 'car', 'box': [10, 0, 0, 5], 'score': .8}],
                         100, 100, PRODUCER, max_detections=2)
    assert r.largest_area[0] == .25
    assert len(audit['rejected_boxes']) == 1 and r.saturated


class FixtureDetector:
    device = 'cpu'
    signature = {'fixture': 'not a vision model'}
    producer_sha256 = PRODUCER
    calls = 0
    def sync(self): pass
    def predict(self, image):
        self.calls += 1
        return [{'label': 'person', 'box': [0, 0, 8, 8], 'score': .9}]


def test_cache_resume_identity_and_integrity(tmp_path):
    image = tmp_path/'image.png'
    Image.new('RGB', (32, 32)).save(image)
    rows = [{'image_id': 'example', 'image_path': str(image),
             'image_sha256': file_hash(image), 'split': 'train'}]
    detector = FixtureDetector()
    out = tmp_path/'run'
    assert cache_all(rows, detector, out)['successful'] == 1
    assert cache_all(rows, detector, out)['successful'] == 1 and detector.calls == 1
    changed = [{**rows[0], 'split': 'test'}]
    with pytest.raises(ValueError, match='mismatch'): cache_all(changed, detector, out)
    db = sqlite3.connect(out/'scene_cache.sqlite')
    db.execute("UPDATE outputs SET checksum='bad'"); db.commit(); db.close()
    with pytest.raises(ValueError, match='integrity'): cache_all(rows, detector, out)


def test_input_failure_is_not_no_detections(tmp_path):
    rows = [{'image_id': 'missing', 'image_path': str(tmp_path/'absent.png')}]
    progress = cache_all(rows, FixtureDetector(), tmp_path/'run')
    assert progress['failed'] == 1 and not progress['all_successful']
    record = json.loads((tmp_path/'run/scene_records.jsonl').read_text())
    assert record['scene']['detector_ok'] is False


def test_mutated_unhashed_source_rejected(tmp_path):
    image = tmp_path/'mutable.png'
    Image.new('RGB', (32,32), 'red').save(image)
    rows = [{'image_id': 'mutable', 'image_path': str(image)}]
    detector = FixtureDetector()
    cache_all(rows, detector, tmp_path/'run')
    Image.new('RGB', (32,32), 'blue').save(image)
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        cache_all(rows, detector, tmp_path/'run')


def test_failed_inputs_can_resume_without_repeating_successes(tmp_path):
    image = tmp_path/'later.png'
    rows = [{'image_id': 'late', 'image_path': str(image)}]
    detector = FixtureDetector()
    assert cache_all(rows, detector, tmp_path/'run')['failed'] == 1
    Image.new('RGB', (32,32)).save(image)
    assert cache_all(rows, detector, tmp_path/'run', retry_failed=True)['successful'] == 1
    db = sqlite3.connect(tmp_path/'run/scene_cache.sqlite')
    assert db.execute('SELECT COUNT(*) FROM previous_attempts').fetchone()[0] == 1
    db.close()
