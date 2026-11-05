"""Runtime ordering and artifact checks using synthetic inputs only."""
import json
import subprocess
import sys

import numpy as np
import pytest

from Scheduler.decoder_scheduler import (
    ARTIFACT_SCHEMA, FEATURE_NAMES, SchedulerCandidate, SemanticScheduler,
    feature_matrix, load_scheduler, weights_digest,
)
from Archived.owners_before_simplification_20261027.SemParam.decoder_parameters import HeaderV3


def candidate(name, importance=.5, source='ue0', sequence=0, available=None, profile=1001):
    values = [0.]*8
    values[3], values[7] = importance, .5
    return SchedulerCandidate(name, source, 0., sequence, profile, tuple(values),
                              tuple([True]*8 if available is None else available))


def test_header_parse_is_authoritative_and_identity_is_not_a_feature():
    flags = (True,)*4 + (False,) + (True,)*3
    header = HeaderV3(1001, 1, 0, 0, (.1,)*8, flags)
    parsed = HeaderV3.from_bytes(header.to_bytes())
    a = SchedulerCandidate.from_header('a', 'ue0', 0, 0, parsed)
    b = SchedulerCandidate.from_header('b', 'ue1', 4, 9, parsed)
    matrix = feature_matrix([a, b], [.5]*8)
    assert matrix.shape == (2, 16)
    np.testing.assert_array_equal(matrix[0], matrix[1])
    assert matrix[0, 0] == 26/255 and matrix[0, 4] == .5 and matrix[0, 12] == 0
    assert not hasattr(a, 'payload') and not hasattr(a, 'labels')


def test_empty_ties_and_profile_errors():
    model = SemanticScheduler('importance')
    assert model.select([]) is None
    assert [d.message_id for d in model.rank([candidate('b'), candidate('a')])] == ['a', 'b']
    with pytest.raises(ValueError, match='profile'):
        model.rank([candidate('a'), candidate('b', profile=1002)])
    with pytest.raises(ValueError, match='duplicate'):
        model.rank([candidate('a'), candidate('a')])


def test_selection_does_not_mutate_queue_and_missing_has_fifo_fallback():
    queue = [candidate('z', .9), candidate('a', .1)]
    assert SemanticScheduler('importance').select(queue).message_id == 'z'
    assert [c.message_id for c in queue] == ['z', 'a']
    missing = [candidate('z', sequence=1, available=[False]*8), candidate('a', available=[False]*8)]
    decision = SemanticScheduler('importance').select(missing)
    assert decision.message_id == 'a' and decision.fallback == 'all_semantics_unavailable'


def test_round_robin_advances_only_on_select():
    model = SemanticScheduler('round_robin')
    queue = [candidate('a0', source='a'), candidate('a1', source='a', sequence=1), candidate('b0', source='b')]
    assert [d.message_id for d in model.rank(queue)] == ['a0', 'b0', 'a1']
    assert model.select(queue).message_id == 'a0'
    assert model.select(queue[1:]).message_id == 'b0'


def test_seeded_random_is_input_order_invariant():
    queue = [candidate(str(i), sequence=i) for i in range(12)]
    a = SemanticScheduler('random', seed=2).rank(queue)
    b = SemanticScheduler('random', seed=2).rank(queue[::-1])
    assert a == b


def test_numpy_network_matches_explicit_formula_and_checks_shapes():
    rng = np.random.default_rng(4)
    weights = [rng.normal(size=(16, 32)), rng.normal(size=(32, 16)), rng.normal(size=(16, 1))]
    biases = [rng.normal(size=32), rng.normal(size=16), rng.normal(size=1)]
    model = SemanticScheduler('mlp', 1001, weights=weights, biases=biases)
    queue = [candidate('a'), candidate('b', .9)]
    x = feature_matrix(queue, [.5]*8)
    expected = np.maximum(np.maximum(x@weights[0]+biases[0], 0)@weights[1]+biases[1], 0)@weights[2]+biases[2]
    np.testing.assert_allclose(model.scores(queue), expected[:, 0])
    with pytest.raises(ValueError):
        SemanticScheduler('mlp', weights=[np.ones((8, 1))], biases=[[0.]])


def test_artifact_checksum_and_feature_compatibility(tmp_path):
    weights, biases = [np.ones((16, 1)).tolist()], [[0.]]
    metadata = dict(schema=ARTIFACT_SCHEMA, header_version=3, feature_names=FEATURE_NAMES,
                    profile_id=1001, kind='linear', medians=[.5]*8, weights=weights,
                    biases=biases, weights_sha256=weights_digest(weights, biases))
    path = tmp_path/'model.json'
    path.write_text(json.dumps(metadata), encoding='utf-8')
    assert load_scheduler(tmp_path, {1001: {}}).select([candidate('a')]).message_id == 'a'
    metadata['weights'][0][0][0] = 2
    path.write_text(json.dumps(metadata), encoding='utf-8')
    with pytest.raises(ValueError, match='checksum'):
        load_scheduler(tmp_path)


def test_scheduler_import_does_not_load_torch():
    process = subprocess.run([sys.executable, '-c',
        "import Scheduler.decoder_scheduler; import sys; assert 'torch' not in sys.modules"],
        capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
