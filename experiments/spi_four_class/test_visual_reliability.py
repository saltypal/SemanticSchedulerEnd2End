"""Diagnostic isolation/recovery fixtures; no claimed visual-model accuracy."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from parameters.qwen_reviewer import QwenAnnotationReviewer

from experiments.spi_four_class import visual_reliability as audit


def test_frozen_panel_is_grouped_and_has_declared_preflight():
    data = json.loads(Path(audit.__file__).with_name('grounding_panel.json').read_text())
    rows = data['rows']
    assert len(rows) == len({r['group'] for r in rows}) == 24
    assert [r['level'] for r in rows[:3]] == ['Critical', 'Critical', 'Low']
    assert all(r['split'] != 'test' for r in rows)
    assert len(audit.select_panel(rows)) == 24


def test_no_evaluator_fields_in_model_payload(tmp_path):
    with pytest.raises(ValueError, match='evaluator'):
        audit.assess(None, ['image.png'], {'instruction': 'x', 'level': 'Critical'}, 'x', tmp_path)


@pytest.mark.parametrize('text_only', [False, True])
def test_call_reaches_actual_reviewer_load_guard(tmp_path, text_only):
    # Use the actual runner, stopping at load: missing modality flags previously
    # failed before this point and were hidden by a permissive fixture.
    reviewer = object.__new__(QwenAnnotationReviewer)
    with patch.object(reviewer, 'load', side_effect=RuntimeError('load reached')):
        with pytest.raises(RuntimeError, match='load reached'):
            audit.assess(reviewer, [] if text_only else ['image.png'],
                         {'hypothetical_scene': 'An intact room.'} if text_only else
                         {'instruction': 'Assess the depicted intervention need.'},
                         'modality_contract', tmp_path, text_only=text_only)


def test_two_teachers_run_without_training_or_promotion(tmp_path):
    rows = json.loads(Path(audit.__file__).with_name('grounding_panel.json').read_text())['rows']
    requests = []
    reviewers = []

    class Fixture:
        provenance = {'model': 'fixture not real VLM'}
        closed = False

        def review_structured(self, paths, payload, prompt, parser, case_id, output, *,
                              protocol, text_only_diagnostic=False, text_only_prefix=None):
            if bool(paths) == bool(text_only_diagnostic):
                raise ValueError('invalid modality contract')
            assert (text_only_prefix is not None) == text_only_diagnostic
            requests.append((paths, payload))
            parsed, _ = parser(json.dumps({'action_needed': False,
                'substantial_consequence': False, 'immediate_threat': False,
                'evidence': 'fixture only'}))
            return {'status': 'complete', 'parsed': parsed, 'image_inputs': paths}

        def close(self):
            self.closed = True

    def loader(spec):
        instance = Fixture()
        reviewers.append(instance)
        return instance

    with patch.object(audit, 'verify_inputs', return_value=(rows, {'layout': 'expanded'})), \
         patch.object(audit, '_images', return_value=tmp_path), \
         patch.object(audit, '_image_path', side_effect=lambda folder, row: row['image_id']):
        output = audit.execute(tmp_path, tmp_path / 'run', teacher_loader=loader)
    status = json.loads((output / 'status.json').read_text())
    assert status['status'] == 'diagnostic_complete'
    assert not status['training_allowed'] and not status['runtime_promotion_allowed']
    assert len(reviewers) == 2 and all(r.closed for r in reviewers)
    assert len(requests) == 2 * (48 + 24)
    assert all(set(payload) == ({'instruction'} if paths else {'hypothetical_scene'})
               for paths, payload in requests)
    assert (output / 'recovery.zip').is_file()
    assert not (output / 'student').exists()


def test_load_failure_persists_status_and_recovery(tmp_path):
    rows = json.loads(Path(audit.__file__).with_name('grounding_panel.json').read_text())['rows']
    def fail(spec):
        raise RuntimeError('fixture load error')
    with patch.object(audit, 'verify_inputs', return_value=(rows, {'layout': 'expanded'})), \
         patch.object(audit, '_images', return_value=tmp_path), \
         pytest.raises(RuntimeError, match='fixture load'):
        audit.execute(tmp_path, tmp_path / 'failed', teacher_loader=fail)
    assert json.loads((tmp_path / 'failed/status.json').read_text())['status'] == 'incomplete'
    assert (tmp_path / 'failed/recovery.zip').is_file()
