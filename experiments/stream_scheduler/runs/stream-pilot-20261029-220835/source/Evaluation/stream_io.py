"""Versioned runtime/reference JSONL joins; keeps supervision outside ingress."""
import json
from pathlib import Path
import numpy as np

from Evaluation.decoder_queue import ReceivedMessage
from Evaluation.scene_environment import SceneEpisode
from Evaluation.stream_simulator import compact_values
from Scheduler.scene_policy import canonical, digest


def load_offered_stream(runtime_path, reference_path=None, *, seed=0, failure_probability=0., delay_cap=10., observation_end=60.):
    """Replay externally supplied compact predictions without importing a classifier.

    Runtime JSONL: schema, source_id, session_id, message_token, sequence,
    arrival_time, parameters, payload_ref, optional metadata. Reference JSONL:
    message_id (the full canonical transport identity), fact_id, priority 0..3,
    weight. Reference-only facts/labels never enter the queue header.
    Service is explicitly synthetic constant 1; no source grouping is inferred.
    """
    if not 0 <= failure_probability <= 1 or not np.isfinite(delay_cap) or delay_cap <= 0:
        raise ValueError('invalid failure probability or follow-up')
    rows = [json.loads(line) for line in Path(runtime_path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if not rows: raise ValueError('empty offered stream')
    allowed = {'schema', 'source_id', 'session_id', 'message_token', 'sequence', 'arrival_time',
               'parameters', 'payload_ref', 'metadata'}
    messages = []
    for row in rows:
        if set(row) - allowed or row.get('schema') != 'stream_received_v1':
            raise ValueError('unexpected runtime schema/fields')
        compact_values(row['parameters'])
        messages.append(ReceivedMessage(row['source_id'], row['session_id'], row['message_token'],
            canonical(row['parameters']).encode(), row['payload_ref'], True, row['arrival_time'], row['sequence']))
    ids = {m.message_id for m in messages}
    if len(ids) != len(messages): raise ValueError('duplicate full transport identity')
    references = {}
    if reference_path:
        for line in Path(reference_path).read_text(encoding='utf-8-sig').splitlines():
            if not line.strip(): continue
            r = json.loads(line)
            if set(r) != {'message_id', 'fact_id', 'priority', 'weight'} or r['message_id'] not in ids:
                raise ValueError('invalid reference join')
            if r['message_id'] in references: raise ValueError('duplicate reference identity')
            if type(r['priority']) is not int or not 0 <= r['priority'] <= 3:
                raise ValueError('reference priority must be 0..3')
            if not isinstance(r['fact_id'], str) or not r['fact_id'] or not np.isfinite(r['weight']) or r['weight'] < 0:
                raise ValueError('invalid reference fact/weight')
            references[r['message_id']] = r
    facts, weights, outcomes, records = {}, {}, set(), []
    rng = np.random.default_rng(seed)
    # Canonical transport identity makes file order irrelevant to hidden outcomes.
    draws = {m.message_id: float(rng.random()) for m in sorted(messages, key=lambda m: m.message_id)}
    for m in messages:
        r = references.get(m.message_id)
        fact = r['fact_id'] if r else None
        facts[m.message_id] = frozenset([fact]) if fact else frozenset()
        if fact:
            if fact in weights and weights[fact] != float(r['weight']): raise ValueError('conflicting fact weights')
            weights[fact] = float(r['weight'])
        if draws[m.message_id] < failure_probability: outcomes.add(m.message_id)
        records.append({'message_id': m.message_id, 'arrival': m.arrival_time,
                        'evaluator': {'fact_id': fact or m.message_id, 'reference_priority': r['priority'] if r else None,
                                      'failed': m.message_id in outcomes, 'failure_draw': draws[m.message_id]}})
    if not np.isfinite(observation_end) or observation_end <= 0 or any(m.arrival_time + delay_cap > observation_end for m in messages):
        raise ValueError('predeclared observation end must allow promised follow-up')
    horizon = float(observation_end)
    episode = SceneEpisode(tuple(messages), facts, {m.message_id: 1. for m in messages}, horizon,
                           weights, frozenset(outcomes), 'external_reference_scope_must_be_declared_by_source')
    manifest = {'schema': 'external_stream_replay_v1', 'config': {'seed': seed, 'delay_cap': delay_cap,
        'failure_probability': failure_probability, 'service_seconds': 'synthetic 1'}, 'records': records,
        'runtime_sha256': digest(rows), 'reference_sha256': digest(references),
        'limitations': ['External reference provenance must be audited before research claims',
                        'No training or automatic dataset split', 'Service and failures are simulated']}
    manifest['sha256'] = digest(manifest)
    return episode, manifest
