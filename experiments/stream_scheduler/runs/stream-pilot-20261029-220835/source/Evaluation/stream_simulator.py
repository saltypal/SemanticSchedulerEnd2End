"""External-parameter streams; no SPI model import, prediction or training.

The compact JSON ingress is an interchange adapter, NOT the 35-byte wire header.
Synthetic worlds exercise plumbing; their labels do not validate image semantics.
"""
from dataclasses import dataclass, asdict
import json
import math
import numpy as np

from Evaluation.decoder_queue import ReceivedMessage
from Evaluation.decoder_service import DecoderService
from Evaluation.scene_environment import SceneEnvironment, SceneEpisode, ActionPolicy
from Scheduler.scene_policy import SceneCandidate, SceneContext, canonical, digest


FIELDS = ('spi/critical', 'spi/high', 'spi/medium', 'spi/normal', 'relevance',
          'novelty', 'instances', 'categories', 'largest_box_fraction', 'union_box_coverage')
REGISTRY_HASH = digest({'schema': 'external_compact_numeric_v1', 'fields': FIELDS})
LEVELS = ('Critical', 'High', 'Medium', 'Low')
PROFILES = (
    ('phone', 'General purpose personal phone', 'Share ordinary everyday images without an assumed emergency mission.'),
    ('incident_phone', 'Citizen reporting phone', 'Document visible incident hazards and meaningful updates.'),
    ('damage_drone', 'Aerial structural inspection camera', 'Inspect visible building damage and supported condition changes.'),
    ('person_drone', 'Aerial person-search camera', 'Search for visible people and relevant surroundings; do not assume distress.'),
    ('flood_drone', 'Flood survey drone', 'Observe inundation and evacuation route access changes.'),
    ('fire_drone', 'Wildfire observation drone', 'Observe visible fire and smoke near routes and infrastructure.'),
    ('road_drone', 'Bridge and road inspection drone', 'Inspect visible structure and access obstructions.'),
    ('car', 'Ordinary moving road camera', 'Observe routine road conditions and obstacles.'),
    ('response_car', 'Emergency response vehicle camera', 'Observe approach route hazards and incident updates.'),
    ('traffic_cctv', 'Fixed intersection camera', 'Observe traffic obstruction and visible incidents.'),
    ('public_cctv', 'Fixed public-space camera', 'Observe visible access and safety changes without inferring danger from occupancy.'),
    ('water_camera', 'Fixed river or underpass camera', 'Observe water coverage and access; physical depth is unknown.'),
    ('robot', 'Industrial inspection robot camera', 'Inspect visible equipment and obstruction changes on the assigned route.'),
    ('farm_drone', 'Agricultural survey drone', 'Inspect visible field changes for routine agriculture.'),
    ('maintenance', 'Technician field camera', 'Document visible infrastructure condition and repair evidence.'),
)


def compact_values(row):
    """Consume a compact parameter export, preserving unknown versus false/zero."""
    if row.get('schema') != 'better_minimal_parameters_v2':
        raise ValueError('unsupported external parameter schema')
    allowed = {'schema', 'spi', 'spi_available', 'spi_display', 'spi_reason', 'relevance',
               'novelty', 'novelty_cold_start', 'capture_time_ms', 'allowed_delay_ms', 'scene'}
    if set(row) - allowed:
        raise ValueError('unknown fields in parameter record; evaluator sidecars are forbidden')
    available_spi = row.get('spi_available')
    if type(available_spi) is not bool or type(row.get('novelty_cold_start')) is not bool:
        raise ValueError('explicit SPI availability and novelty cold-start required')
    level = row.get('spi')
    if level == 'Normal': level = 'Low'
    if available_spi and level not in LEVELS:
        raise ValueError('available SPI requires a recognized supplied class')
    if not available_spi and level is not None:
        raise ValueError('unavailable SPI must not contain a guessed class')
    scene = row.get('scene')
    scene_fields = FIELDS[6:]
    if not isinstance(scene, dict) or set(scene) != set(scene_fields):
        raise ValueError('four frozen global detector measurements required')
    raw = [(float(level == x) if available_spi else None) for x in LEVELS]
    raw += [row.get('relevance'), row.get('novelty')] + [scene[k] for k in scene_fields]
    for i, value in enumerate(raw):
        if value is None: continue
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('finite numeric evidence required')
        lo, hi = (-1., 1.) if i == 4 else ((0., 65535.) if i in (6, 7) else (0., 1.))
        if not lo <= value <= hi: raise ValueError('parameter outside declared range')
        if i in (6, 7) and int(value) != value: raise ValueError('integer detection counts required')
    # Frozen upstream SceneMeasurements invariants, without importing a producer.
    count, categories, largest, coverage = (scene[k] for k in scene_fields)
    if count is not None and categories is not None and (categories > count or count > 0 and categories == 0):
        raise ValueError('inconsistent detector counts')
    if largest is not None and coverage is not None and largest > coverage + 1e-12:
        raise ValueError('largest box exceeds union')
    if count == 0 and (largest is not None or coverage not in (None, 0)):
        raise ValueError('empty detections have no largest box and zero coverage')
    for key in ('capture_time_ms', 'allowed_delay_ms'):
        value = row.get(key)
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError('positive integer capture/delay metadata required')
    if row['novelty_cold_start'] and raw[5] is None:
        raise ValueError('cold-start novelty requires its upstream convention value')
    # Hide cold-start convention values from the model; expose a separate flag.
    cold = tuple(i == 5 and row['novelty_cold_start'] for i in range(len(raw)))
    available = tuple(v is not None and not cold[i] for i, v in enumerate(raw))
    return tuple(float(v) if v is not None and available[i] else 0. for i, v in enumerate(raw)), available, cold


class ParameterQueue:
    """Queue adapter for compact records. Owns no producer and no evaluator truth."""
    def __init__(self, policy):
        self.policy = policy
        self.reset()

    def reset(self):
        self.jobs, self.seen, self.clock = {}, set(), 0.
        self.policy.reset()

    def __len__(self): return len(self.jobs)

    def enqueue(self, message):
        if not isinstance(message, ReceivedMessage) or not message.complete:
            raise ValueError('complete received message required')
        if message.message_id in self.seen: raise ValueError('duplicate transport identity')
        from Evaluation.scene_queue import strict_json
        row = strict_json(message.header_bytes)
        values, available, cold = compact_values(row)
        candidate = SceneCandidate(message.message_id, message.source_id, message.arrival_time,
                                   message.sequence, values, available, cold, REGISTRY_HASH)
        self.jobs[message.message_id] = (message, candidate)
        self.seen.add(message.message_id)

    def ready_snapshot(self, now, trusted_utc=None):
        if not math.isfinite(now) or now < self.clock: raise ValueError('nonmonotonic queue time')
        self.clock = now
        return tuple(c for m, c in self.jobs.values() if m.arrival_time <= now)

    def take(self, selected_id, snapshot):
        if selected_id not in {c.message_id for c in snapshot} or selected_id not in self.jobs:
            raise ValueError('stale or foreign queue selection')
        return self.jobs.pop(selected_id)[0]


class StreamEnvironment(SceneEnvironment):
    """Reuse existing transitions/service/learners with numeric parameter ingress."""
    def __init__(self, episode):
        self.episode = episode
        self.reset()

    def reset(self):
        self.action_policy = ActionPolicy()
        builder = lambda ready, history, attempts, admitted, now: SceneContext(
            tuple(ready), tuple(history), attempts, admitted, now, self.episode.horizon, REGISTRY_HASH)
        self.controller = DecoderService(self.action_policy, queue=ParameterQueue(self.action_policy),
                                         context_builder=builder)
        self.pending = sorted(self.episode.messages, key=lambda m: (m.arrival_time, m.sequence, m.message_id))
        self.now, self.total, self.finished = 0., sum(self.episode.weights.values()), False
        self.covered, self.events, self.transitions = set(), [], []
        self._advance_to_ready()
        return self.observation()


@dataclass(frozen=True)
class StreamConfig:
    seed: int = 1729
    max_ues: int = 12
    messages: int = 48
    capture_window: float = 20.
    failure_probability: float = .05
    repeat_probability: float = .25
    missing_spi_probability: float = .1
    delay_cap: float = 10.
    service_mean: float = 1.
    service_log_sigma: float = 0.
    correlated_outage: bool = False

    def __post_init__(self):
        if type(self.seed) is not int or type(self.max_ues) is not int or not 1 <= self.max_ues <= 100:
            raise ValueError('integer seed and 1..100 UE population required')
        if type(self.messages) is not int or self.messages < 1: raise ValueError('positive message count required')
        for p in (self.failure_probability, self.repeat_probability, self.missing_spi_probability):
            if not math.isfinite(p) or not 0 <= p <= 1: raise ValueError('probabilities must be in [0,1]')
        if any(not math.isfinite(v) or v <= 0 for v in (self.capture_window, self.delay_cap, self.service_mean)):
            raise ValueError('positive finite time configuration required')
        if not math.isfinite(self.service_log_sigma) or self.service_log_sigma < 0:
            raise ValueError('invalid service log sigma')
        if type(self.correlated_outage) is not bool: raise ValueError('boolean outage setting required')


def build_stream(config):
    """Synthetic parameter/world fixture; no images, classifier or VLM is run."""
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(config.seed).spawn(5)]
    population_rng, world_rng, noise_rng, timing_rng, outcome_rng = streams
    ues = []
    for i in range(config.max_ues):
        profile = PROFILES[int(population_rng.integers(len(PROFILES)))]
        join = 0. if i < max(1, config.max_ues // 3) else float(population_rng.uniform(0, config.capture_window * .45))
        leave = float(population_rng.uniform(config.capture_window * .6, config.capture_window))
        pause = (join + .3 * (leave - join), join + .4 * (leave - join))
        ues.append({'source_id': f'ue{i:03}', 'profile_id': profile[0], 'device_description': profile[1],
                    'mission': profile[2], 'join': join, 'leave': leave, 'pause': pause,
                    'origin': [12.9 + float(population_rng.uniform(-.01, .01)), 77.5 + float(population_rng.uniform(-.01, .01))],
                    'velocity': [float(population_rng.uniform(-1e-5, 1e-5)) for _ in range(2)]})
    captures = []
    for i in range(config.messages):
        ue = ues[int(population_rng.integers(len(ues)))]
        capture = float(timing_rng.uniform(ue['join'], ue['leave']))
        if ue['pause'][0] <= capture < ue['pause'][1]: capture = ue['pause'][1]
        captures.append((capture, i, ue))
    captures.sort(key=lambda x: (x[0], x[1]))
    known = {u['source_id']: set() for u in ues}
    world, messages, facts, weights, durations, failures, records = [], [], {}, {}, {}, set(), []
    epoch_ms = 1793232000000  # Declared simulated epoch; not image EXIF.
    for sequence, (capture, _, ue) in enumerate(captures):
        if world and world_rng.random() < config.repeat_probability:
            fact = world[int(world_rng.integers(len(world)))]
        else:
            fact = {'id': f'world{config.seed}/fact{len(world)}',
                    'priority': int(world_rng.choice(4, p=[.12, .23, .25, .4])),
                    'relevance': float(world_rng.uniform(0, 1))}
            world.append(fact)
            weights[fact['id']] = float((3 - fact['priority']) * (fact['relevance'] >= .35))
        # This is noisy synthetic supplied evidence, NOT a learned SPI predictor.
        predicted = fact['priority'] if noise_rng.random() > .2 else int(noise_rng.integers(4))
        available_spi = noise_rng.random() >= config.missing_spi_probability
        cold_start = not known[ue['source_id']]
        novelty = float(fact['id'] not in known[ue['source_id']])
        instances = int(noise_rng.integers(0, 12))
        categories = int(noise_rng.integers(1, min(instances, 4) + 1)) if instances else 0
        largest = float(noise_rng.uniform(.01, .4)) if instances else None
        coverage = float(noise_rng.uniform(largest, min(1., largest * instances))) if instances else 0.
        row = {'schema': 'better_minimal_parameters_v2', 'spi': LEVELS[predicted] if available_spi else None,
               'spi_available': bool(available_spi), 'relevance': float(np.clip(2 * fact['relevance'] - 1 + noise_rng.normal(0, .15), -1, 1)),
               'novelty': novelty, 'novelty_cold_start': cold_start, 'capture_time_ms': epoch_ms + round(capture * 1000),
               'allowed_delay_ms': None,
               'scene': {'instances': instances, 'categories': categories,
                         'largest_box_fraction': largest, 'union_box_coverage': coverage}}
        compact_values(row)
        known[ue['source_id']].add(fact['id'])  # Commit every offered observation.
        extraction = float(timing_rng.uniform(.01, .05)); transport = float(timing_rng.uniform(.01, .1))
        arrival = capture + extraction + transport
        m = ReceivedMessage(ue['source_id'], f'episode{config.seed}', sequence, canonical(row).encode(),
                            {'kind': 'synthetic_opaque_payload', 'simulated_bytes': 4096}, True, arrival, sequence)
        sigma = config.service_log_sigma
        duration = float(timing_rng.lognormal(math.log(config.service_mean) - sigma**2 / 2, sigma))
        draw = float(outcome_rng.random())
        outage = config.correlated_outage and config.capture_window * .45 <= capture < config.capture_window * .55
        failed = bool(draw < config.failure_probability or outage)
        messages.append(m); facts[m.message_id] = frozenset([fact['id']]); durations[m.message_id] = duration
        if failed: failures.add(m.message_id)
        records.append({'message_id': m.message_id, 'capture': capture, 'extraction_ready': capture + extraction,
                        'arrival': arrival, 'location': [ue['origin'][j] + ue['velocity'][j] * capture for j in range(2)],
                        'coordinate_system': 'WGS84', 'location_accuracy_m': 10., 'metadata_origin': 'simulated',
                        'runtime_parameters': row, 'evaluator': {'fact_id': fact['id'], 'reference_priority': fact['priority'],
                        'reference_relevance': fact['relevance'], 'service_duration': duration,
                        'failure_draw': draw, 'failed': failed, 'correlated_outage': bool(outage)}})
    # Predeclared endpoint, not the realized last future arrival exposed to actor.
    horizon = config.capture_window + .15 + config.delay_cap
    episode = SceneEpisode(tuple(messages), facts, durations, horizon, weights, frozenset(failures),
                           'synthetic_world_fixture_not_image_semantics')
    manifest = {'config': asdict(config), 'ues': ues, 'records': records,
                'schema': 'synthetic_stream_v1', 'registry_hash': REGISTRY_HASH,
                'limitations': ['No real RGB stream', 'Reference relevance is a constructed scalar, not actual mission inference',
                                'No SPI classifier', 'No real VLM', 'No codec or radio simulation']}
    manifest['sha256'] = digest(manifest)
    return episode, manifest


def priority_metrics(episode, manifest, events, delay_cap):
    from Evaluation.spi_evaluation import SPIReference, SPIOutcome, evaluate_spi_outcomes
    refs = [SPIReference(r['message_id'], r['evaluator']['fact_id'], r['arrival'],
                         r['evaluator']['reference_priority'], episode.scope) for r in manifest['records']]
    outcomes = [SPIOutcome(e['message_id'], e['start'], e['finish'], bool(e['success'])) for e in events]
    return evaluate_spi_outcomes(refs, outcomes, observation_end=episode.horizon, delay_cap=delay_cap)
