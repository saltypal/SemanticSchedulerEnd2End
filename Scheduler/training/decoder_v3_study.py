"""Controlled observation-coverage study for complete V3 decoder jobs.

Evaluator identities/labels never enter scheduler candidates. Native image
identity is a controlled observation unit, not a verified physical-asset fact.
"""
from dataclasses import dataclass, replace
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import time

import numpy as np

from Archived.owners_before_simplification_20261027.SemParam.decoder_parameters import (PROFILES, ImageEvidence, ParameterExtractor,
    ParameterHeader, TransmitterMemory, fit_temperature, get_profile, freshness, sigmoid, project_nested)
from Scheduler.decoder_scheduler import (ARTIFACT_SCHEMA, FEATURE_NAMES,
    SchedulerCandidate, SemanticScheduler, feature_matrix, weights_digest, load_scheduler, tie_key)

PROTOCOL_ID = 'decoder_v3_controlled_observation_v1'
STRATA = ('no_repeat', 'local25', 'cross25', 'local50', 'cross50')
SEED = 20261022


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def native_utility(labels, profile):
    profile = get_profile(profile)
    if any(name not in labels or labels[name] not in (False, True) for name in profile.label_names):
        raise ValueError('missing/invalid native task labels')
    truth = [int(labels[name]) for name in profile.label_names]
    if any(left < right for left, right in zip(truth, truth[1:])):
        return None  # contradictory nested truth is excluded for this task
    return float(3 * np.dot(truth, profile.importance_weights))


def load_cache(cache):
    cache = Path(cache)
    features, labels = read_jsonl(cache/'features.jsonl'), read_jsonl(cache/'labels.jsonl')
    by_id = {row['image_id']: row for row in labels}
    if len(by_id) != len(labels) or len({row['image_id'] for row in features}) != len(features):
        raise ValueError('duplicate persisted image identities')
    if set(by_id) != {row['image_id'] for row in features}:
        raise ValueError('feature/label cache is incomplete')
    records = []
    for feature in features:
        label = by_id[feature['image_id']]
        if 'labels' in feature or feature['role'] != label['role']:
            raise ValueError('feature truth leakage or role disagreement')
        records.append(dict(evidence=ImageEvidence.from_dict(feature), labels=label['labels'],
                            role=label['role'], group=label['duplicate_group'], image_id=feature['image_id']))
    group_roles = defaultdict(set)
    for row in records:
        group_roles[row['group']].add(row['role'])
    if any(len(roles) > 1 for roles in group_roles.values()):
        raise ValueError('duplicate group crosses data roles')
    return records


def calibrate(records, profile):
    profile = get_profile(profile)
    rows = [row for row in records if row['role'] == 'B' and native_utility(row['labels'], profile) is not None
            and all(name in row['evidence'].logits for name in profile.label_names)]
    logits = np.asarray([[row['evidence'].logits[name] for name in profile.label_names] for row in rows], float).reshape(-1, len(profile.label_names))
    labels = np.asarray([[row['labels'][name] for name in profile.label_names] for row in rows], float).reshape(logits.shape)
    return fit_temperature(logits, labels, profile)


def verify_extraction_cache(cache, records):
    """Fail closed before fitting when a published extraction manifest is present."""
    cache = Path(cache)
    summary_path = cache/'extraction_summary.json'
    if not summary_path.exists():
        return dict(scope='fixture_or_partial_cache',published_extraction_verified=False)
    summary = json.loads(summary_path.read_text())
    manifest = json.loads((cache/'manifest.json').read_text())
    expected = {}
    excluded = set()
    for row in manifest:
        if row['role'].startswith('excluded'):
            if row['role'] == 'excluded_conflicting_labels':
                excluded.add(row['image_id'])
            continue
        if row['image_id'] in expected and expected[row['image_id']]['labels'] != row['labels']:
            raise ValueError('published cache has unresolved conflicting labels')
        expected[row['image_id']] = row
    if set(expected) != {row['image_id'] for row in records} or excluded & set(expected):
        raise ValueError('published cache identities/exclusions differ from manifest')
    if len(records) != summary['result']['written']+summary['result'].get('resumed',0):
        raise ValueError('published cache count differs from extraction summary')
    for row in records:
        source = expected[row['image_id']]
        evidence = row['evidence']
        if row['labels'] != source['labels'] or row['role'] != source['role'] or row['group'] != source['duplicate_group']:
            raise ValueError('published cache labels/splits differ from manifest')
        if np.asarray(evidence.embedding).shape != (512,) or not np.isfinite(evidence.embedding).all():
            raise ValueError('invalid cached CLIP embedding')
        if any(evidence.raw.get('model_revisions',{}).get(name) != revision
               for name,revision in summary['model_revisions'].items()):
            raise ValueError('published cache producer revision mismatch')
        if any(evidence.statuses.get(name) != 'ok' for name in ('clip','ladi','detector')):
            raise ValueError('published cache contains failed image evidence')
        json.dumps(evidence.to_dict(),allow_nan=False)
    return dict(published_extraction_verified=True,unique_images=len(records),excluded_ids=sorted(excluded),
                producer_revisions=summary['model_revisions'],
                manifest_sha256=hashlib.sha256((cache/'manifest.json').read_bytes()).hexdigest())


def calibration_audit(records,profile,calibration):
    """Assess the declared critical claim; never refit using C or test labels."""
    profile = get_profile(profile)
    critical = profile.label_names.index(profile.critical_claim)
    report = dict(claim=profile.critical_claim,reliability_available=calibration.reliable,roles={})
    for role in ('B','C','test'):
        rows = [row for row in records if row['role']==role and native_utility(row['labels'],profile) is not None]
        truth = np.asarray([row['labels'][profile.critical_claim] for row in rows],float)
        probabilities = np.asarray([project_nested(sigmoid(np.asarray([
            row['evidence'].logits[name] for name in profile.label_names])/
            (calibration.temperature if calibration.reliable else 1.)))[critical] for row in rows])
        correct = (probabilities>=.5)==truth
        confidence = np.maximum(probabilities,1-probabilities)
        bins = []
        for index in range(10):
            keep = (confidence>=index/10)&((confidence<(index+1)/10) if index<9 else (confidence<=1))
            bins.append(dict(count=int(keep.sum()),confidence=float(confidence[keep].mean()) if keep.any() else None,
                             accuracy=float(correct[keep].mean()) if keep.any() else None))
        report['roles'][role] = dict(count=len(rows),positives=int(truth.sum()),
            probability_brier=float(np.mean((probabilities-truth)**2)) if len(rows) else None,
            correctness_brier=float(np.mean((confidence-correct)**2)) if len(rows) else None,
            correctness_ECE=sum(row['count']*abs(row['confidence']-row['accuracy']) for row in bins if row['count'])/len(rows) if rows else None,
            bins=bins,scope='fit_partition' if role=='B' else 'held_out_assessment')
    return report


@dataclass(frozen=True)
class Episode:
    episode_id: str
    pool_id: str
    task: str
    stratum: str
    candidates: tuple
    units: tuple
    utilities: tuple
    header_hex: tuple

    def __post_init__(self):
        if not (len(self.candidates) == len(self.units) == len(self.utilities) == len(self.header_hex)):
            raise ValueError('episode record dimensions differ')
        seen = {}
        for unit, utility in zip(self.units, self.utilities):
            if not math.isfinite(utility) or utility < 0 or (unit in seen and seen[unit] != utility):
                raise ValueError('unit weights must be finite, nonnegative and consistent')
            seen[unit] = utility


def make_episode(pool, profile, stratum, seed, calibration=None, pool_id='train'):
    profile = get_profile(profile)
    if stratum not in STRATA or len(pool) < 32:
        raise ValueError('controlled episodes require a supported stratum and >=32 observations')
    rng = np.random.default_rng(seed)
    selected = [pool[i] for i in rng.permutation(len(pool))[:32]]
    if len({row['group'] for row in selected}) != 32:
        raise ValueError('source pool must contain distinct duplicate groups')
    originals_per_ue = 8 if stratum == 'no_repeat' else 6 if stratum.endswith('25') else 4
    originals = [[selected[source*originals_per_ue+i] for i in range(originals_per_ue)] for source in range(4)]
    observations = [list(rows) for rows in originals]
    for source in range(4):
        other = (source + 1) % 4 if stratum.startswith('cross') else source
        for index in range(8-originals_per_ue):
            observations[source].append(originals[other][index % originals_per_ue])
    episode_id = f'{profile.task}:{pool_id}:{stratum}:{seed}'
    entries = []
    for source in range(4):
        memory = TransmitterMemory(f'ue{source}', profile.profile_id)
        for sequence, row in enumerate(observations[source]):
            bundle = ParameterExtractor().extract(row['evidence'], profile, {}, memory, calibration)
            header = ParameterHeader(profile.profile_id, sequence, memory.version, 0, bundle.values, bundle.available)
            wire = header.to_bytes()
            parsed = ParameterHeader.from_bytes(wire)
            memory.commit(row['evidence'])
            entries.append((source, sequence, row, parsed, wire.hex()))
    ready_order = rng.permutation(32)
    candidates, units, utilities, wires = [], [], [], []
    for rank, index in enumerate(ready_order):
        source, sequence, row, header, wire = entries[index]
        candidates.append(SchedulerCandidate.from_header(f'{episode_id}:u{source}:m{sequence}', f'ue{source}', 0., rank, header))
        units.append(row['image_id'])
        utility = native_utility(row['labels'], profile)
        if utility is None:
            raise ValueError('contradictory task truth entered episode pool')
        utilities.append(utility)
        wires.append(wire)
    return Episode(episode_id, str(pool_id), profile.task, stratum,
                   tuple(candidates), tuple(units), tuple(utilities), tuple(wires))


def eligible_records(records, role, profile):
    representatives = {}
    for row in sorted(records, key=lambda row: row['image_id']):
        if row['role'] == role and native_utility(row['labels'], profile) is not None:
            representatives.setdefault(row['group'], row)
    return list(representatives.values())


def evaluation_bank(records, role, profile, calibration, repeats=6):
    rows = eligible_records(records, role, profile)
    rng = np.random.default_rng(SEED)
    rows = [rows[i] for i in rng.permutation(len(rows))]
    episodes = []
    for start in range(0, len(rows)-31, 32):
        pool = rows[start:start+32]
        pool_id = f'{role}-pool-{start//32:03d}'
        for index, stratum in enumerate(STRATA):
            for repetition in range(repeats):
                episodes.append(make_episode(pool, profile, stratum, SEED+1000*start+100*index+repetition,
                                             calibration, pool_id))
    if not episodes:
        raise ValueError(f'no eligible 32-observation evaluation pool for {role}/{get_profile(profile).task}')
    return episodes


def coverage_curve(order, units, utilities, k=10):
    if sorted(order) != list(range(len(units))) or not 1 <= k <= len(units):
        raise ValueError('order must be a full permutation and cutoff valid')
    weights = dict(zip(units, utilities))
    total = math.fsum(weights.values())
    if total == 0:
        return None
    seen, mass, curve = set(), 0., []
    for index in order[:k]:
        if units[index] not in seen:
            mass += weights[units[index]]
        seen.add(units[index])
        curve.append(mass/total)
    return np.asarray(curve)


def early_coverage(order, units, utilities, k=10):
    curve = coverage_curve(order, units, utilities, k)
    return None if curve is None else float(curve.mean())


def pair_data(scores, episode, coverage=True, k=10):
    """Exact detached swap deltas for the controlled one-unit-per-message track."""
    scores = np.asarray(scores, float)
    count = len(episode.units)
    if scores.shape != (count,) or not np.isfinite(scores).all():
        raise ValueError('finite per-message scores required')
    if not coverage:
        left, right = np.triu_indices(count, 1)
        difference = np.asarray(episode.utilities)[left] - np.asarray(episode.utilities)[right]
        keep = difference != 0
        return left[keep], right[keep], (difference[keep]>0).astype(float), np.ones(keep.sum())
    order = sorted(range(count), key=lambda i: (-scores[i], tie_key(episode.candidates[i])))
    units = np.asarray(episode.units, dtype=object)[order]
    utility = np.asarray(episode.utilities)[order]
    total = math.fsum(dict(zip(episode.units, episode.utilities)).values())
    if not total:
        return tuple(np.array([], dtype=int if i < 2 else float) for i in range(4))
    supports = defaultdict(list)
    for rank, unit in enumerate(units, 1):
        if len(supports[unit]) < 2:
            supports[unit].append(rank)
    first = np.asarray([supports[unit][0] for unit in units])
    second = np.asarray([supports[unit][1] if len(supports[unit])>1 else np.inf for unit in units])
    a, b = np.triu_indices(count, 1)
    discount = lambda rank: np.maximum(0., (k-rank+1)/k)
    other_a = np.where(first[a] == a+1, second[a], first[a])
    other_b = np.where(first[b] == b+1, second[b], first[b])
    delta = (utility[a]*(discount(np.minimum(other_a,b+1))-discount(first[a])) +
             utility[b]*(discount(np.minimum(other_b,a+1))-discount(first[b]))) / total
    delta[units[a] == units[b]] = 0
    keep = np.abs(delta) > 1e-12
    order = np.asarray(order)
    return order[a[keep]], order[b[keep]], (delta[keep]<0).astype(float), np.abs(delta[keep])


def frozen_loss_gradient(scores, pairs):
    left, right, target, weights = pairs
    scores = np.asarray(scores, float)
    gradient, hessian = np.zeros_like(scores), np.zeros_like(scores)
    if weights.sum() == 0:
        return None, gradient, hessian
    weight = weights/weights.sum()
    difference = scores[left]-scores[right]
    probability = 1/(1+np.exp(-np.clip(difference,-700,700)))
    loss = float(np.sum(weight*(np.logaddexp(0,difference)-target*difference)))
    direction = weight*(probability-target)
    curvature = weight*probability*(1-probability)
    np.add.at(gradient,left,direction); np.add.at(gradient,right,-direction)
    np.add.at(hessian,left,curvature); np.add.at(hessian,right,curvature)
    return loss, gradient, hessian


def episode_order(model, episode):
    model.reset()
    index = {candidate.message_id:i for i,candidate in enumerate(episode.candidates)}
    return [index[decision.message_id] for decision in model.rank(episode.candidates)]


def score_episode(model, episode):
    order = episode_order(model, episode)
    curve = coverage_curve(order, episode.units, episode.utilities, 10)
    gain = np.asarray(episode.utilities)
    discounts = 1/np.log2(np.arange(10)+2)
    ideal = float(np.sort(gain)[::-1][:10]@discounts)
    ndcg = None if ideal == 0 else float(gain[order[:10]]@discounts/ideal)
    weights = dict(zip(episode.units, episode.utilities))
    high = {unit for unit,weight in weights.items() if weight == 3}
    covered = set(episode.units[i] for i in order[:10])
    return dict(episode_id=episode.episode_id,pool_id=episode.pool_id,task=episode.task,stratum=episode.stratum,
                A10=None if curve is None else float(curve.mean()),
                coverage5=None if curve is None else float(curve[4]),
                coverage10=None if curve is None else float(curve[-1]),ndcg10=ndcg,
                high_recall10=None if not high else len(high&covered)/len(high),
                redundancy10=1-len(covered)/10,
                curve=None if curve is None else curve.tolist(),order=order)


def validation_score(model, episodes):
    groups = defaultdict(list)
    for episode in episodes:
        result = score_episode(model, episode)
        if result['A10'] is not None:
            groups[(episode.pool_id,episode.stratum)].append(result['A10'])
    return float(np.mean([np.mean(values) for values in groups.values()])) if groups else -math.inf


def replay_stream(model, episode, load=1., seed=0, no_backlog=False):
    model.reset()
    rng = np.random.default_rng(seed)
    arrivals = np.arange(len(episode.candidates))*2. if no_backlog else np.cumsum(rng.exponential(1/load,len(episode.candidates)))
    pending, cursor, now, events = [], 0, 0., []
    while cursor < len(arrivals) or pending:
        if not pending and cursor < len(arrivals):
            now = max(now,float(arrivals[cursor]))
        while cursor < len(arrivals) and arrivals[cursor] <= now:
            pending.append(replace(episode.candidates[cursor], arrival_time=float(arrivals[cursor])))
            cursor += 1
        if not pending:
            continue
        decision = model.select(pending)
        selected = next(candidate for candidate in pending if candidate.message_id == decision.message_id)
        pending.remove(selected)
        events.append(dict(message_id=selected.message_id,arrival=selected.arrival_time,start=now,completion=now+1,
                           wait=now-selected.arrival_time,source_id=selected.source_id,unit='abstract_service_tick'))
        now += 1
    return events


def fit_medians(episodes):
    """A-only header values, excluding missing entries in each column."""
    values = [[] for _ in range(8)]
    for episode in episodes:
        for candidate in episode.candidates:
            for index,available in enumerate(candidate.available):
                if available:
                    values[index].append(candidate.values[index])
    return np.asarray([float(np.median(column)) if column else .5 for column in values])


def masked_features(episode,medians,disabled=()):
    matrix = feature_matrix(episode.candidates,medians).astype('float32')
    for index in disabled:
        matrix[:,index] = medians[index]
        matrix[:,index+8] = 0
    return matrix


def export_network(model, kind, profile, medians, folder, disabled=(), metadata=None):
    import torch
    weights, biases = [], []
    for layer in model:
        if isinstance(layer,torch.nn.Linear):
            weights.append(layer.weight.detach().cpu().numpy().T.tolist())
            biases.append(layer.bias.detach().cpu().numpy().tolist())
    artifact = dict(schema=ARTIFACT_SCHEMA,header_version=3,feature_names=FEATURE_NAMES,
                    profile_id=get_profile(profile).profile_id,kind=kind,medians=np.asarray(medians).tolist(),
                    weights=weights,biases=biases,weights_sha256=weights_digest(weights,biases),
                    disabled_fields=list(disabled),protocol_id=PROTOCOL_ID,metadata=metadata or {})
    write_json(Path(folder)/'model.json',artifact)
    return load_scheduler(folder,{get_profile(profile).profile_id:get_profile(profile)})


def fit_network(train, validation, profile, folder, medians, objective='coverage', architecture='mlp',
                rho=1e-4, seed=0, max_epochs=60, patience=10, disabled=()):
    """Train only on header features; checkpoints include optimizer and RNG."""
    import torch
    from torch import nn
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    output = 3 if objective == 'classifier' else 1
    model = nn.Sequential(nn.Linear(16,output)) if architecture == 'linear' else nn.Sequential(
        nn.Linear(16,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,output))
    optimizer = torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0)
    matrices = [torch.from_numpy(masked_features(episode,medians,disabled)) for episode in train]
    best, best_epoch, history = -math.inf, -1, []
    kind = 'classifier' if objective == 'classifier' else 'linear' if architecture == 'linear' else 'regressor' if objective == 'regressor' else 'mlp'
    signature = hashlib.sha256(json.dumps(dict(
        objective=objective,architecture=architecture,rho=rho,seed=seed,disabled=list(disabled),
        medians=np.asarray(medians).tolist(),profile=get_profile(profile).profile_id,
        queue_protocol='shared_fixed_queue_bank',
        train=[(q.header_hex,q.units,q.utilities) for q in train],
        validation=[(q.header_hex,q.units,q.utilities) for q in validation]),sort_keys=True).encode()).hexdigest()
    start_epoch = 0
    if (folder/'last.pt').exists():
        # Only our own experiment artifacts are loaded; never accept untrusted checkpoints.
        checkpoint = torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        if checkpoint.get('signature') != signature:
            raise ValueError('training resume inputs/configuration changed; use a new run directory')
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['torch_rng']); rng.bit_generator.state = checkpoint['numpy_rng']
        random.setstate(checkpoint['python_rng'])
        history = checkpoint['history']
        best, best_epoch = checkpoint['best'], checkpoint['best_epoch']
        start_epoch = checkpoint['epoch']+1
        if checkpoint['epoch']-best_epoch >= patience:
            start_epoch = max_epochs
    for epoch in range(start_epoch,max_epochs):
        model.train()
        epoch_losses, informative = [], 0
        indices = rng.permutation(len(train))
        for start in range(0,len(indices),16):
            batch = indices[start:start+16]
            scores = model(torch.cat([matrices[index] for index in batch]))
            losses, offset = [], 0
            for index in batch:
                episode = train[index]
                n = len(episode.candidates)
                local = scores[offset:offset+n]
                offset += n
                utility = np.asarray(episode.utilities)
                if objective == 'classifier':
                    categories = np.where(utility == 3,2,np.where(utility == 1,1,0))
                    losses.append(nn.functional.cross_entropy(local,torch.as_tensor(categories,dtype=torch.long)))
                elif objective == 'regressor':
                    losses.append(nn.functional.mse_loss(local[:,0],torch.as_tensor(utility/3,dtype=torch.float32)))
                else:
                    left,right,target,weight = pair_data(local[:,0].detach().numpy(),episode,coverage=objective=='coverage')
                    if not len(weight):
                        continue
                    difference = local[left,0]-local[right,0]
                    normalized = torch.as_tensor(weight/weight.sum(),dtype=difference.dtype)
                    targets = torch.as_tensor(target,dtype=difference.dtype)
                    losses.append((normalized*(nn.functional.softplus(difference)-targets*difference)).sum())
                informative += 1
            if not losses:
                continue
            penalty = sum(parameter.square().sum() for parameter in model.parameters())
            loss = torch.stack(losses).mean()+rho*penalty
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        model.eval()
        runtime = export_network(model,kind,profile,medians,folder/'last',disabled)
        score = validation_score(runtime,validation)
        if not math.isfinite(score):
            raise RuntimeError('validation has no defined coverage queries')
        row = dict(epoch=epoch,loss=float(np.mean(epoch_losses)) if epoch_losses else None,
                   informative_queries=informative,validation_A10=score)
        history.append(row)
        improved = score > best+1e-12
        if improved:
            best, best_epoch = score, epoch
        checkpoint = dict(epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),
                          torch_rng=torch.get_rng_state(),numpy_rng=rng.bit_generator.state,
                          python_rng=random.getstate(),objective=objective,architecture=architecture,
                          rho=rho,seed=seed,disabled_fields=list(disabled),protocol_id=PROTOCOL_ID,
                          signature=signature,history=history,best=best,best_epoch=best_epoch)
        if improved:
            torch.save(checkpoint,folder/'best.pt.tmp')
            (folder/'best.pt.tmp').replace(folder/'best.pt')
            export_network(model,kind,profile,medians,folder/'best',disabled,
                           dict(seed=seed,rho=rho,objective=objective,best_epoch=epoch))
        write_json(folder/'history.json',history)
        # Publish the resumable pointer only after its best artifact is durable.
        temporary = folder/'last.pt.tmp'
        torch.save(checkpoint,temporary)
        temporary.replace(folder/'last.pt')
        print(f'{get_profile(profile).task} {objective}/{architecture} seed={seed} rho={rho} epoch={epoch} valA10={score:.6f}',flush=True)
        if epoch-best_epoch >= patience:
            break
    return load_scheduler(folder/'best',{get_profile(profile).profile_id:get_profile(profile)}),best


def tree_objective(predictions, episodes):
    gradient, hessian, cursor = np.zeros_like(predictions),np.zeros_like(predictions),0
    informative = 0
    for episode in episodes:
        n = len(episode.candidates)
        local = predictions[cursor:cursor+n]
        loss,g,h = frozen_loss_gradient(local,pair_data(local,episode))
        if loss is not None:
            gradient[cursor:cursor+n] = g
            hessian[cursor:cursor+n] = h
            informative += 1
        cursor += n
    if informative:
        gradient /= informative; hessian /= informative
    return gradient,np.maximum(hessian,1e-8)


def fit_tree(train,validation,profile,folder,medians,coverage=False,num_leaves=15,max_rounds=200,seed=0):
    import lightgbm as lgb
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    def dataset(episodes):
        matrix = np.concatenate([feature_matrix(episode.candidates,medians) for episode in episodes])
        labels = np.concatenate([np.asarray(episode.utilities,dtype=int) for episode in episodes])
        return lgb.Dataset(matrix,label=labels,group=[len(episode.candidates) for episode in episodes],free_raw_data=False)
    training,selection = dataset(train),dataset(validation)
    def objective(predictions,dataset):
        return tree_objective(predictions,train)
    def metric(predictions,dataset):
        cursor, groups = 0,defaultdict(list)
        for episode in validation:
            n = len(episode.candidates)
            values = predictions[cursor:cursor+n]
            order = sorted(range(n),key=lambda i:(-values[i],tie_key(episode.candidates[i])))
            score = early_coverage(order,episode.units,episode.utilities)
            if score is not None:
                groups[(episode.pool_id,episode.stratum)].append(score)
            cursor += n
        return 'A10',float(np.mean([np.mean(values) for values in groups.values()])),True
    params = dict(objective=objective if coverage else 'lambdarank',metric='None',num_leaves=num_leaves,
                  learning_rate=.05,verbosity=-1,num_threads=2,min_data_in_leaf=10,
                  label_gain=[0,1,2,3],seed=seed,deterministic=True,force_col_wise=True)
    evaluation_history = {}
    booster = lgb.train(params,training,num_boost_round=max_rounds,valid_sets=[selection],feval=metric,
                        callbacks=[lgb.early_stopping(20,first_metric_only=True,verbose=False),lgb.record_evaluation(evaluation_history)])
    write_json(folder/'history.json',evaluation_history)
    path = folder/'tree.txt'
    booster.save_model(str(path))
    metadata = dict(schema=ARTIFACT_SCHEMA,header_version=3,feature_names=FEATURE_NAMES,
                    profile_id=get_profile(profile).profile_id,kind='lightgbm',medians=medians.tolist(),
                    weights=[],biases=[],weights_sha256=weights_digest([],[]),tree_file='tree.txt',
                    tree_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),protocol_id=PROTOCOL_ID,
                    objective='coverage' if coverage else 'lambdarank',num_leaves=num_leaves,seed=seed)
    write_json(folder/'model.json',metadata)
    runtime = load_scheduler(folder,{get_profile(profile).profile_id:get_profile(profile)})
    return runtime,validation_score(runtime,validation)


def save_policy(model,folder):
    """Export deterministic baselines through the same loader as learned models."""
    write_json(Path(folder)/'model.json',dict(schema=ARTIFACT_SCHEMA,header_version=3,
        feature_names=FEATURE_NAMES,profile_id=model.profile_id,kind=model.kind,
        medians=model.medians.tolist(),weights=[],biases=[],weights_sha256=weights_digest([],[]),
        lambda_n=model.lambda_n,seed=model.seed,protocol_id=PROTOCOL_ID))


def pool_means(results,metric='A10'):
    groups = defaultdict(list)
    for row in results:
        if row[metric] is not None:
            groups[(row['pool_id'],row['stratum'])].append(row[metric])
    pools = defaultdict(list)
    for (pool,stratum),values in groups.items():
        pools[pool].append(float(np.mean(values)))
    return {pool:float(np.mean(values)) for pool,values in pools.items()}


def paired_bootstrap(left,right,resamples=10000,seed=SEED):
    keys = sorted(set(left)&set(right))
    if not keys:
        return dict(pools=0,mean_difference=None,ci95=None)
    differences = np.asarray([left[key]-right[key] for key in keys])
    rng = np.random.default_rng(seed)
    boot = differences[rng.integers(0,len(keys),(resamples,len(keys)))].mean(axis=1)
    return dict(pools=len(keys),mean_difference=float(differences.mean()),
                ci95=np.quantile(boot,[.025,.975]).tolist(),unit='disjoint_source_pool_not_disaster_event')


def benchmark_task(models,episodes,folder,strongest):
    import gzip
    import csv
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    results,summary = {},[]
    with gzip.open(folder/'runtime_episodes.jsonl.gz','wt',encoding='utf-8') as runtime, \
         gzip.open(folder/'evaluator_episodes.jsonl.gz','wt',encoding='utf-8') as truth:
        for episode in episodes:
            runtime.write(json.dumps(dict(episode_id=episode.episode_id,header_hex=episode.header_hex,
                candidates=[candidate.__dict__ for candidate in episode.candidates]))+'\n')
            truth.write(json.dumps(dict(episode_id=episode.episode_id,units=episode.units,utilities=episode.utilities,
                                       evidence_scope='native_label_controlled_observation'))+'\n')
    with gzip.open(folder/'decisions.jsonl.gz','wt',encoding='utf-8') as trace:
        for name,model in models.items():
            results[name] = []
            timings = []
            for episode in episodes:
                started = time.perf_counter()
                row = score_episode(model,episode)
                timings.append(time.perf_counter()-started)
                row['method'] = name
                results[name].append(row)
                available = set(range(len(episode.candidates)))
                seen = set()
                for rank,index in enumerate(row['order'],1):
                    new = episode.units[index] not in seen
                    marginal = episode.utilities[index] if new else 0.
                    best = max((episode.utilities[i] if episode.units[i] not in seen else 0. for i in available),default=0.)
                    trace.write(json.dumps(dict(episode_id=episode.episode_id,method=name,rank=rank,
                        eligible=[episode.candidates[i].message_id for i in sorted(available)],
                        selected=episode.candidates[index].message_id,new_weight=marginal,
                        maximum_available_new_weight=best,repeated=not new))+'\n')
                    seen.add(episode.units[index]); available.remove(index)
            row = dict(task=episodes[0].task,method=name,episodes=len(episodes),
                       undefined_A10=sum(result['A10'] is None for result in results[name]),
                       scheduler_plus_metric_ms_median=1000*float(np.median(timings)))
            for metric in ('A10','coverage5','coverage10','ndcg10','high_recall10','redundancy10'):
                values = pool_means(results[name],metric)
                row[metric] = float(np.mean(list(values.values()))) if values else None
                if metric == 'A10':
                    row['pools'] = len(values)
            summary.append(row)
    write_json(folder/'episode_metrics.json',results)
    comparisons = {name:paired_bootstrap(pool_means(rows),pool_means(results[strongest])) for name,rows in results.items()}
    write_json(folder/'paired_vs_selected_baseline.json',dict(baseline=strongest,comparisons=comparisons))
    strata_comparisons = {}
    for stratum in STRATA:
        reference = pool_means([row for row in results[strongest] if row['stratum']==stratum])
        strata_comparisons[stratum] = paired_bootstrap(
            pool_means([row for row in results['neural_coverage'] if row['stratum']==stratum]),reference)
    comparisons['_strata'] = strata_comparisons
    comparisons['_baseline'] = strongest
    write_json(folder/'paired_by_stratum.json',dict(baseline=strongest,comparisons=strata_comparisons))
    with (folder/'comparison.csv').open('w',newline='',encoding='utf-8') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    # Pure scheduling latency excludes evaluator labels and coverage calculations.
    warm = episodes[0].candidates
    latency = {}
    for name,model in models.items():
        for _ in range(3): model.rank(warm)
        measured = []
        for _ in range(30):
            started = time.perf_counter(); model.rank(warm)
            measured.append(time.perf_counter()-started)
        latency[name] = dict(median_ms=1000*float(np.median(measured)),p95_ms=1000*float(np.quantile(measured,.95)),queue_size=32)
    write_json(folder/'scheduler_latency.json',latency)
    # Streaming is a different offered-workload experiment, using fixed source headers.
    stream_rows = []
    with gzip.open(folder/'streaming.jsonl.gz','wt',encoding='utf-8') as handle:
        for episode in episodes[::max(1,len(episodes)//10)]:
            for scenario,load,no_backlog in (('no_backlog',.5,True),('load_0.5',.5,False),('load_1.2',1.2,False)):
                for name,model in models.items():
                    events = replay_stream(model,episode,load=load,seed=SEED,no_backlog=no_backlog)
                    index = {candidate.message_id:i for i,candidate in enumerate(episode.candidates)}
                    order = [index[event['message_id']] for event in events]
                    if len(order) != len(set(order)) or len(order) != len(episode.candidates):
                        raise RuntimeError('stream replay duplicated or lost a message')
                    if no_backlog and order != list(range(len(order))):
                        raise RuntimeError('no-backlog control changed the arrival order')
                    waits = [event['wait'] for event in events]
                    stream_rows.append(dict(episode_id=episode.episode_id,method=name,scenario=scenario,
                        pool_id=episode.pool_id,stratum=episode.stratum,
                        A10=early_coverage(order,episode.units,episode.utilities),
                        mean_wait=float(np.mean(waits)),p95_wait=float(np.quantile(waits,.95)),
                        service_counts={source:sum(event['source_id']==source for event in events)
                                        for source in sorted({event['source_id'] for event in events})}))
                    handle.write(json.dumps(dict(episode_id=episode.episode_id,scenario=scenario,method=name,
                                                 events=events,unit='abstract_service_tick'))+'\n')
    write_json(folder/'streaming_metrics.json',dict(unit='abstract_service_tick',episodes=stream_rows))
    return results,summary,comparisons


def make_figures(task_results,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    tasks = list(task_results)
    figure,axes = plt.subplots(1,len(tasks),figsize=(6*len(tasks),4),squeeze=False)
    for axis,task in zip(axes[0],tasks):
        results,summary,comparisons = task_results[task]
        ordered = sorted(summary,key=lambda row:-(row['A10'] if row['A10'] is not None else -1))
        names = list(dict.fromkeys(['fifo','importance','neural_coverage']+[row['method'] for row in ordered[:3]]))
        for name in names:
            if name not in results: continue
            curves = [row['curve'] for row in results[name] if row['curve'] is not None]
            if curves:
                axis.plot(np.arange(1,11),np.mean(curves,axis=0),label=name)
        axis.set(title=task,xlabel='Messages serviced',ylabel='Weighted observation coverage',ylim=(0,1))
        axis.legend(fontsize=7)
    figure.tight_layout(); figure.savefig(output/'figure1_coverage.png',dpi=200); plt.close(figure)
    figure,axes = plt.subplots(1,len(tasks),figsize=(6*len(tasks),4),squeeze=False)
    for axis,task in zip(axes[0],tasks):
        comparisons = task_results[task][2]
        strata = comparisons['_strata']
        names = [name for name in STRATA if strata[name]['ci95'] is not None]
        for index,name in enumerate(names):
            row = strata[name]; low,high = row['ci95']
            axis.errorbar(row['mean_difference'],index,xerr=[[max(0,row['mean_difference']-low)],
                          [max(0,high-row['mean_difference'])]],fmt='o')
        axis.set_yticks(range(len(names)),names,fontsize=7)
        axis.axvline(0,color='black',linewidth=.8)
        axis.set(title=task,xlabel='CAHR minus '+comparisons['_baseline']+' (A10)')
    figure.tight_layout(); figure.savefig(output/'figure2_paired_comparison.png',dpi=200); plt.close(figure)


def run_study(cache_path,output_dir,config=None):
    config = dict(config or {})
    output = Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    # Explicit updates allow caller overrides without duplicate-key construction.
    defaults = dict(max_epochs=60,patience=10,train_queues=128,seed=0,eval_repeats=6,
                    rho_grid=[1e-5,1e-4,1e-3],tree_leaves=[7,15,31],tree_rounds=200,
                    extra_seeds=[1,2],ablations=True)
    defaults.update(config); resolved = defaults
    if any(not isinstance(resolved[name],int) or resolved[name] < 1
           for name in ('max_epochs','patience','train_queues','eval_repeats','tree_rounds')):
        raise ValueError('positive integer training/evaluation budgets required')
    if any(not math.isfinite(float(rho)) or rho < 0 for rho in resolved['rho_grid']):
        raise ValueError('finite nonnegative L2 grid required')
    write_json(output/'config.json',dict(protocol_id=PROTOCOL_ID,**resolved))
    write_json(output/'status.json',dict(status='running',stage='training'))
    records = load_cache(cache_path)
    audit = dict(cache_hashes={name:hashlib.sha256((Path(cache_path)/name).read_bytes()).hexdigest()
                              for name in ('features.jsonl','labels.jsonl')},
                 roles={role:sum(row['role']==role for row in records) for role in ('A','B','C','test')},
                 evidence_scope='native_labels_controlled_observation_identity',test_used_for_selection=False)
    audit['extraction_verification'] = verify_extraction_cache(cache_path,records)
    audit['queue_protocol'] = 'one shared fixed 128-query A-only bank per task for all methods; repeated each neural epoch'
    write_json(output/'data_audit.json',audit)
    task_results = {}
    all_summary = []
    building_context = None
    try:
        for profile in PROFILES.values():
            folder = output/profile.task; folder.mkdir(exist_ok=True)
            calibration = calibrate(records,profile)
            write_json(folder/'calibration.json',calibration.__dict__)
            train_pool = eligible_records(records,'A',profile)
            if len(train_pool) < 32:
                raise ValueError('insufficient distinct training observations')
            train = [make_episode(train_pool,profile,STRATA[index%5],SEED+index,calibration,f'A-{index}')
                     for index in range(resolved['train_queues'])]
            validation = evaluation_bank(records,'C',profile,calibration,resolved['eval_repeats'])
            medians = fit_medians(train)
            models,selection = {},{}
            for name,kind in (('fifo','fifo'),('round_robin','round_robin'),('random','random'),('importance','importance')):
                models[name] = SemanticScheduler(kind,profile.profile_id,medians,seed=0)
                selection[name] = dict(validation_A10=validation_score(models[name],validation),artifact=str(folder/'models'/name))
                save_policy(models[name],folder/'models'/name)
            heuristics = [(weight,SemanticScheduler('heuristic',profile.profile_id,medians,lambda_n=weight)) for weight in (0,.25,.5,1)]
            weight,model = max(heuristics,key=lambda item:(validation_score(item[1],validation),-item[0]))
            models['fixed_formula'] = model
            save_policy(model,folder/'models/fixed_formula')
            selection['fixed_formula'] = dict(validation_A10=validation_score(model,validation),lambda_n=weight,artifact=str(folder/'models/fixed_formula'))
            specifications = [('classifier','classifier','mlp'),('regressor','regressor','mlp'),
                ('linear_ranknet','ordinary','linear'),('neural_ranknet','ordinary','mlp'),
                ('linear_coverage','coverage','linear'),('neural_coverage','coverage','mlp')]
            best_rho = {}
            for name,objective,architecture in specifications:
                trials = []
                for rho in resolved['rho_grid']:
                    trial = folder/'trials'/name/f'rho_{rho}'
                    model,score = fit_network(train,validation,profile,trial,medians,objective,architecture,
                        rho=rho,seed=0,max_epochs=resolved['max_epochs'],patience=resolved['patience'])
                    trials.append((score,rho,model,trial/'best'))
                score,rho,model,artifact = max(trials,key=lambda row:(row[0],-row[1]))
                models[name] = model; best_rho[name] = rho
                selection[name] = dict(validation_A10=score,rho=rho,artifact=str(artifact))
            for name,coverage in (('lambdamart',False),('coverage_tree',True)):
                trials = []
                for leaves in resolved['tree_leaves']:
                    trial = folder/'trials'/name/f'leaves_{leaves}'
                    model,score = fit_tree(train,validation,profile,trial,medians,coverage,leaves,resolved['tree_rounds'])
                    trials.append((score,leaves,model,trial))
                score,leaves,model,artifact = max(trials,key=lambda row:(row[0],-row[1]))
                models[name] = model
                selection[name] = dict(validation_A10=score,num_leaves=leaves,artifact=str(artifact))
            strongest = max((name for name in models if name != 'neural_coverage'),key=lambda name:selection[name]['validation_A10'])
            selected = max(models,key=lambda name:selection[name]['validation_A10'])
            write_json(folder/'selection.json',dict(methods=selection,selected=selected,strongest_baseline=strongest))
            import shutil
            shutil.copytree(selection[selected]['artifact'],folder/'selected_scheduler',dirs_exist_ok=True)
            # All selection decisions are saved before the locked test bank is constructed.
            test = evaluation_bank(records,'test',profile,calibration,resolved['eval_repeats'])
            write_json(folder/'calibration_audit.json',calibration_audit(records,profile,calibration))
            result = benchmark_task(models,test,folder/'evaluation',strongest)
            task_results[profile.task] = result
            all_summary.extend(result[1])
            if profile.task == 'building_damage':
                building_context = (profile,folder,train,validation,test,medians,best_rho,selection,strongest,models)
        # Complete all three primary task comparisons before auxiliary experiments.
        if building_context is not None:
            profile,folder,train,validation,test,medians,best_rho,selection,strongest,models = building_context
            for seed in resolved['extra_seeds']:
                for name in dict.fromkeys(('neural_coverage',strongest)):
                    target = folder/'seeds'/name/str(seed)
                    spec = next((row for row in specifications if row[0]==name),None)
                    if spec is not None:
                        model,score = fit_network(train,validation,profile,target,medians,spec[1],spec[2],
                            rho=best_rho[name],seed=seed,max_epochs=resolved['max_epochs'],patience=resolved['patience'])
                    elif name in ('lambdamart','coverage_tree'):
                        model,score = fit_tree(train,validation,profile,target,medians,name=='coverage_tree',
                            selection[name]['num_leaves'],resolved['tree_rounds'],seed=seed)
                    else:
                        model = models[name]
                        save_policy(model,target)
                        write_json(target/'seed_qualification.json',dict(
                            status='no_trainable_initialization',seed=seed,method=name,
                            note='Same validation-selected deterministic policy; not an independent trained replicate.'))
                    write_json(target/'test.json',[score_episode(model,episode) for episode in test])
            if resolved['ablations']:
                for disabled in [(index,) for index in range(8)]+[(1,7)]:
                    name = 'without_'+'_'.join(FEATURE_NAMES[index] for index in disabled)
                    model,score = fit_network(train,validation,profile,folder/'ablations'/name,medians,
                        rho=best_rho['neural_coverage'],seed=0,max_epochs=resolved['max_epochs'],patience=resolved['patience'],disabled=disabled)
                    write_json(folder/'ablations'/name/'test.json',[score_episode(model,episode) for episode in test])
        import csv
        with (output/'comparison.csv').open('w',newline='',encoding='utf-8') as handle:
            writer = csv.DictWriter(handle,fieldnames=list(all_summary[0])); writer.writeheader(); writer.writerows(all_summary)
        make_figures(task_results,output)
        write_json(output/'status.json',dict(status='completed',tasks=list(task_results),protocol_id=PROTOCOL_ID,
                                            codec_evaluation=False,statistical_unit='disjoint_source_pool'))
        return all_summary
    except Exception as error:
        write_json(output/'status.json',dict(status='failed',error=repr(error),completed_tasks=list(task_results)))
        raise
