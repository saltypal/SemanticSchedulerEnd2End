"""Show the original eight supported algorithms on the exact same causal headers."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.demo_decoder_scheduler import CACHE, STUDY, cached_evidence, run_demo
from Archived.owners_before_simplification_20261027.SemParam.decoder_parameters import ParameterHeader, PROFILES, TemperatureCalibration
from Scheduler.decoder_scheduler import SchedulerCandidate, load_scheduler

ORIGINAL_METHODS = ('fifo', 'round_robin', 'fixed_formula', 'classifier', 'regressor',
                    'linear_ranknet', 'neural_ranknet', 'lambdamart')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=[p.task for p in PROFILES.values()], default='building_damage')
    args = parser.parse_args()
    task = STUDY/args.task
    selection = json.loads((task/'selection.json').read_text())
    calibration = TemperatureCalibration(**json.loads((task/'calibration.json').read_text()))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT/'results/disaster_decoder_v3/original_comparison'/stamp
    ranknet = STUDY/selection['methods']['neural_ranknet']['artifact'].split('/study/', 1)[1]
    result = run_demo(cached_evidence(CACHE, 3), args.task, ranknet, calibration,
                      output/'headers_and_ranknet.json', 'same real-image cache replay for all methods')
    candidates = []
    for index, row in enumerate(result['headers']):
        header = ParameterHeader.from_bytes(bytes.fromhex(row['header_hex']))
        candidates.append(SchedulerCandidate.from_header(row['message_id'], 'ue-demo', 0., index, header))
    print('\nORIGINAL ALGORITHM COMPARISON: same headers, no re-fitting')
    rankings = {}
    for name in ORIGINAL_METHODS:
        remote = selection['methods'][name]['artifact']
        artifact = STUDY/remote.split('/study/', 1)[1]
        scheduler = load_scheduler(artifact, PROFILES)
        decisions = scheduler.rank(candidates)
        rankings[name] = [dict(message_id=d.message_id, score=d.score) for d in decisions]
        print(f'{name:18} ' + ' -> '.join(f'M{json.loads(d.message_id)[2]}' for d in decisions))
    print('Round-robin matches FIFO here because this tiny walkthrough uses one UE.')
    print('SJF: not evaluated without measured decode costs; equal-cost ordering ties.')
    print('EDF: not evaluated without defensible application deadlines.')
    print('Listwise and contextual bandit remain deferred, as in the original table.')
    print('An order alone is not a performance metric. Full held-out comparisons are saved separately.')
    (output/'comparison.json').write_text(json.dumps(dict(task=args.task,
        matched_information=True, rankings=rankings, training_performed=False), indent=2))
    print(f'SAVED: {output.resolve()}')


if __name__ == '__main__':
    main()
