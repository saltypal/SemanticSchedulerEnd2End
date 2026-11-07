"""Independent numerical fixtures for the planned objective, not runtime code.

Run with --output <unique-run-directory>. No image/SPI/teacher model is loaded.
This does not establish that scene_environment implements these definitions.
"""
import argparse
import json
import math
from pathlib import Path


def checked_units(units, horizon):
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError('invalid horizon')
    result = []
    for release, first_service, weight in units:
        if not math.isfinite(release) or release < 0:
            raise ValueError('invalid release')
        if math.isnan(first_service) or first_service < release:
            raise ValueError('service precedes release')
        if weight is None or not math.isfinite(weight) or weight < 0:
            raise ValueError('unknown or invalid reference weight')
        if release <= horizon:
            result.append((release, first_service, weight))
    return result


def direct_cost(units, horizon):
    return sum(w * (min(s, horizon) - r)
               for r, s, w in checked_units(units, horizon))


def area_cost(units, horizon):
    units = checked_units(units, horizon)
    events = sorted({0., horizon} | {r for r, _, _ in units}
                    | {s for _, s, _ in units if s <= horizon})
    return sum((right - left) * sum(w for r, s, w in units
                                   if r <= (left + right) / 2 < s)
               for left, right in zip(events, events[1:]))


def fact_cost(order, facts):
    """All-ready, unit-service, unit-weight, first-dispatch reference."""
    first = {}
    for dispatch, image in enumerate(order):
        for fact in facts[image]:
            first.setdefault(fact, dispatch)
    return sum(first.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = {
        'mid_service_arrival_and_unserved': ([(0., 1., 4.), (.5, math.inf, 2.)], 5., 13.),
        'future_release_excluded': ([(7., math.inf, 100.)], 5., 0.),
        'immediate_service': ([(0., 0., 8.)], 5., 0.),
        'service_at_horizon': ([(1., 5., 2.)], 5., 8.),
        'coincident_events': ([(0., 1., 2.), (1., 2., 4.)], 3., 6.),
        'empty': ([], 5., 0.),
    }
    checks = {}
    for name, (units, horizon, expected) in cases.items():
        direct, area = direct_cost(units, horizon), area_cost(units, horizon)
        assert math.isclose(direct, expected) and math.isclose(area, expected), name
        checks[name] = {'direct': direct, 'area': area, 'expected': expected}
    for name, units in {
        'unknown_weight_rejected': [(0., 1., None)],
        'pre_release_service_rejected': [(1., 0., 2.)],
        'nan_service_rejected': [(0., math.nan, 2.)],
    }.items():
        try:
            direct_cost(units, 5.)
        except ValueError:
            checks[name] = {'rejected': True}
        else:
            raise AssertionError(name)
    facts = {'A': {'x'}, 'B': {'x'}, 'C': {'y'}}
    repeat_first = fact_cost('ABC', facts)
    new_first = fact_cost('ACB', facts)
    assert (repeat_first, new_first) == (2, 1)
    checks['same_spi_distinct_evidence'] = {'repeat_first': repeat_first, 'new_first': new_first,
                                          'class_mean_wait_both': 1.}
    overlap = {'A': set(range(1, 7)), 'B': {1, 2, 3, 7, 8}, 'C': {4, 5, 6, 9, 10}}
    assert fact_cost('ABC', overlap) == 6 and fact_cost('BCA', overlap) == 5
    checks['greedy_not_universal_optimum'] = {'greedy': 6, 'alternative': 5}
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'passed', 'checks': checks,
              'scope': 'Independent specification arithmetic only; not production integration or semantic validation'}
    (args.output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
