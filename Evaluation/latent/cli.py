"""Run a UE-independent latent batch, or adapt cached prototype artifacts."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Evaluation.latent.experiment import compare_batch, convert_cached_run
from Scheduler.latent.policies import POLICIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--cached-run", type=Path)
    parser.add_argument("--payload-costs", type=Path)
    parser.add_argument("--service-rate", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=POLICIES)
    options = parser.parse_args()
    if bool(options.batch) == bool(options.cached_run):
        parser.error("Provide exactly one of --batch or --cached-run")
    batch = options.batch
    if options.cached_run:
        import uuid
        batch = options.output / "Scheduler" / "inputs" / f"cached-batch-{uuid.uuid4().hex[:8]}.json"
        convert_cached_run(options.cached_run, batch, payload_costs=options.payload_costs,
                           service_rate=options.service_rate, limit=options.limit, seed=options.seed)
    print(compare_batch(batch, options.output, policy_names=options.policies))


if __name__ == "__main__":
    main()
