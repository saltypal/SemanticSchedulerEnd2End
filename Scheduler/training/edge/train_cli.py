"""CLI: ``python -m Scheduler.training.edge.train_cli --algo ppo|scope_ppo|bc|dagger|dqn|scope_dqn|ltr|scope_ltr|semcover|bcedge --table T --profile P --out O``.

Builds a train-split env factory from a merged table (``Evaluation.edge.tables``, imported lazily),
trains with ``trainers`` and saves weights plus a JSON history next to ``--out``.
"""
import argparse
import json


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["ppo", "scope_ppo", "bc", "dagger", "dqn", "scope_dqn", "ltr", "scope_ltr", "semcover", "bcedge"])
    ap.add_argument("--table", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", default="sv_dpp")
    ap.add_argument("--ranker", default="lambdamart", choices=["lambdamart", "ranknet"],
                    help="model for --algo ltr")
    ap.add_argument("--no-lightgbm", action="store_true", help="LambdaMART: use the sklearn fallback")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--n-jobs", type=int, default=200)
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import numpy as np
    from Evaluation.edge import tables
    from Evaluation.edge.contracts import ValueSpec
    from Evaluation.edge.edge_config import load_config
    from Evaluation.edge.env import EdgeEnv
    from Evaluation.edge.receiver import Receiver
    from Scheduler.edge import registry
    from Scheduler.training.edge import trainers
    from Scheduler.training.edge.policy_net import save_model

    rows = [r for r in tables.load_rows(a.table) if r.get("split", "train") == "train"]
    latency = tables.latency_from_profile_json(a.profile)
    spec = ValueSpec.from_config(load_config("sim"))

    def env_factory(seed):
        rng = np.random.default_rng(seed)
        pick = [rows[i] for i in rng.integers(len(rows), size=a.n_jobs)]
        times = np.cumsum(rng.exponential(1.0 / a.rate, size=a.n_jobs))
        jobs = tables.rows_to_jobs(pick, times, rng)
        return EdgeEnv(jobs, latency, spec, a.workers, a.max_batch, Receiver(), seed)

    teacher = registry.build(a.teacher) if a.algo not in ("ppo", "dqn", "bcedge") else None
    kw = dict(iters=a.iters, episodes=a.episodes, seed=a.seed)
    if a.algo == "ppo":
        model, hist = trainers.train_ppo(env_factory, **kw)
    elif a.algo == "scope_ppo":
        model, hist = trainers.train_scope_ppo(env_factory, teacher, **kw)
    elif a.algo == "bc":
        model, hist = trainers.train_bc(env_factory, teacher, episodes=a.episodes, seed=a.seed)
    elif a.algo == "dagger":
        model, hist = trainers.train_dagger(env_factory, teacher, **kw)
    elif a.algo == "dqn":
        model, hist = trainers.train_dqn_plain(env_factory, episodes=a.iters * a.episodes, seed=a.seed,
                                               hidden=a.hidden)
    elif a.algo == "scope_dqn":
        model, hist = trainers.train_scope_dqn(env_factory, teacher, hidden=a.hidden, **kw)
    elif a.algo == "ltr":
        model, hist = trainers.train_ltr(env_factory, teacher, a.ranker, episodes=a.episodes, seed=a.seed,
                                         **({"prefer_lightgbm": False} if a.no_lightgbm and a.ranker == "lambdamart" else {}))
    elif a.algo == "bcedge":
        from Scheduler.training.edge.bcedge_train import train_bcedge
        model, hist = train_bcedge(env_factory, hidden=a.hidden, **kw)
    elif a.algo == "semcover":
        model, hist = trainers.train_semcover(env_factory, teacher, hidden=a.hidden, **kw)
    else:
        model, hist = trainers.train_scope_ltr(env_factory, teacher, hidden=a.hidden, **kw)
    if a.algo in ("dqn", "scope_dqn"):
        import torch
        torch.save(dict(hidden=a.hidden, state=model.state_dict()), a.out)
    elif a.algo in ("ltr", "scope_ltr"):
        model.save(a.out)
    elif a.algo == "bcedge":
        from Scheduler.edge.bcedge import save_net
        save_net(model, a.out)
    elif a.algo == "semcover":
        from Scheduler.training.edge.semcover_train import save_net
        save_net(model, a.out)
    else:
        save_model(model, a.out)
    with open(a.out + ".history.json", "w") as f:
        json.dump(hist, f, indent=1)
    return hist


if __name__ == "__main__":
    main()
