"""Turn a simulator ``State`` into fixed-width numpy features for every learner.

One shared encoder so PPO, DQN, ranking and RLAIF models see identical inputs. Only header-visible
fields are used (never Truth). Per job: 4 tier probs, relevance, dup_prob, u_hat, per-tier slack ratio
(4), age relative to the predicted hard deadline, log age, 13 STS values, payload size.
"""
import numpy as np

JOB_DIM = 4 + 1 + 1 + 1 + 4 + 1 + 1 + 13 + 1
GLOBAL_DIM = 4


def job_features(state):
    """(N, JOB_DIM) matrix in ``state.queue`` order."""
    spec, rows = state.spec, []
    for job in state.queue:
        p = np.asarray(job.header.tier_probs, dtype=float)
        pred = int(p.argmax())
        slack = [max(0.0, (spec.hard_deadline(k) - job.age) / spec.hard_deadline(k)) for k in range(4)]
        sts = list(job.header.sts) + [0.0] * (13 - len(job.header.sts))
        rows.append(np.concatenate([
            p, [job.header.relevance, job.dup_prob, job.u_hat], slack,
            [min(job.age / spec.hard_deadline(pred), 2.0), np.log1p(job.age)], sts[:13],
            [job.header.payload_bytes / 1000.0]]))
    return np.asarray(rows, dtype=float).reshape(len(rows), JOB_DIM)


def global_features(state):
    return np.array([min(len(state.queue) / 50.0, 4.0), state.free_workers / max(state.workers, 1),
                     state.arrival_rate, state.workers / 8.0], dtype=float)
