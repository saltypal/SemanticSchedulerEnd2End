# Version 9 platform timeout recovery

Kaggle user-visible result: canceled after 43213.6 seconds, timeout exceeded, exit 137, two T4 GPUs, 951.33 MB published output.
API confirmed CANCEL_ACKNOWLEDGED.

Recovered final master log: seed 2718 completed PPO and Double DQN (20,000 decisions each), then TD-PER reached 4,000 decisions. This is newer than the stale live-stream snapshot.
The complete master study did not finish.

Recovery in progress:
- Existing monitor retrieves JSON outputs.
- checkpoint-retrieval.log records PT/NPZ/NPY/JSONL/CSV/MD/LOG retrieval.
- status-retrieval.log records targeted stage/status/config recovery.
- All retrieved files go under this run's retrieved/ hierarchy.
- Checkpoint presence alone does not establish successful resume; load/compatibility checks remain required.
- Audit saved statuses and checkpoint state before preparing continuation.
