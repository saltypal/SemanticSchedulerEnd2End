# Pre-SPI CPU pilot audit

- Checkout HEAD: `032c23a3dafc7bfeb39a84ba55193b6b6f3a22a0`; initial working tree was already dirty. No checkout reset/pull/push or production/test edits were made.
- Runtime: `.venv/Scripts/python.exe`, Python 3.12.14, torch 2.14.0+cpu, NumPy 2.5.3, LightGBM 4.6.0, CPU, CUDA unavailable, OMP/MKL threads=1.
- Focused regression: `Test/tests/test_stream_simulator.py Test/tests/test_stream_learning.py -q`: 15 passed in 6.71 s; process 8.28 s.
- Legacy fixture pilot: 3 episodes, 3 epochs, 64 RL steps: exit 0 in 37.84 s, no learner failures; 42 comparison records (14 policies x 3 held-out synthetic episodes). All accounting rows conserve offered items as succeeded+failed+in-flight+queued+not-arrived; J_time values finite (nullable critical metrics remain null when no critical reports are offered).
- Eight non-teacher learners are `fixture_trained`; each has a non-empty `resume.pt` and `portable.json` (16 files verified). Three teacher-dependent arms (`rl_vlm_f`, `vlm_replay`, `visionprefer`) are explicitly blocked because no qualified real VLM labels exist.
- The separate arithmetic fixture passed all 11 checks: mid-service release/unserved=13, future release excluded=0, immediate=0, horizon completion=8, coincident events=6, empty=0; unknown weight/pre-release service/NaN rejected; repeated SPI class gets first priority 2 versus new class 1 with shared class wait 1; greedy 6 versus alternative 5. This is specification arithmetic only, not runtime integration or semantic validation.
- Scheduler and Test Python source hashes: 67 files hashed before and after; 0 changed. The pilot also persisted its own source snapshot/hash manifest. Experimental files added concurrently under `experiments/` were not part of that source-hash set.
- Scope limit: the pilot trains the LEGACY completion-coverage objective. It does not validate the newly specified dispatch waiting/evidence objectives or Deep Sets. No claim of synthetic semantic superiority or convergence. No model, codec, VLM, GPU, Kaggle, SPI training process, or production/test file was touched by this run.

See sibling `.status.json`, `.stdout.log`, `.stderr.log`, `environment.txt`, `source-hashes-{before,after}.json`, `source-hash-diff.json`, `pilot/`, and `waiting-spec/` for detailed evidence.
