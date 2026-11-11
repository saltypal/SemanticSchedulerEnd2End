# Mistakes log: every error, its cause and its fix

Add a row the moment something goes wrong, then update it when fixed. Status: OPEN / FIXED / WATCH.
Newest first. Keep rows short; link the commit or file where useful.

| # | Date | What went wrong | Root cause | Fix | Status |
|---|---|---|---|---|---|
| 36 | 2026-11-11 | Random S2c pair sampling gave only 3 real copies and 0 same-tweet pairs out of 353 | uniform random pairs almost never land on near-duplicates | mined candidates: pHash/SimHash nearest neighbours + same-tweet + random (`dup_pairs.mined_candidates`): 116 copies, 54 same-tweet | FIXED |
| 35 | 2026-11-11 | `obtain_source` would crash right after a successful GitHub clone on Kaggle | `fetch_pinned` returns the folder; caller used it as the commit string (`str + Path`) | banner uses the verified pin; test passes | FIXED |
| 34 | 2026-11-11 | Edge test run took 9 min and one test failed | `source_root` check now falls back to a real git clone when nothing is mounted; the unit test hit the network | test injects a fake clone; asserts the real requirement (source zip must be mounted) | FIXED |
| 33 | 2026-11-11 | pytest exit 4 from PowerShell | PowerShell does not expand `test_edge_*.py` for native commands | pass the expanded file list (`Get-ChildItem ... .FullName`) | FIXED |
| 32 | 2026-11-11 | `edge-publish-runs` said `KAGGLE_API_TOKEN is not set` with the secret attached; token then pasted into a cell | Kaggle Secrets are not env vars (need `UserSecretsClient`), and CLI-pushed runs get no secrets | read via `UserSecretsClient`; run from the web editor; token to be revoked | FIXED (revoke: user) |
| 31 | 2026-11-11 | Publishing A/B/C from notebook output: web "New dataset from output" hid `run/` and 502'd; first helper was slow | thousands of nested shard files; helper copied 2.5 GB then zipped again | `kernel_sources` mount + one uncompressed zip per folder, small first: B 8 s, C 18 s, A 15 min | FIXED |
| 30 | 2026-11-11 | Pilot 2 (single notebook): C_grid crashed `needs samples of at least 2 classes (only class 3)` | pilot sample (first 20 CrisisMMD rows) landed only in one split/tier, so training had only Low images | stratified pilot sampling; grid reports `plumbing only` instead of crashing in pilot | FIXING |
| 29 | 2026-11-11 | Feature pass reported PASS while the whole SQ-GAN stage was skipped | SQ-GAN failure was isolated (by design) but never made the stage fail | `require_sqgan` in full mode; separate SQ-GAN notebook | FIXING |
| 28 | 2026-11-11 | 26 pilot items failed: `no SQ-GAN source root ... src/__init__.py + src/sqgan.py + src/bitstream.py` | the Kaggle source dataset only holds `.upstream/sqgan`; SemanticLossless `src/` was never in it | use the pinned GitHub clone of SemanticLossless (the pinned-sha fetch already worked) + `.upstream/sqgan` from the dataset | FIXING |
| 27 | 2026-11-11 | `sqgan-pinned-source.zip ... found 0`; compute profile looked in `/kaggle/input/<slug>` | Kaggle now mounts datasets at `/kaggle/input/datasets/<owner>/<slug>/`; dataset holds an extracted folder, not a zip | mount discovery everywhere; accept extracted `.upstream/sqgan` | FIXING |
| 26 | 2026-11-11 | Pushing notebook A failed with `409 Conflict` and the title warning; pilot staged as `edge-edge-pilot` | `kernel_slug` added the `edge-` prefix to step names that already start with `edge-`, so the id (edge-edge-...) never matched the title-derived slug | step ids that already carry the prefix are not prefixed again; test added; stale staging folder cleared | FIXED |
| 25 | 2026-11-11 | Back-and-forth on COCO people photos in dataset A | scope changed twice mid-task (added, asked to remove, then asked to keep) | final decision: keep COCO people (Kaggle) + Places365 (Kaggle, all Low) + CrisisMMD; Open Images dropped | FIXED (decision) |
| 24 | 2026-11-11 | Kaggle notebook A stopped at `pip install -e .`: "requires a different Python: 3.13.18 not in '<3.13,>=3.12'" | `pyproject.toml` pins `requires-python >=3.12,<3.13`; Kaggle runs 3.13 | install cell adds `--ignore-requires-python` (no change to repo metadata) | FIXING |
| 23 | 2026-11-11 | Notebook log unreadable: ~18,000 `git clone` "Updating files" lines, pip spinner codes, debugger and nbformat warnings | clone/pip not run quiet; no env flags | quiet clone (`-q`), pip `-q` with output to a file, `PYDEVD_DISABLE_FILE_VALIDATION=1`, `PYTHONWARNINGS=ignore`; shared log helper planned | FIXING |
| 22 | 2026-11-11 | Pilot: SQ-GAN datasets "not mounted" (all SQ-GAN items skipped) | unknown: metadata listed all three datasets and all were ready; checks searched `/kaggle/input` but found nothing. Mount path layout not recorded | pilot now records the `/kaggle/input` tree and searches 4 levels deep and both layouts; re-run pilot | OPEN |
| 21 | 2026-11-11 | Pilot: CLIP and SigLIP2 embedders crashed: `'BaseModelOutputWithPooling' has no attribute 'reshape'` | transformers 5 returns a ModelOutput from `get_image_features` | version-robust embedding helper + dimension assertion | FIXING |
| 20 | 2026-11-11 | Pilot feature pass wrote 0 of 20 images; hashes and shard files missing | one failing stage failed the whole image | each stage isolated; image fails only if unreadable | FIXING |
| 19 | 2026-11-11 | Bare `StopIteration` errors (`next(glob(...))`) hid the real problem | no clear error when a file pattern matched nothing | raise `FileNotFoundError` naming the folder, pattern and files found | FIXING |
| 18 | 2026-11-11 | `lpips` missing on Kaggle | not in the notebook installs | added to pilot and feature-pass notebooks | FIXING |
| 17 | 2026-11-11 | Kaggle gave the pilot slug `edge-pilot-all-stages-...` instead of `edge-pilot` | Kaggle builds the URL slug from the notebook TITLE | title generated from the slug (`Edge Pilot`) | FIXING |
| 16 | 2026-11-10 | Started a dataset dry run that downloaded a 525 MB archive onto the laptop (no free disk) | the real-download path ran locally | stopped, deleted everything; rule: nothing is downloaded locally, all data work runs on Kaggle | FIXED |
| 15 | 2026-11-10 | Uploaded labels only, but the dataset needed images | misread the request | three datasets A (images), B (labels), C (cache); `edge-labels-v1` is superseded | FIXED |
| 14 | 2026-11-10 | Logging-cleanup agents stopped mid-edit | stopped for usage reasons | partial edits reverted (5 files + helper); work restarts in smaller pieces | FIXED |
| 13 | 2026-11-10 | Branch deletions blocked by the harness | destructive git action needs approval | commands listed for the user; nothing deleted | OPEN (needs approval) |
| 12 | 2026-11-10 | Validation event Sri Lanka had only 71 Critical (gate 100) | tiny event | validation moved to California wildfires; Sri Lanka joins training | FIXED |
| 11 | 2026-11-10 | `mission_served` leaked true outcomes (fidelity, true tier) to policies | built from realised values | built from header-only expected tier weight | FIXED |
| 10 | 2026-11-10 | Unlisted event silently hashed into a split | fallback hashed any unknown event | raises `ValueError`; event-less images still hash | FIXED |
| 9 | 2026-11-10 | `whittle_abandon` crashed with "invalid batch size 3" | batch not clipped to a profiled size | uses the shared batch-size helper | FIXED |
| 8 | 2026-11-10 | MILP "upper bound" was not an upper bound | service time rounded up | rounds down | FIXED |
| 7 | 2026-11-10 | SV-DPP never batched under light load | all scores about 0, so B=1 always won the ratio tie | tie broken by expected-value rate | FIXED |
| 6 | 2026-11-10 | DPP policies crashed in the simulator (`KeyError: 'ids'`) | simulator sent per-job events, policies expected batch lists | simulator now sends `ids`/`tier_probs`/`delays` lists | FIXED |
| 5 | 2026-11-10 | First pilot simulation was uninformative: every policy identical, "on-time" column 0 | load far below capacity; wrong metric field used | rerun at true overload with correct metrics (pending real data) | OPEN |
| 4 | 2026-11-10 | 3 old tests failed only on Windows (line endings) | git converts to CRLF on checkout; tests compare exact bytes | `.gitattributes` forces LF; files rewritten once | FIXED (2 remain) |
| 3 | 2026-11-10 | 2 old tests still fail | preserved notebooks embed CRLF copies of sources | left untouched by design (preserve notebooks) | WATCH |
| 2 | 2026-11-10 | Two old tests could not import `Archived...SemParam` | modules deleted from git, never archived | restored from history into `Archived/` | FIXED |
| 1 | 2026-11-10 | Old benchmark was circular (VLM tier defined both labels and cost) | design | whole redesign: human labels, decoder-queue simulator, CTR@D | FIXED |

## Lessons (apply every time)
- Run a pilot before every full run; read the whole report before launching anything big.
- Never download data to the laptop; use Kaggle.
- Check quoting and path rules for each shell (Git Bash rewrites `/home/...`; use PowerShell for `wsl`).
- After editing a config that tracked files also contain, re-check it survived any bulk checkout.
- Keep notebook logs short and readable; the user reads them on the Kaggle website.
| Kaggle kernel title did not slugify to the id (Edge Qwen feedback (S9 ...)), so the first push created a second kernel edge-qwen-feedback-s9-data-s3-kappa-labels | Keep the title equal to the slug words (Edge Qwen Feedback); read the push warning about the title before polling |
