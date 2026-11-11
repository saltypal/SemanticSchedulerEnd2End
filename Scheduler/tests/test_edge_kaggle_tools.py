import json

from scripts.edge import kaggle_ops
from scripts.edge.build_notebook import build_metadata, build_notebook, write_outputs

CFG = {"accounts": {"win": {"username": "u1", "config_dir": "C:/k1", "runner": "native"},
                    "wsl": {"username": "u2", "config_dir": "/k2", "runner": "wsl"}},
       "slug_prefix": "edge-", "repo_url": "https://example.com/r.git"}


def _src(cell):
    return "".join(cell["source"])


def test_cell_order_and_content():
    nb = build_notebook("s1", "pkg.dummy", ["--a", "1"], "T", CFG["repo_url"], requirements=("foo",))
    assert nb["nbformat"] == 4
    cells = nb["cells"]
    assert cells[0]["cell_type"] == "markdown"
    assert "TQDM_DISABLE" in _src(cells[1])
    assert "--depth" in _src(cells[2]) and "rev-parse" in _src(cells[2])
    assert "--ignore-requires-python" in _src(cells[3]) and "'foo'" in _src(cells[3])
    assert "pkg.dummy" in _src(cells[4]) and "'-u'" in _src(cells[4])
    assert "run_manifest.json" in _src(cells[5])


def test_metadata_slug(tmp_path):
    meta = build_metadata(CFG, "win", "s1", "T")
    assert meta["id"] == "u1/edge-s1" and meta["kernel_type"] == "notebook"
    assert meta["dataset_sources"] == [] and meta["is_private"] is True
    nb_path, meta_path = write_outputs(build_notebook("s1", "m", [], "T", "u"), meta, "s1", tmp_path)
    assert json.loads(meta_path.read_text())["code_file"] == "s1.ipynb" and nb_path.exists()


def test_dry_run_push_executes_nothing(tmp_path):
    calls, lines = [], []
    for acct in ("win", "wsl"):
        rc = kaggle_ops.main(["--dry-run", "push", "--account", acct, "--step", "s1"],
                             runner=lambda c, e=None: calls.append(c) or (0, ""), cfg=CFG,
                             log_path=tmp_path / "log.md", out=lines.append)
        assert rc == 0
    assert calls == [] and not (tmp_path / "log.md").exists()
    cmds = [x for x in lines if x.startswith("$")]
    assert len(cmds) == 2 and all("kaggle kernels push" in c for c in cmds)
    assert cmds[1].startswith("$ wsl kaggle")


def test_push_appends_log(tmp_path):
    nb_dir = tmp_path / "nb"
    write_outputs(build_notebook("s1", "m", [], "T", "u"), build_metadata(CFG, "win", "s1", "T"), "s1", nb_dir)
    calls = []

    def fake(cmd, env=None):
        calls.append((cmd, env))
        return 0, "abc123\n" if cmd[0] == "git" else "Kernel version 3 successfully pushed"

    ops = kaggle_ops.Ops(CFG, "win", fake, log_path=tmp_path / "log.md", out=lambda *_: None)
    assert ops.push("s1", nb_dir) == 0
    push_cmd, env = calls[0]
    assert push_cmd[:3] == ["kaggle", "kernels", "push"] and env == {"KAGGLE_CONFIG_DIR": "C:/k1"}
    text = (tmp_path / "log.md").read_text().splitlines()
    assert text[0].startswith("# Edge") and "| s1 | win | u1/edge-s1 | 3 | abc123 |" in text[-1]
    staged = json.loads((nb_dir / "_stage/win/s1/kernel-metadata.json").read_text())
    assert staged["id"] == "u1/edge-s1"


def test_wait_polls_until_complete():
    seq = iter(["running", "running", "KernelWorkerStatus.COMPLETE"])
    ops = kaggle_ops.Ops(CFG, "wsl", lambda c, e=None: (0, next(seq)), out=lambda *_: None)
    sleeps = []
    assert ops.wait("s1", timeout=100, poll=5, sleep=sleeps.append) == 0
    assert sleeps == [5, 5]


WSL_BIN = "/home/Satya/.venvs/semantic-kaggle/bin/kaggle"
CFG2 = {"default_account": "wsl", "slug_prefix": "edge-", "repo_url": "u",
        "accounts": {"win": {"username": "w", "runner": "native", "bin": "kaggle"},
                     "wsl": {"username": "z", "runner": "wsl", "bin": WSL_BIN}}}


def _run(argv, tmp_path, cfg=CFG2):
    calls, lines = [], []
    rc = kaggle_ops.main(argv, runner=lambda c, e=None: calls.append(c) or (0, "sha\n" if c[0] == "git" else "ok"),
                         cfg=cfg, log_path=tmp_path / "log.md", out=lines.append, stage_root=tmp_path / "nb")
    return rc, calls, lines


def test_to_wsl_path():
    assert kaggle_ops.to_wsl_path(r"D:\Bunker\x y\f") == "/mnt/d/Bunker/x y/f"
    assert kaggle_ops.to_wsl_path("C:/a/b") == "/mnt/c/a/b"
    assert kaggle_ops.to_wsl_path("/already/linux") == "/already/linux"
    assert kaggle_ops.to_wsl_path("owner/slug") == "owner/slug"


def test_wsl_prefix_bin_and_path_translation():
    ops = kaggle_ops.Ops(CFG2, "wsl", out=lambda *_: None)
    cmd = ops._cmd(["kernels", "push", "-p", r"D:\a\b", "--path", "E:/c"])
    assert cmd == ["wsl", WSL_BIN, "kernels", "push", "-p", "/mnt/d/a/b", "--path", "/mnt/e/c"]
    assert kaggle_ops.Ops(CFG2, "win", out=lambda *_: None)._cmd(["x", "-p", r"D:\a"]) == ["kaggle", "x", "-p", r"D:\a"]


def test_default_account_is_wsl_from_real_config(tmp_path):
    from scripts.edge.build_notebook import load_kaggle_config
    cfg = load_kaggle_config()
    assert cfg["default_account"] == "wsl" and cfg["accounts"]["win"]["username"] == "satyapaladugu"
    assert cfg["accounts"]["wsl"]["username"] == "satyapaladuguplus" and cfg["accounts"]["win"]["bin"] == "kaggle"
    rc, calls, lines = _run(["--dry-run", "dataset-list", "--mine"], tmp_path)
    assert rc == 0 and calls == [] and lines[0] == f"$ wsl {WSL_BIN} datasets list --mine"


def test_dataset_create_stages_metadata_and_logs(tmp_path):
    src = tmp_path / "My Data"
    src.mkdir()
    (src / "a.txt").write_text("x")
    rc, calls, _ = _run(["dataset-create", "--dir", str(src), "--title", "T", "--public", "--dir-mode", "tar"], tmp_path)
    stage = tmp_path / "nb/_stage/wsl/datasets/my-data"
    meta = json.loads((stage / "dataset-metadata.json").read_text())
    assert meta == {"id": "z/my-data", "title": "T", "licenses": [{"name": "other"}]}
    dcall = calls[0]
    assert (stage / "a.txt").exists() and dcall[:4] == ["wsl", WSL_BIN, "datasets", "create"]
    assert "--public" in dcall and dcall[-2:] == ["--dir-mode", "tar"]
    assert "| dataset-create | wsl | z/my-data |" in (tmp_path / "log.md").read_text()


def test_dataset_metadata_kept_if_present(tmp_path):
    src = tmp_path / "d"
    src.mkdir()
    (src / "dataset-metadata.json").write_text('{"id": "keep/me"}')
    _run(["dataset-version", "--dir", str(src), "--message", "m1", "--account", "win"], tmp_path)
    stage = tmp_path / "nb/_stage/win/datasets/d"
    assert json.loads((stage / "dataset-metadata.json").read_text()) == {"id": "keep/me"}


def test_dataset_version_status_and_dry_run(tmp_path):
    src = tmp_path / "d"
    src.mkdir()
    rc, calls, _ = _run(["dataset-version", "--dir", str(src), "--message", "hi"], tmp_path)
    assert calls[0][3] == "version" and calls[0][-2:] == ["-m", "hi"]
    rc, calls, _ = _run(["dataset-status", "--slug", "foo"], tmp_path)
    assert calls[0][-3:] == ["datasets", "status", "z/foo"] or calls[0][-1] == "z/foo"
    before = len(list(tmp_path.rglob("*")))
    for argv in (["--dry-run", "dataset-create", "--dir", str(src)],
                 ["--dry-run", "dataset-version", "--dir", str(src), "--message", "m"],
                 ["--dry-run", "dataset-status", "--slug", "foo"], ["--dry-run", "dataset-list", "--mine"]):
        rc, calls, lines = _run(argv, tmp_path)
        assert rc == 0 and calls == [] and lines[0].startswith("$ wsl ")
    assert len(list(tmp_path.rglob("*"))) == before


def test_metadata_sources_and_cli(tmp_path):
    meta = build_metadata(CFG, "win", "s1", "T", dataset_sources=["a/b"], kernel_sources=["c/d"])
    assert meta["dataset_sources"] == ["a/b"] and meta["kernel_sources"] == ["c/d"]
    assert build_metadata(CFG, "win", "s1", "T")["kernel_sources"] == []
    from scripts.edge import build_notebook as bn
    bn.main(["--step", "s9", "--entrypoint", "m", "--title", "T", "--out-dir", str(tmp_path),
             "--dataset-source", "a/b", "--dataset-source", "a/c", "--kernel-source", "k/l"])
    m = json.loads((tmp_path / "s9.kernel-metadata.json").read_text())
    assert m["dataset_sources"] == ["a/b", "a/c"] and m["kernel_sources"] == ["k/l"]
    assert m["id"].startswith("satyapaladuguplus/")


def _dataset_nbs():
    from scripts.edge import make_dataset_notebooks as m
    return {n: m.build(n) for n in m.SPECS}


def test_generated_notebook_hygiene():
    for name, (nb, meta) in _dataset_nbs().items():
        cells = nb["cells"]
        ids = [c["id"] for c in cells]
        assert len(set(ids)) == len(ids) and all(len(i) >= 8 and set(i) <= set("0123456789abcdef") for i in ids)
        assert nb["nbformat"] == 4 and nb["nbformat_minor"] >= 5
        code = [c for c in cells if c["cell_type"] == "code"]
        first = _src(code[0])
        for var in ("PYTHONWARNINGS", "PYDEVD_DISABLE_FILE_VALIDATION", "TOKENIZERS_PARALLELISM",
                    "HF_HUB_DISABLE_PROGRESS_BARS", "TQDM_DISABLE"):
            assert var in first
        assert "warnings.filterwarnings('ignore')" in first
        clone, install, run = _src(code[1]), _src(code[2]), _src(code[3])
        assert "--ignore-requires-python" in install and '"-q"' in install and "pip_install.log" in install
        assert "installed in" in install and "[-25:]" in install
        assert '"-q"' in clone and "--no-progress" in clone and "FETCH_HEAD" in clone
        assert "'-u', '-m'" in run and "cwd=REPO" in run


def test_titles_roundtrip_and_sources():
    import re
    from scripts.edge.build_notebook import title_for_slug
    assert title_for_slug("edge-a-download-images") == "Edge A Download Images"
    assert title_for_slug("edge-b-labeling") == "Edge B Labeling"
    nbs = _dataset_nbs()
    for name, (_, meta) in nbs.items():
        assert re.sub(r"[^a-z0-9]+", "-", meta["title"].lower()).strip("-") == name
    a, b = nbs["edge-a-download-images"][1], nbs["edge-b-labeling"][1]
    assert a["kernel_sources"] == [] and a["enable_gpu"] is False
    assert a["dataset_sources"] == ["seaninggg/crisismmd-multimodal-crisis-dataset", "benjaminkz/places365",
                                    "awsaf49/coco-2017-dataset"]   # CrisisMMD + Places365 + COCO inputs
    assert a["enable_internet"] is True
    assert b["kernel_sources"] == ["satyapaladuguplus/edge-a-download-images"] and b["dataset_sources"] == []


def test_kernel_slug_is_not_double_prefixed():
    from scripts.edge import build_notebook as bn
    cfg = {"slug_prefix": "edge-"}
    assert bn.kernel_slug(cfg, "edge-pilot") == "edge-pilot"
    assert bn.kernel_slug(cfg, "s1-dataset") == "edge-s1-dataset"
