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
    assert "git clone --depth 1" in _src(cells[1]) and "rev-parse HEAD" in _src(cells[1])
    assert "pip install -q -e . foo" in _src(cells[2])
    assert "python -u -m pkg.dummy --a 1" in _src(cells[4])
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
