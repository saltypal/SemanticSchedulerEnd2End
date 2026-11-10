"""kaggle_publish with a fake runner and fake secrets client."""
import json
import subprocess

from parameters.edge import kaggle_publish as kp

TOKEN = "tok-SECRET-123"


class Secrets:
    def __init__(self, **values):
        self.values = values

    def get_secret(self, label):
        if label not in self.values:
            raise RuntimeError("no secret")
        return self.values[label]


class Runner:
    def __init__(self, exists, fail=False):
        self.calls, self.envs, self.exists, self.fail = [], [], exists, fail

    def __call__(self, cmd, **kw):
        assert isinstance(cmd, list) and not kw.get("shell")
        self.calls.append(cmd)
        self.envs.append(kw.get("env"))
        if cmd[2] == "status":
            return subprocess.CompletedProcess(cmd, 0 if self.exists else 1,
                                               "ready" if self.exists else "404 Not Found", "")
        if self.fail:
            return subprocess.CompletedProcess(cmd, 1, "", "boom " + TOKEN)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")


def test_no_secrets_skipped(tmp_path):
    runner = Runner(False)
    res = kp.publish_dataset(tmp_path, "s", "T", runner=runner, secrets_client=Secrets())
    assert res["status"] == "skipped" and "Add-ons > Secrets" in res["reason"]
    assert "New dataset from output" in res["reason"]
    assert runner.calls == []


def test_create_when_absent_and_metadata_written(tmp_path):
    runner = Runner(False)
    res = kp.publish_dataset(tmp_path, "s", "T", runner=runner, secrets_client=Secrets(KAGGLE_API_TOKEN=TOKEN))
    assert res == {"status": "created", "url": "https://www.kaggle.com/datasets/satyapaladuguplus/s"}
    assert runner.calls[0] == ["kaggle", "datasets", "status", "satyapaladuguplus/s"]
    assert runner.calls[1][:3] == ["kaggle", "datasets", "create"] and "zip" in runner.calls[1]
    assert json.loads((tmp_path / "dataset-metadata.json").read_text())["id"] == "satyapaladuguplus/s"
    assert runner.envs[1]["KAGGLE_API_TOKEN"] == TOKEN


def test_version_when_exists_with_user_key(tmp_path, capsys):
    runner = Runner(True)
    res = kp.publish_dataset(tmp_path, "s", "T", "msg", runner=runner,
                             secrets_client=Secrets(KAGGLE_USERNAME="u", KAGGLE_KEY="k-SECRET"))
    assert res["status"] == "versioned"
    assert runner.calls[1][:3] == ["kaggle", "datasets", "version"] and "msg" in runner.calls[1]
    assert runner.envs[1]["KAGGLE_USERNAME"] == "u" and runner.envs[1]["KAGGLE_KEY"] == "k-SECRET"
    out = capsys.readouterr()
    assert "k-SECRET" not in out.out + out.err and "k-SECRET" not in json.dumps(res)


def test_failure_scrubs_secret(tmp_path, capsys):
    res = kp.publish_dataset(tmp_path, "s", "T", runner=Runner(False, fail=True),
                             secrets_client=Secrets(KAGGLE_API_TOKEN=TOKEN))
    assert res["status"] == "failed" and TOKEN not in json.dumps(res)
    out = capsys.readouterr()
    assert TOKEN not in out.out + out.err


def test_runner_exception_is_failed(tmp_path):
    def boom(cmd, **kw):
        raise FileNotFoundError("kaggle " + TOKEN)
    res = kp.publish_dataset(tmp_path, "s", "T", runner=boom, secrets_client=Secrets(KAGGLE_API_TOKEN=TOKEN))
    assert res["status"] == "failed" and TOKEN not in json.dumps(res)
