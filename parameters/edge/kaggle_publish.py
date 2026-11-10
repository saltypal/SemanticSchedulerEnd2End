"""Publish a local folder as a Kaggle dataset from inside a Kaggle notebook (create or new version).

Credentials come ONLY from Kaggle Secrets (``kaggle_secrets.UserSecretsClient``): label
``KAGGLE_API_TOKEN`` first, else ``KAGGLE_USERNAME`` + ``KAGGLE_KEY``. They are exported to the
environment of the kaggle subprocess only; never printed, logged or returned. All commands are
argument lists (no shell). Without credentials the call returns status ``skipped`` with a reason
that says how to publish manually.
"""
import json
import os
import subprocess
from pathlib import Path

DEFAULT_OWNER = "satyapaladuguplus"
SKIP_REASON = ("No Kaggle credentials found in Kaggle Secrets. Either add the secret KAGGLE_API_TOKEN "
               "(or KAGGLE_USERNAME and KAGGLE_KEY) in this notebook under Add-ons > Secrets and re-run, "
               "or use 'Save Version' and then 'New dataset from output' in the Kaggle web UI.")


def get_credentials(secrets_client=None):
    """Return an env-var dict for the kaggle CLI, or None when no secrets are available."""
    if secrets_client is None:
        try:
            from kaggle_secrets import UserSecretsClient
            secrets_client = UserSecretsClient()
        except Exception:
            return None

    def get(label):
        try:
            value = secrets_client.get_secret(label)
        except Exception:
            return None
        return value.strip() if isinstance(value, str) and value.strip() else None

    token = get("KAGGLE_API_TOKEN")
    if token:
        return {"KAGGLE_API_TOKEN": token}
    user, key = get("KAGGLE_USERNAME"), get("KAGGLE_KEY")
    if user and key:
        return {"KAGGLE_USERNAME": user, "KAGGLE_KEY": key}
    return None


def _scrub(text, secrets):
    text = text or ""
    for value in secrets:
        if value:
            text = text.replace(value, "***")
    return text[-400:]


def _ensure_metadata(directory, ref, title):
    path = Path(directory) / "dataset-metadata.json"
    if not path.exists():
        path.write_text(json.dumps({"id": ref, "title": title, "licenses": [{"name": "other"}]}, indent=2),
                        encoding="utf-8")


def publish_dataset(directory, slug, title, message="", owner=DEFAULT_OWNER, runner=subprocess.run,
                    secrets_client=None):
    """Create ``owner/slug`` or add a version. Returns {"status": created|versioned|skipped|failed,
    "url" | "reason"}. ``runner`` has the ``subprocess.run`` signature."""
    creds = get_credentials(secrets_client)
    if not creds:
        return {"status": "skipped", "reason": SKIP_REASON}
    secret_values = list(creds.values())
    ref = "%s/%s" % (owner, slug)
    _ensure_metadata(directory, ref, title)
    env = dict(os.environ)
    env.update(creds)
    run = dict(capture_output=True, text=True, env=env)
    try:
        status = runner(["kaggle", "datasets", "status", ref], **run)
        out = (status.stdout or "") + (status.stderr or "")
        exists = status.returncode == 0 and "404" not in out and "not found" not in out.lower()
        if exists:
            cmd = ["kaggle", "datasets", "version", "-p", str(directory),
                   "-m", message or "update", "--dir-mode", "zip"]
        else:
            cmd = ["kaggle", "datasets", "create", "-p", str(directory), "--dir-mode", "zip"]
        done = runner(cmd, **run)
    except Exception as exc:
        return {"status": "failed", "reason": _scrub("%s: %s" % (type(exc).__name__, exc), secret_values)}
    if done.returncode != 0:
        return {"status": "failed", "reason": _scrub((done.stderr or "") + (done.stdout or ""), secret_values)}
    return {"status": "versioned" if exists else "created", "url": "https://www.kaggle.com/datasets/" + ref}
