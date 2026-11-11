"""Builds the backup notebook edge-tunnel: SSH into a live Kaggle session from this PC.

How it works: the notebook starts sshd (key-only login, your public key from the Kaggle Secret SSH_PUBKEY) and a
Cloudflare quick tunnel (no Cloudflare account needed). It prints a *.trycloudflare.com hostname, then keeps the
session alive. On the PC, `scripts/edge/kaggle_tunnel_client.sh <hostname>` connects through the tunnel.

Run it INTERACTIVELY in the Kaggle editor (Run All in a draft session) so the hostname is visible while it runs;
a "Save & Run All" batch run only shows its log after it ends. Same quotas as any Kaggle session (12 h, GPU hours).
Kaggle has no published rule found on SSH tunnels (checked 2026-11-11); use at your own account risk, as a backup.

    python Notebooks/Edge/build_edge_tunnel.py <out_dir> [--account plus|second]
"""
import argparse
import json
from pathlib import Path

ACCOUNTS = {"plus": "satyapaladuguplus", "second": "satyapaladugu"}

INTRO = """# Edge tunnel (backup): SSH into this Kaggle session

1. Add-ons -> Secrets: `SSH_PUBKEY` = the contents of your `~/.ssh/id_ed25519.pub` (public key only).
2. Accelerator as needed (GPU T4 x2), Internet ON. **Run All in the editor** (not Save Version), keep the tab open.
3. Copy the `*.trycloudflare.com` hostname printed below, then on the PC:
   `bash scripts/edge/kaggle_tunnel_client.sh <hostname>`
4. Stop the session from the editor when done (it otherwise holds GPU quota until the 12 h limit).

Password login is disabled; only the key in `SSH_PUBKEY` can log in.
"""

SETUP = r'''import os, subprocess, pathlib, time, re

from kaggle_secrets import UserSecretsClient
pub = UserSecretsClient().get_secret("SSH_PUBKEY").strip()
if not pub.startswith(("ssh-ed25519", "ssh-rsa", "ecdsa-")):
    raise SystemExit("SSH_PUBKEY does not look like a public key (expected ssh-ed25519 ... / ssh-rsa ...)")

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"failed: {cmd}\n{r.stderr[-800:]}")
    return r.stdout

sh("apt-get -qq update && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y openssh-server > /dev/null")
ssh_dir = pathlib.Path("/root/.ssh"); ssh_dir.mkdir(mode=0o700, exist_ok=True)
(ssh_dir / "authorized_keys").write_text(pub + "\n"); os.chmod(ssh_dir / "authorized_keys", 0o600)
pathlib.Path("/etc/ssh/sshd_config.d").mkdir(exist_ok=True)
pathlib.Path("/etc/ssh/sshd_config.d/kaggle.conf").write_text(
    "PasswordAuthentication no\nKbdInteractiveAuthentication no\nPermitRootLogin prohibit-password\n"
    "PubkeyAuthentication yes\nClientAliveInterval 30\n")
sh("mkdir -p /run/sshd && /usr/sbin/sshd")
sh("curl -fsSL -o /usr/local/bin/cloudflared "
   "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x /usr/local/bin/cloudflared")
print("sshd running (key-only); cloudflared", sh("cloudflared --version").strip())
'''

TUNNEL = r'''log = open("/kaggle/working/cloudflared.log", "w")
proc = subprocess.Popen(["cloudflared", "tunnel", "--no-autoupdate", "--url", "ssh://localhost:22"],
                        stdout=log, stderr=subprocess.STDOUT)
host = None
for _ in range(60):
    time.sleep(2)
    m = re.search(r"https://([a-z0-9-]+\.trycloudflare\.com)", open("/kaggle/working/cloudflared.log").read())
    if m:
        host = m.group(1)
        break
if not host:
    raise SystemExit("no tunnel hostname after 120 s; see /kaggle/working/cloudflared.log")
print("=" * 70)
print("TUNNEL HOST:", host)
print("On the PC:  bash scripts/edge/kaggle_tunnel_client.sh", host)
print("=" * 70, flush=True)
'''

KEEPALIVE = r'''# Keeps the session alive while the tunnel is up. Interrupt this cell (or stop the session) to end it.
t0 = time.time()
while proc.poll() is None and time.time() - t0 < 11.5 * 3600:
    time.sleep(60)
    print(f"alive {int((time.time() - t0) / 60)} min", end="\r", flush=True)
print("\ntunnel ended")
'''


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--account", choices=sorted(ACCOUNTS), default="plus")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nb = {"cells": [cell("markdown", INTRO), cell("code", SETUP), cell("code", TUNNEL), cell("code", KEEPALIVE)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (out / "edge-tunnel.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {"id": "%s/edge-tunnel" % ACCOUNTS[a.account], "title": "Edge tunnel", "code_file": "edge-tunnel.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_internet": True, "dataset_sources": [], "kernel_sources": [], "competition_sources": [],
            "model_sources": [], "machine_shape": "NvidiaTeslaT4"}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
