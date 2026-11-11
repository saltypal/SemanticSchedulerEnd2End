"""Emit Notebooks/Edge/edge-live-t4.ipynb: live 2xT4 validation (plan v3 sec. 9) via Evaluation.edge.live_check.

Needs the codec v1 bundle (``satyapaladugu/scheduler-sqgan-v1``, later), the public images and the sim table.
Never pushes anything.
"""
import argparse

from scripts.edge import _nb_common as nc
from scripts.edge import build_notebook as bn

STEP = "edge-live-t4"
TITLE = "Edge Live T4"
ENTRY = "Evaluation.edge.live_check"
DATASET_SOURCES = (nc.SIMTABLE_DS, nc.IMAGES_DS, nc.SQGAN_V1_DS)
REQUIREMENTS = ("pytorch-lightning==1.9.5", "torchmetrics==0.11.4", "transformers==4.46.3", "scipy", "omegaconf",
                "pytorch-msssim==1.0.0", "lpips==0.1.4", "PyYAML", "einops", "pillow")

RUN = '''
import os, subprocess
OUT = "/kaggle/working/runs/" + {step!r}
os.makedirs(OUT, exist_ok=True)
ARGV = ["python", "-u", "-m", {entry!r}, "--table", TABLE, "--profile", PROFILE, "--out", OUT + "/live_check.json",
        "--images-dir", "/kaggle/input"] + {extra!r}
print("$", " ".join(ARGV[3:]), flush=True)
p = subprocess.Popen(ARGV, cwd=REPO, env=dict(os.environ), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(line, end="", flush=True)
if p.wait():
    raise RuntimeError("live_check failed (rc=%d)" % p.returncode)
'''


def build(account=None, branch="main", n_images=1000, policies=("fifo", "edf", "sv_dpp")):
    cfg = bn.load_kaggle_config()
    extra = ["--n-images", str(n_images), "--workers", "2", "--policies", *policies]
    nb = bn.build_notebook(STEP, ENTRY, extra, TITLE, cfg["repo_url"], branch, REQUIREMENTS, True)
    nb["cells"][0] = bn._md(f"# {TITLE}\n\nStep `{STEP}`: 2 x T4, c = 2, {n_images} test images, policies "
                            f"{', '.join(policies)}. Select the T4 x2 accelerator.\n\n{nc.PRIVATE_NOTE}"
                            f"Codec v1 bundle: `{nc.SQGAN_V1_DS}` (attach once published).\n")
    nb["cells"][4:5] = [nc.discover_cell(), bn._code(RUN.format(step=STEP, entry=ENTRY, extra=extra))]
    meta = bn.build_metadata(cfg, account or cfg.get("default_account", "wsl"), STEP, TITLE, True, DATASET_SOURCES)
    meta["machine_shape"] = "NvidiaTeslaT4"
    return nc.finish(nb, meta)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-images", type=int, default=1000)
    ap.add_argument("--policies", nargs="+", default=["fifo", "edf", "sv_dpp"])
    ap.add_argument("--branch", default="main")
    ap.add_argument("--account", default=None, choices=("win", "wsl"))
    ap.add_argument("--out-dir", default=str(bn.OUT_DIR))
    a = ap.parse_args(argv)
    nb, meta = build(a.account, a.branch, a.n_images, a.policies)
    nc.write(nb, meta, STEP, a.out_dir)


if __name__ == "__main__":
    main()
