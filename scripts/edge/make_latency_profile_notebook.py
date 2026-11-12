"""Emit Notebooks/Edge/edge-v1-latency.ipynb (+ kernel metadata): TRUE batched SQ-GAN v1 decode latency, B in {1,2,4,8}.

Short single-GPU notebook (so the whole v1 pass is not rerun): clone main, load the v1 bundle from the fine-tune
kernel output, encode ~200 val images at m 0.25, time ``SQGANDigitalCodec.decode_batch``, write
/kaggle/working/latency_v1_true.json. Account `wsl`, T4 x2 machine, internet on. Never pushes anything.
"""
import argparse
from pathlib import Path

from Evaluation.edge.edge_config import load_config
from scripts.edge import build_notebook as bn

STEP = "edge-v1-latency"
TITLE = "Edge V1 Latency"     # must slugify to the kernel id
DATASET_SOURCES = ("satyapaladuguplus/scheduler-dataset-v1-images",)
OUT_NAME = "latency_v1_true.json"

RUN = '''
import json, sys, time, traceback
from pathlib import Path
try:
    import json, sys, time
    sys.path.insert(0, REPO)
    import numpy as np, torch
    from pathlib import Path
    from Codec.edge import feature_pass as fp
    from Codec.edge.feature_pass_backends import RealSQGAN
    from Evaluation.edge.edge_config import load_config
    M, N_IMAGES, WARMUP, SAMPLES = 0.25, 200, 5, 40
    cfg = json.loads(json.dumps(load_config("feature_pass")))
    cfg["sqgan"]["codec"] = "v1"; cfg["sqgan"]["m_levels"] = [M]
    root = fp.find_data_root(cfg["paths"]["images_marker"], cfg["paths"]["input_base"], cfg["paths"]["discover_depth"])
    rows = [json.loads(l) for l in (root / cfg["paths"]["images_marker"]).read_text().splitlines() if l.strip()]
    val = [r for r in rows if r.get("split") == "val"] or rows
    sq = RealSQGAN(cfg, "cuda")
    codec = sq._codec(M)
    payloads = []
    for r in val[:N_IMAGES]:
        arr, _ = fp.letterbox(fp._load_image(root, r), tuple(cfg["sqgan"]["input_hw"]), cfg["sqgan"]["pad_value"])
        _, handle = sq.segment(arr)
        payloads.append(sq.encode(arr, handle, M)["payload"])
    print("encoded", len(payloads), "payloads")
    samples, vram = {}, {}
    for b in (1, 2, 4, 8):
        batches = [[payloads[(k * b + i) % len(payloads)] for i in range(b)] for k in range(WARMUP + SAMPLES)]
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        out = []
        for k, batch in enumerate(batches):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            codec.decode_batch(batch)
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            if k >= WARMUP:
                out.append(dt)
        samples[str(b)] = out; vram[str(b)] = int(torch.cuda.max_memory_allocated())
        print("B=%d mean %.1f ms" % (b, 1e3 * float(np.mean(out))))
    res = {"samples": samples, "emulated_batching": False, "m": M, "peak_vram_bytes": vram,
           "n_payloads": len(payloads), "gpu": torch.cuda.get_device_name(0), "git_sha": GIT_SHA}
    Path("/kaggle/working/@OUT@").write_text(json.dumps(res))
    print("wrote @OUT@")
except Exception:
    Path("/kaggle/working/@OUT@").write_text(json.dumps({"error": traceback.format_exc(), "emulated_batching": False}))
    raise
'''


def build(branch="main", account=None):
    cfg = bn.load_kaggle_config()
    nb = bn.build_notebook(STEP, "Codec.edge.sqgan_profile", [], TITLE, cfg["repo_url"], branch,
                           ("transformers>=4.49.0", "PyYAML", "lpips"), True)
    nb["cells"][0] = bn._md("# %s\n\nTrue batched decode latency of the v1 codec; writes `%s`.\n" % (TITLE, OUT_NAME))
    nb["cells"][4] = bn._code(RUN.replace("@OUT@", OUT_NAME))
    for c in nb["cells"]:   # keep the clone out of /kaggle/working so the kernel output stays small
        c["source"] = [x.replace("/kaggle/working/repo", "/tmp/repo") for x in c["source"]] if isinstance(c["source"], list) else c["source"].replace("/kaggle/working/repo", "/tmp/repo")
    kernel = load_config("codec")["codecs"]["v1"]["kaggle_kernel"]
    account = account or cfg.get("default_account", "wsl")
    meta = bn.build_metadata(cfg, account, STEP, TITLE, True, DATASET_SOURCES, (kernel,))
    meta["id"] = "%s/%s" % (meta["id"].split("/")[0], STEP)
    meta["machine_shape"] = "NvidiaTeslaT4"
    return nb, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--branch", default="main")
    ap.add_argument("--account", default=None)
    ap.add_argument("--out-dir", default=str(bn.OUT_DIR))
    a = ap.parse_args(argv)
    nb, meta = build(a.branch, a.account)
    for p in bn.write_outputs(nb, meta, STEP, Path(a.out_dir)):
        print(p)


if __name__ == "__main__":
    main()
