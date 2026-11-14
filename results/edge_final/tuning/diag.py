import json, sys
from Evaluation.edge import run_sweep as rs, tables
S = "D:/Bunker/BaseCamp/_claude_scratch"
rows = tables.load_rows(S + "/simtable/simtable_v1.jsonl")
lat = tables.latency_from_profile_json(S + "/simtable/latency_v1.json")
T = dict(V=0.1, hard_purge_prob=0.99, relevance_beta=1.0, tie_tol_frac=0.5)
VAR = {"tuned": T, "V=1.0": dict(T, V=1.0), "tie=0.3": dict(T, tie_tol_frac=0.3), "tie=0.1": dict(T, tie_tol_frac=0.1),
       "beta=0.5": dict(T, relevance_beta=0.5), "purge=0.95": dict(T, hard_purge_prob=0.95),
       "old(untuned)": dict(V=1.0, hard_purge_prob=0.95, relevance_beta=0.5, tie_tol_frac=0.3)}
for name, kw in VAR.items():
    out = rs.run_sweep(rows, lat, "val", ["sv_dpp"], [2], [1.2], 5, 0, policy_kwargs={"sv_dpp": kw},
                       arrival="trace", subsample=1000)
    c = out["cells"][0]
    print("%-14s CTR@D %.3f  CI %s" % (name, c["ctr_ungated"], [round(x, 3) for x in c["ci95"]["ctr_ungated"]]), flush=True)
