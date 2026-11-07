import csv, json, math, pathlib, collections, sys, datetime
root=pathlib.Path("/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End/experiments/stream_scheduler/runs/gap-repair-20261102T221315Z")
out=root/"rate_cpu_full"
val=root/"validation/full"
rows=json.loads((out/"results.json").read_text())
csv_rows=list(csv.DictReader((out/"summary.csv").open(newline="")))
problems=[]
policies={"fifo","round_robin","spi_fifo","spi_novelty"}
def finite_tree(v,path="root"):
    e=[]
    if isinstance(v,float) and not math.isfinite(v): e.append(path)
    elif isinstance(v,dict):
        for k,x in v.items(): e.extend(finite_tree(x,f"{path}.{k}"))
    elif isinstance(v,list):
        for i,x in enumerate(v): e.extend(finite_tree(x,f"{path}[{i}]"))
    return e
print("rows_json",len(rows),"rows_csv",len(csv_rows))
if len(rows)!=320 or len(csv_rows)!=320: problems.append("expected exactly 320 result and CSV rows")
bad=[]
for i,r in enumerate(rows): bad.extend(finite_tree(r,f"row[{i}]"))
print("nonfinite_numeric_values",len(bad))
if bad: problems.append("nonfinite values: "+", ".join(bad[:10]))
groups=collections.defaultdict(list)
for r in rows: groups[r["trace"]].append(r)
print("unique_traces",len(groups))
if len(groups)!=80: problems.append("expected exactly 80 unique traces")
for trace,g in sorted(groups.items()):
    if len(g)!=4 or {r["policy"] for r in g}!=policies:
        problems.append(f"{trace}: policy set or row count mismatch")
    if len({r["offered"] for r in g})!=1: problems.append(f"{trace}: offered count differs across policies")
    if len({(r["ues"],r["rate_design"],r["mode"],r["burst_factor"],r["seed"]) for r in g})!=1:
        problems.append(f"{trace}: trace configuration differs across policies")
    for r in g:
        m=r["metrics"]
        if m["offered"]!=r["offered"]: problems.append(f"{trace}/{r['policy']}: offered mismatch")
        if m["selected"]+m["queued_at_horizon"]+m["not_arrived"]!=m["offered"]:
            problems.append(f"{trace}/{r['policy']}: offered/selected/queued/not-arrived mismatch")
        if m["completed"]+m["in_flight"]!=m["selected"]:
            problems.append(f"{trace}/{r['policy']}: selected/completed/in-flight mismatch")
        if m["successful_completed"]+m["failed"]!=m["completed"]:
            problems.append(f"{trace}/{r['policy']}: completed success/failure mismatch")
        tiers=m["by_reference_tier"]
        if sum(x["offered"] for x in tiers.values())!=m["offered"]:
            problems.append(f"{trace}/{r['policy']}: tier offered sum mismatch")
        if sum(x["selected"] for x in tiers.values())!=m["selected"]:
            problems.append(f"{trace}/{r['policy']}: tier selected sum mismatch")
policy_counts=collections.Counter(r["policy"] for r in rows)
design_counts=collections.Counter(r["rate_design"] for r in rows)
mode_counts=collections.Counter(r["mode"] for r in rows)
burst_counts=collections.Counter(r["burst_factor"] for r in rows)
ue_counts=collections.Counter(r["ues"] for r in rows)
seed_counts=collections.Counter(r["seed"] for r in rows)
print("policy_rows",dict(policy_counts))
print("rate_design_rows",dict(design_counts))
print("mode_rows",dict(mode_counts))
print("burst_rows",dict(burst_counts))
print("ue_rows",dict(ue_counts))
print("seed_rows",dict(seed_counts))
if policy_counts!=collections.Counter({p:80 for p in policies}): problems.append("policy row counts mismatch")
if design_counts!=collections.Counter({"fixed_total":192,"fixed_per_ue":128}): problems.append("rate design row counts mismatch")
if mode_counts!=collections.Counter({"mixed":160,"mission":160}): problems.append("mode row counts mismatch")
if burst_counts!=collections.Counter({1.0:160,4.0:160}): problems.append("burst row counts mismatch")
if ue_counts!=collections.Counter({12:80,50:80,60:80,100:80}): problems.append("UE population row counts mismatch")
if seed_counts!=collections.Counter({1729:320}): problems.append("seed row counts mismatch")
trace_files=list((out/"traces").glob("*.json"))
if len(trace_files)!=80: problems.append(f"expected 80 trace metadata files, found {len(trace_files)}")
else:
    for p in trace_files:
        meta=json.loads(p.read_text())
        cfg=meta.get("config",{})
        if cfg.get("duration")!=200.0: problems.append(f"{p.name}: duration mismatch")
        if cfg.get("seed")!=1729: problems.append(f"{p.name}: seed mismatch")
manifest=json.loads((out/"manifest.json").read_text())
print("incident_status",manifest.get("incident_status"))
if not str(manifest.get("incident_status","")).startswith("unavailable"):
    problems.append("incident mode should remain unavailable without annotations")
audit=json.loads((out/"inputs/data_audit.json").read_text())
print("archive_images",audit["images"],"cached_embeddings",audit["cached_embeddings"],"reference_known",audit["reference_known"])
start=datetime.datetime.fromisoformat((val/"start_utc.txt").read_text().strip().replace("Z","+00:00"))
end=datetime.datetime.fromisoformat((val/"end_utc.txt").read_text().strip().replace("Z","+00:00"))
print("elapsed_seconds",round((end-start).total_seconds(),3))
print("sourcehash_diff_bytes",(val/"sourcehash.diff").stat().st_size)
print("cli_stdout_progress_lines",sum(1 for line in (val/"stdout.log").read_text().splitlines() if line.startswith("STREAM_STUDY ")))
print("cli_exit_code",(val/"exit_code.txt").read_text().strip())
if (val/"sourcehash.diff").stat().st_size: problems.append("source hash changed during execution")
if (val/"exit_code.txt").read_text().strip()!="0": problems.append("CLI exit code was nonzero")
if (val/"stderr.log").stat().st_size: problems.append("CLI stderr was not empty")
if sum(1 for line in (val/"stdout.log").read_text().splitlines() if line.startswith("STREAM_STUDY "))!=80:
    problems.append("expected 80 periodic trace progress lines")
print("failed_checks",len(problems))
for p in problems: print("FAIL",p)
if problems: sys.exit(1)
print("FULL_GRID_VALIDATION_PASS")