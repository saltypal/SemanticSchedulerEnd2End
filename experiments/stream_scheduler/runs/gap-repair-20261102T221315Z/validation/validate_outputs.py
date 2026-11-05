import csv, json, math, pathlib, collections, sys
root=pathlib.Path("/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End/experiments/stream_scheduler/runs/gap-repair-20261102T221315Z")
out=root/"rate_cpu_pilot"
rows=json.loads((out/"results.json").read_text())
csv_rows=list(csv.DictReader((out/"summary.csv").open(newline="")))
problems=[]
expected_policies={"fifo","round_robin","spi_fifo","spi_novelty"}
def finite_tree(value,path="root"):
    errors=[]
    if isinstance(value,float) and not math.isfinite(value): errors.append(path)
    elif isinstance(value,dict):
        for k,v in value.items(): errors.extend(finite_tree(v,f"{path}.{k}"))
    elif isinstance(value,list):
        for i,v in enumerate(value): errors.extend(finite_tree(v,f"{path}[{i}]"))
    return errors
print("json_rows",len(rows),"csv_rows",len(csv_rows))
if len(rows)!=80 or len(csv_rows)!=80: problems.append("expected exactly 80 result and CSV rows")
all_errors=[]
for i,r in enumerate(rows): all_errors.extend(finite_tree(r,f"row[{i}]"))
print("nonfinite_numeric_values",len(all_errors))
if all_errors: problems.extend(["nonfinite values: "+", ".join(all_errors[:10])])
by_trace=collections.defaultdict(list)
for r in rows: by_trace[r["trace"]].append(r)
print("unique_traces",len(by_trace))
if len(by_trace)!=20: problems.append("expected 20 paired trace configurations")
for trace,group in sorted(by_trace.items()):
    if {r["policy"] for r in group}!=expected_policies or len(group)!=4:
        problems.append(f"{trace}: policy assignments incomplete")
    if len({r["offered"] for r in group})!=1:
        problems.append(f"{trace}: offered count differs across policies")
    for r in group:
        m=r["metrics"]
        if m["offered"]!=r["offered"]: problems.append(f"{trace}/{r['policy']}: offered mismatch")
        if m["selected"]+m["queued_at_horizon"]+m["not_arrived"]!=m["offered"]:
            problems.append(f"{trace}/{r['policy']}: offered/selected/queued/not-arrived accounting mismatch")
        if m["completed"]+m["in_flight"]!=m["selected"]:
            problems.append(f"{trace}/{r['policy']}: selected/completed/in-flight mismatch")
        if m["successful_completed"]+m["failed"]!=m["completed"]:
            problems.append(f"{trace}/{r['policy']}: completed success/failure mismatch")
        tiers=m["by_reference_tier"]
        if sum(x["offered"] for x in tiers.values())!=m["offered"]:
            problems.append(f"{trace}/{r['policy']}: reference-tier offered sum mismatch")
        if sum(x["selected"] for x in tiers.values())!=m["selected"]:
            problems.append(f"{trace}/{r['policy']}: reference-tier selected sum mismatch")
counts=collections.Counter(r["policy"] for r in rows)
designs=collections.Counter(r["rate_design"] for r in rows)
modes=collections.Counter(r["mode"] for r in rows)
bursts=collections.Counter(r["burst_factor"] for r in rows)
print("policy_rows",dict(counts))
print("rate_design_rows",dict(designs))
print("mode_rows",dict(modes))
print("burst_rows",dict(bursts))
if counts!=collections.Counter({p:20 for p in expected_policies}): problems.append("unexpected policy row counts")
if designs!=collections.Counter({"fixed_total":48,"fixed_per_ue":32}): problems.append("unexpected rate design counts")
if modes!=collections.Counter({"mixed":40,"mission":40}): problems.append("unexpected mode counts")
if bursts!=collections.Counter({1.0:40,4.0:40}): problems.append("unexpected burst counts")
audit=json.loads((out/"inputs/data_audit.json").read_text())
print("archive_images",audit["images"],"cached_embeddings",audit["cached_embeddings"],"reference_known",audit["reference_known"])
print("failed_checks",len(problems))
for item in problems: print("FAIL",item)
if problems: sys.exit(1)
print("ACCOUNTING_VALIDATION_PASS")
