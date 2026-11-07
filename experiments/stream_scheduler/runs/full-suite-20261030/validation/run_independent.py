exec(open('/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End/experiments/stream_scheduler/runs/full-suite-20261030/validation/run_fixtures.py').read().split('tests=')[0])
rows=[]
for name,fn in [('cost',cost),('targets',target),('lifecycle',lifecycle),('gradients',gradients),('lambdamart',lambdamart)]:
 before=sh();t=time.perf_counter()
 try:v=fn();status='PASS';err=None
 except BaseException:status='FAIL';v=None;err=traceback.format_exc()
 after=sh();rows.append({'name':name,'status':status,'seconds':time.perf_counter()-t,'before':before,'after':after,'source_changed':before!=after,'result':v,'error':err})
 print(json.dumps(rows[-1],allow_nan=False),flush=True)
 if status=='FAIL':break
(O/'independent_checks.json').write_text(json.dumps({'scope':'other independent CPU fixture checks, no training','checks':rows},indent=2,allow_nan=False))