"""Build the local, executable scheduler lesson without changing runtime code."""
from pathlib import Path
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []
def md(text):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))
def code(text):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))

md(r"""
# Semantic decoder scheduler: follow every decision

**A local teaching experiment, not a performance benchmark.** We use the actual
header, queue, service controller, policies, training/export and evaluator modules.
Our four invented messages contain tiny NumPy fixture payloads. Their hand-assigned
parameters and fact labels explain mechanisms; they do not establish extraction
quality or generalization. No VLM, radio link or neural image codec is run here.
At the end we inspect separately labeled, previously executed real SwinJSCC artifacts.

Read the Markdown before running each cell. Use **Restart Kernel and Run All**.
The executable lesson covers task → header → wire bytes → queue → score → dispatch
→ fixture execution → completion/history → coverage → targets/MSE → trained scorer.
""")
md("""
## 1. Setup and reproducibility

Project: SemanticSchedulerBuild, based on its current source snapshot, which can
contain uncommitted changes. Intended kernel: the existing sibling
`SemanticSchedulerEnd2End/.venv` (Python 3.12). Dependency specification:
`pyproject.toml` and `environment/requirements-local-inference.lock`; actual installed versions are recorded below.
Use a local Jupyter installation with that environment's kernel, or execute through
the supplied runner. There is no package installation or model download.

Each run creates a new `results/decoder_context/teaching/<run-id>/` directory.
It holds per-cell text logs, source/environment/config manifest, JSON/CSV results,
figures and the executed notebook copied by the runner. Notebook displays are
preserved in that executed file. Long logs are also written incrementally.
""")
code(r'''
import os, sys, json, time, hashlib, platform, subprocess, importlib.metadata
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager, redirect_stdout, redirect_stderr
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(os.environ.get('SCHEDULER_REPO', r'D:\Bunker\BaseCamp\WirelessCommunication\SemanticCommunication\SemanticSchedulerBuild')).resolve()
sys.path.insert(0, str(ROOT))
RUN = Path(os.environ.get('SCHEDULER_LESSON_RUN', str(ROOT / 'results/decoder_context/teaching' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + str(os.getpid())))))
RUN.mkdir(parents=True, exist_ok=False)
for folder in ('logs', 'metrics', 'figures', 'models'):
    (RUN / folder).mkdir()
class Tee:
    def __init__(self, terminal, file): self.terminal, self.file = terminal, file
    def write(self, value):
        self.terminal.write(value); self.file.write(value); self.file.flush()
        return len(value)
    def flush(self): self.terminal.flush(); self.file.flush()
@contextmanager
def stage(name):
    started = time.perf_counter()
    with (RUN / 'logs' / (name + '.log')).open('w', encoding='utf-8') as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            print(f'START {name} | outputs: {RUN}', flush=True)
            try:
                yield
            except Exception:
                import traceback
                traceback.print_exc()
                raise
            finally:
                print(f'END {name} | elapsed={time.perf_counter()-started:.3f}s', flush=True)
with stage('01_setup'):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display, Image
    plt.rcParams.update({'figure.figsize': (9, 4), 'font.size': 11})
    files = ['Evaluation/decoder_service.py', 'Evaluation/decoder_queue.py',
             'Evaluation/decoder_semantic_benchmark.py', 'Archived/owners_before_simplification_20261027/SemParam/decoder_parameters.py',
             'Scheduler/decoder_scheduler.py', 'Scheduler/contextual_decoder_scheduler.py',
             'Scheduler/training/decoder_context_training.py', 'Archived/owners_before_simplification_20261027/SemParam/decoder_context_profiles.py']
    manifest = {'python': sys.executable, 'version': sys.version, 'hardware': platform.platform(),
                'backend': 'CPU; single-thread numerical libraries', 'repository': str(ROOT),
                'revision': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                'dirty': subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True),
                'packages': {p: importlib.metadata.version(p) for p in ['numpy','pandas','matplotlib','scikit-learn','nbformat','nbclient']},
                'source_sha256': {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
                'scope': 'instructional fixture; no vision extraction, radio, VLM or fresh codec inference'}
    (RUN/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (RUN/'config.json').write_text(json.dumps({'seed': 7, 'profile': 2002, 'K': 3, 'jobs': 4, 'fixture_service_seconds': 1}), encoding='utf-8')
    print(json.dumps(manifest, indent=2))
''')
md(r"""
## 2. Task, images, and the evaluator's answer key

Task: expose **distinct damaged-building evidence** early. A and B show the same
building; C shows a different building; D adds no task fact. These are declared
teaching facts, not labels inferred by the scheduler. Every useful fact has weight 1.
The four colored drawings below are schematic images; real photographs are not
being encoded in this fixture. The scheduler will never receive these drawings or
the fact sets. The payload is an opaque reference from the scheduler's perspective.
""")
code('''
with stage('02_fixture'):
    from Evaluation.decoder_queue import ReceivedMessage
    from Evaluation.decoder_service import DecoderService
    from Archived.owners_before_simplification_20261027.SemParam.decoder_parameters import ParameterHeader
    from Archived.owners_before_simplification_20261027.SemParam.decoder_context_profiles import STUDY_PROFILES
    from Scheduler.decoder_scheduler import SemanticScheduler, PARAMETER_NAMES
    from Scheduler.contextual_decoder_scheduler import StaticPolicyAdapter, DecisionContext, context_matrix, FEATURE_NAMES
    from Evaluation.decoder_semantic_benchmark import coverage_metrics, marginal_gain, exact_sequence
    from Scheduler.training.decoder_context_training import TrainingDecision, fit_policy, matrices
    facts = {'A': {'building_1:damaged'}, 'B': {'building_1:damaged'},
             'C': {'building_2:damaged'}, 'D': set()}
    weights = {'building_1:damaged': 1., 'building_2:damaged': 1.}
    # Fixture payloads stand in for already-received representations, not actual JSCC latents.
    payloads = {name: np.full((16,16,3), value, dtype=np.uint8)
                for name, value in zip(facts, [60,80,170,230])}
    fig, axes = plt.subplots(1,4,figsize=(11,3))
    for ax,(name, image) in zip(axes, payloads.items()):
        ax.imshow(image); ax.set_title(name + ': schematic'); ax.axis('off')
    fig.suptitle('Invented fixture images; A/B share an evaluator fact')
    fig.tight_layout(); fig.savefig(RUN/'figures/fixture_images.png'); plt.show()
    display(pd.DataFrame({'message': list(facts), 'evaluator_only_facts': [sorted(v) for v in facts.values()]}))
''')
md(r"""
## 3. Build and serialize the actual 32-byte header

The order is R, S, Q, I, F, P, C, N. Values are hand-assigned here so the mechanism
is visible. F is unavailable because these static examples have no capture timestamp.
Available zero is a valid observation; unavailable zero means 'do not interpret'.
R = task alignment; S = sender scarcity; Q = claim confidence; I = task importance;
F = freshness; P = visual prominence; C = scene complexity; N = transmitter-KB novelty.
These are intended proxies, not guaranteed measurements of urgency or object identity.

Wire layout: version 1 B + availability 1 B + profile 2 B + token 4 B + KB version
4 B + timestamp 8 B + eight quantized scores 8 B + CRC 4 B = **32 bytes**.
For an available score, q = floor(255v + 0.5), and the receiver uses q/255.
CRC detects accidental corruption; it is not authentication.
""")
code('''
with stage('03_headers'):
    values = {'A': (.95,.6,.9,.95,0,.7,.4,.8),
              'B': (.95,.6,.9,.90,0,.7,.4,.8),
              'C': (.85,.5,.9,.80,0,.6,.3,.7),
              'D': (.10,.1,.8,.10,0,.2,.2,.2)}
    masks = (True, True, True, True, False, True, True, True)
    messages, labels = [], {}
    for token,(name, row) in enumerate(values.items(), 1):
        header = ParameterHeader(2002, token, 0, 0, row, masks, STUDY_PROFILES)
        wire = header.to_bytes()
        received = ParameterHeader.from_bytes(wire, STUDY_PROFILES)
        msg = ReceivedMessage('sender_'+str(token % 2), 'lesson', token, wire,
                              payloads[name], True, 0., token)
        messages.append(msg); labels[msg.message_id] = name
        print(name, '| bytes:', len(wire), '| hex:', wire.hex(), '| I received:', round(received.values[3],6))
    headers_table = pd.DataFrame([ParameterHeader.from_bytes(m.header_bytes, STUDY_PROFILES).values for m in messages], index=list(values), columns=PARAMETER_NAMES)
    display(headers_table)
    headers_table.to_csv(RUN/'metrics/headers.csv')
    print('All messages share profile 2002; full identities include source/session/token.')
''')
md("""
## 4. The header-only observation and direct-I scores

We enqueue complete messages, ask for the ready snapshot, and score each candidate.
Direct I simply uses the received I value. It does not inspect the payload or know
which image is a repeated building. Ties use deterministic arrival/sequence identity.
The context has 39 numerical features: candidate values/masks (16), history masked
means (8), history availability fractions (8), and seven comparison/progress fields.
Header similarity is not verified same-object correspondence.
""")
code('''
with stage('04_observations'):
    baseline = SemanticScheduler('importance', 2002)
    receiver = DecoderService(StaticPolicyAdapter(baseline), 2002, STUDY_PROFILES)
    for message in messages: receiver.enqueue(message)
    candidates = receiver.queue.ready_snapshot(0.)
    context = DecisionContext(candidates, (), 0, 4, 2002)
    matrix = context_matrix(context, [.5]*8)
    print('Ready:', [labels[c.message_id] for c in candidates], '| context shape:', matrix.shape)
    display(pd.DataFrame({'message': [labels[c.message_id] for c in candidates],
                          'I score': baseline.scores(candidates)}))
    display(pd.DataFrame(matrix, index=list(values), columns=FEATURE_NAMES).round(3))
    print('Observation objects contain headers, routing and progress. No evaluator facts or tensors.')
''')
md("""
## 5. Dispatch, fixture execution, completion, and re-selection

Only one job may be in flight. Dispatch removes it from the ready queue; it does
not add it to successful history. Completion commits the dispatch header only on
success. This fixture decoder just copies a NumPy array, so its service duration
of one second is simulated. It does not measure neural decoding latency.
""")
code('''
with stage('05_lifecycle'):
    job = receiver.dispatch(0.)
    print('Selected:', labels[job.message.message_id], '| waiting:', len(receiver), '| history:', len(receiver.history))
    busy_result = receiver.dispatch(.25)
    print('Dispatch while busy:', busy_result)
    fixture_output = np.array(job.message.payload_ref, copy=True)
    record = receiver.complete(job.message.message_id, 'succeeded', 1.)
    print('Fixture output shape:', fixture_output.shape, '| completion:', record)
    print('Successful header history:', [labels[c.message_id] for c in receiver.history])
    updated = DecisionContext(receiver.queue.ready_snapshot(1.), receiver.history, 1, 4, 2002)
    display(pd.DataFrame(context_matrix(updated,[.5]*8), index=[labels[c.message_id] for c in updated.ready_candidates], columns=FEATURE_NAMES).iloc[:,32:].round(3))
''')
md(r"""
## 6. Independent coverage: why A → C can beat A → B

Let W be total offered useful fact weight, fixed before selection. After k jobs,
C_k = distinct successfully exposed fact weight / W. A_K = (C_1 + ... + C_K)/K.
Repeated facts earn no extra weight. Successful fixture output is assumed to preserve
its source facts in this section: this is **ideal-source-evidence evaluation**.
Real reconstructions require separate recovered-fact assessment.

For A,B,C: coverage = [0.5,0.5,1], A_3 = 2/3.
For A,C,B: coverage = [0.5,1,1], A_3 = 5/6.
The second image provides useful evidence sooner in the second ordering.
""")
code('''
with stage('06_coverage'):
    orders = {'A-B-C-D': ['A','B','C','D'], 'A-C-B-D': ['A','C','B','D'], 'D-A-B-C': ['D','A','B','C']}
    results = {name: coverage_metrics(order, facts, weights, k=3) for name,order in orders.items()}
    display(pd.DataFrame({name:{'A_3':r['A_K'], 'coverage@3':r['coverage_at_k'], 'redundancy':r['weighted_redundancy']} for name,r in results.items()}).T)
    fig,ax = plt.subplots()
    for name,result in results.items(): ax.plot([1,2,3],result['curve'], marker='o',label=name)
    ax.set(xlabel='Completed decoder opportunities (fixture)', ylabel='Distinct offered fact coverage', ylim=(0,1.08), xticks=[1,2,3], title='Same facts, different exposure times')
    ax.legend(); fig.tight_layout(); fig.savefig(RUN/'figures/coverage.png'); plt.show()
    print('Evaluator-only exact reference:', exact_sequence(facts,weights,3))
    (RUN/'metrics/hand_orders.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    assert np.isclose(results['A-B-C-D']['A_K'], 2/3)
    assert np.isclose(results['A-C-B-D']['A_K'], 5/6)
''')
md(r"""
## 7. Targets and loss: what does training try to learn?

For a candidate j and delivered history H, the target is its normalized additional
fact weight: y(j,H) = weight(facts(j) minus facts(H)) / W.
After A, B has target 0, C has target 0.5. At runtime the scorer must predict that
from headers alone. Training may access annotations; deployed inference may not.
MSE averages squared prediction errors. The training utility weights candidates so
each decision gets its allocated mass regardless of queue length.
The following predictions are invented to demonstrate the arithmetic, not model outputs.
""")
code('''
with stage('07_loss'):
    target = np.array([marginal_gain(name,['A'],facts,weights) for name in ['B','C','D']])
    prediction = np.array([.3,.4,.1])
    display(pd.DataFrame({'candidate':['B','C','D'], 'target_gain':target, 'invented_prediction':prediction, 'squared_error':(prediction-target)**2}))
    print('Example unweighted MSE =', np.mean((prediction-target)**2))
    print('This loss is computed from annotation targets; no VLM is called.')
''')
md("""
## 8. Fit and export a tiny real scorer

We use the real Ridge training/export module on all subset histories from this
four-job teaching fixture. This intentionally reuses the same fixture for replay:
**it is a mechanics demonstration with no held-out generalization claim**.
Annotation fact IDs create targets only; context inputs remain headers/history.
This is not a new production policy or a recommended parameter selection.
""")
code('''
with stage('08_training'):
    import itertools
    decisions = []
    for count in range(4):
        for indices in itertools.combinations(range(4),count):
            hist = tuple(candidates[i] for i in indices)
            ready = tuple(c for i,c in enumerate(candidates) if i not in indices)
            ctx = DecisionContext(ready,hist,count,4,2002)
            delivered = [labels[c.message_id] for c in hist]
            targets = tuple(marginal_gain(labels[c.message_id],delivered,facts,weights) for c in ready)
            decisions.append(TrainingDecision(ctx,targets,weight=1/15))
    learned = fit_policy(decisions, kind='linear', alpha=.1, provenance={'scope':'same-fixture instructional fit; not held out'})
    print('Training decisions:',len(decisions), '| features:',len(FEATURE_NAMES))
    print('Policy artifact digest:',learned.artifact['sha256'] if hasattr(learned,'artifact') else 'see saved artifact below')
    (RUN/'models/fixture_ridge.json').write_text(json.dumps(learned.artifact,indent=2),encoding='utf-8')
    predicted_gain = learned._predict(context_matrix(updated,learned.artifact['medians']))
    display(pd.DataFrame({'remaining_candidate':[labels[c.message_id] for c in updated.ready_candidates],
                          'actual_model_prediction':predicted_gain,
                          'annotation_target':[marginal_gain(labels[c.message_id],['A'],facts,weights) for c in updated.ready_candidates]}))
    print('Training/export equivalence is checked by fit_policy itself.')
    train_x, train_y, train_weight = matrices(decisions,learned.artifact['medians'],tuple(range(8)))
    train_prediction = learned._predict(train_x)
    training_rows = pd.DataFrame({'target':train_y,'prediction':train_prediction,
                                  'candidate_weight':train_weight,'squared_error':(train_prediction-train_y)**2})
    display(training_rows)
    training_rows.to_csv(RUN/'metrics/training_rows.csv',index=False)
    weighted_sse = float(np.sum(train_weight*(train_prediction-train_y)**2))
    ridge_term = .1*float(np.sum(np.array(learned.artifact['model']['coefficient'])**2))
    print('Actual weighted SSE:',weighted_sse,'| normalized weighted MSE:',weighted_sse/train_weight.sum())
    print('Ridge penalty alpha*sum(beta^2):',ridge_term,'| total fitted objective:',weighted_sse+ridge_term)
    print('Each history has mass 1/15 divided among its candidates; total weight:',train_weight.sum())
    print('The model uses train-only weighted standardization; coefficients refer to standardized features.')
    print('Ridge outputs are unconstrained predictions, not probabilities; ranking uses their order.')
    import inspect
    display(pd.DataFrame({'quantity':['training histories','candidates at initial decision','features per candidate'], 'value':[len(decisions),len(candidates),len(FEATURE_NAMES)]}))
''')
md("""
## 9. Replay policies through the same controller

FIFO, round robin, direct I and the fitted scorer receive identical headers and
fixture payloads. Each re-scores at dispatch. We record every ready list, score,
selection and completion. The evaluator receives a trace only after selection.
Any gain here depends on invented headers and reused fixtures; it is not evidence
that the policy can detect different real buildings.
""")
code('''
with stage('09_policy_replay'):
    policies = {'FIFO':StaticPolicyAdapter(SemanticScheduler('fifo',2002)),
                'RR':StaticPolicyAdapter(SemanticScheduler('round_robin',2002)),
                'I':StaticPolicyAdapter(SemanticScheduler('importance',2002)), 'Fitted Ridge':learned}
    policy_results, traces, candidate_traces = {}, [], []
    for name,policy in policies.items():
        svc = DecoderService(policy,2002,STUDY_PROFILES)
        for message in messages: svc.enqueue(message)
        order = []
        for step in range(4):
            ready = svc.queue.ready_snapshot(float(step))
            current_context = DecisionContext(ready,svc.history,step,4,2002)
            if name == 'Fitted Ridge':
                current_scores = policy.scores(current_context).tolist()
            elif name == 'I':
                current_scores = policy.policy.scores(ready).tolist()
            else:
                current_scores = [None]*len(ready)  # FIFO/RR use ordering rules, not semantic scores.
            history_before = ','.join(labels[c.message_id] for c in svc.history)
            selected = svc.dispatch(float(step))
            token = labels[selected.message.message_id]; order.append(token)
            for candidate,score in zip(ready,current_scores):
                candidate_traces.append({'policy':name,'step':step+1,'candidate':labels[candidate.message_id],
                                         'predicted_score':score,'selected':candidate.message_id==selected.message.message_id,
                                         'successful_history_before':history_before,'attempts_completed':step,
                                         'selection_rule':'numerical score' if score is not None else name+' ordering rule'})
            output = np.array(selected.message.payload_ref,copy=True)
            completion = svc.complete(selected.message.message_id,'succeeded',float(step+1))
            traces.append({'policy':name,'step':step+1,'ready':','.join(labels[c.message_id] for c in ready), 'selected':token,'score':selected.decision.score,'successful_history_size':len(svc.history),'outcome':completion.outcome,'finished_at':completion.finished_at,'fixture_output_shape':str(output.shape)})
        policy_results[name] = {'order':order, **coverage_metrics(order,facts,weights,k=3)}
    display(pd.DataFrame(traces))
    print('Every remaining candidate score for the fitted policy, before each dispatch:')
    display(pd.DataFrame(candidate_traces).query("policy == 'Fitted Ridge'"))
    display(pd.DataFrame({name:{'order':' → '.join(r['order']),'A_3':r['A_K']} for name,r in policy_results.items()}).T)
    pd.DataFrame(traces).to_csv(RUN/'metrics/decision_trace.csv',index=False)
    pd.DataFrame(candidate_traces).to_csv(RUN/'metrics/candidate_scores.csv',index=False)
    (RUN/'metrics/policy_results.json').write_text(json.dumps(policy_results,indent=2),encoding='utf-8')
''')
md("""
## 10. Failure and information limits

A failed decode consumes an opportunity but contributes no fact and no successful
header history. A successful completion only means the execution contract succeeded;
it does not certify the semantic content of a real reconstruction.
If two candidates have identical permitted headers/history relations, a header-only
scorer cannot reliably tell that one contains a novel building. A deterministic
tie-break can choose one but cannot recover missing semantic information.
""")
code('''
with stage('10_failure_checks'):
    failure_receiver = DecoderService(StaticPolicyAdapter(SemanticScheduler('importance',2002)),2002,STUDY_PROFILES)
    for message in messages: failure_receiver.enqueue(message)
    selected = failure_receiver.dispatch(0.)
    failure_receiver.complete(selected.message.message_id,'failed',1.)
    assert len(failure_receiver.history)==0
    statuses = {'A':False,'B':True,'C':True,'D':True}
    failed_metrics = coverage_metrics(['A','B','C','D'],facts,weights,k=3,succeeded=statuses)
    print('History after failed A:',len(failure_receiver.history))
    print('Coverage with failed A:',failed_metrics['curve'],'| A_3:',failed_metrics['A_K'])
    try:
        failure_receiver.complete(selected.message.message_id,'succeeded',1.)
    except ValueError as exc:
        print('Duplicate completion correctly rejected:',exc)
    else:
        raise AssertionError('Duplicate completion was accepted')
    print('Numerical references and lifecycle assertions passed; scope: four-job fixture.')
''')
md("""
## 11. Inspect prior real codec evidence, separately

The following cell reads the existing 24 October SwinJSCC integration outputs.
It does **not** run inference again or turn these images into the fixture labels.
It shows original/reference and reconstruction plus previously measured decode time
and PSNR. Those runs used partial S/I headers and five native-positive images;
they demonstrate integration, not semantic ranking superiority. A cached noisy
continuous latent is a receiver-side artifact, not proof of a packetized digital codec.
""")
code('''
with stage('11_prior_codec_artifacts'):
    prior = ROOT/'results/decoder_context/implementation_ops/20261024T083335Z/actual_policy_replay/summary.json'
    cache = ROOT/'results/decoder_context/implementation_ops/20261024T082550Z/codec_matched_runtime_5_timm/received/manifest.json'
    if prior.exists() and cache.exists():
        saved = json.loads(prior.read_text()); cached = json.loads(cache.read_text())
        metric = saved['results']['FIFO']['trace'][0]['metrics']
        record = next(r for r in cached['records'] if r['image_id']==metric['image_id'])
        reference = cache.parent/record['reference_image']; reconstruction=Path(metric['reconstruction'])
        print('Historical artifact:',prior)
        print('Previously measured PSNR:',metric['psnr_db'],'dB | decode seconds:',metric['decode_seconds'])
        print('Reference SHA verified:',hashlib.sha256(reference.read_bytes()).hexdigest()==record['reference_sha256'])
        fig,axes=plt.subplots(1,2,figsize=(10,5))
        for ax,path,title in zip(axes,[reference,reconstruction],['Prior codec input (letterboxed)','Prior SwinJSCC reconstruction']):
            ax.imshow(plt.imread(path)); ax.set_title(title); ax.axis('off')
        fig.tight_layout(); fig.savefig(RUN/'figures/prior_real_codec.png'); plt.show()
        (RUN/'metrics/prior_codec_reference.json').write_text(json.dumps({'artifact':str(prior),'artifact_sha256':hashlib.sha256(prior.read_bytes()).hexdigest(),'metrics':metric},indent=2),encoding='utf-8')
    else:
        print('Historical codec artifacts unavailable: this section skipped explicitly.')
''')
md("""
## 12. What to inspect and change yourself

1. Inspect `headers.csv`: explain why F is unavailable and why I changes slightly on wire.
2. Inspect `decision_trace.csv`: explain when a selected message enters history.
3. Compare the coverage curves: explain why another view may add zero declared facts.
4. Change the invented I values so C comes before B. Re-run all into a new run directory.
5. Add a new supported fact to B. Recalculate its marginal gain after A.
6. Ask what information the header would need to distinguish A/B from A/C in real data.

The next experiment needs independent annotated images, causal extracted parameters,
grouped training/validation/final splits and separately assessed reconstruction facts.
This lesson deliberately has none of those scientific validation claims. VLM feedback,
PPO, genuine parameter extraction, source encoding and radio transport are not executed.
Runtime code links: `Evaluation/decoder_service.py`, `Scheduler/contextual_decoder_scheduler.py`,
`Evaluation/decoder_semantic_benchmark.py`, `Scheduler/training/decoder_context_training.py`.
""")
code('''
with stage('12_finish'):
    summary = {'status':'complete','scope':'instructional fixture plus read-only historical codec display',
               'training_decisions':len(decisions),'fresh_neural_codec_inference':False,
               'vlm_calls':0,'output_directory':str(RUN), 'policy_A3':{name:r['A_K'] for name,r in policy_results.items()}}
    (RUN/'validation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print('Read the saved executed notebook; figures and metrics are also standalone files.')
''')
notebook = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec':{'display_name':'Python (project .venv)', 'language':'python','name':'python3'},'language_info':{'name':'python'}})
destination = ROOT/'notebooks/teaching/semantic_scheduler_walkthrough.ipynb'
destination.parent.mkdir(parents=True,exist_ok=True)
nbf.validate(notebook)
nbf.write(notebook,destination)
print(destination)
