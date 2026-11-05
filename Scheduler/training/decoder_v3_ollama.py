"""Local Ollama deployment and bounded visual review inside Kaggle Linux."""
import base64
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
import urllib.request

from Archived.owners_before_simplification_20261027.SemParam.decoder_extractors import parse_qwen_assessment
from Archived.owners_before_simplification_20261027.SemParam.decoder_costs import qwen_decoder_matrix_estimate

OLLAMA_VERSION = '0.40.8'
OLLAMA_ARCHIVE_SHA256 = '0bb17cc19e9c1339ddbe0cba55148bf068f8f19dec8ec5fd531461aaa207282a'
QWEN_MODEL = 'qwen3-vl:4b-instruct-q4_K_M'
QWEN_DIGEST = '84f7ed8042fa7cdbd902f7dcb7d0370e032095047cc3244179f263b77b5bc318'


def request_json(path, value=None, timeout=900):
    body = None if value is None else json.dumps(value).encode()
    request = urllib.request.Request('http://127.0.0.1:11434'+path, data=body,
                                     headers={'Content-Type':'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        result = json.load(response)
    if 'error' in result:
        raise RuntimeError(result['error'])
    return result


def gpu_snapshot():
    result = {}
    for name, query in (('devices','--query-gpu=uuid,name,memory.used,utilization.gpu'),
                        ('processes','--query-compute-apps=gpu_uuid,pid,used_gpu_memory')):
        process = subprocess.run(['nvidia-smi',query,'--format=csv,noheader'], capture_output=True,text=True)
        result[name] = process.stdout.strip() if process.returncode == 0 else process.stderr.strip()
    return result


class OllamaQwenReviewer:
    """A separately versioned GGUF producer; never substituted for HF outputs."""
    def __init__(self, root='/kaggle/temp/decoder_v3_ollama', log_dir=None, require_two_gpus=True):
        import torch
        if require_two_gpus and torch.cuda.device_count() < 2:
            raise RuntimeError('The requested Ollama run requires two visible GPUs')
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir) if log_dir else self.root
        self.log_dir.mkdir(parents=True, exist_ok=True)
        installation = self.root/'installation'
        binary = installation/'bin/ollama'
        marker = installation/'verified.json'
        if not (binary.exists() and marker.exists() and
                json.loads(marker.read_text()).get('archive_sha256') == OLLAMA_ARCHIVE_SHA256):
            archive = self.root/'ollama-linux-amd64.tar.zst'
            url = f'https://github.com/ollama/ollama/releases/download/v{OLLAMA_VERSION}/ollama-linux-amd64.tar.zst'
            subprocess.run(['curl','--fail','--location','--retry','5','--retry-all-errors',
                            '--continue-at','-','--output',str(archive),url],check=True)
            hasher = hashlib.sha256()
            with archive.open('rb') as source:
                for chunk in iter(lambda:source.read(1024*1024), b''):
                    hasher.update(chunk)
            if hasher.hexdigest() != OLLAMA_ARCHIVE_SHA256:
                raise RuntimeError('Ollama official archive checksum mismatch')
            import zstandard
            import tempfile
            staging = Path(tempfile.mkdtemp(prefix='install-',dir=self.root))
            with archive.open('rb') as source, zstandard.ZstdDecompressor().stream_reader(source) as reader:
                with tarfile.open(fileobj=reader, mode='r|') as bundle:
                    for member in bundle:
                        target = (staging/member.name).resolve()
                        if not target.is_relative_to(staging.resolve()):
                            raise ValueError('archive member escapes local installation')
                        bundle.extract(member, staging, filter='data')
            if not (staging/'bin/ollama').is_file():
                raise RuntimeError('Ollama archive lacks the expected executable')
            (staging/'verified.json').write_text(json.dumps(dict(archive_sha256=OLLAMA_ARCHIVE_SHA256)))
            if installation.exists():
                installation.rename(self.root/f'incomplete-installation-{time.time_ns()}')
            staging.rename(installation)
        environment = os.environ.copy()
        environment.update(OLLAMA_HOST='127.0.0.1:11434', OLLAMA_MODELS=str(self.root/'models'),
                           CUDA_VISIBLE_DEVICES='0,1', OLLAMA_SCHED_SPREAD='1',
                           OLLAMA_NUM_PARALLEL='1', OLLAMA_MAX_LOADED_MODELS='1',
                           OLLAMA_CONTEXT_LENGTH='4096', OLLAMA_NO_CLOUD='1')
        self.log = (self.log_dir/'ollama_server.log').open('a',encoding='utf-8')
        self.server = subprocess.Popen([str(binary),'serve'],env=environment,stdout=self.log,stderr=subprocess.STDOUT)
        try:
            for _ in range(60):
                if self.server.poll() is not None:
                    raise RuntimeError('Ollama server exited; see ollama_server.log')
                try:
                    version = request_json('/api/version', timeout=2)
                    break
                except Exception:
                    time.sleep(1)
            else:
                raise RuntimeError('Ollama server did not become ready')
            if version.get('version') != OLLAMA_VERSION:
                raise RuntimeError(f'Ollama version mismatch: {version}')
            request_json('/api/pull',dict(model=QWEN_MODEL,stream=False),timeout=3600)
            tags = request_json('/api/tags')['models']
            actual = next(row for row in tags if row['name'] == QWEN_MODEL)
            if actual['digest'].removeprefix('sha256:') != QWEN_DIGEST:
                raise RuntimeError('Ollama Qwen tag digest changed from the registered producer')
            self.provenance = dict(backend='ollama',version=version,model=QWEN_MODEL,digest=QWEN_DIGEST,
                                   environment={key:environment[key] for key in ('CUDA_VISIBLE_DEVICES','OLLAMA_SCHED_SPREAD','OLLAMA_NUM_PARALLEL')},
                                   gpu_before=gpu_snapshot(),model_details=request_json('/api/show',dict(model=QWEN_MODEL)))
            (self.log_dir/'ollama_producer.json').write_text(json.dumps(self.provenance,indent=2),encoding='utf-8')
        except Exception:
            self.close()
            raise

    def assess(self, image):
        image = image.convert('RGB').copy()
        image.thumbnail((512,512))
        buffer = BytesIO()
        image.save(buffer,format='JPEG',quality=95)
        prompt = (
            'Assess visible disaster evidence only; do not invent urgency, casualties, location or time. '
            'Return JSON {"tasks":{task_name:{"assessable":true/false,"relevance_grade":0..3,'
            '"evidence_grade":0..3,"visible_evidence":[short strings]}}}. '
            'Include all three tasks: building_damage, road_damage, structural_flooding. '
            'Relevance:0=none,1=context,2=partial,3=direct. Building evidence:0=no damage supported,'
            '1=uncertain,2=affected below minor,3=minor or greater. Road evidence:0=no damage supported,'
            '1/2=uncertain/partial,3=visible road damage. Flooding:0=none,1/2=nonstructural or uncertain,'
            '3=visible structural flooding. Set assessable=false where unsupported. At most two evidence strings per task.'
        )
        messages = [dict(role='user',content=prompt,images=[base64.b64encode(buffer.getvalue()).decode()])]
        started, attempts, responses = time.perf_counter(), [], []
        for _ in range(2):
            response = request_json('/api/chat',dict(model=QWEN_MODEL,messages=messages,stream=False,
                format='json',think=False,keep_alive='10m',options=dict(temperature=0,seed=20261022,num_predict=512,num_ctx=4096)))
            text = response['message']['content']
            attempts.append(text)
            counts = {key:response.get(key) for key in ('prompt_eval_count','eval_count','total_duration','load_duration','prompt_eval_duration','eval_duration')}
            counts['arithmetic_estimate'] = qwen_decoder_matrix_estimate(response.get('prompt_eval_count',0),response.get('eval_count',0))
            responses.append(counts)
            try:
                assessment = parse_qwen_assessment(text)
                gpu = gpu_snapshot()
                (self.log_dir/'ollama_gpu_after_inference.json').write_text(json.dumps(gpu,indent=2),encoding='utf-8')
                return dict(status='ok',annotation_tier='vlm_proposed',assessment=assessment,raw_attempts=attempts,
                            seconds=time.perf_counter()-started,backend='ollama',model_id=QWEN_MODEL,
                            model_digest=QWEN_DIGEST,input_resolution=list(image.size),inference=responses,gpu=gpu)
            except (ValueError,TypeError,KeyError):
                messages += [dict(role='assistant',content=text),dict(role='user',content='Repair to the requested JSON schema. JSON only.')]
        return dict(status='invalid_output',annotation_tier='unavailable',raw_attempts=attempts,
                    seconds=time.perf_counter()-started,backend='ollama',model_id=QWEN_MODEL,inference=responses)

    def close(self):
        if hasattr(self,'server') and self.server.poll() is None:
            self.server.terminate()
            try:
                self.server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server.kill()
        if hasattr(self,'log'):
            self.log.close()
