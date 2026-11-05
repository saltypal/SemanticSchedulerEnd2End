"""Incremental notebook/cell evidence and cloud stage entry points."""
from pathlib import Path
from contextlib import contextmanager,redirect_stdout,redirect_stderr
import json
import sys
import time
import traceback
from .artifacts import Tee,write_json


class CellRecorder:
    def __init__(self,cells,output):
        self.cells=cells;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.results={};self.flush()

    def flush(self):
        import nbformat as nbf
        notebook=nbf.v4.new_notebook()
        for index,source in enumerate(self.cells):
            result=self.results.get(index);cell=nbf.v4.new_code_cell(source)
            if result:
                cell.execution_count=index+1
                log=Path(result['log']);value=log.read_text(encoding='utf-8') if log.exists() else ''
                cell.outputs=[nbf.v4.new_output('stream',name='stdout',text=value)]
                if result.get('error'):cell.outputs.append(nbf.v4.new_output('error',ename=result['error'],evalue=result['message'],traceback=result['traceback']))
            notebook.cells.append(cell)
        nbf.write(notebook,self.output/'executed.ipynb')
        write_json(self.output/'cell_status.json',self.results)

    @contextmanager
    def cell(self,index):
        started=time.monotonic();path=self.output/('cell_'+str(index)+'.log')
        result={'status':'running','log':str(path),'seconds':None};self.results[index]=result;self.flush()
        try:
            with path.open('w',encoding='utf-8') as handle,redirect_stdout(Tee(sys.stdout,handle)),redirect_stderr(Tee(sys.stderr,handle)):
                print('CELL START',index,'log',path,flush=True)
                try:yield
                except BaseException as error:
                    result.update(status='failed',error=type(error).__name__,message=str(error),traceback=traceback.format_exc().splitlines())
                    traceback.print_exc();raise
                else:result['status']='complete'
                finally:print('CELL END',index,'seconds',time.monotonic()-started,flush=True)
        finally:
            result['seconds']=time.monotonic()-started;self.flush()


def locate(suffix):
    candidates=list(Path('/kaggle/input').glob('**/'+suffix))
    if len(candidates)!=1:raise ValueError('expected exactly one upstream '+suffix+'; found '+str(candidates))
    return candidates[0]


def execute(stage,source_root,output):
    root=Path(source_root);output=Path(output)
    if stage=='prepare':
        from .cloud_data import prepare
        result=prepare(output,maximum=10000,download_bytes=20_000_000_000,seconds=10800,
                       exposure_ledger=root/'exposure_ledger.json')
    elif stage=='extract':
        from .cloud_extract import extract
        result=extract(locate('data/records.json'),output,gpu_hours=7.5,batch_size=16)
    elif stage=='select':
        from .cloud_study import select_campaign
        from .extractor import VisualEvidence
        records=json.loads(locate('data/records.json').read_text())
        ledger=json.loads(locate('data/exposure_ledger.json').read_text())
        audit=json.loads(locate('data/exposure_audit.json').read_text())
        if audit['ledger_sha256']!=ledger['sha256']:raise ValueError('upstream exposure ledger differs')
        caches={}
        for backbone in ('clip','siglip'):
            cache={}
            for line in locate('features/'+backbone+'.jsonl').read_text().splitlines():
                row=json.loads(line);cache[row['image_id']]=VisualEvidence.from_dict(row['evidence'])
            if set(cache)!={r['image_id'] for r in records}:raise ValueError('feature cache ownership does not match data')
            caches[backbone]=cache
        result=select_campaign(records,caches,output,seconds=21600,
                               exposed_ids=[r['image_id'] for r in ledger['entries']])
    else:raise ValueError('unknown cloud stage')
    print('Stage finished:',stage,'artifact root:',result,flush=True)
    return result
