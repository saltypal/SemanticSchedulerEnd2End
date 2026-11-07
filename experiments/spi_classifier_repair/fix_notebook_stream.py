"""Wrapper-only v19 repair; preserve source ZIP/model-stage cache identities."""
import ast,hashlib,io,json,shutil,tempfile,time,types
from pathlib import Path
import nbformat

def build():
    root=Path(__file__).resolve().parents[2];exp=root/'experiments/spi_classifier_repair'
    old=exp/'runs/medic_training_20261028T214102Z';stage=exp/'runs'/('medic_training_streamfix_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()));stage.mkdir(parents=True)
    for name in ('source.zip','SOURCE_MANIFEST.json'):shutil.copy2(old/name,stage/name)
    nb=nbformat.read(old/'notebook.ipynb',as_version=4)
    source=nb.cells[1].source
    needle=" def write(self,s): self.stream.write(s);self.file.write(s);return len(s)"
    if source.count(needle)!=1:raise ValueError('unexpected notebook stream wrapper')
    source=source.replace(needle," def isatty(self): return False\n def __getattr__(self,name): return getattr(self.stream,name)\n"+needle)
    nb.cells[1].source=source
    # Regression executes the exact staged wrapper, not a different helper.
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='Tee')
    namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),'staged_Tee','exec'),namespace)
    with tempfile.TemporaryDirectory() as tmp:
        stream=io.StringIO();tee=namespace['Tee'](stream,Path(tmp)/'log')
        assert tee.isatty() is False;assert tee.encoding==stream.encoding
        # This reproduces the Transformers5.5 loading-report condition.
        def style(text,color):return f'color:{text}' if tee.isatty() else text
        assert style('UNEXPECTED:','orange')=='UNEXPECTED:'
        assert tee.write('loading report\n')==15;tee.flush();assert stream.getvalue()=='loading report\n';tee.file.close()
    for cell in nb.cells:
        if cell.cell_type=='code':compile(cell.source,'notebook','exec')
    nbformat.validate(nb);nbformat.write(nb,stage/'notebook.ipynb')
    nbformat.write(nb,exp/'notebooks/notebook9_medic_spi_training.ipynb')
    metadata=json.loads((old/'kernel-metadata.json').read_text())
    metadata['dataset_sources'].append('satyapaladugu/semanticscheduler-spi-resume-v19')
    (stage/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    (stage/'VALIDATION.json').write_text(json.dumps({'kernel':nb.metadata.kernelspec.name,'code_compiled':True,'nbformat':True,
        'exact_staged_Tee_regression_passed':True,'source_zip_identical_to_v19':hashlib.sha256((stage/'source.zip').read_bytes()).hexdigest()==hashlib.sha256((old/'source.zip').read_bytes()).hexdigest(),
        'change_scope':'notebook logging wrapper only; no model-stage recipe changes','remote_submitted':False},indent=2))
    print(stage);return stage

if __name__=='__main__':build()
