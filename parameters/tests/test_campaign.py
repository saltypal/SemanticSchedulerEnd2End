import json
import logging
from pathlib import Path
from io import StringIO
import pytest
from PIL import Image
from parameters.contracts import digest
from parameters.exposure import apply_ledger,fingerprint
from parameters.cloud_notebook import CellRecorder
from parameters.artifacts import Tee


def test_tee_survives_stale_closed_stage_stream(tmp_path):
    terminal = StringIO()
    stale_path = tmp_path / 'closed-stage.log'
    with stale_path.open('w', encoding='utf-8') as stale_file:
        stale_stream = Tee(terminal, stale_file)
        stale_stream.write('before close\n')

    current_log = StringIO()
    current_stream = Tee(stale_stream, current_log)
    current_stream.write('after close\n')
    current_stream.flush()

    assert terminal.getvalue() == 'before close\nafter close\n'
    assert current_log.getvalue() == 'after close\n'


def test_logger_retains_stream_after_notebook_stage_closes(tmp_path, capsys):
    terminal = StringIO()
    logger = logging.Logger('retained-stage-stream')
    with (tmp_path / 'stage.log').open('w', encoding='utf-8') as stage_file:
        stream = Tee(terminal, stage_file)
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)
        logger.warning('during stage')

    # Transformers keeps the handler after the capture context has exited.
    # Python logging otherwise swallows the write error and prints a traceback.
    logger.warning('after stage')
    handler.flush()
    assert terminal.getvalue() == 'during stage\nafter stage\n'
    assert (tmp_path / 'stage.log').read_text(encoding='utf-8') == 'during stage\n'
    assert capsys.readouterr().err == ''


def test_exposure_propagates_through_duplicate_components(tmp_path):
    path=tmp_path/'image.png';Image.new('RGB',(10,10),'red').save(path)
    entry={'image_id':'seen',**fingerprint(path)}
    ledger={'schema':'better_exposure_v1','entries':[entry],'scope':'fixture'}
    ledger['sha256']=digest(ledger)
    rows=[{'image_id':'seen','image_path':str(path),'source':'ladi','group':'g'},
          {'image_id':'derivative','image_path':str(path),'source':'medic','group':'g'},
          {'image_id':'other','image_path':str(path),'source':'medic','group':'h'}]
    report=apply_ledger(rows,ledger)
    assert [r['development_exposed'] for r in rows]==[True,True,False]
    assert report['excluded_parents']==2
    ledger['scope']='tampered'
    with pytest.raises(ValueError):apply_ledger(rows,ledger)


def test_notebook_logs_and_failure_are_preserved(tmp_path):
    recorder=CellRecorder(['print(1)','raise ValueError()'],tmp_path)
    with recorder.cell(0):print('real cell output')
    with pytest.raises(ValueError):
        with recorder.cell(1):raise ValueError('expected fixture failure')
    status=json.loads((tmp_path/'cell_status.json').read_text())
    assert status['0']['status']=='complete' and status['1']['status']=='failed'
    import nbformat
    notebook=nbformat.read(tmp_path/'executed.ipynb',as_version=4)
    assert notebook.cells[0].execution_count==1
    assert 'real cell output' in notebook.cells[0].outputs[0].text
    assert notebook.cells[1].outputs[-1].ename=='ValueError'
