import csv
from pathlib import Path
import pytest
from EncDec.Evaluation.kodak_benchmark import CSV_COLUMNS, EncodedSample
from EncDec.Evaluation.benchmark_report import summarize_compatible, plot_quality
from EncDec.Evaluation.audited_benchmark import benchmark_with_outcomes

def row(run='r', version='v'):
    value={key:None for key in CSV_COLUMNS}
    value.update(run_id=run,method='codec',method_version=version,operating_point='same',device='cpu',image_id='image',psnr_db=20,ssim=.8)
    return value

def test_versions_and_runs_never_average_together():
    assert len(summarize_compatible([row(),row('r2'),row(version='v2')]))==3
    with pytest.raises(ValueError,match='Duplicate'):
        summarize_compatible([row(),row()])

def test_plot_rejects_mixed_runs(tmp_path):
    with pytest.raises(ValueError,match='one run'):
        plot_quality([row(),row('r2')],tmp_path/'bad.png')

def test_decode_failure_is_not_quality_measurement(tmp_path):
    from PIL import Image
    path=tmp_path/'test.png'
    Image.new('RGB',(16,16)).save(path)
    class Broken:
        name='broken'; operating_point='test'; device='cpu'; version='v1'
        def complexity(self): return {}
        def encode(self,*args): return EncodedSample(None,payload_bits=10)
        def decode(self,*args): raise RuntimeError('CRC failed')
    rows,outcomes=benchmark_with_outcomes(Broken(),[path],run_id='test',project_root=tmp_path)
    assert rows[0]['psnr_db'] is None and rows[0]['ssim'] is None
    assert outcomes[0]['status']=='failed' and 'CRC' in outcomes[0]['error']
    assert set(rows[0])==set(CSV_COLUMNS)
