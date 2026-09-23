import subprocess
import sys
from INFRA.configuration import ROOT

def test_image_runtime_import_does_not_import_codec():
    code="import sys; import SemParam.runtime; assert not any(name == 'EncDec' or name.startswith('EncDec.') for name in sys.modules); assert 'torch' not in sys.modules"
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)

def test_codebook_import_does_not_load_swin():
    code="import sys; import SemParam.Codebooks.pipeline; assert not any(name.startswith('EncDec.Models') for name in sys.modules)"
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)
