"""Exercise the real training loop on CPU with a tiny test-only model."""
from types import SimpleNamespace
import pytest


def test_cpu_training_updates_weights_and_persists_checkpoint(tmp_path):
    torch = pytest.importorskip('torch')
    from PIL import Image
    from EncDec.Models.SwinJSCC.trainer import train_phase, TrainingPhaseConfig
    from EncDec.Models.SwinJSCC.training_utils import DataParallelPlan

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.tensor(0.5))

        def forward(self, images, snr, rate):
            return images * self.gain, None

    module = TinyModel()
    adapter = SimpleNamespace(config=SimpleNamespace(base_fixed_c=96), build_training_module=lambda: module)
    for directory in ('train', 'validation'):
        (tmp_path/directory).mkdir()
        Image.new('RGB', (128, 128), (200, 100, 50)).save(tmp_path/directory/'fixture.png')
    phase = TrainingPhaseConfig('base', epochs=1, batch_size_per_gpu=1, crop_size=128)
    result = train_phase(adapter, phase, tmp_path/'train', tmp_path/'validation', (10,), (96,),
                         DataParallelPlan(requested_gpu_count=0, device='cpu'), tmp_path/'output', resume=False)
    assert module.gain.item() > 0.5
    assert result.best_epoch == 0
    assert (tmp_path/'output/base_resume.pt').is_file()


def test_invalid_accelerator_fails():
    from INFRA.configuration import resolve_config
    with pytest.raises(ValueError, match='TPU'):
        resolve_config(overrides={'device': 'tpu'})
