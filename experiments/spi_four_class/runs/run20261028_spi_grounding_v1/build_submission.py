import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from parameters.build_spi_pipeline_notebook import build
output = Path('experiments/spi_four_class/runs/run20261028_spi_grounding_v1/submission')
print(build(output, 'run20261028_spi_grounding_v1', server_seconds=3600, diagnostic=True))
