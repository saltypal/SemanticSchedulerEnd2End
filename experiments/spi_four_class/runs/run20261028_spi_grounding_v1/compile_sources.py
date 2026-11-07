from pathlib import Path
files=['experiments/spi_four_class/visual_reliability.py','parameters/build_annotation_notebook.py','parameters/build_spi_pipeline_notebook.py']
for f in files: compile(Path(f).read_text(encoding='utf-8'),f,'exec')
print({'compiled':files})
