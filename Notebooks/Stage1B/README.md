# Stage 1B semantic-header notebooks

`02_transformer_vs_codebook_semantic_header.ipynb` is the executable showcase. It processes the fixed sequence below in order so the local NetworkX baseline can establish history:

```text
BuildingFire.jpeg -> BuildingFire2.jpeg -> Mumbai-Drone-Photo-1.jpeg -> sunflower.jpeg
```

The notebook renders one aligned four-column figure: image, semantic parameters, semantic-header preview, and Stage 1A native-latent/communication-VQ payload summary. The header is a logical sidecar; it does not change existing Stage 1A VQ packet bit accounting.

The notebook also loads the optional Stage 1C semantic parameter-codebook artifact and compares its parameter values, full-series latency, CPU/GPU memory, artifact size, and semantic-header bytes with the Transformer path. It records limits explicitly: the codebook is demo-only unless trained and evaluated on a proper independent corpus.

Run the generator after changing its repository-backed notebook source:

```powershell
python scripts\build_stage1b_notebooks.py
```
