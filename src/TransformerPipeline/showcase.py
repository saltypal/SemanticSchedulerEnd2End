"""One compact figure for the sequential Stage 1B semantic-header demonstration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from INFRA.Artifacts import SemanticPacketArtifact, SemanticParameterBundle


@dataclass(frozen=True, slots=True)
class ShowcaseRecord:
    image_path: Path
    bundle: SemanticParameterBundle
    packet: SemanticPacketArtifact
    label: str


def _value_text(bundle: SemanticParameterBundle, name: str) -> str:
    parameter = bundle.values.get(name)
    if parameter is None or not parameter.available:
        return "unavailable"
    assert parameter.value is not None
    return f"{parameter.value:.3f}"


def plot_semantic_showcase(records: Iterable[ShowcaseRecord], output_path: str | Path | None = None):
    """Render image, parameters, header, and payload in aligned visual columns."""

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from PIL import Image

    items = list(records)
    if not items:
        raise ValueError("Showcase plotting requires at least one record.")
    figure = plt.figure(figsize=(5.4 * len(items), 13.5), constrained_layout=True)
    grid = GridSpec(4, len(items), figure=figure, height_ratios=(3.2, 2.6, 1.2, 1.3))
    score_names = ("freshness", "novelty", "relevance", "reliability", "urgency", "task_importance")
    for column, record in enumerate(items):
        image_axis = figure.add_subplot(grid[0, column])
        with Image.open(record.image_path) as image:
            image_axis.imshow(image.convert("RGB"))
        image_axis.set_title(record.label, fontsize=11, fontweight="bold")
        image_axis.axis("off")

        parameter_axis = figure.add_subplot(grid[1, column])
        scores = []
        labels = []
        for name in score_names:
            parameter = record.bundle.values.get(name)
            labels.append(name.replace("task_", "task "))
            scores.append(parameter.value if parameter is not None and parameter.available else 0.0)
        colors = ["#276FBF" if score > 0.0 else "#B9C0C9" for score in scores]
        positions = list(range(len(labels)))
        parameter_axis.barh(positions, scores, color=colors)
        parameter_axis.set_yticks(positions, labels, fontsize=8)
        parameter_axis.set_xlim(0.0, 1.0)
        parameter_axis.invert_yaxis()
        parameter_axis.grid(axis="x", alpha=0.25)
        aoi = _value_text(record.bundle, "AoI_seconds")
        parameter_axis.set_title(f"AoI_seconds: {aoi}", fontsize=10)
        for position, score, name in zip(positions, scores, score_names, strict=True):
            parameter = record.bundle.values.get(name)
            label = "unavailable" if parameter is None or not parameter.available else f"{score:.3f}"
            parameter_axis.text(min(score + 0.02, 0.83), position, label, va="center", fontsize=7)

        header_axis = figure.add_subplot(grid[2, column])
        header_axis.axis("off")
        header_axis.set_title("Semantic header", fontsize=10)
        header_axis.text(
            0.01,
            0.95,
            f"{len(record.packet.header_bytes)} bytes\n{record.packet.header_bytes.hex()[:72]}…",
            va="top",
            family="monospace",
            fontsize=8,
            wrap=True,
        )

        payload_axis = figure.add_subplot(grid[3, column])
        payload_axis.axis("off")
        payload_axis.set_title("Stage 1A payload summary", fontsize=10)
        latent = record.packet.native_latent
        vq = record.packet.communication_vq_packet
        latent_text = "native latent unavailable"
        if latent is not None:
            latent_text = f"latent shape={tuple(latent.tensor.shape)}\nactive values={latent.active_real_values()}"
        vq_text = "VQ packet unavailable"
        if vq is not None:
            vq_text = f"VQ bits/image={vq.payload_bits_per_image}\nindices shape={tuple(vq.transmitted.shape)}"
        payload_axis.text(0.01, 0.95, f"{latent_text}\n{vq_text}", va="top", fontsize=8, family="monospace")
    figure.suptitle("Stage 1B semantic header paired with Stage 1A payloads", fontsize=15, fontweight="bold")
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=180, bbox_inches="tight")
    return figure
