"""Construct and display semantic-header sidecars without changing Stage 1A packets."""

from __future__ import annotations

from typing import Any

from INFRA.Artifacts import LatentArtifact, SemanticPacketArtifact, SemanticParameterBundle, TransmissionArtifact
from SemParam.headers.semantic import SemanticHeaderCodec


def build_semantic_packet(
    bundle: SemanticParameterBundle,
    *,
    codec: SemanticHeaderCodec | None = None,
    native_latent: LatentArtifact | None = None,
    communication_vq_packet: TransmissionArtifact | None = None,
) -> SemanticPacketArtifact:
    active_codec = codec or SemanticHeaderCodec()
    header_bytes = active_codec.encode(bundle)
    return SemanticPacketArtifact(
        header_bytes=header_bytes,
        parameter_bundle=bundle,
        native_latent=native_latent,
        communication_vq_packet=communication_vq_packet,
        header_schema_version=1,
        metadata={"semantic_header_is_logical_sidecar": True},
    )


def _shape_and_preview(values: Any, limit: int = 8) -> tuple[str, str]:
    shape = tuple(int(item) for item in getattr(values, "shape", ()))
    detached = getattr(values, "detach", None)
    if callable(detached):
        values = detached().cpu().reshape(-1).tolist()
    else:
        reshape = getattr(values, "reshape", None)
        values = reshape(-1).tolist() if callable(reshape) else list(values)
    preview = ", ".join(f"{float(item):.4g}" for item in values[:limit])
    return str(shape), preview


def format_semantic_packet(packet: SemanticPacketArtifact, codec: SemanticHeaderCodec | None = None) -> str:
    """Return a bounded, ordered report suitable for the CLI and notebooks."""

    active_codec = codec or SemanticHeaderCodec()
    decoded = active_codec.decode(packet.header_bytes)
    lines = ["SEMANTIC HEADER", f"  bytes={len(packet.header_bytes)} bits={packet.semantic_header_bits}"]
    for field in decoded.fields:
        value = "unavailable" if not field.available else f"{field.value:.6f} {field.units or ''}".rstrip()
        lines.append(f"  {field.name}: {value} [{field.producer_level.value}]")
    lines.append(f"  hex_preview={packet.header_bytes.hex()[:96]}")

    lines.append("NATIVE LATENT PAYLOAD")
    if packet.native_latent is None:
        lines.append("  unavailable: no Stage 1A latent was attached")
    else:
        shape, preview = _shape_and_preview(packet.native_latent.tensor)
        lines.append(f"  shape={shape} active_real_values={packet.native_latent.active_real_values()}")
        lines.append(f"  preview=[{preview}]")

    lines.append("COMMUNICATION VQ PAYLOAD")
    if packet.communication_vq_packet is None:
        lines.append("  unavailable: no Stage 1A communication VQ packet was attached")
    else:
        header = packet.communication_vq_packet.metadata.get("header", {})
        bits = packet.communication_vq_packet.payload_bits_per_image
        _, preview = _shape_and_preview(packet.communication_vq_packet.transmitted)
        lines.append(f"  codebook_id={header.get('codebook_id', '<unknown>')} payload_bits_per_image={bits}")
        lines.append(f"  index_preview=[{preview}]")
        lines.append("  note=communication VQ framing remains separate from the semantic header")
    return "\n".join(lines)


def print_semantic_packet(packet: SemanticPacketArtifact, codec: SemanticHeaderCodec | None = None) -> None:
    print(format_semantic_packet(packet, codec=codec))
