"""Single-pass export of the pre-mask SQ-GAN image-encoder grid.

The captured tensor is exactly what the frozen image encoder hands to the
masker, immediately before masking and VQ. Capture wraps the shared
``codec.encode`` path with a forward hook, so the payload bytes come from the
identical code path as a baseline encode and the encoder runs once for both
the payload and the grid. The grid stays a transmitter-side local value for
the current image; nothing here caches or transmits it.

The pinned bundle confirms the tensor shape at runtime; the handoff's
1x256x16x32 is the expected shape, not an assumption enforced here.
"""
import contextlib
from threading import get_ident


@contextlib.contextmanager
def prefeature_capture(codec):
    """Temporarily record the outputs of ``codec.model.encoder``.

    Yields ``(holder, calls)`` where ``holder['features']`` is a detached copy
    of the last encoder output and ``calls`` lists one shape entry per encoder
    invocation.
    """
    holder = {}
    calls = []
    calling_thread = get_ident()

    def record(module, inputs, output):
        if get_ident() != calling_thread:
            return
        calls.append(tuple(getattr(output, "shape", ())))
        holder["features"] = output.detach().clone() if hasattr(output, "detach") else output

    handle = codec.model.encoder.register_forward_hook(record)
    try:
        yield holder, calls
    finally:
        handle.remove()


def encode_with_features(codec, image, semantic=None):
    """Encode once and return both the SQ01 payload and the pre-mask grid.

    Returns ``{"payload", "features", "encoder_calls", "feature_shape",
    "feature_dtype"}``. Raises if the encoder did not run exactly once or did
    not produce a 4-D spatial grid.
    """
    with prefeature_capture(codec) as (holder, calls):
        payload = codec.encode(image, semantic)
    if len(calls) != 1:
        raise RuntimeError(
            "encode_with_features requires exactly one encoder call, observed %d"
            % len(calls))
    features = holder.get("features")
    if features is None or not hasattr(features, "shape"):
        raise RuntimeError("encoder produced no capture-able pre-mask grid")
    if len(features.shape) != 4:
        raise RuntimeError(
            "pre-mask grid must be a 4-D spatial tensor, got shape %r"
            % (tuple(features.shape),))
    return {
        "payload": payload,
        "features": features,
        "encoder_calls": len(calls),
        "feature_shape": [int(size) for size in features.shape],
        "feature_dtype": str(features.dtype),
    }
