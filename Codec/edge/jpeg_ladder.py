"""JPEG quality-ladder fallback codec (Pillow). Same table fields as codec_table; STS unavailable."""
import io

QUALITIES = (10, 20, 40, 60, 80)


def encode_jpeg(image, quality):
    """image: PIL.Image or HxWx3 uint8 array -> JPEG bytes."""
    from PIL import Image
    if not hasattr(image, "save"):
        image = Image.fromarray(image)
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue()


def decode_jpeg(payload):
    from PIL import Image
    return Image.open(io.BytesIO(payload)).convert("RGB")


def ladder_rows(items, judge_fn, qualities=QUALITIES):
    """items: iterable of (image_id, image). judge_fn(original, recon)->0/1. One row per (image, quality)."""
    rows = []
    for image_id, image in items:
        for q in qualities:
            payload = encode_jpeg(image, q)
            rows.append({"image_id": image_id, "payload_bytes": len(payload), "sts": [], "sts_available": False,
                         "fidelity": int(judge_fn(image, decode_jpeg(payload))), "codec": "jpeg", "quality": int(q)})
    return rows
