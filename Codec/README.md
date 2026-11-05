# Codec

Image encoders, decoders, preprocessing, latent transport adapters and codec
training. Existing production source was reused from the preserved EncDec owner;
no codec was retrained or run during the reorganization.

Use `Models/`, `integration/` and `runtime.py`. Reconstruction metrics are in
`Evaluation/codec_metrics`. Original notebooks/tests remain in the archived
EncDec directory. The old directory is an exact source snapshot, not active code
ownership. Model assets remain in the root models/ and references/ directories.


## Active decoder-scheduler codec
SQ-GAN is the selected digital source codec: build_scheduler_codec() in integration/sqgan_payload.py reads config/scheduler_codec.json.
encode(rgb) can use the bundled frozen transmitter segmentation teacher; encode(rgb, semantic) accepts an explicit paired map.
decode(payload_bytes) needs no original RGB or semantic map. Native input/output is256x512 RGB.
The existing SQ01 experimental framing is reused from the verified semantic-codec source; it is not the paper entropy coder.
CRC detects corruption; it does not correct errors. No analog channel, SNR, retransmission or exact-pixel guarantee is implied.
The large shared checkpoint/codebooks are setup assets, not sent with each image.
