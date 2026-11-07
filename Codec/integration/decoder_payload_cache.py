"""Strict Swin received-latent caching before decoder scheduling, no fallback."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import traceback


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def image_seed(identity, seed, checkpoint_sha, snr_db, rate):
    value = json.dumps([str(identity), int(seed), checkpoint_sha, float(snr_db), int(rate)])
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'big') % (2**63-1)


def letterbox_image(path, size):
    from PIL import Image
    if type(size) is not int or size < 1:
        raise ValueError('positive fixed model input size required')
    with Image.open(path) as opened:
        if opened.getexif().get(274, 1) != 1:
            raise ValueError('transform noncanonical EXIF image and annotations before caching')
        image = opened.convert('RGB')
    width, height = image.size
    factor = min(size/width, size/height)
    new_width, new_height = max(1, round(width*factor)), max(1, round(height*factor))
    offset = ((size-new_width)//2, (size-new_height)//2)
    result = Image.new('RGB', (size, size), (0, 0, 0))
    result.paste(image.resize((new_width, new_height), Image.Resampling.BICUBIC), offset)
    return result, dict(source_size=[width, height], output_size=[size, size],
                        scale_x=new_width/width, scale_y=new_height/height,
                        offset_x=offset[0], offset_y=offset[1], operation='resize_and_pad_no_crop')


def transform_points(points, transform):
    return [(x*transform['scale_x']+transform['offset_x'],
             y*transform['scale_y']+transform['offset_y']) for x, y in points]


def synchronize(device):
    import torch
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def cache_received(adapter, rows, output, checkpoint_sha, device='cpu', snr_db=10., rate=96, seed=42,
                   execution_scope='measured_checkpoint_inference'):
    import numpy as np
    import torch
    from Evaluation.Artifacts import ImageArtifact
    from Evaluation.Channels.awgn import AWGNChannel
    from Scheduler.training.decoder_v3_study import write_json
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('received payload cache must be new or empty')
    if not math.isfinite(snr_db):
        raise ValueError('finite SNR required')
    output.mkdir(parents=True, exist_ok=True)
    size = tuple(adapter.config.initial_image_size)
    if len(size) != 2 or size[0] != size[1]:
        raise ValueError('square model input required')
    records, identities = [], set()
    for row in rows:
        identity, path = str(row['image_id']), Path(row['image_path']).resolve()
        if identity in identities:
            raise ValueError('duplicate source identity')
        identities.add(identity)
        print(f'Encode/cache received latent image={identity}', flush=True)
        image, geometry = letterbox_image(path, size[0])
        tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).float().to(device)/255.
        artifact = ImageArtifact(tensor=tensor, sample_ids=[identity], source='decoder_context')
        synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            latent = adapter.encode(artifact, snr_db=snr_db, rate=rate)
        synchronize(device)
        encode_seconds = time.perf_counter()-started
        local_seed = image_seed(identity, seed, checkpoint_sha, snr_db, rate)
        devices = [torch.device(device).index or 0] if str(device).startswith('cuda') else []
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.manual_seed(local_seed)
            if devices:
                torch.cuda.manual_seed_all(local_seed)
            transmission = AWGNChannel().transmit(latent, snr_db)
        name = hashlib.sha256(identity.encode()).hexdigest()
        payload, reference = output/f'{name}.pt', output/f'{name}_reference.png'
        torch.save(dict(tensor=latent.tensor.detach().cpu(),
                        rate_mask=None if latent.rate_mask is None else latent.rate_mask.detach().cpu(),
                        received=transmission.received.detach().cpu(), source_model=latent.source_model,
                        rate_tokens=latent.rate_tokens, metadata=latent.metadata,
                        channel_uses=transmission.channel_uses_per_image,
                        channel_metadata=transmission.metadata,
                        checkpoint_sha256=checkpoint_sha, snr_db=float(snr_db)), payload)
        image.save(reference)
        records.append(dict(image_id=identity, source_image=str(path), source_sha256=sha256_file(path),
                            received_payload=payload.name, payload_sha256=sha256_file(payload),
                            reference_image=reference.name, reference_sha256=sha256_file(reference),
                            geometry=geometry, seed=local_seed, snr_db=float(snr_db), rate=rate,
                            checkpoint_sha256=checkpoint_sha, encode_seconds=encode_seconds,
                            channel_uses=transmission.channel_uses_per_image,
                            latent_tensor_bytes=latent.tensor.numel()*latent.tensor.element_size(),
                            payload_file_bytes=payload.stat().st_size, digital_payload_bits=None,
                            execution_scope=execution_scope))
        write_json(output/'manifest.json', dict(schema='received_latent_cache_v1', complete=False, records=records))
    if not records:
        raise ValueError('no source images supplied')
    result = dict(schema='received_latent_cache_v1', complete=True, device=str(device), records=records,
                  channel_before_scheduling=True, execution_scope=execution_scope)
    write_json(output/'manifest.json', result)
    return result


def decode_cached(adapter, record, cache_root, output, checkpoint_sha, device='cpu'):
    import numpy as np
    import torch
    from PIL import Image
    from Evaluation.Artifacts import LatentArtifact, TransmissionArtifact
    root = Path(cache_root).resolve()
    payload = (root/record['received_payload']).resolve()
    reference_path = (root/record['reference_image']).resolve()
    if (not payload.is_relative_to(root) or not reference_path.is_relative_to(root)
            or record['checkpoint_sha256'] != checkpoint_sha
            or sha256_file(payload) != record['payload_sha256']
            or sha256_file(reference_path) != record['reference_sha256']):
        raise ValueError('received payload/reference or checkpoint mismatch')
    saved = torch.load(payload, map_location=device, weights_only=True)
    if saved['checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('embedded checkpoint mismatch')
    latent = LatentArtifact(saved['tensor'], saved['rate_mask'], saved['source_model'],
                            saved['rate_tokens'], saved['metadata'])
    transmission = TransmissionArtifact(saved['tensor'], saved['received'], 'awgn', saved['snr_db'],
                                        saved['channel_uses'], metadata=saved['channel_metadata'])
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        reconstruction = adapter.reconstruct_received(transmission, latent).tensor
    synchronize(device)
    elapsed = time.perf_counter()-started
    if reconstruction.shape[0] != 1 or not torch.isfinite(reconstruction).all():
        raise ValueError('invalid reconstruction')
    values = reconstruction[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    with Image.open(reference_path) as source:
        reference = np.asarray(source.convert('RGB'), dtype=np.float64)/255.
    if reference.shape != values.shape:
        raise ValueError('reconstruction geometry mismatch')
    mse = float(np.mean((values-reference)**2))
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(values*255).astype(np.uint8)).save(destination)
    return dict(image_id=record['image_id'], decode_seconds=elapsed,
                psnr_db=None if mse == 0 else float(-10*math.log10(mse)), exact_pixels=bool(mse == 0),
                reconstruction=str(destination), payload_sha256=record['payload_sha256'],
                execution_scope=record['execution_scope'], semantic_preservation='not_independently_assessed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('records', 'output', 'model-dir', 'upstream-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--limit', type=int, choices=(1, 5), default=1)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be new/empty')
    args.output.mkdir(parents=True, exist_ok=True)
    from Scheduler.training.decoder_v3_study import write_json
    import sys
    write_json(args.output/'preflight.json', dict(interpreter=sys.executable, python=sys.version,
                                                strict_runtime=True, device=args.device, limit=args.limit))
    try:
        from Codec.Models.SwinJSCC.model_loader import load_stage1_swinjscc
        print('Validate locked checkpoint, upstream source and runtime', flush=True)
        adapter = load_stage1_swinjscc(args.model_dir, device=args.device, upstream_root=args.upstream_root)
        for module in adapter._require_built():
            module.eval()
        checkpoint = sha256_file(args.model_dir/'stage1_swinjscc_full.pt')
        rows = json.loads(args.records.read_text(encoding='utf-8'))[:args.limit]
        received = cache_received(adapter, rows, args.output/'received', checkpoint, device=args.device)
        metrics = [decode_cached(adapter, record, args.output/'received', args.output/f'reconstruction_{i}.png',
                                 checkpoint, args.device) for i, record in enumerate(received['records'])]
        write_json(args.output/'decode_metrics.json', metrics)
        write_json(args.output/'status.json', dict(status='complete', actual_codec_executed=True, count=len(metrics)))
    except Exception as error:
        write_json(args.output/'failure.json', dict(type=type(error).__name__, message=str(error),
                                                   traceback=traceback.format_exc(), actual_codec_executed=False))
        raise


if __name__ == '__main__':
    main()
