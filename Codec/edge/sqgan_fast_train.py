"""Faster copy of the Semantic_Lossless SQ-GAN training loop (src/sqgan_train.py ``execute``, pinned a370de1).

Same model (``sqgan_train.build``), data, losses, optimizers, schedulers, codebook EMA, checkpoint format and
log lines. Speed changes (each switchable in config/edge/sqgan_finetune.yaml ``train``):
  * reuse_forward: the discriminator pass reuses the generator pass's inputs/reconstructions instead of a second
    full forward (author loss detaches both, vqperceptual.py:274-275); the discriminator sees reconstructions from
    before the generator step (standard GAN practice). One forward => one codebook-EMA update per step.
  * amp: fp16 autocast + one GradScaler per optimizer (T4 tensor cores). Overflow steps are skipped by the scaler
    instead of raising; 50 consecutive skips raise.
  * one GPU->CPU sync for the finite-gradient check instead of one per parameter.
  * num_workers DataLoader workers with pinned memory (images were decoded on the main thread).
``evaluate`` is delegated to the upstream ``execute`` unchanged.
"""
import json
import time
from contextlib import nullcontext


def _finite_grads(params):
    import torch
    grads = [p.grad for p in params if p.grad is not None]
    return bool(torch.stack([torch.isfinite(g).all() for g in grads]).all()) if grads else True


def execute_fast(sqgan_train, stage, options, run):
    import torch
    from src.dataset import ParentImages
    from src.ema import ReplicaEMA
    from src.runtime import write_json
    if stage == "evaluate":
        return sqgan_train.execute(stage, options, run)
    model = sqgan_train.build(options)
    device = next(model.parameters()).device
    write_json(run / "model.json", model.metadata)
    pilot = stage == "pilot"
    if not pilot and not (run / "pilot_passed.json").exists():
        raise RuntimeError("SQ-GAN requires successful adversarial pilot before training")
    target = run / "pilot" if pilot else run
    target.mkdir(exist_ok=True)
    (target / "checkpoints").mkdir(exist_ok=True)
    checkpoint = target / "checkpoints" / "last.pt"
    if checkpoint.exists():
        raise FileExistsError("Refusing to overwrite SQ-GAN run")
    amp = bool(options.get("amp", True)) and device.type == "cuda"
    reuse = bool(options.get("reuse_forward", True))
    workers = int(options.get("num_workers", 4))
    dataset = ParentImages(options["manifest"], "train", [256, 512])
    batch = int(options.get("batch_size", 2))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=workers,
                                         pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    if options.get("data_parallel", True) and device.type == "cuda":
        if torch.cuda.device_count() < 2 or batch < 2:
            raise RuntimeError("SQ-GAN DataParallel requires two GPUs and batch>=2")
        parallel = torch.nn.DataParallel(model)
    else:
        parallel = model
    discriminator = list(model.core.loss.discriminator.parameters())
    params = list(model.parameters())
    lr = float(options.get("learning_rate", 1.25e-5))
    model.core.learning_rate, model.core.min_learning_rate = lr, 0.
    model.core.steps_per_epoch = len(loader)
    model.core.training_steps = len(loader) * int(options.get("epochs", 150))
    (opt_g, opt_d), scheduler_configs = model.core.configure_optimizers()
    schedulers = [c["scheduler"] if isinstance(c, dict) else c for c in scheduler_configs]
    scalers = [torch.amp.GradScaler("cuda", enabled=amp), torch.amp.GradScaler("cuda", enabled=amp)]
    ema = ReplicaEMA({"image": model.core.quantize.codebook, "semantic": model.core.quantize_semantic.codebook})
    write_json(target / "fast_loop.json", {"amp_fp16": amp, "reuse_forward_for_discriminator": reuse,
                                           "num_workers": workers, "source": "Codec/edge/sqgan_fast_train.py"})
    autocast = (lambda: torch.autocast("cuda", dtype=torch.float16)) if amp else nullcontext
    started = last_save = time.monotonic()
    steps = epoch = skipped = consecutive_skips = 0

    def save():
        tmp = checkpoint.with_suffix(".tmp")
        torch.save({"state_dict": model.state_dict(), "generator_optimizer": opt_g.state_dict(),
                    "discriminator_optimizer": opt_d.state_dict(), "step": steps, "epoch": epoch,
                    "schedulers": [s.state_dict() for s in schedulers], "scalers": [s.state_dict() for s in scalers],
                    "torch_rng": torch.get_rng_state(), "options": options, "resume_supported": False}, tmp)
        tmp.replace(checkpoint)

    def done():
        return (pilot and steps >= 2) or time.monotonic() - started >= float(options.get("max_seconds", 14400))

    def update(idx, optimizer, out):
        nonlocal skipped, consecutive_skips
        for p in discriminator:
            p.requires_grad_(idx == 1)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            loss, _ = model.core.loss(codebook_loss=out["qloss"].mean(), semantic=out["semantic"],
                                      inputs=out["inputs"], reconstructions=out["xrec"], optimizer_idx=idx,
                                      global_step=steps, last_layer=model.core.get_last_layer(), split="train")
        if not torch.isfinite(loss):
            if not amp:
                raise FloatingPointError("Nonfinite SQ-GAN objective")
            skipped += 1
            consecutive_skips += 1
            optimizer.zero_grad(set_to_none=True)
            return float("nan")
        scaler = scalers[idx]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if _finite_grads(params):
            consecutive_skips = 0
        elif not amp:
            raise FloatingPointError("Nonfinite SQ-GAN gradient")
        else:
            skipped += 1
            consecutive_skips += 1
        scaler.step(optimizer)            # skips the step itself when grads are inf/nan (amp)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        return float(loss.detach())

    for epoch in range(1, int(options.get("epochs", 150)) + 1):
        parallel.train()
        for item in loader:
            image = item["image"].to(device, non_blocking=True)
            losses = {}
            with autocast():
                out = parallel(image)
            losses["generator_loss"] = update(0, opt_g, out)
            counts = ema.update()
            if not {"image", "semantic"}.issubset(counts):
                raise RuntimeError("Missing image/semantic codebook EMA observations")
            if not reuse:
                with torch.no_grad(), autocast():
                    out = parallel(image)
                counts = ema.update()
            else:
                out = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in out.items()}
            losses["discriminator_loss"] = update(1, opt_d, out)
            del out
            if consecutive_skips >= 50:
                raise FloatingPointError("50 consecutive non-finite fp16 updates; rerun with train.amp: false")
            steps += 1
            event = {"stage": "adversarial", "epoch": epoch, "step": steps,
                     "elapsed_seconds": time.monotonic() - started, "amp_skipped_updates": skipped, **losses}
            with (target / "training.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event) + "\n")
            for s in schedulers:
                s.step()
            if steps <= 2 or steps % 10 == 0:
                rate = (time.monotonic() - started) / steps
                print("SQ-GAN step %d, discriminator loss %.6f, %.2f s/step, fp16 skips %d"
                      % (steps, losses["discriminator_loss"], rate, skipped), flush=True)
            if time.monotonic() - last_save >= 300:
                save()
                last_save = time.monotonic()
            if done():
                break
        save()
        if done():
            break
    if not steps:
        raise RuntimeError("No SQ-GAN updates")
    if pilot and steps < 2:
        raise RuntimeError("SQ-GAN pilot did not complete two adversarial updates")
    ema.close()
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"], strict=True)
    write_json(target / "training_summary.json", {"steps": steps, "epochs_entered": epoch,
               "elapsed_seconds": time.monotonic() - started, "strict_reload": True, "source_codec_only": True,
               "amp_skipped_updates": skipped, "teacher_cost_included_in_elapsed_time": True,
               "paper_convergence_reproduced": False, "fast_loop": True})
