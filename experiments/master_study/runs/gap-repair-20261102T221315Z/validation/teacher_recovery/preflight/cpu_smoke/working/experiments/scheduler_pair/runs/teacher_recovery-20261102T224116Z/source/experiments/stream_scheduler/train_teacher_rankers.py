"""Train parameter-only students; every requested arm receives a durable outcome."""
import argparse
import hashlib
import importlib.util
import json
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("boot", Path(__file__).with_name("runtime_bootstrap.py"))
boot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boot)
boot.install(ROOT)
import numpy as np
from Scheduler.semantic_learning_policy import NumpyRanker
from Scheduler.training.semantic_learning_models import fit_supervised
from experiments.stream_scheduler.teacher_pipeline import accepted_samples

STUDENT_ARMS = ("linear_choice", "mlp_choice", "ranknet", "deepsets_ranknet",
                "set_transformer_ranknet", "temporal_deepsets_ranknet", "visionprefer")


def agreement(artifact, samples):
    correct = []
    policy = NumpyRanker(artifact)
    for sample in samples:
        score = policy.raw(sample.snapshot)
        target = sample.pairs[0][2]
        if target == .5:
            continue
        difference = float(score[0] - score[1])
        # An exact predicted tie is not a correct strict preference for either side.
        correct.append(float(difference > 0 if target == 1. else difference < 0))
    if not correct:
        raise ValueError("no strict held-out preference support")
    value = float(np.mean(correct))
    return value, {"strict_pairs": len(correct), "silver_agreement": value}


def train_students(prepared, teacher_output, output, *, seeds=(1729,),
                   device="cpu", epochs=50, model_seconds=180, resume=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=resume)
    ledger = {arm + "/" + str(seed): {"status": "not_started"} for seed in seeds for arm in STUDENT_ARMS}
    if resume and (output/"status.json").exists():
        ledger.update(json.loads((output/"status.json").read_text()))
    def persist():
        (output / "status.json").write_text(json.dumps(ledger, indent=2))
    persist()
    try:
        identity={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in
          (("contexts",Path(prepared)/"contexts.pickle"),("requests",Path(prepared)/"requests.jsonl"),("responses",Path(teacher_output)/"responses.jsonl"))}
        binding=output/"input_identity.json"
        if resume and (not binding.exists() or json.loads(binding.read_text())!=identity):
            raise ValueError("P resume input identity mismatch")
        binding.write_text(json.dumps(identity,indent=2))
        cache, roles, aspects = accepted_samples(prepared, teacher_output, report_output=output/'label_audit')
    except Exception as error:
        (output / "label_error.txt").write_text(traceback.format_exc())
        for key in ledger:
            ledger[key] = {"status": "blocked_labels", "reason": str(error)}
        persist()
        return ledger
    strict = lambda xs: sum(sample.pairs[0][2] != .5 for sample in xs)
    for seed in seeds:
        for arm in STUDENT_ARMS:
            key = arm + "/" + str(seed)
            dest = output / arm / str(seed)
            if resume and ledger[key].get("status","").startswith("evaluated"):continue
            data = aspects if arm == "visionprefer" else roles
            support = {"strict_train": strict(data["train"]),
                       "strict_validation": strict(data["validation"])}
            if arm=="visionprefer":support["label_variant"]="VisionPrefer-inspired partial-common-aspect adaptation"
            if support["strict_train"] < 16 or support["strict_validation"] < 8:
                ledger[key] = {"status": "blocked_labels", **support,
                               "reason": "require at least16 strict train and8 strict validation preferences"}
                persist()
                continue
            try:
                model, history = fit_supervised(
                    data["train"], cache["preprocessing"], arm, dest,
                    lambda artifact: agreement(artifact, data["validation"]),
                    seed=seed, epochs=epochs, seconds=model_seconds, task="P", device=device,
                    resume=dest/"resume.pt" if resume and (dest/"resume.pt").exists() else None)
                ledger[key] = {"status": "partial_budget" if model["source"]["stop"] == "partial_budget"
                               else "evaluated_silver_validation", **support,
                               "validation": agreement(model, data["validation"])[1],
                               "final_test_opened": False}
            except Exception as error:
                dest.mkdir(parents=True, exist_ok=True)
                (dest / "error.txt").write_text(traceback.format_exc())
                ledger[key] = {"status": "failed", "error": str(error), **support}
            persist()
            print("P_STUDENT", key, ledger[key], flush=True)
    return ledger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--teacher-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--model-seconds", type=float, default=None)
    parser.add_argument("--seeds", default="1729,2718,31415")
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args()
    train_students(args.prepared, args.teacher_output, args.output,
                   seeds=tuple(map(int, args.seeds.split(","))), device=args.device,
                   epochs=args.epochs, model_seconds=args.model_seconds,resume=args.resume)


if __name__ == "__main__":
    main()
