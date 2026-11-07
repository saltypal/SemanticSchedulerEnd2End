"""Frozen-feature SPI training and portable NumPy inference.

CLI: python -m parameters.spi train --features features.npz --labels labels.json
     --output NEW_RUN [--epochs 50]
No automatic dataset/model downloads. Labels carry their source and rubric hash.
"""
from dataclasses import dataclass, asdict
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from .artifacts import Run, write_json
from .contracts import digest
from .spi_labels import LEVELS


def normalize(features):
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("finite N x D features required")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms <= 1e-10):
        raise ValueError("zero image embeddings are unavailable")
    return x / norms


def validate_levels(levels):
    """Explicit ordered subset; never infer output classes from validation labels."""
    levels = tuple(levels)
    if len(levels) < 2 or levels != tuple(n for n in LEVELS if n in levels):
        raise ValueError("SPI levels must be an ordered subset of at least two canonical levels")
    return levels


def metrics(probabilities, labels, levels=LEVELS):
    levels = validate_levels(levels)
    classes = len(levels)
    p = np.asarray(probabilities, float)
    y = np.asarray(labels)
    if p.shape != (len(y), classes) or not len(y) or y.dtype.kind not in "iu" or np.any((y < 0) | (y >= classes)):
        raise ValueError("nonempty configured-class probabilities and integer ordinal labels required")
    if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("invalid class probabilities")
    pred = p.argmax(1)
    confusion = np.zeros((classes, classes), int)
    np.add.at(confusion, (y, pred), 1)
    rows = {}
    for index, name in enumerate(levels):
        tp = int(confusion[index, index]); offered = int(confusion[index].sum()); selected = int(confusion[:, index].sum())
        rows[name] = {"support": offered, "precision": tp / selected if selected else None,
                      "recall": tp / offered if offered else None,
                      "f1": 2 * tp / (offered + selected) if offered + selected else None}
    q = np.eye(classes)[y]
    serious = np.array([levels[i] in ("High", "Critical") for i in y])
    supported=[row for row in rows.values() if row['support']]
    critical_index=levels.index('Critical') if 'Critical' in levels else None
    critical_mask=y==critical_index if critical_index is not None else np.zeros(len(y),dtype=bool)
    critical_bad=sum(levels[int(pred[i])] in ('Low','Medium') for i in np.flatnonzero(critical_mask))
    return {"n": len(y), "levels": list(levels), "ordinal_scale": "configured output order", "accuracy": float(np.mean(pred == y)),
            "balanced_accuracy":float(np.mean([r['recall'] for r in supported])),
            "macro_f1":float(np.mean([r['f1'] or 0. for r in rows.values()])),
            "brier_multiclass":float(np.square(p-q).sum(1).mean()),
            "critical_to_low_medium":critical_bad,
            "critical_to_low_medium_rate":critical_bad/int(critical_mask.sum()) if critical_mask.any() else None,
            "log_loss": float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean()),
            "rps": float(np.square(np.cumsum(p, axis=1)[:, :-1] - np.cumsum(q, axis=1)[:, :-1]).mean()),
            "ordinal_mae": float(np.abs(pred-y).mean()), "confusion_true_rows_predicted_columns": confusion.tolist(),
            "high_critical_underpredictions": int(np.sum(serious & (pred < y))), "per_class": rows}


class SPIClassifier:
    """No neural framework in exported inference. Higher ordinal index = higher SPI."""
    def __init__(self, artifact):
        if artifact.get("schema") not in ("better_spi_head_v1", "better_spi_head_v2", "better_spi_linear_v1"):
            raise ValueError("unsupported SPI artifact")
        self.levels = validate_levels(artifact.get("levels", ()))
        if artifact["schema"] == "better_spi_head_v1" and self.levels != LEVELS:
            raise ValueError("legacy SPI artifacts require four canonical levels")
        if artifact.get("sha256") != digest({k: v for k, v in artifact.items() if k != "sha256"}):
            raise ValueError("SPI artifact checksum mismatch")
        self.artifact = artifact
        self.linear=artifact['schema']=='better_spi_linear_v1'
        self.w2,self.b2=(np.asarray(artifact[k],np.float32) for k in ('w2','b2'))
        d=artifact['input_dimension']
        if self.linear:
            if not np.isfinite(self.w2).all() or not np.isfinite(self.b2).all() or (self.w2.shape,self.b2.shape)!=((len(self.levels),d),(len(self.levels),)):
                raise ValueError('invalid linear SPI weights')
            return
        self.w1,self.b1=(np.asarray(artifact[k],np.float32) for k in ('w1','b1'))
        h=artifact['hidden_dimension']
        if any(not np.isfinite(v).all() for v in (self.w1, self.b1, self.w2, self.b2)):
            raise ValueError("nonfinite SPI weights")
        if (self.w1.shape, self.b1.shape, self.w2.shape, self.b2.shape) != ((h,d),(h,),(len(self.levels),h),(len(self.levels),)):
            raise ValueError("SPI weight shape mismatch")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf8")))

    def predict_proba(self, features, feature_signature):
        if feature_signature != self.artifact["feature_signature"]:
            raise ValueError("SPI encoder/preprocessing signature mismatch")
        x = normalize(features)
        if x.shape[1] != self.artifact["input_dimension"]:
            raise ValueError("SPI feature dimension mismatch")
        if self.linear:
            logits=x @ self.w2.T+self.b2
        else:
            z = x @ self.w1.T + self.b1
            # torch.nn.GELU(approximate='tanh'), exactly the training activation.
            z = .5*z*(1+np.tanh(np.sqrt(2/np.pi)*(z+.044715*z**3)))
            logits = z @ self.w2.T + self.b2
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        return p/p.sum(1, keepdims=True)

    def predict(self, feature, feature_signature):
        p = self.predict_proba(np.asarray(feature).reshape(1,-1), feature_signature)[0]
        index = int(p.argmax())
        level = self.levels[index]
        canonical_index = LEVELS.index(level)
        return {"available": True, "level": level, "ordinal_index": canonical_index,
                "output_index": index, "supported_levels": list(self.levels),
                "scheduling_class": 3-canonical_index, "scores": dict(zip(self.levels, map(float, p))),
                "score_interpretation": "student model scores; not human-calibrated urgency",
                "rubric_sha256": self.artifact["rubric_sha256"], "model_sha256": self.artifact["sha256"]}


@dataclass(frozen=True)
class TrainingConfig:
    levels: tuple[str, ...] = LEVELS
    seed: int = 17
    epochs: int = 50
    patience: int = 8
    batch_size: int = 64
    learning_rate: float = .001
    weight_decay: float = .0001
    hidden_dimension: int = 128
    dropout: float = .1
    seconds: int = 600


def prepare(features_path, labels_path, levels=LEVELS):
    """Validate label units, join exactly, and reject group/hash leakage."""
    levels = validate_levels(levels)
    with np.load(features_path, allow_pickle=False) as cache:
        ids = list(map(str, cache["image_ids"].tolist()))
        x = normalize(cache["embeddings"])
        signature = str(cache["feature_signature"].item())
        hashes = list(map(str, cache["image_sha256"].tolist()))
    if len(ids) != len(set(ids)) or len(ids) != len(x) or not signature:
        raise ValueError("invalid feature cache identity")
    rows = json.loads(Path(labels_path).read_text(encoding="utf8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("label manifest must be a nonempty list")
    if len(hashes)!=len(ids) or len(set(hashes))!=len(hashes):raise ValueError("duplicate image content in feature cache")
    seen = set(); groups = {}; lookup = dict(zip(ids, range(len(ids)))); rubric_hashes = set(); sources = set()
    indices = {s: [] for s in ("train", "validation", "test")}; targets = {s: [] for s in indices}; excluded = []
    for r in rows:
        identity = r["image_id"]
        if identity in seen: raise ValueError("duplicate labelled image_id")
        seen.add(identity)
        split = r["split"]; group = r["group"]
        if split not in indices or not isinstance(group, str) or not group:
            raise ValueError("explicit split and nonempty duplicate/event group required")
        if group in groups and groups[group] != split: raise ValueError("group crosses splits")
        groups[group] = split
        if identity not in lookup: raise ValueError("label has no cached image feature")
        if r.get("image_sha256")!=hashes[lookup[identity]]:raise ValueError("label/image content hash mismatch")
        if split == "test":
            indices[split].append(lookup[identity])
            continue  # Do not inspect held-out label values or support during fitting.
        if r.get("label_source") not in ("vlm_pseudo", "human_image") or r.get("label_unit") != "whole_image":
            raise ValueError("SPI only accepts explicit whole-image human or pseudo labels; no message/damage remapping")
        if r.get("available") is not True:
            if r.get("level") is not None or not r.get("reason"): raise ValueError("invalid unavailable label")
            excluded.append({"image_id": identity, "reason": r["reason"]}); continue
        if r.get("level") not in LEVELS or not r.get("rubric_sha256"):
            raise ValueError("supported priority and rubric hash required")
        if r["level"] not in levels:
            raise ValueError("available label outside configured output classes; do not silently drop or remap it")
        rubric_hashes.add(r["rubric_sha256"]); sources.add(r["label_source"])
        indices[split].append(lookup[identity]); targets[split].append(levels.index(r["level"]))
    if len(rubric_hashes) != 1 or len(sources) != 1:
        raise ValueError("one rubric and one declared label-source type per experiment; do not pool incompatible targets")
    if len(indices["train"]) < len(levels) or set(targets["train"]) != set(range(len(levels))):
        raise ValueError("all configured SPI classes need actual training support; do not invent missing labels")
    if not indices["validation"]: raise ValueError("grouped validation labels required")
    return x, ids, indices, targets, {"feature_signature": signature, "rubric_sha256": next(iter(rubric_hashes)),
        "label_source": next(iter(sources)), "levels": list(levels), "excluded": excluded,
        "support": {s: ({"unopened_images":len(indices[s])} if s == "test" else
                         {n: targets[s].count(i) for i,n in enumerate(levels)}) for s in indices}}


def train(features_path, labels_path, output, config=None):
    import torch
    config = config or TrainingConfig()
    if (config.epochs <= 0 or config.patience <= 0 or config.batch_size <= 0 or config.hidden_dimension <= 0
            or not 0 <= config.dropout < 1 or not np.isfinite(config.learning_rate) or config.learning_rate <= 0):
        raise ValueError("invalid training settings")
    # Validate before creating a run; do not leave a success-looking empty artifact.
    x, ids, ix, ys, identity = prepare(features_path, labels_path, config.levels)
    run = Run(output, {**asdict(config), **identity, "loss": "unweighted_cross_entropy", "device": "cpu"})
    run.manifest["input_hashes"] = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (features_path,labels_path)}
    run.manifest["label_claim"] = "human-image evaluation" if identity["label_source"] == "human_image" else "teacher/rubric agreement; not human urgency accuracy"
    write_json(run.path/"manifest.json", run.manifest)
    torch.manual_seed(config.seed); torch.set_num_threads(1)
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1],config.hidden_dimension),
        torch.nn.GELU(approximate="tanh"),torch.nn.Dropout(config.dropout),torch.nn.Linear(config.hidden_dimension,len(identity["levels"])))
    optimizer = torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    xt = torch.from_numpy(x[ix["train"]]); yt = torch.tensor(ys["train"],dtype=torch.long)
    xv = torch.from_numpy(x[ix["validation"]]); yv = torch.tensor(ys["validation"],dtype=torch.long)
    rng = np.random.default_rng(config.seed); best = float("inf"); stale = 0; history = []; best_state = None
    with run.stage("train"):
        for epoch in range(config.epochs):
            run.budget(); model.train(); order = rng.permutation(len(xt)); loss_sum = 0.
            for start in range(0,len(order),config.batch_size):
                batch=order[start:start+config.batch_size]; optimizer.zero_grad()
                loss=torch.nn.functional.cross_entropy(model(xt[batch]),yt[batch])
                if not torch.isfinite(loss): raise FloatingPointError("nonfinite SPI loss")
                loss.backward(); optimizer.step(); loss_sum += loss.item()*len(batch)
            model.eval()
            with torch.inference_mode(): validation_loss=float(torch.nn.functional.cross_entropy(model(xv),yv))
            if not np.isfinite(validation_loss): raise FloatingPointError("nonfinite SPI validation loss")
            history.append({"epoch":epoch+1,"train_loss":loss_sum/len(xt),"validation_loss":validation_loss})
            write_json(run.path/"metrics/history.json",history)
            if validation_loss < best-1e-8:
                best=validation_loss; stale=0; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            else: stale+=1
            if epoch == 0 or (epoch+1)%5 == 0: print("SPI epoch",epoch+1,"validation CE",validation_loss,flush=True)
            if stale>=config.patience: break
        model.load_state_dict(best_state); model.eval()
        artifact={"schema":"better_spi_head_v2","levels":identity["levels"],"input_dimension":x.shape[1],
            "unsupported_levels":[n for n in LEVELS if n not in identity["levels"]],
            "hidden_dimension":config.hidden_dimension,"feature_signature":identity["feature_signature"],
            "rubric_sha256":identity["rubric_sha256"],"label_source":identity["label_source"],
            "target":"visible response attention under one fixed rubric", "human_calibrated":False,
            "w1":model[0].weight.detach().tolist(),"b1":model[0].bias.detach().tolist(),
            "w2":model[3].weight.detach().tolist(),"b2":model[3].bias.detach().tolist()}
        artifact["sha256"]=digest(artifact)
        write_json(run.path/"models/spi.json",artifact)
        exported=SPIClassifier.load(run.path/"models/spi.json")
        p=exported.predict_proba(x[ix["validation"]],identity["feature_signature"])
        with torch.inference_mode(): reference=model(xv).softmax(1).numpy()
        if not np.allclose(p,reference,rtol=1e-4,atol=1e-6): raise RuntimeError("exported SPI differs from training model")
        prior=np.bincount(ys["train"],minlength=len(identity["levels"]))/len(ys["train"])
        report={"student":metrics(p,np.asarray(ys["validation"]),identity["levels"]),
            "training_prior":metrics(np.tile(prior,(len(p),1)),np.asarray(ys["validation"]),identity["levels"]),
            "test_evaluated":False,"claim":run.manifest["label_claim"],"support":identity["support"],
            "export_max_absolute_difference":float(np.max(np.abs(p-reference))),
            "epochs_completed":len(history),"best_validation_loss":best}
        write_json(run.path/"metrics/validation.json",report)
        write_json(run.path/"status.json",{"status":"trained","model_sha256":artifact["sha256"],
            "scope":run.manifest["label_claim"],"final_test_opened":False})
    return exported, report


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="command",required=True)
    fit=sub.add_parser("train"); fit.add_argument("--features",required=True); fit.add_argument("--labels",required=True)
    fit.add_argument("--output",required=True); fit.add_argument("--epochs",type=int,default=50)
    fit.add_argument("--levels",nargs="+",choices=LEVELS,default=list(LEVELS),help="Explicit output order; never automatically inferred")
    args=parser.parse_args(); _,report=train(args.features,args.labels,args.output,TrainingConfig(epochs=args.epochs,levels=tuple(args.levels)))
    print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
