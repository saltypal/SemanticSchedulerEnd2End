"""Build the Colab/local parameter extraction demonstration notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "SemParam" / "notebooks" / "04_all_parameter_extraction_demo.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip("\n").splitlines(True),
    }


cells = [
    markdown(
        """
        # Semantic parameter extraction demo

        This notebook demonstrates the complete parameter contract on the three car/fire photographs. It follows `docs/PARAMETER_SPECIFICATION_V2.md`.

        It produces model-derived semantic evidence, context relevance and embedding novelty; rule-derived demonstration impact and urgency; and a transparent image-quality sufficiency proxy. Visible extent is represented as unavailable because this small demo does not include a validated target/effect segmentation model. The notebook does not claim calibrated probabilities or trained scheduler performance.

        Run locally when the repository images exist. In Google Colab, run the upload cell and upload the same three files.
        """
    ),
    code(
        """
        # Fresh Colab runtime: uncomment this installation line if needed.
        # !pip -q install transformers accelerate pandas matplotlib pillow opencv-python

        import os
        import json
        import hashlib
        from pathlib import Path
        from datetime import datetime, timezone

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from PIL import Image
        import cv2
        import torch
        from transformers import CLIPModel, CLIPProcessor

        SEED = 42
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print("device:", DEVICE)
        """
    ),
    code(
        """
        LOCAL_IMAGE_DIR = Path(r"D:/Bunker/BaseCamp/WirelessCommunication/SemanticComms/SemanticSchedulerEnd2End/data")
        EXPECTED_NAMES = ["car_Fire.jpg", "car_fire_extinguish.jpg", "car_fire_extinguish2.jpg"]
        image_paths = [LOCAL_IMAGE_DIR / name for name in EXPECTED_NAMES]

        if not all(path.exists() for path in image_paths):
            try:
                from google.colab import files
                print("Upload the three JPG files.")
                uploaded = files.upload()
                image_paths = [Path(name).resolve() for name in uploaded.keys()]
            except ImportError as exc:
                missing = [str(path) for path in image_paths if not path.exists()]
                raise FileNotFoundError("Missing local images and this is not Colab: " + str(missing)) from exc

        image_paths = sorted(image_paths, key=lambda p: p.name.lower())
        assert len(image_paths) == 3, image_paths
        print("images:")
        for path in image_paths:
            print(" -", path)
        """
    ),
    markdown(
        """
        ## Model and vocabulary

        The first implementation uses the open-source Hugging Face checkpoint `openai/clip-vit-base-patch32`. The same image embedding is reused for semantic evidence, context relevance and novelty. CLIP prompt scores are similarity evidence, not calibrated probabilities.
        """
    ),
    code(
        """
        MODEL_NAME = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

        SCENE_CONCEPTS = ["highway", "normal road", "residential street", "industrial site", "indoor room", "landscape", "vehicle scene"]
        ENTITY_CONCEPTS = ["car", "person", "building", "road", "smoke", "flame", "water", "debris", "traffic signal"]
        EVENT_CONCEPTS = [
            "normal scene with no visible emergency", "visible vehicle fire", "visible collision",
            "visible road obstruction", "visible flooding", "visible structural damage",
        ]
        RELATION_CONCEPTS = [
            "car on or beside a road", "flame or smoke overlapping a vehicle",
            "person near a vehicle", "debris on a road", "water over a road",
        ]
        TASK_TEXT = "identify visible vehicle fire or road hazard evidence for response planning"

        def normalise(x):
            return x / x.norm(dim=-1, keepdim=True)

        @torch.inference_mode()
        def encode_images(images):
            inputs = processor(images=images, return_tensors="pt").to(DEVICE)
            features = model.get_image_features(**inputs)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            if isinstance(features, tuple):
                features = features[0]
            return normalise(features).cpu().numpy()

        @torch.inference_mode()
        def encode_texts(texts):
            inputs = processor(text=texts, return_tensors="pt", padding=True).to(DEVICE)
            features = model.get_text_features(**inputs)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            if isinstance(features, tuple):
                features = features[0]
            return normalise(features).cpu().numpy()

        def prompt_scores(image_embedding, concepts):
            prompts = [f"a photograph showing {concept}" for concept in concepts]
            text_embeddings = encode_texts(prompts)
            scores = text_embeddings @ image_embedding
            exp_scores = np.exp((scores - scores.max()) / 0.07)
            proxy = exp_scores / exp_scores.sum()
            order = np.argsort(-scores)
            return [(concepts[i], float(scores[i]), float(proxy[i])) for i in order]
        """
    ),
    code(
        """
        images = [Image.open(path).convert("RGB") for path in image_paths]
        image_embeddings = encode_images(images)
        task_embedding = encode_texts([TASK_TEXT])[0]

        fig, axes = plt.subplots(1, len(images), figsize=(15, 4))
        for ax, image, path in zip(axes, images, image_paths):
            ax.imshow(image)
            ax.set_title(path.name)
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        print("embedding shape:", image_embeddings.shape)
        """
    ),
    code(
        """
        def image_quality_proxy(image):
            arr = np.asarray(image)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            bad_exposure = float(((gray <= 5) | (gray >= 250)).mean())
            sharpness_proxy = np.clip(laplacian_variance / 500.0, 0.0, 1.0)
            exposure_proxy = 1.0 - bad_exposure
            return {
                "blur_laplacian_variance": laplacian_variance,
                "exposure_bad_fraction": bad_exposure,
                "sufficiency_proxy": float(0.5 * sharpness_proxy + 0.5 * exposure_proxy),
            }

        def hash_image(path):
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]

        def nearest_history_distance(index, embeddings):
            if index == 0:
                return np.nan, 0, "no_history"
            similarities = embeddings[index] @ embeddings[:index].T
            max_similarity = float(np.max(similarities))
            return float((1.0 - max_similarity) / 2.0), index, "exact_cosine_history"

        def task_impact_rule(event_name, context_similarity):
            if event_name in {"visible vehicle fire", "visible collision"} and context_similarity > 0:
                return 3
            if event_name in {"visible road obstruction", "visible flooding"}:
                return 2
            return 1

        def urgency_rule(event_name):
            if event_name in {"visible vehicle fire", "visible collision"}:
                return 3
            if event_name in {"visible road obstruction", "visible flooding"}:
                return 2
            return 1

        def significance_proxy(event_probability, prior=0.20):
            # Binary toy task: report hazard or no hazard; unit error loss.
            prior_hazard = prior >= 0.5
            posterior_hazard = event_probability >= 0.5
            old_loss = event_probability if not prior_hazard else 1.0 - event_probability
            new_loss = event_probability if not posterior_hazard else 1.0 - event_probability
            return float(max(0.0, old_loss - new_loss))
        """
    ),
    code(
        """
        records = []
        for i, (path, image, embedding) in enumerate(zip(image_paths, images, image_embeddings)):
            scene_candidates = prompt_scores(embedding, SCENE_CONCEPTS)
            entity_candidates = prompt_scores(embedding, ENTITY_CONCEPTS)
            event_candidates = prompt_scores(embedding, EVENT_CONCEPTS)
            relation_candidates = prompt_scores(embedding, RELATION_CONCEPTS)
            context_similarity = float(embedding @ task_embedding)
            novelty, history_count, novelty_method = nearest_history_distance(i, image_embeddings)
            quality = image_quality_proxy(image)
            top_event, event_cosine, event_probability_proxy = event_candidates[0]
            records.append({
                "image": path.name,
                "image_hash": hash_image(path),
                "scene_id": scene_candidates[0][0],
                "scene_score_cosine": scene_candidates[0][1],
                "entity_1": entity_candidates[0][0],
                "entity_1_score": entity_candidates[0][1],
                "event_id": top_event,
                "event_score_cosine": event_cosine,
                "relation_1": relation_candidates[0][0],
                "relation_1_score": relation_candidates[0][1],
                "context_similarity": context_similarity,
                "novelty_distance": novelty,
                "history_count": history_count,
                "novelty_method": novelty_method,
                "extent_fraction": np.nan,
                "extent_basis": "unavailable_without_validated_masks",
                "impact_level": task_impact_rule(top_event, context_similarity),
                "impact_source": "demo_rule",
                "urgency_level": urgency_rule(top_event),
                "urgency_source": "demo_task_policy",
                "sufficiency_probability": np.nan,
                "sufficiency_proxy": quality["sufficiency_proxy"],
                "sufficiency_source": "uncalibrated_image_quality_proxy",
                "event_probability_proxy": event_probability_proxy,
                "significance_proxy": significance_proxy(event_probability_proxy),
                **quality,
            })

        results = pd.DataFrame(records)
        results
        """
    ),
    markdown(
        """
        ## Reading the output

        - Concept labels are categorical; their IDs are not priorities.
        - `context_similarity` is a raw image/task cosine.
        - `novelty_distance` uses the images earlier in this demo as delivered history.
        - Impact and urgency are demonstration rules, not learned visual predictions.
        - `extent_fraction` is unavailable without validated target/effect masks.
        - `sufficiency_proxy` is an image-quality proxy; calibrated sufficiency remains unavailable.
        - `significance_proxy` demonstrates the significance-paper arithmetic with an uncalibrated event proxy. It is not a project result.
        """
    ),
    code(
        """
        display_columns = [
            "image", "scene_id", "entity_1", "event_id", "relation_1",
            "context_similarity", "novelty_distance", "impact_level",
            "urgency_level", "extent_fraction", "sufficiency_probability",
            "sufficiency_proxy", "significance_proxy",
        ]
        results[display_columns].round(4)
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].bar(results["image"], results["context_similarity"])
        axes[0].set_title("Task relevance")
        axes[0].set_ylabel("CLIP cosine")
        axes[0].tick_params(axis="x", rotation=35)

        ordered = results.sort_values("significance_proxy", ascending=False)
        axes[1].bar(ordered["image"], ordered["significance_proxy"], color="darkorange")
        axes[1].set_title("Toy significance proxy ordering")
        axes[1].set_ylabel("proxy value")
        axes[1].tick_params(axis="x", rotation=35)
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        OUTPUT_DIR = Path(os.environ.get("SEMANTIC_RESULTS_DIR", "./semantic_demo_results"))
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        results.to_csv(OUTPUT_DIR / "parameter_extraction_demo.csv", index=False)
        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": MODEL_NAME,
            "device": DEVICE,
            "task": TASK_TEXT,
            "parameter_schema": "SemanticHeaderV2-demo",
            "extent_status": "unavailable_without_validated_masks",
            "sufficiency_status": "proxy_only_not_calibrated",
            "significance_status": "toy_proxy_not_measurement",
        }
        (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print("saved:", OUTPUT_DIR.resolve())
        """
    ),
    markdown(
        """
        ## Limitations and next step

        This is a compact extraction demonstration, not the final production extractor. CLIP prompt matching is a bootstrap method, not a substitute for detector validation, relation labels or calibration. The next full extractor should add YOLO-World boxes, selective SAM 2 masks, versioned task profiles, trained impact/urgency heads where labels justify them, and a calibrated answerability model.

        Tomorrow's RL notebook should consume context relevance, delivered-history novelty, task impact, urgency and task evidence sufficiency. Semantic evidence and visible extent remain outside that RL state as agreed. The significance-paper value should be computed only after defining the receiver belief, action set and loss table.
        """
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {OUTPUT} ({len(cells)} cells)")
