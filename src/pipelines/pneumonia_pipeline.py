"""Pneumonia image inference pipeline with Grad-CAM explainability."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config.paths import resolve_path
from src.explainability.explanation_rules import get_confidence_band
from src.explainability.gradcam_service import find_first_existing_fallback, generate_gradcam
from src.models.pneumonia_model_loader import load_pneumonia_artifacts
from src.preprocessing.pneumonia_preprocess import preprocess_pneumonia_image
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class PneumoniaPredictionResult:
    label: str
    probability: float
    confidence_band: str
    positive_class_probability: float
    gradcam_success: bool
    gradcam_overlay: np.ndarray | None
    gradcam_heatmap: np.ndarray | None
    fallback_image_path: Path | None
    original_image: np.ndarray


def _extract_prediction(probabilities: np.ndarray, decision_threshold: float = 0.5) -> tuple[int, float, float]:
    """Normalize model output to class index, selected class probability, and pneumonia probability."""
    if probabilities.ndim != 2 or probabilities.shape[0] != 1:
        raise ValueError(f"Unexpected prediction shape: {probabilities.shape}")

    if probabilities.shape[1] == 1:
        pneumonia_prob = float(probabilities[0, 0])
        threshold = float(np.clip(decision_threshold, 0.0, 1.0))
        class_index = int(pneumonia_prob >= threshold)
        selected_prob = pneumonia_prob if class_index == 1 else 1.0 - pneumonia_prob
        return class_index, selected_prob, pneumonia_prob

    if probabilities.shape[1] >= 2:
        class_index = int(np.argmax(probabilities[0]))
        selected_prob = float(probabilities[0, class_index])
        pneumonia_prob = float(probabilities[0, 1])
        return class_index, selected_prob, pneumonia_prob

    raise ValueError("Model prediction output has unsupported format.")


def predict_pneumonia(file_bytes: bytes) -> PneumoniaPredictionResult:
    """Run pneumonia prediction and generate Grad-CAM with fallback logic."""
    artifacts = load_pneumonia_artifacts()
    model = artifacts["model"]
    task = artifacts["task_manifest"]

    if model is None:
        # Module is in degraded mode (TF missing or model file absent).
        # We still decode + preprocess the image so the UI can show the upload.
        image_size = tuple(task["image_size"])
        original_image, _ = preprocess_pneumonia_image(
            file_bytes, image_size=image_size)
        fallback_paths = [
            resolve_path(path)
            for path in task.get("static_figures", {}).get("gradcam_fallback", [])
        ]
        fallback_path = find_first_existing_fallback(fallback_paths)
        reason = artifacts.get("degraded_reason", "model_unavailable")
        LOGGER.warning(
            "Pneumonia module running degraded: %s. Reason=%s",
            reason,
            "tensorflow_not_installed"
            if reason == "tensorflow_not_installed"
            else "model_file_missing_or_invalid",
        )
        return PneumoniaPredictionResult(
            label="Unavailable",
            probability=0.0,
            confidence_band="unknown",
            positive_class_probability=0.0,
            gradcam_success=False,
            gradcam_overlay=None,
            gradcam_heatmap=None,
            fallback_image_path=fallback_path,
            original_image=original_image,
        )

    image_size = tuple(task["image_size"])
    decision_threshold = float(task.get("decision_threshold", 0.5))

    original_image, model_batch = preprocess_pneumonia_image(
        file_bytes, image_size=image_size)
    probabilities = model.predict(model_batch, verbose=0)
    class_index, selected_prob, pneumonia_prob = _extract_prediction(
        probabilities,
        decision_threshold=decision_threshold,
    )

    confidence_band = get_confidence_band(selected_prob)
    label = task["labels"][class_index]

    LOGGER.info("Starting Grad-CAM generation for current pneumonia prediction.")
    gradcam = generate_gradcam(
        model=model,
        preprocessed_batch=model_batch,
        original_image=original_image,
        class_index=class_index,
    )

    fallback_paths = [resolve_path(
        path) for path in task["static_figures"]["gradcam_fallback"]]
    fallback_path = None
    overlay = None
    heatmap = None

    if gradcam["success"]:
        overlay = gradcam["overlay"]
        heatmap = gradcam["heatmap"]
        gradcam_success = True
        LOGGER.info(
            "Grad-CAM generation succeeded for current pneumonia prediction.")
    else:
        gradcam_success = False
        fallback_path = find_first_existing_fallback(fallback_paths)
        if fallback_path is not None:
            LOGGER.warning(
                "Grad-CAM generation failed; using fallback image %s. Error=%s",
                fallback_path.name,
                gradcam.get("error"),
            )
        else:
            LOGGER.warning(
                "Grad-CAM generation failed and no fallback image is available. Error=%s",
                gradcam.get("error"),
            )

    LOGGER.info(
        "Pneumonia prediction complete: label=%s, selected_probability=%.4f, pneumonia_probability=%.4f, decision_threshold=%.3f",
        label,
        selected_prob,
        pneumonia_prob,
        decision_threshold,
    )
    return PneumoniaPredictionResult(
        label=label,
        probability=float(np.clip(selected_prob, 0.0, 1.0)),
        confidence_band=confidence_band,
        positive_class_probability=float(np.clip(pneumonia_prob, 0.0, 1.0)),
        gradcam_success=gradcam_success,
        gradcam_overlay=overlay,
        gradcam_heatmap=heatmap,
        fallback_image_path=fallback_path,
        original_image=original_image,
    )
