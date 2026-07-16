"""Tests for pneumonia preprocessing utilities."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from src.pipelines.pneumonia_pipeline import _extract_prediction
from src.preprocessing.pneumonia_preprocess import preprocess_pneumonia_image


def test_pneumonia_image_preprocess_shape() -> None:
    random_image = np.random.randint(
        0, 255, size=(300, 280, 3), dtype=np.uint8)
    image = Image.fromarray(random_image)
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    original, preprocessed_batch = preprocess_pneumonia_image(
        buffer.getvalue(), image_size=(224, 224))

    assert original.ndim == 3
    assert original.shape[2] == 3
    assert preprocessed_batch.shape == (1, 224, 224, 3)


def test_pneumonia_image_preprocess_keeps_raw_pixel_scale() -> None:
    image = Image.fromarray(np.full((64, 64, 3), 180, dtype=np.uint8))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")

    _, batch = preprocess_pneumonia_image(
        buffer.getvalue(), image_size=(224, 224))

    assert batch.dtype == np.float32
    assert float(batch.min()) >= 0.0
    assert float(batch.max()) <= 255.0


def test_extract_prediction_single_output_respects_custom_threshold() -> None:
    probs = np.array([[0.80]], dtype=np.float32)
    class_index, selected_prob, pneumonia_prob = _extract_prediction(
        probs, decision_threshold=0.925)

    assert class_index == 0
    assert np.isclose(pneumonia_prob, 0.80, atol=1e-6)
    assert np.isclose(selected_prob, 0.20, atol=1e-6)

    probs_high = np.array([[0.96]], dtype=np.float32)
    class_index_hi, selected_prob_hi, pneumonia_prob_hi = _extract_prediction(
        probs_high, decision_threshold=0.925)

    assert class_index_hi == 1
    assert np.isclose(pneumonia_prob_hi, 0.96, atol=1e-6)
    assert np.isclose(selected_prob_hi, 0.96, atol=1e-6)
