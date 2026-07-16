"""Pneumonia image preprocessing for DenseNet121 inference."""

from __future__ import annotations

import numpy as np

from src.utils.image_utils import decode_uploaded_image, image_to_numpy, resize_image


def preprocess_pneumonia_image(file_bytes: bytes, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return original RGB image and model batch matching notebook training input scale.

    The exported DenseNet model already includes preprocess_input in its graph,
    so inference should pass raw RGB pixels (0-255) as float32.
    """
    pil_image = decode_uploaded_image(file_bytes)
    resized = resize_image(pil_image, image_size)

    original_array = image_to_numpy(pil_image)
    resized_array = image_to_numpy(resized)
    batch = np.expand_dims(resized_array, axis=0).astype(np.float32)
    return original_array, batch
