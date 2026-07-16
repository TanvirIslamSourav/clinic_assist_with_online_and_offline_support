"""Grad-CAM service for pneumonia model explainability."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import cm
from PIL import Image

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

try:
    import tensorflow as tf  # noqa: E402
except ImportError:  # Tensorflow is optional; pneumonia module runs degraded.
    tf = None  # type: ignore[assignment]


def _require_tf():
    """Return the lazily imported tf module or raise a friendly error."""
    if tf is None:
        raise RuntimeError(
            "TensorFlow is not installed; Grad-CAM is unavailable in this build."
        )
    return tf


def _find_gap_layer(model: Any) -> Any:
    """Find the GlobalAveragePooling2D layer before the model output."""
    tf = _require_tf()
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            return layer
    raise ValueError("No GlobalAveragePooling2D layer found in model.")


def generate_gradcam(
    model: Any,
    preprocessed_batch: np.ndarray,
    original_image: np.ndarray,
    class_index: int | None = None,
    alpha: float = 0.45,
) -> dict[str, Any]:
    """
    Generate Grad-CAM visualization matching the notebook implementation.

    Uses a proper Keras Model with connected outputs to ensure correct gradient flow.
    This avoids the disconnected computation graph issue in manual layer-by-layer passes.

    Args:
        model: Loaded Keras/TensorFlow model
        preprocessed_batch: Preprocessed image batch (1, 224, 224, 3)
        original_image: Original image for overlay (uint8 or similar)
        class_index: Target class index (0=Normal, 1=Pneumonia for sigmoid, or specific class for softmax)
        alpha: Overlay blending factor (0=full image, 1=full heatmap)

    Returns:
        Dict with success flag, heatmap, overlay, and raw_heatmap arrays
    """
    if tf is None:
        return {
            "success": False,
            "error": "tensorflow_not_installed",
        }
    try:
        # Step 1: Find GlobalAveragePooling2D layer (last conv output before pooling)
        gap_layer = _find_gap_layer(model)
        LOGGER.debug("Using GAP layer: %s", gap_layer.name)

        # Step 2: Create a gradient model with dual outputs:
        # [0] = feature maps before GlobalAveragePooling2D (shape: [1, H, W, C])
        # [1] = model output (shape: [1, num_classes] or [1, 1])
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[gap_layer.input, model.output],
        )

        # Step 3: Run forward pass and compute gradients within GradientTape context
        with tf.GradientTape() as tape:
            feature_maps, predictions = grad_model(
                preprocessed_batch, training=False
            )

            # Step 4: Extract target class probability for gradient computation
            if isinstance(predictions, (list, tuple)):
                predictions = predictions[0]
            predictions = tf.convert_to_tensor(predictions)

            # Handle different output shapes (sigmoid vs softmax)
            if predictions.shape[-1] == 1:
                # Sigmoid output: single channel
                loss = predictions[0, 0]
            else:
                # Softmax output or multi-class: select target class
                if class_index is None:
                    class_index = tf.argmax(predictions[0]).numpy().item()
                loss = predictions[0, class_index]

        # Step 5: Compute gradients of loss w.r.t. feature maps
        grads = tape.gradient(loss, feature_maps)
        if grads is None:
            raise ValueError("Gradients for Grad-CAM could not be computed.")

        # Step 6: Global Average Pooling of gradients (across spatial dimensions)
        # Result shape: (num_channels,) - one weight per feature map
        pooled_grads = tf.reduce_mean(
            grads, axis=(1, 2))  # Pool spatial dims (H, W)
        pooled_grads = pooled_grads[0]  # Remove batch dimension

        # Step 7: Weighted sum of feature maps
        # Multiply each channel by its gradient weight and sum
        feature_maps = feature_maps[0]  # Remove batch dimension
        heatmap = tf.reduce_sum(feature_maps * pooled_grads, axis=-1)

        # Step 8: ReLU to keep only positive contributions (excitation)
        heatmap = tf.maximum(heatmap, 0)

        # Step 9: Normalize to [0, 1] range
        heatmap_max = tf.reduce_max(heatmap)
        heatmap = heatmap / (heatmap_max + tf.keras.backend.epsilon())
        heatmap_np = heatmap.numpy()

        # Step 10: Resize heatmap to match original image dimensions
        height, width = original_image.shape[:2]
        heatmap_resized = tf.image.resize(
            heatmap_np[np.newaxis, ..., np.newaxis],
            (height, width)
        ).numpy().squeeze()

        # Step 11: Apply jet colormap to heatmap
        color_map = cm.get_cmap("jet")(heatmap_resized)[..., :3]

        # Step 12: Normalize base image to [0, 1]
        if original_image.dtype != np.float32:
            base_image = original_image.astype(np.float32) / 255.0
        else:
            base_image = original_image.copy()
            if base_image.max() > 1.0:
                base_image = base_image / 255.0

        # Step 13: Create overlay: weighted blend of image and heatmap
        overlay = np.clip(
            (1.0 - alpha) * base_image + alpha * color_map,
            0.0,
            1.0
        )

        # Convert to uint8 for display
        overlay_uint8 = np.uint8(overlay * 255.0)
        heatmap_uint8 = np.uint8(color_map * 255.0)

        LOGGER.info("Grad-CAM generation succeeded. Heatmap range: [%.4f, %.4f]",
                    heatmap_resized.min(), heatmap_resized.max())

        return {
            "success": True,
            "heatmap": heatmap_uint8,
            "overlay": overlay_uint8,
            "raw_heatmap": heatmap_resized,
        }
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Grad-CAM generation failed.")
        return {
            "success": False,
            "error": str(exc),
        }


def find_first_existing_fallback(fallback_paths: list[Path]) -> Path | None:
    """Return the first fallback Grad-CAM image that exists."""
    for path in fallback_paths:
        if path.exists() and path.is_file():
            return path
    return None
