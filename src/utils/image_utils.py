"""Image decoding and preprocessing utilities."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError


def decode_uploaded_image(file_bytes: bytes) -> Image.Image:
    """Decode uploaded bytes into a Pillow image and normalize color channels."""
    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    # Some real-world JPEG exports can be slightly truncated but still usable.
    prior_truncated_setting = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with Image.open(BytesIO(file_bytes)) as image:
            image.load()
            normalized = image.convert("RGB") if image.mode != "RGB" else image
            return normalized.copy()
    except (UnidentifiedImageError, OSError, ValueError):
        # Fallback decoder for JPEG variants Pillow may reject in some environments.
        try:
            import tensorflow as tf

            tensor = tf.io.decode_image(
                file_bytes,
                channels=3,
                expand_animations=False,
            )
            decoded = tensor.numpy()
            if decoded.ndim != 3 or decoded.shape[2] != 3:
                raise ValueError("Uploaded image has unsupported dimensions.")
            return Image.fromarray(decoded.astype(np.uint8), mode="RGB")
        except Exception as exc:  # noqa: BLE001
            raise ValueError("Uploaded file is not a valid image.") from exc
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = prior_truncated_setting


def resize_image(image: Image.Image, image_size: tuple[int, int]) -> Image.Image:
    """Resize image to model-required spatial dimensions."""
    return image.resize(image_size)


def image_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL image to float32 numpy array."""
    return np.asarray(image, dtype=np.float32)
