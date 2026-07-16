"""Heart preprocessing helpers for inference-time dataframe construction."""

from __future__ import annotations

import pandas as pd

from src.features.heart_features import add_heart_engineered_features


def prepare_heart_dataframe(raw_payload: dict[str, float], expected_raw_features: list[str]) -> pd.DataFrame:
    """Construct one-row dataframe from ordered raw input fields."""
    row = {feature: raw_payload[feature] for feature in expected_raw_features}
    base_df = pd.DataFrame([row])
    return add_heart_engineered_features(base_df)
