"""Heart feature engineering used prior to pipeline inference."""

from __future__ import annotations

import pandas as pd


def add_heart_engineered_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Recreate domain engineered features from the training notebook."""
    engineered = dataframe.copy()
    engineered["age_maxhr"] = engineered["age"] * engineered["thalach"]
    engineered["hr_reserve"] = engineered["thalach"] - \
        (220 - engineered["age"])
    engineered["bp_high"] = (engineered["trestbps"] > 140).astype(int)
    engineered["bp_low"] = (engineered["trestbps"] < 90).astype(int)
    engineered["chol_high"] = (engineered["chol"] > 240).astype(int)
    engineered["age_sq"] = engineered["age"] ** 2
    engineered["senior"] = (engineered["age"] > 60).astype(int)
    engineered["oldpeak_hr"] = engineered["oldpeak"] * engineered["thalach"]
    return engineered
