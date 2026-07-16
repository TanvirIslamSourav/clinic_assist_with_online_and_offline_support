"""Maps technical feature names to clinician-friendly phrases."""

from __future__ import annotations

FEATURE_NAME_MAP = {
    "Pregnancies": "pregnancy history",
    "Glucose": "glucose level",
    "BloodPressure": "blood pressure",
    "SkinThickness": "skin thickness proxy",
    "Insulin": "insulin level",
    "BMI": "body mass index",
    "DiabetesPedigreeFunction": "family history-related diabetes risk",
    "Age": "age-related baseline risk",
    "Age_Glucose": "age-related glucose interaction",
    "BMI_Insulin": "combined obesity and insulin-related risk",
    "Glucose_Preg": "glucose and pregnancy interaction",
    "BMI_Glucose": "combined obesity and glucose-related risk",
    "Glucose_sq": "nonlinear glucose burden",
    "BMI_sq": "nonlinear obesity burden",
    "Insulin_Glucose": "insulin-to-glucose balance",
    "Age_group": "age group category",
    "High_Glucose": "high glucose indicator",
    "Obese": "obesity-related risk",
    "High_BP": "high blood pressure indicator",
    "age": "age",
    "sex": "sex category",
    "cp": "chest pain pattern",
    "trestbps": "resting blood pressure",
    "chol": "cholesterol level",
    "fbs": "fasting blood sugar indicator",
    "restecg": "resting ECG finding",
    "thalach": "maximum heart rate",
    "exang": "exercise-induced angina",
    "oldpeak": "exercise ST depression",
    "slope": "ST slope",
    "ca": "major vessel count",
    "thal": "thalassemia test category",
    "age_maxhr": "age and maximum heart-rate interaction",
    "hr_reserve": "heart rate reserve pattern",
    "bp_high": "high blood pressure flag",
    "bp_low": "low blood pressure flag",
    "chol_high": "high cholesterol flag",
    "age_sq": "nonlinear age effect",
    "senior": "senior age indicator",
    "oldpeak_hr": "combined stress-related cardiac burden",
}


def humanize_feature(feature_name: str) -> str:
    """Return clinician-friendly phrase for a feature name."""
    return FEATURE_NAME_MAP.get(feature_name, feature_name.replace("_", " ").lower())
