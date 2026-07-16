"""Rule-based text generation for clinician-facing model explanations."""

from __future__ import annotations

from config.paths import load_thresholds
from src.explainability.explanation_templates import join_phrases, probability_sentence
from src.explainability.feature_name_mapping import humanize_feature


def get_confidence_band(probability: float) -> str:
    """Map probability to a confidence band using configurable thresholds."""
    thresholds = load_thresholds()["confidence_bands"]
    if probability >= float(thresholds["high"]):
        return "high confidence"
    if probability >= float(thresholds["moderate"]):
        return "moderate confidence"
    return "lower confidence"


def _normalize_contributors(features: list[str], min_items: int = 2, max_items: int = 4) -> list[str]:
    normalized = [humanize_feature(feature)
                  for feature in features][:max_items]
    if not normalized:
        return []
    if len(normalized) < min_items:
        return normalized
    return normalized


def generate_tabular_explanation(
    task_name: str,
    predicted_label: str,
    predicted_probability: float,
    confidence_band: str,
    positive_contributors: list[str],
    negative_contributors: list[str] | None = None,
) -> str:
    """Generate deterministic explanation text for tabular tasks."""
    positive = _normalize_contributors(positive_contributors)
    negative = _normalize_contributors(
        negative_contributors or [], min_items=1, max_items=3)

    intro = (
        f"For {task_name}, the model predicts {predicted_label.lower()} with {confidence_band}. "
        f"{probability_sentence(predicted_probability, confidence_band)}"
    )

    if positive:
        positive_text = join_phrases(positive)
        contributor_line = f"The strongest contributing factors were {positive_text}."
    else:
        contributor_line = "No dominant contributors were identified from the available rule-based signals."

    if negative:
        negative_text = join_phrases(negative)
        opposing_line = f"Some factors appeared to reduce the predicted risk, including {negative_text}."
    else:
        opposing_line = ""

    footer = (
        "This explanation reflects model behavior and should be reviewed alongside clinical findings. "
        "It is not a final diagnosis."
    )

    text_parts = [intro, contributor_line]
    if opposing_line:
        text_parts.append(opposing_line)
    text_parts.append(footer)
    return " ".join(text_parts)


def generate_image_explanation(
    predicted_label: str,
    predicted_probability: float,
    confidence_band: str,
) -> str:
    """Generate deterministic explanation text for image model outputs."""
    if predicted_label.upper() == "PNEUMONIA":
        finding_text = "a pneumonia-like pattern"
    else:
        finding_text = "a normal-like chest X-ray pattern"

    return (
        f"The image model predicts {finding_text} with {confidence_band}. "
        f"{probability_sentence(predicted_probability, confidence_band)} "
        "The Grad-CAM heatmap highlights image regions that most influenced the model output. "
        "These highlighted areas indicate model attention only and should not be interpreted as a confirmed diagnosis without clinical review."
    )
