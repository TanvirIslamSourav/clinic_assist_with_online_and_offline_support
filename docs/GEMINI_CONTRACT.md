# Gemini contract

Everything the LLM layer is allowed to do. If a change would violate this file, it violates
rule 1 in `CLAUDE.md`.

## The distinction this whole file protects

Gemini's job is to help **the doctor** decide. Gemini has no job in helping **the model** decide.

Those sound similar and are opposites. The first is the product: a clinician staring at `0.53`
and three SHAP bars needs someone to tell them what the model actually saw, whether the heatmap
is looking at lung or clavicle, and what would move the number. That is real, substantial work
and Gemini is good at it. The second is a system that launders an LLM guess into a clinical
probability. The line between them is architectural, not a matter of prompt wording.

```
inputs -> saved artifact -> probability + threshold -> evidence bundle -> [Gemini] -> validator -> UI
                                                              |                                    ^
                                                              +------ rule-based summary ----------+
                                                                       (offline path)
```

Gemini consumes the bundle. It never sits upstream of it. Cut the Gemini box and everything to
its left still reaches the UI.

## Online vs offline

| | Offline (`GEMINI_ENABLED=false`, no key, timeout, 429, validator reject) | Online |
| --- | --- | --- |
| Probability | identical | identical |
| Band / abstention | identical | identical |
| SHAP contributions | identical | identical |
| Summary | rule-based, deterministic | validated narrative |
| Grad-CAM | rendered | rendered **+ concordance audit** |
| OOD gate | basic format check | Gemini vision check |
| Q&A panel | absent | present |
| Bangla | rule-based template | narrated |

Nothing in the left column is degraded output. It is the product. The right column is a better
day.

## Model and SDK

- SDK: `google-genai`.
- Narration, concordance, Q&A: **Gemini 3.5 Flash**. OOD gate: **Gemini 3.1 Flash-Lite**.
- Do **not** target a Pro model. Pro is not on the free tier and the app must run on a free key.
- Rate limits are **per Google Cloud project, not per API key** — extra keys buy nothing. Live
  limits are visible in AI Studio; Google no longer publishes a stable public table, so **never
  hardcode RPM/RPD numbers**. Treat 429 as normal and back off. Daily quotas reset midnight
  Pacific.

## Client (`src/llm/client.py`)

- Key from `GEMINI_API_KEY` (env or `st.secrets`). Absent key => `GEMINI_ENABLED=false`, no crash.
- `temperature=0`, `response_mime_type="application/json"`, `response_schema=<schema>`.
- Hard timeout **8s**. On timeout: log, return `None`, caller falls back.
- Retry on 429/5xx with exponential backoff (1s, 2s, 4s), max 3 attempts, then give up quietly.
- `st.cache_data` keyed on `sha256(canonical_json(bundle))`. Identical case => zero extra calls.
- Never log bundle contents at INFO. Log the hash.

## Evidence bundle

Built by the pipeline. The **only** thing Gemini is given.

```json
{
  "schema_version": "1.0",
  "module": "heart",
  "model_id": "best_Heart_pipe",
  "model_version": "1.2",
  "probability": 0.53,
  "threshold": 0.50,
  "abstain": true,
  "abstain_reason": "within_threshold_margin",
  "band": null,
  "inputs": [
    {"field": "cp", "label": "Chest pain type", "value": 3,
     "display_value": "Asymptomatic", "unit": null}
  ],
  "contributions": [
    {"field": "cp", "label": "Chest pain type", "signed_value": 0.18, "direction": "increases"}
  ],
  "contributions_source": "static_shap",
  "missing_fields": [],
  "population_caveat": "Trained on the UCI Cleveland cohort; not validated on this population.",
  "language": "en"
}
```

- `contributions_source`: `live_shap` | `static_shap` | `unavailable`. If `unavailable`, the
  narrative must not name any driver.
- Pneumonia bundles add `gradcam_available`, `ood_gate` (`{passed, reason}`), and `concordance`
  (`{verdict, note}`, verdict ∈ `plausible` | `implausible` | `unchecked`).
- `language`: `en` | `bn`.

## Feature 1 — narration

System instruction must enforce all of:

1. Use **only** the bundle. Add no clinical facts, epidemiology, or context from training data.
2. Reproduce `probability` and `threshold` **verbatim**. Never round, restate, or recompute.
3. Never state or imply a diagnosis. Never name a drug, dose, test protocol, or treatment.
4. If `abstain` is true, assert no direction. Say the result does not discriminate.
5. If `contributions_source` is `unavailable`, attribute the result to nothing.
6. Always state what the model did not see (exam, history, labs absent from `inputs`).
7. Max 90 words. Plain clinical register. No hedging filler, no reassurance.
8. If `language` is `bn`, output Bangla. Numbers stay in Western digits.

```json
{"type":"object","properties":{
  "summary":{"type":"string"},
  "limitation":{"type":"string"},
  "drivers_used":{"type":"array","items":{"type":"string"}}},
 "required":["summary","limitation"]}
```

`drivers_used` exists so the validator can check the model only cited fields in the bundle.

## Feature 2 — Grad-CAM concordance audit

The most valuable thing Gemini does here. Send the X-ray **and** the Grad-CAM overlay. Ask one
narrow question: is the highlighted region inside the lung fields, or is it on the clavicle, the
image border, a laterality marker, or burnt-in text?

This audits the *explanation*, not the patient. It catches the shortcut-learning failure that
gets CXR papers retracted. Returns `{verdict, note}`. An `implausible` verdict renders a caution
beside the heatmap and **never changes the probability**.

## Feature 3 — OOD gate

Flash-Lite, before inference. Is this a frontal chest radiograph? Returns
`{is_cxr: bool, reason: str}`. If false, refuse to run the model and say why. Must reject
`black.jpg` and `white.jpg` — keep both as test fixtures. Offline, degrade to a
dimension/format check plus a caution note.

## Feature 4 — Q&A panel

- Scoped to the current bundle only. Same system instruction plus: refuse anything not derivable
  from the bundle with a fixed sentence, and never advise on management.
- Suggested questions, not an empty box.
- **Counterfactuals ("what would change this?") re-run the real model** with perturbed inputs.
  Gemini phrases the computed result only. The model does the arithmetic. This is the clearest
  case of the architecture doing useful work rather than restricting it.
- Cap 5 turns per case.

## Validator (`src/llm/validator.py`) — mandatory

Runs on every response. Any failure => discard the LLM output, render the rule-based summary, log
`VALIDATOR_REJECT` with the reason. Never show a rejected narrative.

1. **Parses** as JSON against the schema.
2. **Number fidelity** — every numeric token in `summary` appears in the bundle.
3. **Driver fidelity** — every `drivers_used` entry matches a `field` in `contributions`.
4. **Direction agreement** — if `abstain` is false, direction matches `probability >= threshold`.
   If `abstain` is true, no directional claim is present.
5. **Banned content** — reject on treatment/diagnosis language (drug names, `mg`, `dose`,
   `prescribe`, `diagnosis of`, `confirms`, `rule out`). One constant, one place.
6. **Length** — `summary` ≤ 90 words.

Rejections are a metric, not an embarrassment. Surface the count in About / Methodology.

## Degraded modes

Surfaced as a chip per `docs/UI_SPEC.md`. Never a traceback.

| State | Trigger | Behaviour |
| --- | --- | --- |
| `full` | key present, call ok, validator ok | model output + narrative |
| `offline` | no key, timeout, 429, validator reject | model output + rule-based summary |
| `artifact_missing` | artifact absent or hash mismatch | module disabled with an explanation |
