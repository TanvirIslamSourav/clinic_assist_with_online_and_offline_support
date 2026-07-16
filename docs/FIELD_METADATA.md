# Field metadata

Forms are **rendered from data, not hand-written**. Every input field is described once in
`config/manifest.json` and the page renders whatever is there. Adding a unit or fixing a label
must never require touching a page file.

## Why this exists

The current heart form shows `cp`, `thal`, `restecg`, `ca`, `oldpeak` as bare integer dropdowns.
A cardiologist cannot fill that in. The diabetes form shows `BloodPressure` and `Glucose` with no
units, which is worse than ugly — it is a clinical hazard (see below).

## Schema

Under `manifest.json` → `modules.<module>.fields`, an **ordered** list:

```json
{
  "field": "trestbps",
  "label": "Resting blood pressure",
  "unit": "mm Hg",
  "type": "number",
  "min": 80,
  "max": 220,
  "step": 1,
  "default": 130,
  "help": "Measured on admission, seated at rest."
}
```

```json
{
  "field": "cp",
  "label": "Chest pain type",
  "unit": null,
  "type": "category",
  "options": [
    {"value": 0, "label": "Typical angina"},
    {"value": 1, "label": "Atypical angina"},
    {"value": 2, "label": "Non-anginal pain"},
    {"value": 3, "label": "Asymptomatic"}
  ],
  "default": 3
}
```

`type` ∈ `number` | `integer` | `category` | `boolean`. Widget mapping is in `docs/UI_SPEC.md`.

## The encoding rule — read this before writing any option list

**The `options` above illustrate the shape. They are not a source of truth.**

The UCI heart data circulates in several encodings. `cp` is 1–4 in the original Cleveland file and
0–3 in the widely mirrored Kaggle version. `thal` appears as 3/6/7 in some copies and 0–3 in
others. Copying a mapping from a blog post — or from an assistant's memory — has roughly a coin
flip's chance of being wrong, and a wrong mapping is *more* dangerous than a raw integer, because
the clinician now trusts it.

Required procedure:

1. Open `notebook_reference/xai_medical_diagnosis_FINAL.ipynb` (or `notebook_extracted.py`) and
   find where the heart features are encoded, or where the raw CSV is loaded.
2. Derive the mapping from that code. If the pipeline holds a fitted encoder, read `categories_`.
3. **Report the derived mapping to the human for confirmation before writing labels.**
4. If the notebook does not settle the mapping unambiguously, stop and say so. Do not guess.

Lock it in with tests:

- `test_option_values_match_encoder` — the set of `value`s per categorical field equals the
  categories the artifact was fitted on. Fails loudly if an artifact is swapped.
- `test_field_order_matches_model_input` — field order equals the model's expected feature order.
  Silent column reordering is the classic way to get plausible, wrong predictions.
- `test_every_field_has_label_and_type` — no field ships bare.

## Diabetes: two traps that must be surfaced in the UI

- **`BloodPressure` is diastolic only.** A clinician will type 120 meaning systolic. Label it
  `Diastolic blood pressure (mm Hg)` and cap `max` near 130 so a systolic value is rejected.
- **`Glucose` is 2-hour OGTT plasma glucose**, not a random fingerstick. Label it
  `Plasma glucose, 2-hr OGTT (mg/dL)` and say so in `help`. A random glucose entered here is
  out-of-distribution input the model will answer anyway.

Remaining units: `SkinThickness` triceps skinfold (mm), `Insulin` 2-hr serum insulin (µU/mL),
`BMI` (kg/m²), `DiabetesPedigreeFunction` (unitless), `Age` (years), `Pregnancies` (count,
integer — currently rendering as `1.00`, which is a bug).

## Population caveat

Each module carries a `population_caveat` string in the manifest, rendered on the page and passed
into the evidence bundle. Diabetes: trained on adult women of Pima heritage — a real limitation
for a Bangladeshi clinic. Pneumonia: validated on paediatric chest X-rays from a single hospital
source. Naming the limitation yourself is stronger than having a judge find it.

## Sample cases

Under `modules.<module>.sample_cases`, a list of `{name, values}`. Required set:

- `Low risk — clear`
- `High risk — clear`
- `Near threshold` (the abstention demo)
- `Missing values` (exercises the imputer path)

Nobody will type eleven fields during a demo.
