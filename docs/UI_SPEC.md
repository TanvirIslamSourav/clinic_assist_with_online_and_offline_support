# UI spec

What the clinician sees. The current dashboard is a research notebook with input boxes; this
describes what it becomes.

## Principles

- A doctor with 90 seconds must be able to fill the form, read the result, and know how much to
  trust it. Everything else is secondary.
- The model's number is the largest thing on the result. Gemini's prose is visibly secondary,
  labelled, and removable.
- Never show a raw traceback, a dataset column name, or an unlabelled integer.
- Style your own wrapper classes only. Never target Streamlit's generated class names.

## Navigation

`st.navigation` / `st.Page`, replacing the sidebar radio and the `if/elif` block in `app.py`.
Pages: Home, Diabetes, Heart disease, Pneumonia, About / Methodology.

## Home

Currently ~60% empty. Replace with:

- Title + one-line purpose.
- Four cards (`st.container(border=True)` in `st.columns`), one per module, each with a one-line
  "when to use this" and a button that navigates.
- The connectivity chip (below). Home is where the doctor learns whether they're online today.
- No LLM call on this page.

## Forms (diabetes, heart)

Rendered from `config/manifest.json` metadata — never hand-written. See `docs/FIELD_METADATA.md`.

- `Load sample case ▾` selectbox above the form. Four presets: low risk, high risk, near
  threshold, missing values.
- Widget per type: bounded continuous → `st.slider`; precision-sensitive (`oldpeak`,
  `DiabetesPedigreeFunction`) → `st.number_input`; categorical → `st.selectbox` with
  `format_func` mapping the stored integer to its label; boolean → `st.radio` Yes/No.
- Every label carries its unit. Every field with a gotcha carries `help=`.
- Population caveat rendered under the form heading, from the manifest.
- Submit inside `st.form` so partial edits don't trigger reruns.

## Result card

Renders **directly under the submit button**. Static research figures move to a collapsed
`st.expander("Research figures")` at the bottom of the page. Learning curves leave the clinician
pages entirely and live in About / Methodology.

Order, top to bottom:

1. **The number.** Probability at ~32px, with `threshold` and `model_version` beneath it in muted
   12px. This is the biggest element on the card.
2. **Band or abstention.** Either a risk band chip, or — when `abstain` is true — an amber
   "Inconclusive — clinician review required" chip plus one line explaining that the result sits
   within the margin and does not discriminate. Never render a band when abstaining.
3. **Top contributors.** Three horizontal bars, signed. Red = pushes risk up, teal = pulls down.
   Labels are the human field names, with the patient's entered value. Omit this block entirely
   when `contributions_source` is `unavailable` — never show an empty chart.
4. **AI summary** (online only). Distinct background (`surface-1`), tagged with a badge reading
   "AI-generated summary", the model name, and a validation tick reading "figures match model
   output". Absent when offline; the rule-based summary takes the same slot with no badge.
5. **Question chips** (online only). Three or four suggested questions plus the Bangla toggle.
6. **Disclaimer**, attached here at the bottom of the card — not at the top of the page. Wording
   is module-specific: name what the model did *not* see.

Build with one `st.markdown(..., unsafe_allow_html=True)` block against your own CSS classes, or
`st.container(border=True)` + `st.columns` + `st.metric`. Either is fine; don't mix.

## Connectivity chip

Always visible (sidebar footer + result card corner). Three states, from
`docs/GEMINI_CONTRACT.md`:

| State | Chip | Meaning |
| --- | --- | --- |
| `full` | green "AI assist online" | model output + validated narrative |
| `offline` | grey "Offline — rule-based" | model output + rule-based summary. Not an error. |
| `artifact_missing` | red "Module unavailable" | module disabled with an explanation |

`offline` is a normal operating state and must not look like a failure. No red, no warning icon,
no toast. A rural clinic will live here.

## Pneumonia page

- Upload → **OOD gate runs before inference** (online). If the image isn't a frontal chest
  radiograph, refuse to run the model and say why. Offline, fall back to a basic dimension/format
  check and a caution note.
- Result card as above, with the Grad-CAM overlay beside the original.
- **Concordance verdict** rendered as a note beside the heatmap. An `implausible` verdict shows an
  amber caution ("attention falls outside the lung fields — treat this explanation with
  suspicion") and **never alters the probability**.
- Learning curves are not on this page.

## Q&A panel (online only)

`st.chat_input` / `st.chat_message`, below the result card, collapsed by default.

- Opens with suggested questions as buttons, not an empty text box.
- Scoped to the current case. Out-of-scope questions get the fixed refusal from the contract.
- 5-turn cap, then a "start a new case" note.
- Absent entirely when offline. Do not render a disabled box.

## About / Methodology

Everything the clinician pages shed: learning curves, AUC figures, dataset provenance, SHAP/LIME
galleries, the Kaggle notebook link, and the validator rejection count. Add the honest limitation
paragraph: what each model was trained on, and what that means for this population.
