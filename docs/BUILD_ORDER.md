# Build order

One task per session. Do not start the next unprompted. Each task ships with tests and leaves the
app runnable.

Scope: **UI overhaul + Gemini online assistance.** Tasks 1–4 make the app usable by a doctor.
Tasks 5–8 add the online layer. Task 4 is the last point at which stopping still yields a
coherent product; everything after is additive by design.

---

### Task 1 — Field metadata contract

Add `modules.<module>.fields`, `sample_cases`, and `population_caveat` to `config/manifest.json`
per `docs/FIELD_METADATA.md`. Add a typed loader in `src/utils/`.

**Derive the heart encodings from the notebook and report them for confirmation before writing
any option labels.** No UI changes in this task.

Done when: `test_option_values_match_encoder`, `test_field_order_matches_model_input`, and
`test_every_field_has_label_and_type` pass.

### Task 2 — Render both tabular forms from metadata

Rewrite the diabetes and heart forms to build widgets from the metadata per `docs/UI_SPEC.md`.
Correct widget per type, units in labels, `help` text, min/max enforced, `format_func` for
categoricals. Delete the hand-written field code. Add the `Load sample case` selectbox and the
population caveat line.

This is the task that turns the app from a notebook with input boxes into something a doctor can
use. It is worth more than any Gemini feature.

Done when: no bare dataset column name appears anywhere in the UI.

### Task 3 — Page structure and result card

Per `docs/UI_SPEC.md`: `st.navigation` replaces the sidebar radio and the `if/elif` block in
`app.py`. Results render directly under the submit button. Static figures move into a collapsed
expander; learning curves move off the clinician pages into About / Methodology. Disclaimer moves
from page-top to attached-to-result with module-specific wording. Build the result card component
(number, band, contributor bars) and the connectivity chip in `src/components/`. Rebuild Home.

At this point the app should look like a different product with zero Gemini involvement.

### Task 4 — Abstention and artifact integrity

Add `abstention_margin` to `thresholds.json`. Within ±margin of the threshold, render
"Inconclusive — clinician review required" instead of a band, and set `abstain: true`. Add SHA-256
per artifact in the manifest, verified on load; mismatch disables the module with an explanation
rather than predicting.

### Task 5 — LLM client, isolated

`src/llm/client.py` per `docs/GEMINI_CONTRACT.md`: key handling, `GEMINI_ENABLED` flag, temp 0,
JSON schema output, 8s timeout, backoff, cache by bundle hash. Plus `.env.example`.
**No UI, no prompts, no pages touched.** Tests mock the API entirely.

Done when: with no key set, every test passes and `streamlit run app.py` behaves exactly as after
task 4.

### Task 6 — Evidence bundle, narration, validator

Bundle builder in `src/pipelines/`. Prompt builder and `src/llm/validator.py` per the contract.
Wire into the tabular pages: validated narrative when online, rule-based summary otherwise.
Connectivity chip goes live. Bangla toggle.

Done when: a golden-file test pins the prompt for a fixed bundle, and the validator rejects each
of a hand-written set of bad responses (invented number, wrong direction, drug name, abstain
violation, cited driver not in the bundle).

### Task 7 — Pneumonia OOD gate and Grad-CAM concordance

Flash-Lite gate before inference; concordance audit after. `black.jpg` and `white.jpg` must be
rejected by the gate — add them as test fixtures. An `implausible` verdict renders a caution
beside the heatmap and never alters the probability. Offline path degrades to a format check.

The concordance audit is the most distinctive thing in the project. Do not cut it before task 8.

### Task 8 — Scoped Q&A panel

`st.chat_input` / `st.chat_message` per `docs/UI_SPEC.md`. Suggested questions, bundle-only
grounding, fixed refusal for out-of-scope, 5-turn cap. Counterfactuals re-run the real model;
Gemini phrases the computed result only.

---

## Later, not now

Case PDF export · paper-slip photo intake · authenticated clinician workspace · audit log ·
calibration views · model version registry.
