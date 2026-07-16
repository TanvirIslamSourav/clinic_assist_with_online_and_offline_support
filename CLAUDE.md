# CLAUDE.md

Read this before every task. It is short on purpose — detailed contracts live in `docs/` and
should be read on demand, not memorised.

## What this is

A Streamlit clinical decision-support dashboard for doctors in low-resource settings. Three
modules: diabetes risk, heart disease risk, and pneumonia pattern detection from chest X-rays.
It reuses trained research artifacts and never retrains at runtime.

**Offline-first.** The app is fully functional with no network. When a connection is available,
Gemini adds a layer of assistance on top: it explains the model's evidence in plain language,
audits the Grad-CAM heatmap, answers questions about the case, and phrases counterfactuals. That
layer is additive. It is never load-bearing.

The target user is a doctor or medical officer at a rural health complex, possibly on an
unreliable connection, who needs a second opinion in under a minute. Not a data scientist.

## Non-negotiable rules

Invariants. If a task appears to require breaking one, **stop and ask**.

1. **Gemini assists the clinician's decision. It never participates in the model's decision.**
   The probability comes from a saved artifact, always. Gemini receives a finished evidence
   bundle and produces prose. It cannot see, change, or invent a number, a class, or a decision.
   There is no code path where an LLM response becomes a prediction — not as a fallback, not when
   the artifact is missing, not "just for the demo". Giving Gemini a large, useful job in the
   doctor's reasoning is the goal; giving it any job in the model's arithmetic is the failure.

2. **The app works with Gemini switched off.** `GEMINI_ENABLED=false` must produce a complete,
   usable app with rule-based summaries. Every Gemini call has a timeout and a fallback. Network
   failure is the expected case, not the edge case. Test it that way.

3. **No real patient data through a free-tier key.** Free-tier Gemini usage can be used to improve
   Google's products and may be seen by human reviewers. Demo data is synthetic or de-identified.
   `DATA_MODE` must be `demo` unless a billed key is configured.

4. **Never invent an encoding.** Dataset integer codes (`cp=0`, `thal=2`, …) mean different things
   in different published versions of these datasets. Every label mapping must be derived from the
   notebook's own encoder or preprocessing code and verified by a test. A confidently mislabelled
   field is worse than a raw code.

5. **Every result carries its disclaimer, attached to the result** — not to the top of the page.
   Never present output as a diagnosis. Never let any output recommend a treatment, drug, or dose.

6. **Artifacts are read-only.** No training, no fitting, no `.fit()` at runtime. Paths come from
   `config/manifest.json`, never hardcoded.

7. **Secrets never enter the repo.** `.env` and `.streamlit/secrets.toml` only. If you need a key
   to test, mock the client.

## Commands

```bash
pip install -r requirements.txt
streamlit run app.py
pytest -q
```

## Layout

```
app.py                  router only — no logic
config/manifest.json    artifact paths, field metadata, label maps, sample cases
config/thresholds.json  decision thresholds + abstention margins
src/pages/              one file per page. UI only.
src/components/         shared UI (result card, status chip, disclaimer). UI only.
src/pipelines/          inference orchestration, evidence bundle construction
src/preprocessing/      artifact loading, transforms
src/explainability/     SHAP / Grad-CAM / rule-based summaries
src/llm/                Gemini client, prompts, validator. Isolated here.
src/utils/              logging, session state
static_outputs/         research figures (fallback visuals)
tests/
```

**Boundaries:** UI never calls Gemini directly — it calls a pipeline and renders the result
object. `src/llm/` never imports from `src/pages/`. If a page grows an `if probability > x`
branch, that logic belongs in a pipeline.

## Conventions

- Type hints on every public function. `from __future__ import annotations` at the top.
- `st.cache_resource` for model loading (never reload TensorFlow per rerun). `st.cache_data` for
  pure transforms and Gemini responses.
- Log via `src/utils/logger.get_logger(__name__)`. Never `print`.
- Never show a raw traceback to a user. Catch, log, render a degraded-mode message.
- Tests mock the Gemini API. We test our prompt builder and our validator, not Google's service.
- Style your own wrapper classes. **Never target Streamlit's generated CSS class names** — they
  change between versions and silently break on upgrade.

## Where to look

| Doing this | Read |
| --- | --- |
| Anything the user sees | `docs/UI_SPEC.md` |
| Anything touching Gemini | `docs/GEMINI_CONTRACT.md` |
| Form fields, labels, units, encodings | `docs/FIELD_METADATA.md` |
| Picking the next task | `docs/BUILD_ORDER.md` |

## Working style

- One task from `docs/BUILD_ORDER.md` at a time. Do not start the next unprompted.
- Write the test alongside the code, not after.
- Do not refactor code unrelated to the current task.
- If a task is underspecified, ask one question rather than guessing.
- Report what changed in 3–5 lines. No summary essays.
