# Notes for Copilot

Superseded. All architecture rules, safety invariants, and conventions now live in
[`CLAUDE.md`](./CLAUDE.md), with detailed contracts in [`docs/`](./docs/).

Kept as a pointer so that any agent or contributor landing here is redirected rather than
following the old guidance, which stated "keep rule-based explanation deterministic (no LLM/API
calls)". That rule has been **replaced, not abandoned**: the prediction path is still fully
deterministic and offline, and the rule-based summary is still what renders when there is no
connection. Gemini is an online assistance layer that sits strictly downstream of the model
output and cannot influence it. See rule 1 in `CLAUDE.md`.
