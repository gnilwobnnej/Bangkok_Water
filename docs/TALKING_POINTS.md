# Talking points: AI fluency, agents and the ML inside

Short notes for presenting the Bangkok Flood Simulator. Figures are as of 7-8 Nov 2026; 2026 scores shift as the
season goes on, so run the `docs-consistency` subagent before presenting.

## 1. The project in one line
An interactive flood-risk map of Bangkok. It's a "what-if" water-level simulator, checked against real floods seen
by satellites and by residents, and it updates itself daily from radar, rain, river flow and citizen reports.

## 2. The models (what predicts what)
| Model | Method | Learns from | Tested on |
|---|---|---|---|
| Bathtub simulator | Rule-based: low ground connected to rivers floods; walls hold until overtopped | none (physics-style rule) | every flood |
| Risk index | Hand-weighted blend of elevation, nearness to water, population | none | every flood (it scores below random, a useful lesson) |
| **River-flood ML** | **LightGBM** gradient-boosted trees | 2011 MODIS flood + 9 radar wet seasons (2017–2025) | 15 radar passes in 2026 it never saw: ROC-AUC 0.91–0.96 vs bathtub 0.66–0.85 |
| **Street-flood ML** | LightGBM, plus building and population features | Residents' Traffy Fondue reports, 2022–2025 | 2026 storms: 0.78 in the dense city, best method on 22 of 22 storm days |
| Flood-report classifier | TF-IDF on character 2–4-grams + logistic regression (Thai has no spaces between words) | reports Traffy did tag | backup when Traffy's category feed is down (~85% right) |
| Translation | Helsinki-NLP opus-mt-th-en (MarianMT), run locally | pretrained | gives the gist; place names often wrong |
| Satellite flood maps | Sentinel-1 radar change detection (dark AND darker than the same weeks last year) | rule + thresholds | the ground truth for the river model |

## 3. How the models were made (5 steps)
1. **Labels:**
   - **River model:** 10 floods. The 2011 MODIS flood plus 9 Sentinel-1 radar wet seasons (2017–2025), each cell 1 flooded / 0 dry / 255 unknown.
   - **Street model:** residents' Traffy reports on storm days.
2. **Features per ~65 m cell:** from Copernicus elevation, OpenStreetMap waterways, ESA WorldCover, and (street model) Google Open Buildings and population.
3. **Train:** LightGBM, gradient-boosted decision trees (400 trees, one algorithm for both models), on balanced samples. That's 2 M cells for the river model (equal per flood) and 75 k for the street model.
4. **Test:** spatial block cross-validation, leave-one-flood-out, and 2026 held out completely, with 95% bootstrap intervals and comparison against the bathtub and risk index.
5. **Explain and publish:** SHAP explains the predictions. The model is retrained on all training floods and saved to `models/`. Its maps and report go into the app, and the daily workflow re-scores it.

## 4. The ML craft (what makes the scores trustworthy)
- **Features:**
  - **River model:** elevation, height above nearest drainage, local relative elevation, slope, distance to canals and to the Chao Phraya, and land-cover shares.
  - **Street model adds:** small dips in the ground (300 m), building count and height, population, built-up share.
- **Leakage control:**
  - **Spatial block cross-validation:** 4 km blocks, held out across every year at once.
  - **Leave-one-flood-out:** train on every flood but one, then test on that one.
  - **2026 kept out of training:** used only as an honest test year.
- **Uncertainty everywhere:** 95% block-bootstrap intervals on every score, Monte Carlo runs for elevation error (a "chance flooded" layer), and a map of where the cross-validation models disagree.
- **Explainability:** SHAP values show what drives each model. For the river model that's mostly elevation.
- **Honest design choices:**
  - **Population left out of the river model:** with it, the model learned "dense city = dry", a dangerous lesson.
  - **Case-control labels for street floods:** "dry" only counts where people report other problems, so the labels don't just learn where people live.
  - **Ideas that failed:** FABDEM elevation and radar "double bounce" didn't help, and the docs report that.
- **Right model for the place:** the street model is used inside the dense city, the river model outside it. Each is shown where it was tested and won.

## 5. AI fluency: how agents were actually used
- **Claude Code (Claude Opus 5.5) as a collaborator, not an autopilot:** it read the codebase, proposed options with costs, and built only what was approved.
- **Cost awareness:** paid add-ons (Claude API classification at about $0.10/day, an auto-fix bot) were costed and then declined. Everything now runs inside the existing plan, and the rule is written down in `CLAUDE.md`.
- **Context engineering:** `CLAUDE.md` records the rules an agent can't infer (Python 3.9 and 3.14, bot-owned data, no secrets, thresholds in `config.py`).
- **Specialist subagents:**
  - **`pipeline-doctor`:** traced 4 failed data runs to 3 root causes in under a minute.
  - **`data-auditor`:** found stale metadata and data-growth drift.
  - **`docs-consistency`:** found about 30 outdated figures across 4 docs.
  - **`test-writer`:** wrote 22 tests and flagged 3 edge cases.
- **Guardrails, not trust:** hooks block writes to raw data and secrets, run the related tests after every edit, and run the full test suite before any push.
- **Verify the verifier:** agent findings were rechecked against the data before editing, and a second agent pass caught 4 contradictions the first edit missed.
- **Humans keep the irreversible steps:** changes go out as pull requests, and the permission system blocked an unreviewed merge, so the human merged.
- **Knowing when *not* to use an agent:** a failed run caused by a GitHub 504 timeout just needed a re-run, not code changes.

## 6. One-sentence takeaways
- "The ML is only as good as its test: blocks, held-out years and error bars on everything."
- "Satellites told us where rivers spread; residents told us where streets fill. Two models, each used where it earned it."
- "Agents did the legwork — diagnosing, auditing, testing — while guardrails and a human kept the final say."
