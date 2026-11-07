# Bangkok Flood Simulator

Streamlit app (`app.py`) and data pipeline for Bangkok flood risk. The README covers features and setup;
`docs/METHODS.md` covers the science. This file lists the rules an agent can't work out from the code.

## Rules
- **Python 3.9 and 3.14.** CI tests both (`.github/workflows/ci.yml`): 3.9 matches local development, 3.14 matches
  Streamlit Community Cloud. Keep `from __future__ import annotations` and avoid syntax newer than 3.9 (`match`,
  `X | Y` outside annotations, etc.).
- **`data/processed/` is committed on purpose** (~280 MB, most of it `buildings/`): the hosted app reads it. `data/raw/` is gitignored
  downloads; never edit it, change the prepare/fetch script instead.
- **The bot owns some data.** `.github/workflows/update-data.yml` commits `data/processed/radar_flood/`, and
  `traffy/` plus `flood_labels/traffy_*.tif`, every day. Pull before working on those files and don't hand-edit them.
- **No secrets in the repo.** The Earth Engine key lives only in the GitHub secret `GEE_SERVICE_ACCOUNT_KEY`.
- **No paid APIs.** Don't add Anthropic or other paid API calls to scripts or workflows.
- **Thresholds and paths live in `src/config.py`.** The docs and app text quote them (alert thresholds, Traffy storm
  rules). When you change a value, update `README.md`, `docs/METHODS.md` and the `app.py` text that cite it.
- **Tests:** `pytest` (config in `pytest.ini`; `tests/conftest.py` has a 20×20 synthetic study area). The smoke test
  `tests/test_app_smoke.py` runs the whole app on the real processed data.

## Local environment
- `python` is Anaconda Python 3.9. Its pyarrow 19 can't read parquet files that CI writes ("Repetition level
  histogram size mismatch"), so Traffy tests and the smoke test fail locally until pyarrow is upgraded.
- `gh` is installed and logged in. Use `gh run list --workflow update-data.yml` and `gh run view <id> --log-failed`
  to look at failed data updates.

## Project agents and hooks
- Subagents in `.claude/agents/`: `pipeline-doctor` (diagnose and fix a failed Update data run), `data-auditor`
  (read-only sanity check of `data/processed/`).
- Hooks in `.claude/settings.json` (script `.claude/hooks/project_hooks.py`):
  - block writes to `data/raw/`, `.streamlit/secrets.toml` and key-like `*.json` files;
  - run the tests that import a `src/`/`scripts/` file after it is edited;
  - run `pytest` before Claude's own `git push`.
