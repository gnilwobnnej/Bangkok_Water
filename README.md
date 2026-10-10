# Bangkok Flood Simulator

Interactive flood-risk map and "what-if" water-level simulator for Bangkok, Thailand.
Drag the water level up and watch which districts go under first, how many people are affected,
and where the tipping points are.

## Quick start
```bash
pip install -r requirements.txt
python scripts/prepare_data.py   # one-time download (~60 MB) → data/processed/
streamlit run app.py
```

## Deploying (Streamlit Community Cloud)
`data/processed/` (~235 MB) is committed on purpose so the hosted app has its data; `data/raw/` stays local.
Radar, rain and river data update themselves daily (next section). Other layers change rarely: re-run their
prepare scripts locally, then commit and push `data/processed/`.
`shap` is only needed for `train_model.py`: `pip install shap` before training.

## Automatic daily updates
`.github/workflows/update-data.yml` runs every day at 09:30 Bangkok time, and on demand from the Actions tab
(**Update data → Run workflow**). It:
- fetches new Sentinel-1 passes, rainfall and river flow;
- re-scores the models with `evaluate_current.py`;
- commits `data/processed/radar_flood/` as github-actions[bot], which redeploys the hosted app.

If Earth Engine fails, it still updates rain and river flow, then marks the run as failed so GitHub emails you.

- **History is kept:** new passes are merged into `summary.json` instead of replacing it (use `--replace` to
  start over). Each pass adds about 0.45 MB to the repo, roughly 30 MB a year.
- **Banner:** the app shows an "Unusual conditions" banner when any of these hold (thresholds in
  `src/config.py`):
  - the latest pass (within 14 days) has at least 2× the typical flooded area;
  - the Chao Phraya is at or above 130% of normal;
  - 3-day rain is at or above 100 mm.

**One-time setup** (Earth Engine login for GitHub):
1. In Google Cloud project `third-node-510023-u2`, create a service account. Give it the roles
   *Earth Engine Resource Writer* and *Service Usage Consumer*. (Writer is needed: downloading
   images requires `earthengine.thumbnails.create`, which the read-only Viewer role lacks.)
2. Create a JSON key for it, and keep it out of this folder.
3. In the GitHub repo, go to Settings → Secrets and variables → Actions. Add the secret
   `GEE_SERVICE_ACCOUNT_KEY` with the whole JSON as its value, then delete the local key file.
4. In Settings → Actions → General → Workflow permissions, choose *Read and write*.

Locally, `fetch_current_flood.py` uses this key too if `GEE_SERVICE_ACCOUNT_KEY` is set; otherwise it uses your
own `earthengine authenticate` login.

## Features
- **Water-level slider (0–3 m)** with live flooded area, people affected, and districts at risk
- **"Bathtub + connectivity" flood model**: low ground floods only if linked to a river/canal (toggleable),
  optionally held back by the river walls, King's Dike and airport dike until the water tops them
- **Flood risk index** blending low elevation, proximity to waterways, and population density, with adjustable weights
- **Interactive map**: flood-depth, ground-elevation (adjustable colour range + relief shading) and risk overlays, rivers & canals, district hover tooltips
- **Elevation profile**: histogram of land area by height vs. the water level, lowest districts
- **Animation** of rising water from 0 to 3 m
- District ranking chart, flood-vs-level curve, and a sortable district table

## Flood defences (optional)
```bash
python scripts/prepare_defences.py   # ~1 min; OpenStreetMap roads -> data/processed/defences.tif + .geojson
```
The sidebar switch **Include flood defences** adds Bangkok's main flood barriers to the river-connected
simulation:

| Defence | Route | Crest (m above sea level) |
|---|---|---|
| King's Dike | Bangkok's northern boundary from the river → Phahon Yothin, Sai Mai, Hathai Rat, Rom Klao, King Kaew and Tamru-Bang Phli roads → Old Sukhumvit Road at Samut Prakan | 2.5 |
| Chao Phraya river walls (both banks) | Bangkok's northern boundary to Samut Prakan | 3.0 upstream, 2.8 middle, 2.5 downstream |
| Suvarnabhumi Airport dike | Ring around the airport | 3.5 |

- **How walls work in the model:** a wall blocks water until the level rises above its crest. Canals inside the
  walls don't flood the land around them, because pumps and gates keep them low. Only rivers and canals
  connected to the map edge (the Gulf and the rivers running off the map) feed the flood.
- **Sources:** routes follow published descriptions (Royal Development Projects Board; Think of Living, 2011),
  traced along OpenStreetMap roads. Crest heights are the lower end of published ranges: BMA flood-wall figures,
  JICA's 2.47 m design level for Rom Klao Road, and Airports of Thailand for the airport dike. Every line carries
  its source; hover it on the map. Edit `SECTIONS`, `RIVER_WALLS` or `AIRPORT_DIKE` in the script to change an
  assumption.
- **Caveats:**
  - The northern closure along Bangkok's boundary is approximate. It keeps Don Mueang dry, but Don Mueang
    flooded in 2011.
  - Walls are one crest height per section.
  - The west bank's (Thonburi) polder dikes, pumps and tunnels aren't modelled.
  - Crest heights are above Thai mean sea level and the DEM is above the EGM2008 geoid. These differ by a few
    tens of cm, and no correction is applied.

**Effect at 2.0 m (Bangkok's 50 districts):** flooded area falls from 678 to 490 km², and people affected from
1.93 M to 0.80 M. At 2.5 m the King's Dike is overtopped, and the results return to the no-defence numbers.

**Against real floods:** on the 2011 flood, ROC-AUC rises from 0.668 to 0.722. The gain holds in all 5
cross-validation folds. Results for the radar passes are in the Validation tab. To re-score 2011 without
retraining the ML model, run `python scripts/train_model.py --baselines-only`.

## Machine-learning model (optional)
A LightGBM model learns flood susceptibility from where Bangkok **actually flooded**, instead of hand-set
weights: the **2011 flood** (Global Flood Database event DFO_3850, MODIS 250 m) and **nine wet seasons,
2017-2025**, mapped from Sentinel-1 radar. Every flood counts equally (200,000 labelled cells sampled from each).
Floods from 2026 on (`TEST_YEARS_FROM` in `src/config.py`) are never trained on, so the recent radar passes stay an
honest test. A second model trained on 2011 only, as before, is kept for comparison.

```bash
# 1. 2011 labels: download event 3850's map from global-flood-database.cloudtostreet.ai (no account) ...
python scripts/prepare_ml_data.py --label-file data/raw/gfd_dfo_3850.tif
#    ... or fetch it with a free Google Earth Engine login (run `earthengine authenticate` once)
python scripts/prepare_ml_data.py --gee-project YOUR_CLOUD_PROJECT_ID
# 2. Wet-season radar labels 2017-2025 (~10 min per year; passes are cached in data/raw/flood_archive/)
python scripts/build_flood_archive.py --gee-project YOUR_CLOUD_PROJECT_ID
# 3. Train + evaluate (~7 min); prints scores per flood vs the existing methods
python scripts/train_model.py
```
Each radar season combines its August-November passes, each compared with the same weeks of the year before:
a cell is **flooded** if at least half of it was flooded on 2 or more passes, **dry** if radar saw it and it was
always under 10% flooded, and unknown otherwise. The labels are in `data/processed/flood_labels/` (radar_YYYY.tif,
same 1/0/255 format as flood_2011.tif; `summary.json` lists the passes used). In those seasons 2-5% of the
land flooded, against 21% in 2011.

Evaluation: **spatial 5-fold CV** on 4 km blocks, held out of every flood year at once, and **leave one flood
out** (train on the other nine, test on all of the tenth, with 95% block-bootstrap intervals).

**Results (ROC-AUC; PR-AUC in brackets):**

| Test | ML, all floods | ML, 2011 only | Bathtub + defences | Bathtub | Risk index |
|---|---|---|---|---|---|
| 2011 flood, spatial CV | 0.853 (0.58) | **0.872 (0.68)** | 0.721 (0.33) | 0.667 (0.29) | 0.449 (0.18) |
| 2017-2025 seasons, spatial CV (mean) | **0.975 (0.58)** | 0.918 (0.27) | 0.795 (0.09) | 0.749 (0.07) | 0.421 (0.03) |
| Each season left out (range) | **0.972-0.987** | 0.901-0.931 | 0.769-0.807 | | |
| 2026 radar passes, never trained on (8) | **0.91-0.96** | 0.82-0.91 | 0.75-0.86 | 0.70-0.85 | 0.39-0.57 |

On the latest 2026 pass (27 September) the 95% intervals don't overlap: 0.955-0.968 for the new model against
0.892-0.923 for the 2011-only model. **The trade-off:** the new model is slightly worse on the extreme 2011 flood
(0.853 vs 0.872), because nine of its ten floods are ordinary seasons. It is the better guide to a typical year;
the 2011 map stays the guide to a rare, extreme one.

Population density and built-up land are excluded by default: with them the 2011 model scores 0.893, but its top
factor becomes "dense city = dry", which mostly reflects flood defences and satellites missing urban floodwater.

The app then gains "ML flood susceptibility" and "Observed flood (2011)" layers and a **ML model** tab listing the
floods it was trained on. Caveats: radar and MODIS both under-detect flooding between buildings; the seasonal
baseline hides places that flood every year and can hide a flood in the year after a wet one; the model doesn't
know about short cloudburst flooding.

## Current flood from satellite radar (optional)
Maps where Bangkok is flooding **now** from Sentinel-1 radar (sees through cloud; a new pass every few days),
using the same free Earth Engine project as the ML model:

```bash
python scripts/fetch_current_flood.py --gee-project YOUR_CLOUD_PROJECT_ID   # default: last 45 days
python scripts/evaluate_current.py      # scores every method against each radar pass (~few min)
```
Each pass is compared with the same weeks last year from the same orbit (`--baseline dry` compares with
January-March instead), with permanent water removed (JRC Global Surface Water). The app gains a
"Radar flood" layer with a date picker, a "Flooded now" column and a **Current flood** tab with rainfall
(Open-Meteo, no key) and Chao Phraya river flow vs its 1996-2025 normal (GloFAS via the Open-Meteo Flood API,
no key), plus a table of every pass with its rain and river flow. A daily GitHub Actions run picks up new
passes automatically (see "Automatic daily updates"). To update by hand, re-run both commands; new passes are
added to the archive. `python scripts/fetch_current_flood.py --weather-only` refreshes just the rain and
river data.

**27 Sep 2026 pass** (after ~200 mm of rain on 25-27 Sep): 99 km² of unusual water across greater Bangkok,
32 km² inside the city, mostly eastern farmland (Nong Chok 8.3%, Lat Krabang 5.9%). Tested against it, the
2011-trained ML model scores ROC-AUC 0.91 (0.94 within Bangkok), the bathtub 0.70 and the risk index 0.39.

## Validation
The **Validation** tab gathers every test of the methods (ML model, bathtub with and without defences, risk index) on
floods (or places) they never saw: every training flood (spatial cross-validation), each training flood left out in
turn, and every recent radar pass. It shows:
- how often the ML model beats the bathtub, and on how many passes the difference is clear of noise;
- whether training on more floods helped: the all-floods model against the 2011-only model, flood by flood;
- ROC-AUC or PR-AUC over time with shaded 95% intervals, and each method's difference from the bathtub;
- a table of every method x event with the random-guess baseline, cells tested and flooded share, as a CSV
  download;
- how the labels, scores and intervals work, and the known biases.

The intervals come from a **spatial block bootstrap** in `evaluate_current.py` (`src/validation.py`): 4 km blocks
are resampled 200 times and every method is re-scored on each resample, so neighbouring cells, which flood
together, aren't counted as independent evidence. Each method's difference from the bathtub uses the same
resamples and gets its own interval. `train_model.py` does the same for the left-out floods. For spatial CV the
range shown is the spread over the 5 folds.

## 3D buildings (optional)
Every building in Bangkok in 3D, coloured by flood impact under the simulated level, the latest radar pass,
the ML model or the 2011 flood, with hospitals, schools, police and fire stations highlighted.

```bash
python scripts/prepare_buildings.py --gee-project YOUR_CLOUD_PROJECT_ID               # all 50 districts (~1 h)
python scripts/prepare_buildings.py --gee-project ID --districts "Nong Chok" "Pathum Wan"   # or just some
```
Footprints: Google Open Buildings v3 (confidence >= 0.7); heights: Open Buildings 2.5D Temporal (latest year);
facilities: OpenStreetMap. The **Buildings 3D** tab shows one district at a time (full footprints, or a
lighter column mode for large districts / online use) and a whole-city view of ~520 m columns. In the radar
view a building counts as affected when floodwater was detected within ~65 m (radar can't see water between
buildings).

## Data (all free, no API keys; layers marked * need a free Google Earth Engine account)
| Layer | Source |
|---|---|
| Elevation | Copernicus GLO-30 DEM (AWS open data) |
| Population | WorldPop 2020 constrained, 100 m |
| Rivers & canals | OpenStreetMap (Overpass API) |
| Districts | geoBoundaries THA ADM2 |
| 2011 flood (ML labels) | Global Flood Database event DFO_3850, MODIS 250 m (website download or Earth Engine*) |
| Land cover (ML features) | ESA WorldCover 2021, 10 m (AWS open data) |
| Current flood | Sentinel-1 radar* (10 m, mapped at 20 m); permanent water from JRC Global Surface Water* |
| Rainfall | Open-Meteo weather-model estimate, daily |
| River flow | GloFAS via the Open-Meteo Flood API, Chao Phraya at Nonthaburi, daily; normal = 1996-2025 median |
| Buildings | Google Open Buildings v3 footprints + 2.5D Temporal heights* |
| Critical facilities | OpenStreetMap (hospitals, clinics, schools, universities, police, fire stations) |
| Flood defences | Routes: Royal Development Projects Board, OpenStreetMap roads; crests: BMA, JICA, Airports of Thailand |

## Project layout
```
app.py                          Streamlit UI
scripts/prepare_data.py         Download + align all data onto a ~65 m grid
scripts/prepare_ml_data.py      2011 flood labels + land cover for the ML model
scripts/build_flood_archive.py  Wet-season radar flood labels, 2017-2025
scripts/train_model.py          Train + evaluate the LightGBM flood models
scripts/fetch_current_flood.py  Sentinel-1 radar flood maps, rainfall and river flow
scripts/evaluate_current.py     Score every method against the radar maps
scripts/prepare_buildings.py    Building footprints, heights and facilities per district
scripts/prepare_defences.py     River walls, King's Dike and airport dike -> crest-height grid
scripts/capture_screenshots.py  Screenshots for the presenter guide (app on port 8599)
scripts/build_guide_pdf.py      docs/presenter_guide.html -> PDF
src/config.py                   Study area, grid, file paths
src/data_loader.py              Load processed data
src/risk.py                     Risk index
src/simulate.py                 Flood simulation (optionally with defences) + impact stats
src/defences.py                 Defence lines -> crest grid, load lines for the map
src/features.py                 ML features
src/ml.py                       Load the ML model outputs
src/current_flood.py            Load radar passes, rainfall and river flow
src/validation.py               Block-bootstrap intervals and the Validation tab's tables
src/buildings.py                Load buildings, flood impact per building
src/viz.py                      Map layers, animation frames, charts
src/viz3d.py                    3D building maps (pydeck)
tests/                          pytest suite (synthetic study area + app smoke test)
```

## Tests
```bash
pip install -r requirements-dev.txt
pytest                      # ~30 s
```
Unit tests use a tiny synthetic 20×20 study area (`tests/conftest.py`): river, lowland, a ridge, a sealed
pocket and high ground. They cover the flood simulation, risk index, radar labels, per-pass weather, the alert rules,
the bootstrap scores (checked against scikit-learn), the wet-season labels and the multi-flood training sets.
`tests/test_app_smoke.py` runs the whole app headlessly on the real data, and skips itself if
`data/processed/` is missing. GitHub Actions (`.github/workflows/ci.yml`) runs everything on every push, on
Python 3.9 (local development) and 3.14 (what Streamlit Community Cloud runs, with newer pandas and numpy).

## Limitations
Exploratory model, not a forecast: flood walls and dikes only with the defences switch, one crest per
section; pumps, drainage tunnels and gate capacity aren't modelled;
water is static (no flow, rainfall timing, or tides); the DEM is a surface model that includes
some buildings; population is a 2020 estimate.

## Presenting
`docs/Bangkok_Flood_Simulator_Presenter_Guide.pdf` is a presenter guide with screenshots, a step-by-step
demo script, and when to change the risk-index weights. Regenerate it with
`scripts/capture_screenshots.py` (needs the app running on port 8599) and `scripts/build_guide_pdf.py`
(both need `pip install playwright` and Google Chrome).
