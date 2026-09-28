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
To refresh (e.g. new radar passes), re-run the prepare scripts locally, then commit and push `data/processed/`.
`shap` is only needed for `train_model.py`: `pip install shap` before training.

## Features
- **Water-level slider (0–3 m)** with live flooded area, people affected, and districts at risk
- **"Bathtub + connectivity" flood model**: low ground floods only if linked to a river/canal (toggleable)
- **Flood risk index** blending low elevation, proximity to waterways, and population density, with adjustable weights
- **Interactive map**: flood-depth, ground-elevation (adjustable colour range + relief shading) and risk overlays, rivers & canals, district hover tooltips
- **Elevation profile**: histogram of land area by height vs. the water level, lowest districts
- **Animation** of rising water from 0 to 3 m
- District ranking chart, flood-vs-level curve, and a sortable district table

## Machine-learning model (optional)
A LightGBM model learns flood susceptibility from where Bangkok **actually flooded in 2011**
(Global Flood Database event DFO_3850, MODIS 250 m) instead of hand-set weights. It's evaluated with spatial
block cross-validation against the hand-weighted risk index and the bathtub simulation.

```bash
# 1. Labels: download event 3850's map from global-flood-database.cloudtostreet.ai (no account) ...
python scripts/prepare_ml_data.py --label-file data/raw/gfd_dfo_3850.tif
#    ... or fetch it with a free Google Earth Engine login (run `earthengine authenticate` once)
python scripts/prepare_ml_data.py --gee-project YOUR_CLOUD_PROJECT_ID
# 2. Train + evaluate (~2 min); prints scores vs the existing methods
python scripts/train_model.py
```
**Results (spatial 5-fold CV on 4 km blocks, 846k labelled cells, 21% flooded):**

| Method | ROC-AUC | PR-AUC |
|---|---|---|
| ML model (terrain + land cover) | 0.867 | 0.669 |
| Bathtub simulation | 0.668 | 0.288 |
| Hand-weighted risk index (default weights) | 0.448 | 0.181 |
| Random guess | 0.5 | 0.211 |

Population density and built-up land are excluded by default: with them the model scores 0.893, but its top
factor becomes "dense city = dry", which mostly reflects flood defences and MODIS missing urban floodwater.

The app then gains "ML flood susceptibility" and "Observed flood (2011)" layers and a **ML model** tab.
Caveats: MODIS under-detects flooding between buildings, the model learned from one river/tidal flood, and
it implicitly learns 2011-era flood defences.

## Current flood from satellite radar (optional)
Maps where Bangkok is flooding **now** from Sentinel-1 radar (sees through cloud; a new pass every few days),
using the same free Earth Engine project as the ML model:

```bash
python scripts/fetch_current_flood.py --gee-project YOUR_CLOUD_PROJECT_ID   # default: last 45 days
python scripts/evaluate_current.py      # scores every method against each radar pass
```
Each pass is compared with the same weeks last year from the same orbit (`--baseline dry` compares with
January-March instead), with permanent water removed (JRC Global Surface Water). The app gains a
"Radar flood" layer with a date picker, a "Flooded now" column and a **Current flood** tab with rainfall
(Open-Meteo, no key). Re-run both commands to pick up new passes.

**27 Sep 2026 pass** (after ~200 mm of rain on 25-27 Sep): 99 km² of unusual water across greater Bangkok,
32 km² inside the city, mostly eastern farmland (Nong Chok 8.3%, Lat Krabang 5.9%). Tested against it, the
2011-trained ML model scores ROC-AUC 0.91 (0.94 within Bangkok), the bathtub 0.70 and the risk index 0.39.

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

## Data (all free, no API keys)
| Layer | Source |
|---|---|
| Elevation | Copernicus GLO-30 DEM (AWS open data) |
| Population | WorldPop 2020 constrained, 100 m |
| Rivers & canals | OpenStreetMap (Overpass API) |
| Districts | geoBoundaries THA ADM2 |

## Project layout
```
app.py                  Streamlit UI
scripts/prepare_data.py Download + align all data onto a ~65 m grid
src/config.py           Study area, grid, file paths
src/data_loader.py      Load processed data
src/risk.py             Risk index
src/simulate.py         Flood simulation + impact stats
src/viz.py              Map layers, animation frames, charts
```

## Limitations
Exploratory model, not a forecast: ignores flood walls, pumps, drainage tunnels and water gates;
water is static (no flow, rainfall timing, or tides); the DEM is a surface model that includes
some buildings; population is a 2020 estimate.

## Presenting
`docs/Bangkok_Flood_Simulator_Presenter_Guide.pdf` is a presenter guide with screenshots, a step-by-step
demo script, and when to change the risk-index weights. Regenerate it with
`scripts/capture_screenshots.py` (needs the app running on port 8599) and `scripts/build_guide_pdf.py`
(both need `pip install playwright` and Google Chrome).
