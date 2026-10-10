# Bangkok Flood Simulator: methods

A short, citable description of the data, models, validation and limitations. The README covers how to run
everything; this file explains what the numbers mean. All results below are as of 10 October 2026.

## 1. Study area and grid
- **Area:** greater Bangkok, 100.30–100.95°E, 13.48–14.00°N (about 18 M people).
- **"Bangkok (50 districts)":** the city proper (about 10.9 M people, 1,581 km²), using geoBoundaries THA ADM2.
- **Grid:** every layer is aligned to one 0.0006° grid (about 65 m; 1,083 × 867 cells, EPSG:4326). Cell areas
  allow for latitude.

## 2. Data
| Layer | Source | Notes |
|---|---|---|
| Elevation | Copernicus GLO-30 DEM | The lowest 30 m value in each grid cell, which reduces building and tree bias. A surface model, rounded to 0.5 m steps over many flat areas. |
| Population | WorldPop 2020, constrained, 100 m | Modelled residential population |
| Rivers and canals | OpenStreetMap (`waterway=river/canal`) | Flood sources in the bathtub model |
| Land cover | ESA WorldCover 2021, 10 m | Fractions per cell (ML features) |
| 2011 flood | Global Flood Database, event DFO_3850 (MODIS, 250 m) | ML labels |
| Wet seasons 2017–2025 | Sentinel-1 GRD via Google Earth Engine | ML labels (section 5) |
| Current flood (2026) | Sentinel-1, the same classification | Test data only |
| Permanent water | JRC Global Surface Water, at least 50% occurrence | Removed from radar labels |
| Rain | Open-Meteo: forecast API (recent), historical archive / ERA5 (past seasons) | Model estimates at 13.80°N 100.75°E |
| River flow | GloFAS via the Open-Meteo Flood API, Chao Phraya at Nonthaburi | Model estimate. Its "normal" is the 1996–2025 median for the date. |
| Flood defences | Routes from published descriptions, traced on OpenStreetMap roads | Crest heights from BMA, JICA and Airports of Thailand (low end of published ranges) |
| Buildings | Google Open Buildings v3 and 2.5D Temporal heights | Used only in the 3D view |

## 3. Bathtub flood simulation (`src/simulate.py`)
- **Basic rule:** a cell floods at water level *L* if its elevation is at most *L*. Depth = *L* − elevation.
- **River-connected mode (default):** low cells flood only if they belong to an 8-connected region that touches a
  river or canal cell. Sealed pockets stay dry.
- **Rain-ponding mode:** every cell at or below *L* floods.
- **With defences:** the defence lines are drawn onto the grid with every cell they touch, so water spreading
  diagonally can't slip through a one-cell-wide line.
  - A defence cell blocks water while its crest is above *L*.
  - Only rivers and canals connected to the map edge without crossing a standing wall feed the flood. Canals
    inside a walled area are assumed to be kept low by pumps and gates.
- **Outputs:** flooded area and people for greater Bangkok, and the same plus % flooded for each of the 50 districts.

## 4. Risk index (`src/risk.py`)
- **Blend:** a weighted average of three factors, each scaled 0–1 per cell:
  - low elevation: 2nd–98th percentile stretch, inverted;
  - nearness to a waterway: exponential decay, about 37% at 1 km;
  - log population density.
- **Weights:** set by the user. Only their ratios matter.
- **Meaning:** the index is relative, not a probability.

## 5. Flood labels
- **2011:** the Global Flood Database map, resampled to the grid. A cell is *flooded* if at least 50% of it is
  flooded, and *dry* otherwise. It is unknown if it is permanent water (at least 50%) or had too few cloud-free
  views to call it dry. 21.1% of 846,343 labelled cells flooded.
- **Radar wet seasons** (`scripts/build_flood_archive.py`):
  - **Classification:** each August–November pass is classified in Earth Engine, after a 30 m focal-median speckle
    filter. A pixel is flooded if VV backscatter is below −18 dB and at least 3 dB darker than the baseline. The
    baseline is the same orbit's median over ±21 days around the same date a year earlier.
  - **Seen:** a cell is seen if radar data covers at least 50% of it and permanent water covers under 50%.
  - **Season labels:** a cell is *flooded* if it is at least 50% flooded on 2 or more passes. It is *dry* if seen and
    always under 10% flooded. Anything else is unknown.
  - **Result:** 2.2–5.1% of about 755,000 labelled cells flooded per season.
- **Current passes (2026):** the same rule applied to each pass on its own.

## 6. ML susceptibility model (`scripts/train_model.py`)
**Features** (static, describing the place, not the weather):
- elevation, height above the nearest waterway, and elevation relative to the 1 km surroundings;
- slope;
- distance to any river or canal, and distance to the Chao Phraya;
- fractions of cropland, open water, trees, grassland and wetland.

Population density and built-up land are left out on purpose. With them the model learns "dense city = dry",
which mostly reflects flood defences and satellites missing water between buildings.

**Training:**
- LightGBM: 400 trees, learning rate 0.05, 31 leaves, at least 200 samples per leaf, 0.8 row and column
  subsampling, and positives weighted to balance the classes.
- Ten events: 2011 and 2017–2025. 200,000 labelled cells are sampled from each event, and weights make every event
  count equally.
- Floods from 2026 on are never trained on.
- A second model trained on 2011 alone is kept for comparison.

**Output:** the flood probability per cell, the SHAP importance of each feature, and per-district means.

## 7. Validation (`src/validation.py`, `scripts/evaluate_current.py`)
**Scores:**
- ROC-AUC: the chance a flooded cell scores higher than a dry one.
- PR-AUC (average precision). A random guess scores the flooded share.

Each method is scored on cells it has not seen:
- the ML probability;
- the bathtub onset, i.e. the lowest level at which a cell floods (negated, so earlier flooding scores higher);
- the risk index.

**Three tests:**
1. **Spatial cross-validation:** 4 km blocks are split into 5 folds. A block is held out of every event at once, so
   a model is never trained on a place it is tested on, in any year. Out-of-fold predictions are pooled and scored
   per event.
2. **Leave one event out:** train on nine events, then score every labelled cell of the tenth.
3. **2026 radar passes:** no model has seen them.

**Intervals:** a spatial block bootstrap.
- 4 km blocks are resampled with replacement 200 times, and every method is re-scored on each resample. The 95%
  interval is the 2.5th to 97.5th percentile.
- Every method uses the same resamples, so the difference between two methods also gets an interval. A difference
  counts as "clear" when that interval excludes zero.

**Results (ROC-AUC):**

| Test | ML, all floods | ML, 2011 only | Bathtub + defences | Bathtub | Risk index |
|---|---|---|---|---|---|
| 2011, spatial CV | 0.853 | **0.872** | 0.721 | 0.667 | 0.449 |
| 2017–2025, spatial CV (mean) | **0.975** | 0.918 | 0.795 | 0.749 | 0.421 |
| Each season left out | **0.972–0.987** | 0.901–0.931 | 0.769–0.807 | | |
| 9 passes, 2026 (greater Bangkok) | **0.91–0.96** | 0.82–0.91 | 0.74–0.86 | 0.69–0.85 | 0.37–0.57 |

- **Against the bathtub:** the all-floods model is clearly better on all 9 passes in both areas.
- **Against the 2011-only model on 27 Sep 2026:** 0.955–0.968 vs 0.892–0.923; the intervals don't overlap.
- **Trade-off:** the all-floods model is slightly weaker on the extreme 2011 flood. Nine of its ten events are
  ordinary seasons.

**Event context** (`scripts/event_context.py`; not a model input):
- 2011's August–November rain was 107% of normal.
- But the Chao Phraya was at or above 130% of normal for 122 days, against 0–55 days in 2017–2025.
- So 2011 was a river flood from upstream, not a local-rain flood.

## 8. Uncertainty
### Elevation error (`scripts/dem_uncertainty.py`, `src/uncertainty.py`)
**Method:**
- The bathtub model is rerun 50 times.
- Each run adds a zero-mean Gaussian random field to the DEM: standard deviation σ = 0.7 m, with correlation
  exp(−(r/300 m)²). The field is made by Gaussian-smoothing white noise with a kernel s.d. of 150 m, then rescaling
  it to σ.
- Defence crests are kept fixed.
- Reported: the 5th, 50th and 95th percentiles of flooded area, people, and districts more than 25% flooded, for
  each level and mode. Also the share of runs in which each cell floods, at 0.5, 1.0 … 3.0 m.

**Basis:**
- **Method:** simulating plausible versions of a global DEM with correlated error follows Hawker et al. (2018).
- **Error size:** σ is an assumption. Copernicus GLO-30 has a mean absolute error of about 1.6 m in built-up areas
  (Hawker et al. 2022), mostly from buildings, much of which taking the cell minimum removes.
- **Sensitivity run:** the pessimistic case (σ = 1.4 m) is kept for comparison.

**Findings (river-connected, greater Bangkok):**

| Level | Single run | 90% range, σ 0.7 m | 90% range, σ 1.4 m |
|---|---|---|---|
| 1.0 m, area | 1,024 km² | 951–984 km² | 1,135–1,174 km² |
| 1.0 m, people | 1.02 M | 1.26–1.34 M | 2.24–2.36 M |
| 1.5 m, area | 1,553 km² | 1,511–1,543 km² | |
| 1.5 m, people | 2.11 M | 2.47–2.57 M | |

- **Totals are stable within a given σ, but σ matters a lot.** Doubling it nearly doubles the people affected
  at 1.0 m.
- **Individual cells are uncertain.** At 1.5 m only 69 km² of Bangkok's 50 districts floods in at least 90% of runs.
  Another 299 km² floods in 50–90% of runs, and 487 km² in 10–50%.
- **The single run lies outside its own range.** Because of the DEM's 0.5 m steps, whole plateaus sit exactly at the
  water line. Small errors also open paths to populated land.

### ML model disagreement
- **What it is:** the per-cell standard deviation of the flood probability across the five spatial-CV models
  (`data/processed/ml_disagreement.tif`). The median is 0.015 and the 99th percentile 0.09.
- **What it shows:** sensitivity to which places the model learned from. It doesn't cover label or feature error.

## 9. Limitations
- **Static water:** no flow speed, rain timing, tides or duration. Pumps, drainage tunnels, gate capacity and the
  west bank's polder dikes are not modelled. Each defence section has a single crest height.
- **Elevation:** the DEM is a surface model, ground subsidence is ignored, and the elevation-error size is assumed.
- **Labels:** MODIS and radar both under-detect water between buildings, so the dense city looks drier than it was.
  Flooded fields and aquaculture can look like floods. The seasonal baseline hides land that floods every year, and
  a wet year can hide floods in the year after.
- **ML model:** it learns *where* floods happen, not *when*. It knows nothing about short cloudbursts in the city,
  and it under-rates places that flood only in extreme years.
- **Tests aren't independent:** passes a few days apart see the same flood, so a run of wins is a trend, not
  separate trials. Most detected flooding is farmland on the city edge, so the scores mostly describe the rural fringe.
- **Uncertainty shown is partial:** the ranges cover elevation error and training-data sensitivity only, so the
  real uncertainty is larger.

## References
- Hawker L, Bates P, Neal J, Rougier J (2018). Perspectives on digital elevation model (DEM) simulation for flood
  modeling in the absence of a high-accuracy open access global DEM. *Frontiers in Earth Science* 6:233.
  doi:10.3389/feart.2018.00233
- Hawker L, Uhe P, Paulo L, Sosa J, Savage J, Sampson C, Neal J (2022). A 30 m global map of elevation with forests
  and buildings removed. *Environmental Research Letters* 17:024016. doi:10.1088/1748-9326/ac4d4f
- Tellman B, et al. (2021). Satellite imaging reveals increased proportion of population exposed to floods.
  *Nature* 596:80–86 (Global Flood Database).
- Ke G, et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. *NeurIPS* 30.
- Lundberg SM, Lee S-I (2017). A unified approach to interpreting model predictions. *NeurIPS* 30 (SHAP).
