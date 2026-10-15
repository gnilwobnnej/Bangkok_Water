# Seeing floods between buildings: data for Bangkok's dense city

Written 14 October 2026. Radar (Sentinel-1) and MODIS miss most water between buildings, so in every training
flood the dense city looks drier than it was. The ML model then learns "dense city = dry", which is why
population and built-up land are left out of it. This note ranks the free data that could fill the gap for
**rain and street flooding**, as training labels and as an independent test, and says what was built.

## The sources, best first

| # | Source | What it gives | Strength | Weakness | Status |
|---|---|---|---|---|---|
| 1 | **Traffy Fondue** (BMA's citizen-report app), public API, no key | 44,442 flood reports since 2022, each with a point, time (to the second), photo and text; plus ~1.5 M reports of other problems | Densest exactly where satellites are blind; catches floods that drain within hours | Reports follow people and app use; some "flood" reports are chronic drain complaints | **Built** (below) |
| 2 | **FABDEM** (Copernicus DEM with buildings and trees removed; Hawker et al. 2022) | A better ground surface in the dense city | Fixes the DEM's building bias (~1.6 m in built-up areas) | A better input, not flood observations; non-commercial licence | Tested: helps street floods, hurts river floods; not the default |
| 3 | **BMA Drainage and Sewerage Dept** flood points and rain gauges | ~737 known flood-prone points; rain and street-water sensors behind floodmanagement.bangkok.go.th | Official, long record | No bulk download found; may need a data request | Next |
| 4 | **Sentinel-1 "double bounce"** in the existing radar pipeline | Flooded streets make radar *brighter* (water + wall reflection) | Same data and code already used | A pass every few days misses most cloudbursts; depends on wall orientation | Tested: too weak to use |
| 5 | **Sentinel-1 InSAR coherence** (ASF HyP3, free Earthdata login) | Coherence drops where water fills streets | The most reliable satellite urban-flood signal (UrbanSARFloods; Mason et al. 2021) | Heavy processing; same timing problem | Later, river-scale events only |
| — | Commercial radar or optical imagery (ICEYE, Capella, Planet) | Sub-metre images | — | Paid | Not used |
| — | Copernicus EMS / UNOSAT maps | Rapid flood maps | — | Large river floods only | Not used |

## What was built: street-flood labels from Traffy Fondue

1. **Download** (`scripts/fetch_traffy.py`). Every flood report (the API's `problem_type=น้ำท่วม` filter), and
   every report of any kind on every third day, which gives where people report problems at all. Times are
   Bangkok local. One-day pages avoid the API's failures on deep paging.
2. **Storm days** (`src/traffy.storm_days`). A day with at least 15 flood reports, at least 3× the usual daily
   count for the time of year, and at least 10 mm of rain at the wettest of 12 points over Bangkok (ERA5) that
   day or the day before. 164 storm days from 2022 to 2026 hold 62% of all flood reports. The rain rule drops
   16 bursts of complaints on dry days.
3. **Labels by case-control** (`scripts/build_traffy_labels.py`), one map per year on the 65 m grid:
   - **flooded:** a flood report on a storm day;
   - **dry:** no flood report all year, but at least 5 reports of other problems, so people there use the app;
   - **unknown:** everything else.

   Taking dry cells only where people report keeps both classes in the same busy, built-up places. Otherwise a
   model could score well just by learning where people live.
4. **A separate street-flood model** (`train_model.py --kind street`), trained on 2022-2025 and tested on 2026.
   2026 includes the 25-29 September storm (~140 mm in a day, 10,000+ flood reports). The model keeps
   population and built-up land, and adds small dips in the ground (300 m) and building density and height.
5. **Validation** (`scripts/evaluate_street.py`). Every method is scored on the 2026 reports, for the year and
   storm by storm, in Bangkok and in the **dense city** (cells ≥ 90% built-up), which radar can't test.

## Results
| Test (ROC-AUC) | ML street | ML, all floods (radar) | Bathtub + defences | Risk index |
|---|---|---|---|---|
| 2022-2025, each year left out | **0.71-0.74** | | 0.52-0.56 | |
| 2026, Bangkok | **0.783** | 0.671 | 0.596 | 0.591 |
| 2026, dense city | **0.792** (0.752-0.821) | 0.669 | 0.562 | 0.576 |
| 2026 storm days, dense city: best method | **25 of 25** | | | |
| 25-29 Sep 2026 storm, dense city | **0.81-0.82** | 0.66-0.75 | 0.54-0.61 | 0.58-0.61 |

The labels pass a sanity check: flooded cells turn up within 300 m of known flood roads (Ratchadaphisek,
Pattanakarn, Sukhumvit 101/1, Chan Road, Lat Phrao 101, Ngam Wong Wan) in most years. The street model's top
factors are distance to the Chao Phraya, building count, ground elevation and local dips. Population and
built-up land add under 0.01. Details: README ("Street flooding from residents' reports") and the app's
Validation tab.

## FABDEM vs Copernicus elevation (`scripts/compare_dem.py`)
FABDEM (Copernicus with buildings and trees removed, from the Earth Engine community catalog) puts the dense city
a median 0.72 m lower (Bangkok as a whole: 0.18 m lower; greater Bangkok: no change). The bathtub model was run on
both, with and without defences, and scored on every flood with labels. Each change has a 95% block bootstrap
interval, and every one of the 29 excludes zero:

| Test (ROC-AUC, bathtub + defences) | Copernicus | FABDEM |
|---|---|---|
| 2011 flood (MODIS) | **0.721** | 0.669 |
| Wet seasons 2017-2025 (radar) | **0.77-0.81** | 0.64-0.70 |
| 2026 radar passes | **0.74-0.86** | 0.61-0.82 |
| Street flooding 2022-2025, dense city | 0.51-0.53 | **0.57-0.61** |
| Street flooding 2026, dense city | 0.562 | **0.675** |

FABDEM is better on all 10 street-flood tests and worse on all 19 river-flood tests. Lowering the city makes the
bathtub flood it earlier, which matches where rain pools in streets but not where the river spreads. The river
floods are what the bathtub is for, so **Copernicus stays the default** (`config.DEM_SOURCE`).
`prepare_data.py --dem fabdem --dem-only` switches. FABDEM elevation could still become a street-model feature.

## Sentinel-1 double bounce (`scripts/double_bounce.py`)
Built-up ground (ESA WorldCover) flagged where VV backscatter rose by at least 3 dB over the median of the same
orbit's scenes in the previous 60 days. Tested on every pass since 2022 that came within 6 hours of at least 20
flood reports on a storm day (17 wet passes, 16 with enough cells to score), and on 12 quiet dry-season passes as
controls. Flooded = a flood report within 6 hours of the pass; dry = that year's Traffy "dry" label.

| | Double bounce | ML street model |
|---|---|---|
| ROC-AUC on the 16 wet passes, median (range) | 0.52 (0.44-0.70) | **0.82 (0.66-0.94)** |
| Passes where the interval is above a coin flip (0.5) | 3 of 16 | |
| Passes where ML street is clearly better | | 13 of 16 |
| 27 Sep 2026 storm (1,044 flooded cells) | 0.69 (0.65-0.71) | **0.81** |

Built-up Bangkok does brighten more on wet passes (median 3.6% of the ground vs 1.7% on dry-season controls), but
not specifically where streets flooded: rain-soaked roofs and roads also brighten. Only the largest storm, with
widespread standing water at pass time, shows a usable signal. Two passes a day (06:00 and 18:00) rarely catch
water that drains in hours. **Not added** to the radar pipeline or the app.

## Sources
- Traffy Fondue: [overview](https://www.thailand.go.th/issue-focus-detail/001_05_025),
  [one million complaints](https://pr-bangkok.com/?p=583017); API: `publicapi.traffy.in.th/teamchadchart-stat-api/geojson/v1`.
- FABDEM V1-2: Earth Engine `projects/sat-io/open-datasets/FABDEM`
  ([community catalog](https://gee-community-catalog.org/projects/fabdem/)); CC BY-NC-SA 4.0.
- BMA flood risk platform: [CivicDataLab](https://civicdatalab.in/work/climateaction/bma-flood-risk-data-platform/);
  flood-prone points: [Thai PBS World](https://www.thaipbsworld.com/?p=294390).
- Urban SAR flood mapping: [UrbanSARFloods](https://arxiv.org/html/2406.04111v1);
  [Mason et al. 2021, J. Appl. Remote Sens. 15:032003](https://spiedigitallibrary.org/journals/journal-of-applied-remote-sensing/volume-15/issue-3/032003/Floodwater-detection-in-urban-areas-using-Sentinel-1-and-WorldDEM/10.1117/1.JRS.15.032003.pdf);
  [ISPRS Annals 2025](https://isprs-annals.copernicus.org/articles/X-G-2025/847/2025/).
- Hawker L et al. (2022). A 30 m global map of elevation with forests and buildings removed. *Environ. Res.
  Lett.* 17:024016.
