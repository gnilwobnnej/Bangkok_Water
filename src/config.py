"""Shared settings: study area, analysis grid, and file locations."""
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# Bangkok metropolitan bounding box (lon/lat, EPSG:4326)
WEST, SOUTH, EAST, NORTH = 100.30, 13.48, 100.95, 14.00
CRS = "EPSG:4326"

# Analysis grid resolution in degrees (~65 m at Bangkok's latitude)
RES = 0.0006
WIDTH = round((EAST - WEST) / RES)
HEIGHT = round((NORTH - SOUTH) / RES)

DEM_FILE = PROCESSED_DIR / "dem.tif"
DEM_SOURCE = "copernicus"  # scripts/prepare_data.py --dem: "copernicus" (surface) or "fabdem" (buildings removed)
POP_FILE = PROCESSED_DIR / "population.tif"
WATER_FILE = PROCESSED_DIR / "waterways.tif"
DISTRICTS_FILE = PROCESSED_DIR / "districts.geojson"
WATERWAY_LINES_FILE = PROCESSED_DIR / "waterways.geojson"

# Machine-learning model (scripts/prepare_ml_data.py, scripts/train_model.py)
MODELS_DIR = ROOT / "models"
FLOOD_2011_FILE = PROCESSED_DIR / "flood_2011.tif"        # 1 flooded, 0 dry, 255 unknown
LANDCOVER_FILE = PROCESSED_DIR / "landcover_fractions.tif"
ML_PROB_FILE = PROCESSED_DIR / "ml_susceptibility.tif"             # trained on every flood event
ML_PROB_2011_FILE = PROCESSED_DIR / "ml_susceptibility_2011.tif"   # trained on 2011 only (the "before" model)
ML_REPORT_FILE = PROCESSED_DIR / "ml_report.json"
ML_MODEL_FILE = MODELS_DIR / "flood_lgbm.txt"
ML_DISAGREEMENT_FILE = PROCESSED_DIR / "ml_disagreement.tif"  # std of the CV fold models' probabilities
# Street-flood model (train_model.py --kind street), from Traffy Fondue reports; Bangkok's 50 districts, NaN elsewhere
ML_STREET_PROB_FILE = PROCESSED_DIR / "ml_street_susceptibility.tif"
ML_STREET_DISAGREEMENT_FILE = PROCESSED_DIR / "ml_street_disagreement.tif"
ML_STREET_REPORT_FILE = PROCESSED_DIR / "ml_street_report.json"
ML_STREET_MODEL_FILE = MODELS_DIR / "flood_street_lgbm.txt"

# Elevation uncertainty (scripts/dem_uncertainty.py)
DEM_UNCERTAINTY_DIR = PROCESSED_DIR / "dem_uncertainty"
DEM_UNCERTAINTY_FILE = DEM_UNCERTAINTY_DIR / "summary.json"  # P5/P50/P95 curves per mode
# ...plus chance_<mode>.tif: % of runs flooded, one uint8 band per level in src/uncertainty.PROB_LEVELS
# The same with twice the error (--sigma 1.4), curves only: shows how much the assumed error size matters
DEM_UNCERTAINTY_PESSIMISTIC_FILE = PROCESSED_DIR / "dem_uncertainty_pessimistic" / "summary.json"
LABEL_NODATA = 255
# Wet-season radar flood labels, one per year (scripts/build_flood_archive.py), same 1/0/255 format
FLOOD_LABELS_DIR = PROCESSED_DIR / "flood_labels"        # radar_YYYY.tif + summary.json
ARCHIVE_YEARS = range(2017, 2026)                        # 2017-2025
TEST_YEARS_FROM = 2026  # floods from this year on are kept out of training: evaluate_current.py tests on them

# Current flood from Sentinel-1 radar (scripts/fetch_current_flood.py, scripts/evaluate_current.py)
RADAR_DIR = PROCESSED_DIR / "radar_flood"            # one GeoTIFF per pass date
RADAR_SUMMARY_FILE = RADAR_DIR / "summary.json"
RADAR_EVAL_FILE = RADAR_DIR / "evaluation.json"

# "Unusual conditions" banner (src/current_flood.py conditions_alert)
ALERT_FLOOD_RATIO = 2.0      # latest pass flooded area >= this x the median of earlier passes
ALERT_MIN_EARLIER_PASSES = 3  # ...judged only once there are this many earlier passes to compare with
ALERT_PASS_MAX_AGE_DAYS = 14  # ...and only while the latest pass is this recent
ALERT_RIVER_PCT = 130        # Chao Phraya flow >= this % of normal for the date
ALERT_RAIN_3DAY_MM = 100     # rain over the last 3 days >= this

# Flood defences (scripts/prepare_defences.py): river walls and dikes as crest heights (m above mean sea level)
DEFENCES_FILE = PROCESSED_DIR / "defences.tif"          # float32 crest height, NaN = no defence
DEFENCE_LINES_FILE = PROCESSED_DIR / "defences.geojson"  # the lines, with name, crest_m, source and note

# Street flooding from Traffy Fondue citizen reports (scripts/fetch_traffy.py, scripts/build_traffy_labels.py)
TRAFFY_DIR = PROCESSED_DIR / "traffy"
TRAFFY_REPORTS_DIR = TRAFFY_DIR / "flood_reports"       # every flood report (point, time, district, text),
                                                        # one YYYY-MM.parquet per month so daily updates stay small
TRAFFY_OTHER_FILE = TRAFFY_DIR / "other_counts.tif"       # reports of anything else per cell, one band per year
TRAFFY_STORMS_FILE = TRAFFY_DIR / "summary.json"          # storm days and their rain, label counts per year
TRAFFY_YEARS = range(2022, datetime.date.today().year + 1)  # reports start in 2022; TEST_YEARS_FROM on are tests
TRAFFY_STORM_MIN_REPORTS = 15   # a storm day has >= this many flood reports...
TRAFFY_STORM_SPIKE = 3.0        # ...and >= this x the typical daily count for the time of year...
TRAFFY_STORM_MIN_RAIN_MM = 10   # ...and >= this much rain (wettest point in the city) that day or the day before
TRAFFY_MIN_STORM_REPORTS = 1    # label flooded: >= this many flood reports on storm days in the year
TRAFFY_MIN_OTHER_REPORTS = 5    # label dry: no flood report, and >= this many reports of other problems
TRAFFY_EVAL_FILE = TRAFFY_DIR / "evaluation.json"  # every method scored on the test years' reports
DENSE_BUILT_UP_PCT = 90  # "dense city": cells at least this % built-up (ESA WorldCover), ~30% of Bangkok

# Buildings (scripts/prepare_buildings.py)
BUILDINGS_DIR = PROCESSED_DIR / "buildings"              # one GeoParquet per district
BUILDINGS_GRID_FILE = PROCESSED_DIR / "buildings_grid.tif"  # band 1 building count, band 2 mean height (m)
FACILITIES_FILE = PROCESSED_DIR / "facilities.geojson"
