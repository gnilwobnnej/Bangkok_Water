"""Prepare building footprints, heights and critical facilities for the 3D buildings tab.

    python scripts/prepare_buildings.py --gee-project YOUR_CLOUD_PROJECT_ID              # all 50 districts
    python scripts/prepare_buildings.py --gee-project ID --districts "Nong Chok" "Pathum Wan"

Data (free; footprints and heights via Google Earth Engine, facilities via OpenStreetMap):
    Google Open Buildings v3 polygons      footprints, kept at confidence >= 0.7
    Google Open Buildings 2.5D Temporal    building height (4 m), latest year
    OpenStreetMap                          hospitals, clinics, schools, universities, police, fire stations

Outputs (data/processed/):
    buildings_grid.tif          building count and mean height per analysis cell (whole-city 3D view),
                                rebuilt from all prepared districts on every run
    buildings/<district_id>.parquet   footprints with height, area and analysis-grid row/col
    facilities.geojson          critical facilities with district and analysis-grid row/col
Districts already prepared are skipped unless --refresh is given.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
import shapely
from rasterio.features import rasterize
from rasterio.merge import merge

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import HEADERS, OVERPASS_URLS, PROFILE  # noqa: E402
from src import config as C  # noqa: E402

FOOTPRINTS = "GOOGLE/Research/open-buildings/v3/polygons"
HEIGHTS = "GOOGLE/Research/open-buildings-temporal/v1"
MIN_CONFIDENCE = 0.7          # Google's suggested threshold for precise footprints
HEIGHT_SCALE = 10             # heights downloaded as uint16 in 0.1 m steps (towers are well over 100 m)
MAX_BUILDINGS = 80_000        # above this, drop footprints under SMALL_M2 to keep districts light
SMALL_M2 = 30
SIMPLIFY_DEG = 0.00001        # ~1 m
HEIGHT_TILE_DEG = 0.1         # ~11 km tiles: 4 m uint16 stays well under the 50 MB download cap
FACILITY_TYPES = {
    "hospital": "Hospital", "clinic": "Clinic", "school": "School", "kindergarten": "School",
    "university": "University", "college": "University", "police": "Police", "fire_station": "Fire station",
}


def latest_heights(ee, region):
    """Latest-year mosaic plus its native 4 m projection (mosaic() itself drops the projection)."""
    col = ee.ImageCollection(HEIGHTS).filterBounds(region)
    latest = col.filter(ee.Filter.eq("inference_time_epoch_s", col.aggregate_max("inference_time_epoch_s")))
    return latest.mosaic(), latest.first().select("building_height").projection()


def build_grid():
    """Building count and mean height per analysis cell, from every prepared district.

    Built locally (aggregating the city's fine 2.5D pixels in Earth Engine exceeds its interactive
    limits), so the whole-city view counts exactly the same buildings as the district view.
    """
    count = np.zeros(C.HEIGHT * C.WIDTH)
    height_sum = np.zeros(C.HEIGHT * C.WIDTH)
    files = sorted(C.BUILDINGS_DIR.glob("*.parquet"))
    for f in files:
        df = pd.read_parquet(f, columns=["row", "col", "height_m"])
        idx = df["row"].to_numpy(int) * C.WIDTH + df["col"].to_numpy(int)
        count += np.bincount(idx, minlength=count.size)
        height_sum += np.bincount(idx, weights=df["height_m"].to_numpy(), minlength=count.size)
    mean_height = np.divide(height_sum, count, out=np.zeros_like(count), where=count > 0)
    with rasterio.open(C.BUILDINGS_GRID_FILE, "w", dtype="float32", **{**PROFILE, "count": 2}) as dst:
        dst.write(count.reshape(C.HEIGHT, C.WIDTH).astype("float32"), 1)
        dst.write(mean_height.reshape(C.HEIGHT, C.WIDTH).astype("float32"), 2)
        dst.set_band_description(1, "building_count")
        dst.set_band_description(2, "mean_height_m")
    print(f"  grid from {len(files)} district(s): {count.sum():,.0f} buildings")


def download(url: str, dest: Path):
    r = requests.get(url, timeout=900)
    if not r.ok:  # Earth Engine explains failures in the response body
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    dest.write_bytes(r.content)


def ee_geometry(ee, geom):
    return ee.Geometry(json.loads(gpd.GeoSeries([geom]).to_json())["features"][0]["geometry"])


def download_heights(ee, region, bounds, dest: Path):
    """4 m building heights for a district, fetched in tiles (Earth Engine caps one download at 50 MB)."""
    img = latest_heights(ee, region)[0].select("building_height")
    img = img.multiply(HEIGHT_SCALE).round().clamp(0, 65535).toUint16()
    west, south, east, north = bounds
    nx = max(1, int(np.ceil((east - west) / HEIGHT_TILE_DEG)))
    ny = max(1, int(np.ceil((north - south) / HEIGHT_TILE_DEG)))
    tiles = []
    for i in range(nx):
        for j in range(ny):
            tile = dest.with_name(f"{dest.stem}_{i}_{j}.tif")
            w, e = west + (east - west) * i / nx, west + (east - west) * (i + 1) / nx
            s, n = south + (north - south) * j / ny, south + (north - south) * (j + 1) / ny
            download(img.getDownloadURL({"region": ee.Geometry.Rectangle([w, s, e, n]), "scale": 4,
                                         "crs": "EPSG:4326", "format": "GEO_TIFF"}), tile)
            tiles.append(tile)
    sources = [rasterio.open(t) for t in tiles]
    mosaic, transform = merge(sources)
    profile = sources[0].profile
    for src in sources:
        src.close()
    profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform, compress="deflate")
    with rasterio.open(dest, "w", **profile) as dst:
        dst.write(mosaic)
    for t in tiles:
        t.unlink()


def prepare_district(ee, row, raw_dir: Path, refresh: bool):
    out = C.BUILDINGS_DIR / f"{row.district_id}.parquet"
    if out.exists() and not refresh:
        return None
    t0 = time.time()
    region = ee_geometry(ee, row.geometry)
    footprints_raw = raw_dir / f"{row.district_id}_footprints.geojson"
    heights_raw = raw_dir / f"{row.district_id}_height.tif"
    if refresh or not footprints_raw.exists():
        fc = ee.FeatureCollection(FOOTPRINTS).filterBounds(region).filter(ee.Filter.gte("confidence", MIN_CONFIDENCE))
        download(fc.getDownloadURL(filetype="geojson", selectors=["area_in_meters", "confidence", ".geo"]),
                 footprints_raw)
    if refresh or not heights_raw.exists():
        download_heights(ee, region, row.geometry.bounds, heights_raw)

    b = gpd.read_file(footprints_raw)
    b = b[b.geometry.notna() & ~b.geometry.is_empty].reset_index(drop=True)
    # Keep buildings whose centre lies in this district (filterBounds also returns edge-crossers).
    centroids = b.geometry.representative_point()
    b = b[centroids.within(row.geometry)].reset_index(drop=True)
    centroids = centroids[centroids.within(row.geometry)].reset_index(drop=True)

    # Mean height over each footprint from the 4 m height raster.
    with rasterio.open(heights_raw) as src:
        h = src.read(1).astype("float32") / HEIGHT_SCALE
        ids = rasterize(((g, i + 1) for i, g in enumerate(b.geometry)), out_shape=h.shape,
                        transform=src.transform, fill=0, dtype="int32", all_touched=True)
    n = len(b) + 1
    sums = np.bincount(ids.ravel(), weights=h.ravel(), minlength=n)
    cnt = np.bincount(ids.ravel(), minlength=n)
    height = np.divide(sums, cnt, out=np.zeros(n), where=cnt > 0)[1:]

    df = gpd.GeoDataFrame({
        "height_m": height.astype("float32"),
        "area_m2": b["area_in_meters"].astype("float32"),
        "lon": centroids.x.astype("float64"), "lat": centroids.y.astype("float64"),
    }, geometry=shapely.set_precision(b.geometry.simplify(SIMPLIFY_DEG).values, 1e-6), crs=C.CRS)
    dropped = 0
    if len(df) > MAX_BUILDINGS:
        small = df["area_m2"] < SMALL_M2
        dropped = int(small.sum())
        df = df[~small].reset_index(drop=True)
    df["row"] = np.clip(((C.NORTH - df["lat"]) / C.RES).astype(int), 0, C.HEIGHT - 1).astype("int16")
    df["col"] = np.clip(((df["lon"] - C.WEST) / C.RES).astype(int), 0, C.WIDTH - 1).astype("int16")
    df.to_parquet(out)
    return {"district_id": int(row.district_id), "district": row.district, "buildings": len(df),
            "dropped_small": dropped, "seconds": round(time.time() - t0, 1),
            "tall_share": float((df["height_m"] >= 20).mean())}


def prepare_facilities(districts: gpd.GeoDataFrame):
    print("[3/3] Critical facilities (OpenStreetMap)")
    types = "|".join(FACILITY_TYPES)
    query = (f'[out:json][timeout:180];nwr["amenity"~"^({types})$"]'
             f'({C.SOUTH},{C.WEST},{C.NORTH},{C.EAST});out center tags;')
    data = None
    for url in OVERPASS_URLS:
        try:
            r = requests.post(url, data={"data": query}, headers=HEADERS, timeout=200)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            print(f"  {url} failed: {e}")
    if data is None:
        raise RuntimeError("All Overpass mirrors failed; try again later.")
    rows = []
    for el in data["elements"]:
        lat, lon = (el["lat"], el["lon"]) if "lat" in el else (el.get("center", {}).get("lat"), el.get("center", {}).get("lon"))
        if lat is None:
            continue
        tags = el.get("tags", {})
        rows.append({"name": tags.get("name:en") or tags.get("name") or "(unnamed)",
                     "type": FACILITY_TYPES[tags["amenity"]], "lon": lon, "lat": lat})
    fac = gpd.GeoDataFrame(rows, geometry=gpd.points_from_xy([r["lon"] for r in rows], [r["lat"] for r in rows]),
                           crs=C.CRS)
    fac = gpd.sjoin(fac, districts[["district_id", "geometry"]], how="left", predicate="within").drop(columns="index_right")
    fac["district_id"] = fac["district_id"].fillna(0).astype(int)
    fac["row"] = np.clip(((C.NORTH - fac["lat"]) / C.RES).astype(int), 0, C.HEIGHT - 1)
    fac["col"] = np.clip(((fac["lon"] - C.WEST) / C.RES).astype(int), 0, C.WIDTH - 1)
    fac.to_file(C.FACILITIES_FILE, driver="GeoJSON")
    print(f"  {len(fac):,} facilities ({(fac.district_id > 0).sum():,} inside Bangkok): "
          + ", ".join(f"{k} {v}" for k, v in fac["type"].value_counts().items()))


def main(project, names, refresh, skip_facilities):
    import ee

    ee.Initialize(project=project)
    C.BUILDINGS_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = C.RAW_DIR / "buildings"
    raw_dir.mkdir(parents=True, exist_ok=True)
    districts = gpd.read_file(C.DISTRICTS_FILE)

    todo = districts if not names else districts[districts["district"].isin(names)]
    missing = set(names or []) - set(todo["district"])
    if missing:
        raise SystemExit(f"Unknown district name(s): {', '.join(sorted(missing))}")
    print(f"[1/3] Buildings for {len(todo)} district(s)")
    log_path = C.BUILDINGS_DIR / "summary.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    for row in todo.sort_values("district").itertuples():
        try:
            rec = prepare_district(ee, row, raw_dir, refresh)
        except Exception as e:  # keep going; report at the end
            print(f"  {row.district}: FAILED ({str(e)[:150]})")
            continue
        if rec is None:
            print(f"  {row.district}: already prepared")
            continue
        log[str(rec["district_id"])] = rec
        log_path.write_text(json.dumps(log, indent=2))
        print(f"  {rec['district']:22s} {rec['buildings']:7,} buildings"
              + (f" (dropped {rec['dropped_small']:,} under {SMALL_M2} m²)" if rec["dropped_small"] else "")
              + f"  {rec['seconds']:.0f}s")

    print("[2/3] Whole-city building grid")
    build_grid()
    if not skip_facilities and (refresh or not C.FACILITIES_FILE.exists()):
        prepare_facilities(districts)
    print("Done.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gee-project", required=True, help="Google Cloud project id registered for Earth Engine")
    ap.add_argument("--districts", nargs="*", help="district names (default: all 50)")
    ap.add_argument("--refresh", action="store_true", help="re-download and rebuild everything requested")
    ap.add_argument("--skip-facilities", action="store_true")
    args = ap.parse_args()
    main(args.gee_project, args.districts, args.refresh, args.skip_facilities)
