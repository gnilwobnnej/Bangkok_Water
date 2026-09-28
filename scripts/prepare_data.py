"""Download, clip and align all input data onto one analysis grid.

Run once before starting the app:
    python scripts/prepare_data.py

Outputs (data/processed/):
    dem.tif            terrain elevation in metres (EGM2008), Float32
    population.tif     people per grid cell, Float32
    waterways.tif      1 where a river/canal crosses the cell, UInt8
    districts.geojson  Bangkok's 50 districts (khet)
    waterways.geojson  river/canal centre lines for the map
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy.ndimage import median_filter
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402

DEM_URL = (
    "https://copernicus-dem-30m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_10_N13_00_E100_00_DEM/Copernicus_DSM_COG_10_N13_00_E100_00_DEM.tif"
)
POP_URL = (
    "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/"
    "2020/BSGM/THA/tha_ppp_2020_constrained.tif"
)
GEOBOUNDARIES_API = "https://www.geoboundaries.org/api/current/gbOpen/THA/{level}/"
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
HEADERS = {"User-Agent": "bangkok-flood-simulator/1.0 (educational data project)"}

TRANSFORM = from_origin(C.WEST, C.NORTH, C.RES, C.RES)
PROFILE = dict(
    driver="GTiff", height=C.HEIGHT, width=C.WIDTH, count=1,
    crs=C.CRS, transform=TRANSFORM, compress="deflate",
)


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"  cached: {dest.name}")
        return dest
    print(f"  downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        tmp.replace(dest)
    return dest


def warp_to_grid(src_path: Path, resampling: Resampling) -> tuple[np.ndarray, float]:
    """Reproject a raster onto the analysis grid; returns (array, source pixel size in degrees)."""
    out = np.full((C.HEIGHT, C.WIDTH), np.nan, dtype="float32")
    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1), destination=out,
            src_nodata=src.nodata, dst_nodata=np.nan,
            dst_transform=TRANSFORM, dst_crs=C.CRS, resampling=resampling,
        )
        return out, abs(src.transform.a)


def write_raster(path: Path, arr: np.ndarray, dtype: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", dtype=dtype, **PROFILE) as dst:
        dst.write(arr.astype(dtype), 1)


def prepare_dem():
    print("[1/4] Elevation (Copernicus GLO-30)")
    raw = download(DEM_URL, C.RAW_DIR / "copernicus_dem_N13_E100.tif")
    # Copernicus is a surface model (includes buildings/trees). Taking the *minimum* 30 m sample
    # in each ~65 m cell, then a small median filter, approximates bare ground between buildings.
    dem, _ = warp_to_grid(raw, Resampling.min)
    dem = median_filter(np.nan_to_num(dem, nan=0.0), size=3)
    write_raster(C.DEM_FILE, dem, "float32")
    print(f"  elevation range: p2={np.percentile(dem, 2):.2f} m  "
          f"median={np.median(dem):.2f} m  p98={np.percentile(dem, 98):.2f} m")


def prepare_population():
    print("[2/4] Population (WorldPop 2020, 100 m constrained)")
    raw = download(POP_URL, C.RAW_DIR / "tha_ppp_2020_constrained.tif")
    pop, src_res = warp_to_grid(raw, Resampling.bilinear)
    # Values are people per source pixel; rescale by pixel-area ratio to keep totals consistent.
    pop = np.clip(np.nan_to_num(pop, nan=0.0), 0, None) * (C.RES / src_res) ** 2
    write_raster(C.POP_FILE, pop, "float32")
    print(f"  total population in study area: {pop.sum():,.0f}")


def prepare_districts():
    print("[3/4] District boundaries (geoBoundaries)")

    def fetch(level):
        meta = requests.get(GEOBOUNDARIES_API.format(level=level), timeout=60).json()
        return gpd.read_file(meta["simplifiedGeometryGeoJSON"])

    provinces = fetch("ADM1")
    is_bkk = provinces["shapeName"].str.contains("bangkok|krung thep", case=False)
    bangkok = provinces[is_bkk].geometry.union_all()
    districts = fetch("ADM2")
    inside = districts.geometry.representative_point().within(bangkok)
    districts = districts[inside][["shapeName", "geometry"]].rename(columns={"shapeName": "district"})
    districts = districts.reset_index(drop=True)
    districts["district_id"] = np.arange(1, len(districts) + 1)
    districts.to_file(C.DISTRICTS_FILE, driver="GeoJSON")
    print(f"  {len(districts)} districts")


def prepare_waterways():
    print("[4/4] Rivers and canals (OpenStreetMap)")
    query = (
        f'[out:json][timeout:180];'
        f'way["waterway"~"^(river|canal)$"]({C.SOUTH},{C.WEST},{C.NORTH},{C.EAST});'
        f'out geom;'
    )
    data = None
    for url in OVERPASS_URLS:
        try:
            r = requests.post(url, data={"data": query}, headers=HEADERS, timeout=200)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:  # try the next mirror
            print(f"  {url} failed: {e}")
    if data is None:
        raise RuntimeError("All Overpass mirrors failed; try again later.")

    rows = []
    for el in data["elements"]:
        coords = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
        if len(coords) >= 2:
            tags = el.get("tags", {})
            rows.append({
                "name": tags.get("name:en") or tags.get("name") or "",
                "kind": tags.get("waterway"),
                "geometry": LineString(coords),
            })
    lines = gpd.GeoDataFrame(rows, crs=C.CRS)
    lines.to_file(C.WATERWAY_LINES_FILE, driver="GeoJSON")

    mask = rasterize(
        ((g, 1) for g in lines.geometry), out_shape=(C.HEIGHT, C.WIDTH),
        transform=TRANSFORM, fill=0, all_touched=True, dtype="uint8",
    )
    write_raster(C.WATER_FILE, mask, "uint8")
    print(f"  {len(lines)} waterway segments, {mask.sum():,} grid cells")


if __name__ == "__main__":
    print(f"Grid: {C.WIDTH} x {C.HEIGHT} cells at {C.RES} deg")
    prepare_dem()
    prepare_population()
    prepare_districts()
    prepare_waterways()
    print("Done. Start the app with:  streamlit run app.py")
