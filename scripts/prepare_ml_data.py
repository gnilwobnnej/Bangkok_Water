"""Prepare training data for the ML flood-susceptibility model.

Labels come from the Global Flood Database event DFO_3850 (Thailand, Aug 2011 - Jan 2012, MODIS 250 m).
Pick one source:

    # Route 1 - no account: download the event map from global-flood-database.cloudtostreet.ai
    python scripts/prepare_ml_data.py --label-file data/raw/gfd_dfo_3850.tif

    # Route 2 - free Google Earth Engine login (run `earthengine authenticate` once first)
    python scripts/prepare_ml_data.py --gee-project YOUR_CLOUD_PROJECT_ID

Outputs (data/processed/):
    flood_2011.tif           1 = flooded, 0 = dry, 255 = unknown (permanent water / too few clear views)
    landcover_fractions.tif  % of each cell in each ESA WorldCover class (one band per class)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import PROFILE, TRANSFORM  # noqa: E402
from src import config as C  # noqa: E402

GFD_COLLECTION = "GLOBAL_FLOOD_DB/MODIS_EVENTS/V1"
GFD_EVENT_ID = 3850
GFD_BANDS = ["flooded", "clear_views", "jrc_perm_water"]
WORLDCOVER_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_N12E099_Map.tif"
)
# ESA WorldCover class codes kept as features
LANDCOVER_CLASSES = {
    "built_up": 50, "cropland": 40, "water": 80, "tree": 10, "grassland": 30, "wetland": 90,
}


def warp_band(src, band: int, resampling=Resampling.nearest) -> np.ndarray:
    """Warp one band onto the grid. The source nodata flag is ignored: GFD exports mark 0
    ("not flooded") as nodata, which would discard every dry cell. Cells outside the source stay NaN."""
    out = np.full((C.HEIGHT, C.WIDTH), np.nan, dtype="float32")
    # Warping an in-memory array (not the file band) is what actually drops the file's nodata flag.
    reproject(
        source=src.read(band).astype("float32"), destination=out,
        src_transform=src.transform, src_crs=src.crs, src_nodata=None, dst_nodata=np.nan,
        dst_transform=TRANSFORM, dst_crs=C.CRS, resampling=resampling,
    )
    return out


def fetch_from_gee(project: str) -> Path:
    import ee

    ee.Initialize(project=project)
    img = (ee.ImageCollection(GFD_COLLECTION)
           .filter(ee.Filter.eq("id", GFD_EVENT_ID)).first().select(GFD_BANDS))
    pad = 0.05
    region = ee.Geometry.Rectangle([C.WEST - pad, C.SOUTH - pad, C.EAST + pad, C.NORTH + pad])
    url = img.getDownloadURL({
        "region": region, "scale": 250, "crs": "EPSG:4326", "format": "GEO_TIFF", "filePerBand": False,
    })
    dest = C.RAW_DIR / f"gfd_dfo_{GFD_EVENT_ID}_gee.tif"
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    dest.write_bytes(r.content)
    with rasterio.open(dest, "r+") as dst:  # Earth Engine exports carry no band names
        for i, name in enumerate(GFD_BANDS, start=1):
            dst.set_band_description(i, name)
    print(f"  downloaded {dest.name} from Earth Engine")
    return dest


def prepare_labels(label_path: Path, min_clear_views: int):
    print(f"[1/2] Flood labels from {label_path.name}")
    with rasterio.open(label_path) as src:
        names = [d.lower() if d else "" for d in src.descriptions]
        if not any(names) and src.count == len(GFD_BANDS):
            names = list(GFD_BANDS)  # unnamed 3-band GFD export: bands are in the requested order
        print(f"  bands: {src.count} {names}  crs: {src.crs}  res: {src.res}")
        if src.crs is None:
            raise SystemExit("Label file has no georeference; use the Earth Engine route instead.")

        def band(name, default):
            return names.index(name) + 1 if name in names else default

        flooded = warp_band(src, band("flooded", 1))
        clear = warp_band(src, band("clear_views", 0)) if "clear_views" in names else None
        perm = warp_band(src, band("jrc_perm_water", 0)) if "jrc_perm_water" in names else None

    label = np.where(flooded >= 0.5, 1, 0).astype("uint8")
    unknown = np.isnan(flooded)
    if perm is not None:
        unknown |= perm >= 0.5
    if clear is not None:
        dry_but_unseen = (label == 0) & (np.nan_to_num(clear) < min_clear_views)
        print(f"  dry cells dropped for < {min_clear_views} clear views: {dry_but_unseen.sum():,}")
        unknown |= dry_but_unseen
    label[unknown] = C.LABEL_NODATA

    with rasterio.open(C.FLOOD_2011_FILE, "w", dtype="uint8", nodata=C.LABEL_NODATA, **PROFILE) as dst:
        dst.write(label, 1)
    n_f, n_d, n_u = (label == 1).sum(), (label == 0).sum(), (label == C.LABEL_NODATA).sum()
    print(f"  flooded {n_f:,}  dry {n_d:,}  unknown {n_u:,}  -> flooded share of known cells "
          f"{n_f / max(n_f + n_d, 1):.1%}")


def prepare_landcover():
    print("[2/2] Land cover (ESA WorldCover 2021, 10 m)")
    raw = C.RAW_DIR / "worldcover_bangkok.tif"
    if not raw.exists():
        # The source is a Cloud-Optimized GeoTIFF, so only Bangkok's window is downloaded.
        with rasterio.open("/vsicurl/" + WORLDCOVER_URL) as src:
            window = from_bounds(C.WEST - 0.01, C.SOUTH - 0.01, C.EAST + 0.01, C.NORTH + 0.01, src.transform)
            data = src.read(1, window=window)
            profile = src.profile.copy()
            profile.update(width=data.shape[1], height=data.shape[0],
                           transform=src.window_transform(window), compress="deflate")
        raw.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(raw, "w", **profile) as dst:
            dst.write(data, 1)
        print(f"  downloaded {data.shape[1]} x {data.shape[0]} window")

    with rasterio.open(raw) as src:
        classes = src.read(1)
        src_transform, src_crs = src.transform, src.crs
    bands = []
    for name, code in LANDCOVER_CLASSES.items():
        out = np.zeros((C.HEIGHT, C.WIDTH), dtype="float32")
        reproject(
            source=((classes == code) * 100).astype("uint8"), destination=out,
            src_transform=src_transform, src_crs=src_crs,
            dst_transform=TRANSFORM, dst_crs=C.CRS, resampling=Resampling.average,
        )
        bands.append(out)
        print(f"  {name:10s} {out.mean():5.1f}% of study area")

    with rasterio.open(C.LANDCOVER_FILE, "w", dtype="float32",
                       **{**PROFILE, "count": len(bands)}) as dst:
        for i, (name, arr) in enumerate(zip(LANDCOVER_CLASSES, bands), start=1):
            dst.write(arr, i)
            dst.set_band_description(i, name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src_group = ap.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--label-file", type=Path, help="GFD event GeoTIFF downloaded from the website")
    src_group.add_argument("--gee-project", help="Google Cloud project id registered for Earth Engine")
    ap.add_argument("--min-clear-views", type=int, default=3,
                    help="drop 'dry' cells seen clearly fewer times than this during the event")
    args = ap.parse_args()

    label_path = args.label_file if args.label_file else fetch_from_gee(args.gee_project)
    prepare_labels(label_path, args.min_clear_views)
    prepare_landcover()
    print("Done. Next: python scripts/train_model.py")
