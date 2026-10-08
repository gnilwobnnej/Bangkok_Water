"""Flood defences: the King's Dike, the Chao Phraya river walls and the Suvarnabhumi Airport dike.

    python scripts/prepare_defences.py

Needs data/processed/ from scripts/prepare_data.py. Downloads road and dike lines from OpenStreetMap, then writes
  data/processed/defences.geojson   the lines, with name, crest_m, source and note (shown on the map)
  data/processed/defences.tif       crest height per grid cell (m above mean sea level, NaN = no defence)

OpenStreetMap barely maps Bangkok's dikes, so the routes come from published descriptions, traced along the roads
they follow, and the crest heights from the sources below. Where sources give a range, the lower figure is used.
Every section and its source is listed in SECTIONS / RIVER_WALLS / AIRPORT_DIKE: edit those to change an
assumption, then re-run this script.

Datum: the crest heights are above Thai mean sea level and the DEM (Copernicus GLO-30) is above the EGM2008 geoid.
The two differ by a few tens of cm around Bangkok; no correction is applied.
"""
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import shapely
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation, maximum_filter
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge, nearest_points, substring, unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config as C  # noqa: E402
from src.defences import crest_grid  # noqa: E402
from prepare_data import HEADERS, OVERPASS_URLS, TRANSFORM, write_raster  # noqa: E402

SRC = {
    "rdpb": "Royal Development Projects Board, 'Eastern King's Dike' project (km.rdpb.go.th/Project/View/6635)",
    "thinkofliving": "Think of Living, 'Where are the dike lines?', 12 Oct 2011 (route by road)",
    "jica": "JICA, Study on the BMA Subcenters Program, ch. 5.3 (Rom Klao Road dike designed at 2.47 m MSL)",
    "bma_walls": "BMA flood walls: upstream +3.00-3.50, middle +2.80-3.00, downstream +2.50-2.80 m MSL "
                 "(BMA, 'Flood Mitigation and Management in Bangkok', UNESCAP; Pattaya Mail, 10 Oct 2024)",
    "aot": "Airports of Thailand: 23.5 km dike, built at 3.0 m and raised to 3.5 m in 2011 (aviationpros.com, 2011)",
}

KINGS_DIKE_CREST = 2.5  # 2.47 m MSL as designed (JICA); raised to +3.0 in places after 2011 (BMA)
KINGS_DIKE_SOURCE = f"Route: {SRC['rdpb']}; {SRC['thinkofliving']}. Crest: {SRC['jica']}; {SRC['bma_walls']}."

# King's Dike, north to south. Each section is an OSM road, clipped to a box (south, west, north, east).
SECTIONS = [
    ("ถนนพหลโยธิน", "Phahon Yothin Road", (13.915, 100.60, 13.99, 100.64)),  # north end: to the BMA boundary
    ("ถนนสายไหม", "Sai Mai Road", (13.90, 100.62, 13.94, 100.72)),
    ("ถนนหทัยราษฎร์", "Hathai Rat Road", (13.80, 100.70, 13.93, 100.73)),
    ("ถนนสุวินทวงศ์", "Suwinthawong Road (short link)", (13.80, 100.715, 13.83, 100.74)),
    ("ถนนร่มเกล้า", "Rom Klao Road", (13.71, 100.72, 13.83, 100.76)),
    ("ถนนกิ่งแก้ว", "King Kaew Road", (13.59, 100.69, 13.73, 100.76)),
    ("ถนนตำหรุ-บางพลี", "Tamru-Bang Phli Road", (13.50, 100.67, 13.61, 100.71)),
    ("ถนนสุขุมวิท", "Old Sukhumvit Road (to the river at Samut Prakan)", (13.49, 100.585, 13.605, 100.69)),
]

# Chao Phraya flood walls on both banks, by latitude band (BMA's upstream / middle / downstream sections).
# They run from just north of the BMA boundary down to Old Sukhumvit Road at Samut Prakan.
RIVER_WALLS = [  # (name, south, north, crest_m)
    ("Chao Phraya river walls (upstream)", 13.80, 13.86, 3.0),
    ("Chao Phraya river walls (middle)", 13.70, 13.80, 2.8),
    ("Chao Phraya river walls (downstream)", 13.585, 13.70, 2.5),
]

AIRPORT_DIKE = ("Suvarnabhumi Airport dike", 3.5, SRC["aot"])


def overpass(query: str) -> list:
    for attempt in range(3):
        for url in OVERPASS_URLS:
            try:
                r = requests.post(url, data={"data": query}, headers=HEADERS, timeout=150)
                r.raise_for_status()
                return r.json()["elements"]
            except Exception as e:  # try the next mirror
                print(f"  {url} failed: {str(e)[:80]}")
        time.sleep(15)
    raise RuntimeError("All Overpass mirrors failed; try again later.")


def lines_of(elements) -> MultiLineString:
    return MultiLineString([[(p["lon"], p["lat"]) for p in e["geometry"]] for e in elements
                            if len(e.get("geometry", [])) >= 2])


def kings_dike(river: LineString) -> list:
    """The dike as a list of line pieces, north to south, joined end to end."""
    pieces = []
    for thai, eng, (s, w, n, e) in SECTIONS:
        print(f"  {eng}")
        els = overpass(f'[out:json][timeout:120];way["highway"]["name"="{thai}"]({s},{w},{n},{e});out geom;')
        road = linemerge(lines_of(els)).intersection(shapely.box(w, s, e, n))
        if road.is_empty:
            raise RuntimeError(f"No OSM geometry for {eng}")
        pieces.append((eng, road))

    # North closure: from where Phahon Yothin leaves Bangkok, along the BMA's northern boundary to the river
    bma = unary_union(gpd.read_file(C.DISTRICTS_FILE).geometry)
    outline = bma.exterior
    pieces[0] = (pieces[0][0], pieces[0][1].intersection(bma.buffer(0.002)))  # only the part inside Bangkok
    start = pieces[0][1].intersection(outline)
    start = start if start.geom_type == "Point" else max(start.geoms, key=lambda p: p.y)
    river_points = river.intersection(outline)
    end = max(getattr(river_points, "geoms", [river_points]), key=lambda p: p.y)
    a, b = sorted([outline.project(start), outline.project(end)])
    arcs = [substring(outline, a, b), unary_union([substring(outline, b, outline.length),
                                                    substring(outline, 0, a)])]
    north = min(arcs, key=lambda g: g.length)  # the short way round, across the north of the city
    pieces.insert(0, ("BMA northern boundary (closure to the river, approximate)", north))

    # Join each piece to the next, and both ends to the river, so the line has no gaps
    joined = []
    for (name, g), (_, nxt) in zip(pieces, pieces[1:] + [(None, None)]):
        joined.append((name, g))
        if nxt is not None:
            joined.append((name + " (join)", LineString(nearest_points(g, nxt))))
    joined.append(("Old Sukhumvit Road (join to river)", LineString(nearest_points(pieces[-1][1], river))))
    return joined


def main():
    waterways = gpd.read_file(C.WATERWAY_LINES_FILE)
    river = linemerge(unary_union(waterways[waterways["name"] == "Chao Phraya River"].geometry))
    river_line = max(getattr(river, "geoms", [river]), key=lambda g: g.length)

    print("[1/3] King's Dike (OpenStreetMap roads)")
    rows = []
    for name, g in kings_dike(river_line):
        rows.append({"name": "King's Dike: " + name, "crest_m": KINGS_DIKE_CREST, "kind": "dike",
                     "source": KINGS_DIKE_SOURCE,
                     "note": ("Approximate: Bangkok's northern boundary stands in for the northern defence line."
                              if "boundary" in name else "Elevated road on an embankment."),
                     "geometry": g})

    print("[2/3] Suvarnabhumi Airport dike (OpenStreetMap)")
    els = overpass(f'[out:json][timeout:120];way["man_made"="dyke"]'
                   f'({C.SOUTH},{C.WEST},{C.NORTH},{C.EAST});out tags geom;')
    airport = [e for e in els if "airport" in e.get("tags", {}).get("note", "").lower()]
    name, crest, source = AIRPORT_DIKE
    for e in airport:
        rows.append({"name": name, "crest_m": crest, "kind": "dike", "source": source,
                     "note": "Ring dike around the airport.", "geometry": lines_of([e])})
    dikes = gpd.GeoDataFrame(rows, crs=C.CRS)

    print("[3/3] Chao Phraya river walls and the crest-height grid")
    shape = (C.HEIGHT, C.WIDTH)
    river_cells = rasterize([(river, 1)], out_shape=shape, transform=TRANSFORM, fill=0,
                            all_touched=True, dtype="uint8").astype(bool)
    # Dikes: drawn cell by cell along each line, then widened by one cell so no corner gaps remain
    dike_crest = crest_grid(dikes, shape, TRANSFORM)
    widened = maximum_filter(np.nan_to_num(dike_crest, nan=-np.inf), size=3)
    crest = np.where(np.isfinite(widened), widened, np.nan).astype("float32")
    # River walls: the ring of cells along both banks (the cells next to the river, including canal mouths,
    # which are closed by gates during floods)
    lats = C.NORTH - (np.arange(C.HEIGHT) + 0.5) * C.RES
    bank = binary_dilation(river_cells, structure=np.ones((3, 3), bool)) & ~river_cells
    wall_rows = []
    for wname, south, north, wcrest in RIVER_WALLS:
        band = (lats >= south) & (lats < north)
        cells = bank & band[:, None]
        crest[cells] = np.fmax(crest[cells], wcrest)
        piece = river.intersection(shapely.box(C.WEST, south, C.EAST, north))
        wall_rows.append({"name": wname, "crest_m": wcrest, "kind": "river wall", "source": SRC["bma_walls"],
                          "note": "Both banks. Drawn along the river's centre line on the map.", "geometry": piece})
    crest[river_cells] = np.nan  # never block the river itself

    lines = gpd.GeoDataFrame(pd.concat([dikes, gpd.GeoDataFrame(wall_rows, crs=C.CRS)], ignore_index=True), crs=C.CRS)
    lines.to_file(C.DEFENCE_LINES_FILE, driver="GeoJSON")
    write_raster(C.DEFENCES_FILE, crest, "float32")
    km = lines.to_crs(32647).length / 1000
    for kind, total in km.groupby(lines["kind"]).sum().items():
        print(f"  {kind}: {total:.0f} km")
    print(f"  {np.isfinite(crest).sum():,} grid cells with a defence. Wrote {C.DEFENCES_FILE.name} and "
          f"{C.DEFENCE_LINES_FILE.name}")



if __name__ == "__main__":
    main()
