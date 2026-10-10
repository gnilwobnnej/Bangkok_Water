"""Bangkok Flood Simulator — interactive flood-risk map and 'what-if' water-level explorer.

    streamlit run app.py
"""
import time

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src import config as C
from src.data_loader import load_study_area, missing_files
from src.defences import load_defence_lines
from src.buildings import (
    available_districts, district_counts, impact_modes, load_building_grid, load_buildings, load_facilities,
    values_at,
)
from src.current_flood import (
    DRY_PCT, FLOODED_PCT, conditions_alert, district_flooded_pct, load_evaluation, load_pass, load_summary,
)
from src.viz3d import city_deck, district_deck, district_view, footprint_coords
from src.ml import ML_2011, ML_ALL, load_ml
from src.risk import compute_factors, district_mean, district_risk, risk_index
from src.simulate import level_curve, simulate
from src.uncertainty import load_dem_uncertainty, load_flood_chance
from src.validation import CI, REFERENCE, TEST_CV, TEST_RADAR, loeo_table, validation_table, wins
from src.viz import (
    basemap_rgb, build_map, difference_chart, elevation_histogram, feature_importance_chart, flood_frame,
    flood_timeline_chart, fmt_people, level_curve_chart, model_comparison_chart, score_over_time_chart,
    top_districts_chart,
)

MAX_LEVEL = 3.0
LEVELS = np.round(np.arange(0, MAX_LEVEL + 0.001, 0.1), 1)

st.set_page_config(page_title="Bangkok Flood Simulator", layout="wide")

missing = missing_files()
if missing:
    st.error(
        "Data not prepared yet. Run `python scripts/prepare_data.py` first.\n\n"
        "Missing: " + ", ".join(f.name for f in missing)
    )
    st.stop()


@st.cache_resource(show_spinner="Loading elevation, population and waterways…")
def get_area():
    return load_study_area()


@st.cache_resource
def get_factors():
    return compute_factors(get_area())


@st.cache_resource
def get_basemap():
    return basemap_rgb(get_area())


@st.cache_resource
def get_district_elevation():
    return district_mean(get_area(), get_area().dem, "elevation")


def file_version(*paths) -> tuple:
    """Modification times of data files. Passed to cached loaders so that new data (e.g. a pushed update)
    is a cache miss even when the Streamlit process isn't restarted."""
    return tuple(p.stat().st_mtime if p.exists() else None for p in paths)


@st.cache_resource(max_entries=1)
def get_ml(version: tuple):
    return load_ml()


@st.cache_resource(max_entries=1)
def get_radar(version: tuple):
    return load_summary()


@st.cache_resource(max_entries=1)
def get_radar_eval(version: tuple):
    return load_evaluation()


@st.cache_resource(max_entries=32)
def get_radar_pass(date: str, version: tuple):
    return load_pass(date)


@st.cache_resource(show_spinner="Loading buildings…")
def get_buildings(district_id: int):
    b = load_buildings(district_id)
    return b, footprint_coords(b)


@st.cache_resource
def get_building_grid():
    return load_building_grid()


@st.cache_resource
def get_facilities():
    return load_facilities()


@st.cache_data(show_spinner="Computing flood curve…")
def get_curve(connected: bool, defences: bool):
    return level_curve(get_area(), LEVELS, connected, defences)


@st.cache_resource
def get_defence_lines():
    return load_defence_lines()


@st.cache_resource(max_entries=2)
def get_dem_uncertainty(path, version: tuple):
    return load_dem_uncertainty(path)


@st.cache_resource(max_entries=4)
def get_flood_chance(connected: bool, defences: bool, level: float, version: tuple):
    return load_flood_chance(connected, defences, level)


area = get_area()
factors = get_factors()
ml = get_ml(file_version(C.ML_PROB_FILE, C.ML_REPORT_FILE))
radar_version = file_version(C.RADAR_SUMMARY_FILE)
radar = get_radar(radar_version)
unc_version = file_version(C.DEM_UNCERTAINTY_FILE)
dem_unc = get_dem_uncertainty(C.DEM_UNCERTAINTY_FILE, unc_version)

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Scenario")
    level = st.slider(
        "Water level (m above sea level)", 0.0, MAX_LEVEL, 1.0, 0.1,
        help="Height the river, canals and sea rise to. Much of Bangkok sits only 0–2 m above sea level.",
    )
    connected = st.toggle(
        "Water must flow from rivers/canals", value=True,
        help="On: only low ground connected to a river or canal floods. "
             "Off: every cell below the level floods (like heavy rain pooling in low spots).",
    )
    use_defences = False
    if area.defences is not None:
        use_defences = st.toggle(
            "Include flood defences", value=False, disabled=not connected,
            help="River walls along the Chao Phraya, the King's Dike around eastern Bangkok and the "
                 "Suvarnabhumi Airport dike hold water out until it rises above their crest (2.5–3.5 m). "
                 "Canals inside them are pumped, so they don't flood the land they run through. "
                 "Only applies when water must flow from rivers/canals.",
        ) and connected
    animate = st.button("Animate rising water (0 → 3 m)", use_container_width=True, type="primary")

    st.header("Risk index weights")
    w_elev = st.slider("Low elevation", 0.0, 1.0, 0.5, 0.05)
    w_water = st.slider("Near river / canal", 0.0, 1.0, 0.3, 0.05)
    w_pop = st.slider("Population density", 0.0, 1.0, 0.2, 0.05)

    st.header("Map layers")
    show_depth = st.checkbox("Flood depth", value=True)
    show_elev = st.checkbox("Ground elevation (with relief shading)", value=False)
    elev_range = st.slider(
        "Elevation colour range (m)", -2.0, 15.0, (0.0, 5.0), 0.5, disabled=not show_elev,
        help="Bangkok is very flat, so a narrow range makes small height differences visible. "
             "Anything outside the range gets the end colour.",
    )
    show_chance = dem_unc is not None and st.checkbox(
        "Chance flooded (elevation uncertainty)", value=False,
        help="Share of 50 simulations, each with plausible errors added to the elevation data, in which a "
             "cell floods. Shown for the nearest of 0.5, 1.0 … 3.0 m. Follows the connectivity and defence "
             "settings above.")
    show_risk = st.checkbox("Risk index", value=False)
    show_ml = show_obs = show_disagree = False
    if ml is not None:
        show_ml = st.checkbox("ML flood susceptibility", value=False,
                              help="Probability of flooding learned from where Bangkok flooded in 2011 and in "
                                   "the radar-mapped wet seasons of 2017–2025.")
        if ml.disagreement is not None:
            show_disagree = st.checkbox(
                "ML model disagreement", value=False,
                help="How much five versions of the model, each trained with a different fifth of the map held "
                     "out, disagree about each cell (standard deviation of their flood probabilities). Bright = "
                     "the prediction depends on which places it learned from, so trust it less.")
        show_obs = st.checkbox("Observed flood (2011)", value=False,
                               help="Satellite-mapped flood extent, Aug 2011 – Jan 2012 (MODIS, 250 m).")
    show_radar, radar_date = False, None
    if radar is not None:
        radar_passes = radar[0].set_index("date")
        show_radar = st.checkbox("Radar flood (Sentinel-1)", value=False,
                                 help="Unusual water seen by satellite radar on the chosen date.")

        def pass_label(d):
            p = radar_passes.loc[d]
            rain = "" if np.isnan(p["rain_day_mm"]) else f", {p['rain_day_mm']:.0f} mm rain"
            return f"{d} ({p['flooded_km2']:.0f} km² flooded{rain})"

        radar_date = st.selectbox(
            "Radar pass date", radar_passes.index.tolist()[::-1], format_func=pass_label,
            help="Sentinel-1 passes over Bangkok every few days. Also sets 'Flooded now' in tooltips and tables.",
        )
    show_waterways = st.checkbox("Rivers & canals", value=False)
    show_defences = use_defences or st.checkbox(
        "Flood defences", value=False, disabled=get_defence_lines() is None,
        help="Orange = holding at this water level, dashed red = overtopped. Hover a line for its crest "
             "height and source. Always shown while defences are included in the simulation.")
    dark = st.checkbox("Dark basemap", value=True)

# ---------------- compute ----------------
result = simulate(area, level, connected, use_defences)
risk = risk_index(factors, w_elev, w_water, w_pop)
table = (
    area.districts[["district_id", "district"]]
    .merge(get_district_elevation().reset_index(), on="district_id")
    .merge(district_risk(area, risk).reset_index(), on="district_id")
    .merge(result.by_district.reset_index(), on="district_id")
)
if ml is not None:
    table = table.merge(ml.districts[["district_id", "ml_score", "observed_2011_pct"]], on="district_id")
radar_pass = get_radar_pass(radar_date, radar_version) if radar_date else None
if radar_pass is not None:
    table = table.merge(district_flooded_pct(area, radar_pass), on="district_id")
bkk = area.district_ids > 0

# ---------------- header ----------------
st.title("Bangkok Flood Simulator")
st.caption(
    "Raise the water level and watch where Bangkok floods first. Built on satellite elevation "
    "(Copernicus GLO-30), population (WorldPop 2020), and rivers & canals (OpenStreetMap)."
)
if radar is not None:
    alerts = conditions_alert(radar[0], radar[1], radar[2])
    if alerts:
        st.warning("**Unusual conditions right now**\n\n" + "\n".join(f"- {a}" for a in alerts)
                   + "\n\nSee the **Current flood** tab for the details.", icon="⚠️")


def show_metrics(res, tbl, unc=None):
    """Headline numbers; `unc` (a row of the elevation-uncertainty curve) adds the 90% range under each."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Water level", f"{res.level:.1f} m")
    rng = {}
    if unc is not None:
        rng = {"area": f"range {unc['area_km2_p5']:,.0f}–{unc['area_km2_p95']:,.0f} km²",
               "people": f"range {fmt_people(unc['people_p5'])}–{fmt_people(unc['people_p95'])}",
               "districts": f"range {unc['districts_25_p5']:.0f}–{unc['districts_25_p95']:.0f}"}
        c1.caption("Ranges: 90% of simulations with elevation error (Flood curve tab)")
    c2.metric("Flooded area", f"{res.area_km2:,.0f} km²", rng.get("area"), delta_color="off")
    c3.metric("People affected", fmt_people(res.people), rng.get("people"), delta_color="off")
    c4.metric("Districts > 25% flooded", f"{(tbl['pct_flooded'] > 25).sum()} / {len(tbl)}", rng.get("districts"),
              delta_color="off")


def unc_at(lv):
    return dem_unc.at(connected, use_defences, lv) if dem_unc is not None else None


if animate:
    stage = st.empty()
    base = get_basemap()
    for lv in LEVELS:
        frame_res = simulate(area, float(lv), connected, use_defences)
        frame_tbl = table[["district_id"]].merge(frame_res.by_district.reset_index(), on="district_id")
        with stage.container():
            show_metrics(frame_res, frame_tbl, unc_at(float(lv)))
            st.image(flood_frame(base, frame_res.depth), use_container_width=True,
                     caption=f"Water level {lv:.1f} m — darker blue = deeper water")
        time.sleep(0.15)
    st.info("Animation finished. Use the slider for the interactive map at a specific level.")
    st.stop()

show_metrics(result, table, unc_at(level))

# ---------------- map ----------------
chance = get_flood_chance(connected, use_defences, level, unc_version) if show_chance else None
fmap = build_map(
    area, table,
    depth=result.depth if show_depth else None,
    risk=risk if show_risk else None,
    show_waterways=show_waterways, dark=dark,
    # Keep the colour range at least 0.5 m wide if both handles land on the same value.
    elevation_range=(elev_range[0], max(elev_range[1], elev_range[0] + 0.5)) if show_elev else None,
    ml_prob=ml.prob if show_ml else None,
    observed=ml.observed if show_obs else None,
    radar=radar_pass["flood_pct"] if show_radar else None,
    radar_label=f"Radar flood {radar_date}",
    defences=get_defence_lines() if show_defences else None, level=level,
    disagreement=ml.disagreement if show_disagree else None,
    chance=chance[0] if chance else None,
    chance_label=f"Chance flooded at {chance[1]:.1f} m" if chance else "",
)
st_folium(fmap, height=620, use_container_width=True, returned_objects=[])
st.caption("Hover a district for details. Toggle layers in the sidebar or the map's layer control."
           + (f" The chance-flooded layer is for {chance[1]:.1f} m, the nearest level it was computed for."
              if chance and abs(chance[1] - level) > 0.01 else ""))

# ---------------- analysis tabs ----------------
tab_top, tab_elev, tab_curve, tab_now, tab_3d, tab_ml, tab_valid, tab_table, tab_method = st.tabs([
    "Most affected districts", "Elevation profile", "Flood curve", "Current flood",
    "Buildings 3D", "ML model", "Validation", "District table", "Method & limitations",
])
with tab_3d:
    prepared = available_districts()
    bgrid = get_building_grid()
    if not prepared or bgrid is None:
        st.info(
            "No building data yet. The 3D view shows every building coloured by flood impact, using Google "
            "Open Buildings footprints and heights (free Google Earth Engine project needed):\n\n"
            "`python scripts/prepare_buildings.py --gee-project YOUR_PROJECT`\n\nThen reload this page."
        )
    else:
        modes = impact_modes(result, radar_pass, radar_date, ml)
        c1, c2, c3 = st.columns([1.2, 2, 1.2])
        view_kind = c1.radio("View", ["District", "Whole city"], horizontal=True, key="b3d_view")
        mode = modes[c2.selectbox("Colour buildings by", list(modes), format_func=lambda k: modes[k].label,
                                  key="b3d_mode",
                                  help="Follows the sidebar: the water level, the connectivity switch and the radar date.")]
        height_scale = c3.slider("Height exaggeration", 1.0, 5.0, 3.0, 0.5, key="b3d_scale",
                                 help="Bangkok is mostly low-rise, so stretching heights makes the 3D easier to read.")
        per_district = district_counts(area, bgrid[0], mode).merge(area.districts[["district_id", "district"]])
        per_district = per_district[per_district["district_id"].isin(prepared)]
        facilities = get_facilities()

        if view_kind == "Whole city":
            total, hit = per_district["buildings"].sum(), per_district["affected"].sum()
            m1, m2, m3 = st.columns(3)
            m1.metric("Buildings mapped", f"{total:,.0f}")
            m2.metric("Affected", f"{hit:,.0f}", f"{hit / max(total, 1):.1%} of buildings", delta_color="off")
            if facilities is not None:
                fac_in = facilities[facilities["district_id"] > 0]
                m3.metric("Critical facilities affected",
                          f"{(values_at(fac_in, mode.grid) >= mode.affected_min).sum()} / {len(fac_in)}")
            st.pydeck_chart(city_deck(area, bgrid[0], mode, height_scale), use_container_width=True, height=620)
            st.caption("Each column is a ~520 m block: height = buildings affected, redder = larger share affected. "
                       "Drag to pan, right-drag (or Ctrl-drag) to tilt and rotate, scroll to zoom.")
            if len(prepared) < len(area.districts):
                st.caption(f"Building data prepared for {len(prepared)} of {len(area.districts)} districts.")
        else:
            ranked = per_district.sort_values("affected", ascending=False)
            options = ranked["district_id"].tolist()
            names = dict(zip(per_district["district_id"], per_district["district"]))
            d1, d2 = st.columns([2, 1])
            district_id = d1.selectbox(
                "District", options, key="b3d_district",
                format_func=lambda i: f"{names[i]} ({ranked.set_index('district_id').loc[i, 'affected']:,.0f} buildings affected)",
                help="Sorted by buildings affected under the current colouring.")
            buildings, coords = get_buildings(district_id)
            light_default = len(buildings) > 40_000
            shapes = d2.radio("Building shapes", ["Footprints", "Light (faster)"], horizontal=True,
                              index=1 if light_default else 0, key=f"b3d_shapes_{district_id}",
                              help="Light draws each building as a small square column at its true height: "
                                   "about 3× less data, useful online and for the largest districts.")
            v = values_at(buildings, mode.grid)
            hit = v >= mode.affected_min
            fac_d = None if facilities is None else facilities[facilities["district_id"] == district_id]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Buildings", f"{len(buildings):,}")
            m2.metric("Affected", f"{hit.sum():,}", f"{hit.mean():.1%}", delta_color="off")
            if mode.key != "2011":
                m3.metric(mode.severe_label, f"{(v >= mode.severe_min).sum():,}")
            if fac_d is not None:
                fv = values_at(fac_d, mode.grid)
                m4.metric("Critical facilities affected", f"{(fv >= mode.affected_min).sum()} / {len(fac_d)}")
            deck = district_deck(buildings, coords if shapes == "Footprints" else None, mode, height_scale,
                                 fac_d, district_view(buildings, hit))
            st.pydeck_chart(deck, use_container_width=True, height=620)
            st.caption("Coloured = affected, grey = not affected. Tall thin columns are critical facilities "
                       "(pink = affected; white = hospital/clinic, yellow = school/university, blue = police/fire). "
                       "The view opens on the affected area; scroll out to see the whole district. "
                       "Hover for details. Right-drag (or Ctrl-drag) to tilt and rotate.")
            if fac_d is not None and len(fac_d):
                fac_hit = fac_d.assign(value=values_at(fac_d, mode.grid))
                fac_hit = fac_hit[fac_hit["value"] >= mode.affected_min]
                if len(fac_hit):
                    st.subheader(f"Critical facilities affected in {names[district_id]}")
                    st.dataframe(fac_hit[["name", "type", "value"]].sort_values("type"), hide_index=True,
                                 use_container_width=True,
                                 column_config={"name": "Name", "type": "Type",
                                                "value": st.column_config.NumberColumn(mode.value_label, format="%.2f")})
        st.markdown("""
**Read with care**
- Footprints and heights are **AI-estimated from satellite imagery** (Google Open Buildings), so some are wrong or missing.
- "Affected" means the building stands in a ~65 m cell that floods under the chosen view. It doesn't mean water
  entered the building, and ground floors may be raised.
- Radar can't see water between buildings, so in the radar view a building counts as affected when floodwater was
  detected **within ~65 m** of it. That still under-counts in the dense centre.
- Facilities come from OpenStreetMap and may be incomplete.
""")
with tab_now:
    if radar is None:
        st.info(
            "No radar flood maps yet. They show where Bangkok is flooding now, from Sentinel-1 satellite radar. "
            "To fetch them (free Google Earth Engine project needed):\n\n"
            "1. `python scripts/fetch_current_flood.py --gee-project YOUR_PROJECT`\n"
            "2. `python scripts/evaluate_current.py`\n\nThen reload this page."
        )
    else:
        passes, rain, river, summary = radar
        sel = passes.set_index("date").loc[radar_date]
        baseline_text = ("the same weeks last year, so normal seasonal water such as planted rice paddies "
                         "is mostly excluded" if summary["method"]["baseline"] == "seasonal"
                         else "the dry season (January–March), so all new water counts, including planted paddies")
        st.markdown(
            f"**Sentinel-1 radar** has passed over Bangkok **{len(passes)} times** since {passes['date'].iloc[0]}. "
            f"Radar sees through cloud, and open water shows up dark. On **{radar_date}** it found "
            f"**{sel['flooded_km2']:.0f} km²** of unusual water across greater Bangkok "
            f"(**{sel['flooded_km2_bangkok']:.0f} km²** inside Bangkok's 50 districts). "
            f"Each pass is compared with {baseline_text}."
        )
        weather = []
        if not np.isnan(sel["rain_day_mm"]):
            three_day = "" if np.isnan(sel["rain_3day_mm"]) else f" ({sel['rain_3day_mm']:.0f} mm over 3 days)"
            weather.append(f"**{sel['rain_day_mm']:.0f} mm** of rain fell that day{three_day}")
        if not np.isnan(sel["discharge_m3s"]):
            weather.append(f"the Chao Phraya at Nonthaburi was flowing at **{sel['discharge_m3s']:,.0f} m³/s** "
                           f"(**{sel['discharge_pct_normal']:.0f}%** of normal for the date)")
        if weather:
            st.markdown(f"On {radar_date}, " + "; ".join(weather) + ".")
        st.plotly_chart(flood_timeline_chart(passes, rain, river, radar_date), use_container_width=True)
        sources = f"Rainfall: {summary.get('rainfall_source', 'Open-Meteo')}."
        if len(river):
            sources += f" River flow: {summary['river_source']}."
        st.caption(sources + " Tick **Radar flood (Sentinel-1)** in the sidebar to see the selected pass on the map.")

        st.subheader("Every radar pass")
        pass_cols = ["date", "direction", "flooded_km2", "flooded_km2_bangkok", "rain_day_mm", "rain_3day_mm"]
        if len(river):
            pass_cols += ["discharge_m3s", "discharge_pct_normal"]
        st.dataframe(
            passes[pass_cols].iloc[::-1].assign(direction=passes["direction"].str.title()),
            hide_index=True, use_container_width=True,
            column_config={
                "date": "Date",
                "direction": "Orbit",
                "flooded_km2": st.column_config.NumberColumn("Flooded, greater Bangkok (km²)", format="%.1f"),
                "flooded_km2_bangkok": st.column_config.NumberColumn("Flooded, Bangkok (km²)", format="%.1f"),
                "rain_day_mm": st.column_config.NumberColumn("Rain that day (mm)", format="%.0f"),
                "rain_3day_mm": st.column_config.NumberColumn("Rain, 3 days (mm)", format="%.0f"),
                "discharge_m3s": st.column_config.NumberColumn("River flow (m³/s)", format="%.0f"),
                "discharge_pct_normal": st.column_config.NumberColumn("River vs normal", format="%.0f%%"),
            },
        )

        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"Most flooded districts on {radar_date}")
            st.dataframe(
                table.nlargest(10, "flooded_now_pct")[["district", "flooded_now_pct", "flooded_now_km2"]],
                hide_index=True, use_container_width=True,
                column_config={
                    "district": "District",
                    "flooded_now_pct": st.column_config.ProgressColumn(
                        "Flooded now (%)", min_value=0, max_value=max(1.0, float(table["flooded_now_pct"].max())),
                        format="%.1f%%"),
                    "flooded_now_km2": st.column_config.NumberColumn("km²", format="%.1f"),
                },
            )
        ev = get_radar_eval(file_version(C.RADAR_EVAL_FILE))
        with c2:
            st.subheader("Did the models see it coming?")
            if ev is None:
                st.write("Run `python scripts/evaluate_current.py` to test the models against the radar maps.")
            else:
                scope = st.radio("Area", ["Greater Bangkok", "Bangkok (50 districts)"], horizontal=True,
                                 key="radar_scope")
                day = ev[(ev["date"] == radar_date) & (ev["scope"] == scope)]
                if day.empty:
                    st.write("Too little flooding on this date to score the models.")
                else:
                    st.plotly_chart(model_comparison_chart(day[["method", "roc_auc", "pr_auc"]]),
                                    use_container_width=True)
                    st.caption(
                        f"None of these methods has seen this flood. {day['flooded_share'].iloc[0]:.1%} of "
                        f"radar-visible cells flooded, so a random guess scores PR-AUC "
                        f"{day['flooded_share'].iloc[0]:.3f}. ROC-AUC 0.5 = random."
                    )
        if ev is not None:
            st.subheader("How well each method matched every radar pass")
            st.plotly_chart(score_over_time_chart(ev, scope), use_container_width=True)
            st.caption("The **Validation** tab has every score with its 95% interval, the 2011 results and a download.")
        st.markdown("""
**Read with care**
- Radar misses much of the water **between buildings**, so flooding in the dense city is under-counted.
- Smooth surfaces such as **runways and flooded rice fields** can look like floodwater. Much of the detected
  water is in farmland, which is also where the ML model expects flooding, so part of the match comes from that.
- Each pass is a **snapshot** taken every few days, not a live feed. Passes from the two orbits view the ground
  from different angles, so compare dates from the same orbit for the cleanest trend.
- Rainfall figures are weather-model estimates, not rain-gauge readings. Pass dates are in UTC: descending
  passes happen around 06:10 Bangkok time the *next* morning (so all of "that day's" rain fell before them),
  ascending passes around 18:30 the same day (so the evening's rain came after them).
- River flow comes from the GloFAS river model, not a gauge, and reads high on the Chao Phraya. Compare it with
  its own normal (the dashed line) rather than trusting the exact m³/s.
""")
with tab_ml:
    if ml is None:
        st.info(
            "The machine-learning model hasn't been trained yet. It learns flood susceptibility from where "
            "Bangkok actually flooded in 2011. To build it:\n\n"
            "1. `python scripts/prepare_ml_data.py --label-file data/raw/gfd_dfo_3850.tif` "
            "(or `--gee-project YOUR_PROJECT`)\n"
            "2. `python scripts/train_model.py`\n\nThen reload this page."
        )
    else:
        cv = ml.cv.set_index("method")
        ml_methods = [m for m in cv.index if m.startswith("ML model")]
        ml_name = ML_ALL if ML_ALL in cv.index else ml_methods[0]
        ml_auc = cv.loc[ml_name, "roc_auc"]
        best_other = cv.drop(index=ml_methods)["roc_auc"].idxmax()
        events = ml.report.get("events", [])
        factors = ('elevation, waterways and land cover' if ml.report.get('excluded_features')
                   else 'elevation, waterways, land cover and population')
        if len(events) > 1:
            radar_years = [e["year"] for e in events if e["id"].startswith("radar")]
            st.markdown(
                f"A gradient-boosted tree model (LightGBM) trained on **{len(events)} floods**: the 2011 flood "
                f"(MODIS) and the radar-mapped wet seasons of **{min(radar_years)}–{max(radar_years)}**, "
                f"{ml.report['cells_per_event']:,} labelled cells from each so every flood counts equally "
                f"(**{ml.report['n_training_cells']:,} cells** in all). Instead of hand-set weights, it learns how "
                f"{factors} relate to real flooding. Floods from {ml.report['test_years_from']} on are kept out "
                "of training, so the **Validation** tab can test on them."
            )
            with st.expander("The floods it was trained on, with each season's rain and river"):
                ctx = [e.get("context", {}) for e in events]
                cols = {"Flood": [e["name"] for e in events],
                        "Flooded": [e["flooded_share"] * 100 for e in events]}
                if any(ctx):
                    cols.update({
                        "Rain": [c.get("rain_total_mm") for c in ctx],
                        "Rain vs normal": [c.get("rain_pct_normal") for c in ctx],
                        "Wettest 3 days": [c.get("rain_max_3day_mm") for c in ctx],
                        "River peak": [c.get("river_peak_m3s") for c in ctx],
                        "Peak vs normal": [c.get("river_peak_pct_normal") for c in ctx],
                        "High-river days": [c.get("river_days_above_alert") for c in ctx],
                    })
                cols.update({"Labelled cells": [e["n_cells"] for e in events],
                             "Source": [e["source"] for e in events]})
                st.dataframe(
                    pd.DataFrame(cols), hide_index=True, use_container_width=True,
                    column_config={
                        "Flooded": st.column_config.NumberColumn("Flooded (% of labelled)", format="%.1f%%"),
                        "Rain": st.column_config.NumberColumn("Rain, Aug–Nov (mm)", format="%.0f"),
                        "Rain vs normal": st.column_config.NumberColumn(format="%.0f%%"),
                        "Wettest 3 days": st.column_config.NumberColumn("Wettest 3 days (mm)", format="%.0f"),
                        "River peak": st.column_config.NumberColumn("River peak (m³/s)", format="%.0f"),
                        "Peak vs normal": st.column_config.NumberColumn("River peak vs normal", format="%.0f%%"),
                        "High-river days": st.column_config.NumberColumn(
                            f"Days river ≥ {C.ALERT_RIVER_PCT}% of normal", format="%d"),
                    })
                st.caption("Radar seasons: August–November passes, each compared with the same weeks of the year "
                           "before. A cell is flooded if at least half of it was flooded on 2 or more passes, and "
                           "dry if it was seen and always under 10% flooded.")
                meta = ml.report.get("event_context")
                if meta:
                    st.caption(
                        f"Weather and river for context only; the model doesn't use them. August–November. Rain: "
                        f"{meta['rain_source']}; normal = {meta['rain_normal_mm']:.0f} mm, the "
                        f"{meta['normal_years'][0]}–{meta['normal_years'][1]} median. River: {meta['river_source']}, "
                        "compared with the median for each date. Both are model estimates, not gauge readings. "
                        "2011 stands out for its river, not its local rain: the flood came down the Chao Phraya "
                        "from the north.")
        else:
            st.markdown(
                f"A gradient-boosted tree model (LightGBM) trained on **{ml.report['n_training_cells']:,} grid "
                f"cells** labelled flooded or dry in the **2011 flood** ({ml.report['flooded_share']:.0%} "
                f"flooded). Instead of hand-set weights, it learns how {factors} relate to real flooding."
            )
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Does it beat the existing methods? (2011 flood)")
            st.plotly_chart(model_comparison_chart(ml.cv), use_container_width=True)
            gap = ml_auc - cv.loc[best_other, "roc_auc"]
            verdict = (f"The ML model scores **{gap:+.2f} ROC-AUC** vs the best existing method ({best_other})."
                       if abs(gap) >= 0.01 else
                       f"The ML model performs about the same as the {best_other.lower()}.")
            if cv.loc["Risk index (default weights)", "roc_auc"] < 0.5:
                verdict += (" The hand-weighted risk index scores *below random* for 2011: it ranks dense, "
                            "canal-rich inner districts highest, but those were largely protected and stayed dry.")
            st.caption(
                f"{verdict} Error bars = {'95% interval (spatial block bootstrap)' if 'roc_auc_lo' in ml.cv else 'spread over the folds'}. "
                f"Scores come from spatial cross-validation: the map is cut into "
                f"{ml.report['cv']['block_m'] / 1000:.0f} km blocks and each model is tested on blocks it never saw "
                "(held out of every flood year at once). ROC-AUC 0.5 = random, 1.0 = perfect. PR-AUC rewards "
                f"finding flooded cells without false alarms; a random guess scores {ml.report['flooded_share']:.2f}."
                + (" The **Validation** tab has the scores for every flood." if len(events) > 1 else "")
            )
        with c2:
            st.subheader("What drives the prediction?")
            st.plotly_chart(feature_importance_chart(ml.features), use_container_width=True)
            st.caption("SHAP values: how much each factor moves the prediction on average. "
                       "Red = higher values make flooding more likely; blue = less likely.")
        if ml.report.get("excluded_features"):
            st.caption("Left out on purpose: " + ", ".join(ml.report["excluded_features"]) + ". With them, the model "
                       "learns \"dense city = dry\", which mostly reflects 2011 flood defences and satellites "
                       "missing water between buildings. Removing them cost only a little accuracy.")
        st.subheader("Districts: predicted vs observed")
        comp = table[["district", "ml_score", "observed_2011_pct", "risk"]].sort_values("ml_score", ascending=False)
        st.dataframe(
            comp, hide_index=True, use_container_width=True, height=300,
            column_config={
                "district": "District",
                "ml_score": st.column_config.ProgressColumn("ML susceptibility", min_value=0, max_value=1, format="%.2f"),
                "observed_2011_pct": st.column_config.NumberColumn("Flooded in 2011 (% of mapped area)", format="%.0f%%"),
                "risk": st.column_config.NumberColumn("Hand-weighted risk index", format="%.2f"),
            },
        )
        st.markdown("""
**Read with care**
- The 2011 map comes from MODIS satellites at 250 m. They see water poorly between buildings, so **flooding in dense
  urban areas is probably under-counted**, and the model may partly learn "built-up = dry".
- It learned from the 2011 river flood and nine **ordinary wet seasons** (radar, 2017–2025). It predicts a typical
  year's flooding well but under-rates places that flood only in an extreme year like 2011, and it doesn't know
  about short cloudburst flooding in the city, which radar mostly misses.
- The radar seasons compare each pass with the same weeks of the year before, so a place that floods every
  year (like planted rice paddies) isn't counted as flooded, and a wet year can hide floods in the year after.
- Some coastal "flooding" in the 2011 map is likely shrimp and fish ponds, which MODIS can't tell apart from floodwater.
- Areas protected by flood walls stayed dry in 2011, so the model implicitly learns those defences.
  That's useful, but defences built since 2011 aren't included.
- Tick **ML model disagreement** in the sidebar to see where the prediction is least stable: five versions of the
  model, each trained without a different fifth of the map, give different answers there.
""")
with tab_valid:
    ev_all = get_radar_eval(file_version(C.RADAR_EVAL_FILE))
    vt = validation_table(ev_all, ml.report if ml is not None else None)
    if vt.empty:
        st.info("Nothing to validate yet. Train the ML model (`python scripts/train_model.py`) and/or fetch and "
                "score radar passes (`python scripts/fetch_current_flood.py`, then "
                "`python scripts/evaluate_current.py`).")
    else:
        radar_rows = vt[vt["test"] == TEST_RADAR]
        loeo = loeo_table(ml.report if ml is not None else None)
        st.markdown(
            "How well does each method say **where** Bangkok floods? Every score here is on floods, or places, "
            "the method never saw: the **training floods** (the ML model is tested on 4 km blocks held out of "
            "its training)"
            + (f", **each training flood left out in turn** ({len(loeo['event'].unique())} floods)"
               if not loeo.empty else "")
            + " and "
            + (f"**{radar_rows['date'].nunique()} recent Sentinel-1 radar passes** from {radar_rows['date'].min()} "
               f"to {radar_rows['date'].max()}, which no model was trained on." if not radar_rows.empty
               else "no radar passes yet.")
            + " The bathtub simulation and the risk index have no training data, so every flood is new to them."
        )
        v1, v2 = st.columns(2)
        vscope = v1.radio("Area", ["Greater Bangkok", "Bangkok (50 districts)"], horizontal=True,
                          key="valid_scope")
        vmetric = v2.radio("Score", ["roc_auc", "pr_auc"], horizontal=True, key="valid_metric",
                           format_func={"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}.get)
        metric_name = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}[vmetric]
        ml_name = ML_ALL if ev_all is None or ML_ALL in set(ev_all["method"]) else "ML model (trained on 2011)"

        if ev_all is not None and ml_name in set(ev_all["method"]):
            w = wins(ev_all, ml_name, vscope, vmetric)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"ML beats the bathtub ({metric_name})", f"{w['better']} of {w['passes']} passes")
            if w["has_intervals"]:
                m2.metric("Clearly better", w["clearly_better"],
                          help=f"Passes where the {CI}% interval of the difference is entirely above zero.")
                m3.metric("Clearly worse", w["clearly_worse"],
                          help=f"Passes where the {CI}% interval of the difference is entirely below zero.")
            cv11 = vt[(vt["test"] == TEST_CV) & vt["event"].str.startswith("2011")].set_index("method")
            if {ml_name, REFERENCE} <= set(cv11.index):
                gap = cv11.loc[ml_name, vmetric] - cv11.loc[REFERENCE, vmetric]
                m4.metric(f"2011 flood: ML minus bathtub ({metric_name})", f"{gap:+.2f}",
                          help="Spatial cross-validation: every cell scored by a model that never saw its 4 km block.")

        def_name = "Bathtub + defences"
        if ev_all is not None and def_name in set(ev_all["method"]):
            w = wins(ev_all, def_name, vscope, vmetric)
            d1, d2, d3, d4 = st.columns(4)
            d1.metric(f"Defences improve the bathtub ({metric_name})", f"{w['better']} of {w['passes']} passes",
                      help="The bathtub simulation with the river walls, King's Dike and airport dike, compared "
                           "with the plain bathtub.")
            if w["has_intervals"]:
                d2.metric("Clearly better", w["clearly_better"],
                          help=f"Passes where the {CI}% interval of the difference is entirely above zero.")
                d3.metric("Clearly worse", w["clearly_worse"],
                          help=f"Passes where the {CI}% interval of the difference is entirely below zero.")
            cv11 = vt[(vt["test"] == TEST_CV) & vt["event"].str.startswith("2011")].set_index("method")
            if {def_name, REFERENCE} <= set(cv11.index):
                gap = cv11.loc[def_name, vmetric] - cv11.loc[REFERENCE, vmetric]
                d4.metric("2011 flood: gain from defences", f"{gap:+.2f}",
                          help="Spatial cross-validation: every cell scored by a model that never saw its 4 km block.")

        if ev_all is not None and not ev_all.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader(f"{metric_name} on every radar pass")
                st.plotly_chart(score_over_time_chart(ev_all, vscope, vmetric, intervals=True),
                                use_container_width=True)
                st.caption(f"Shading = {CI}% interval. "
                           + ("ROC-AUC 0.5 = random, 1.0 = perfect." if vmetric == "roc_auc" else
                              "The dotted line is a random guess, which scores the share of cells flooded. "
                              "Flooding is rare on most passes, so PR-AUC is low for every method; compare "
                              "it with that line."))
            with c2:
                st.subheader("Difference from the bathtub simulation")
                st.plotly_chart(difference_chart(ev_all, vscope, REFERENCE, vmetric), use_container_width=True)
                st.caption(f"Above zero = better than the bathtub. A bar that doesn't cross zero is a difference "
                           f"the {CI}% interval says isn't just chance.")

        if not loeo.empty:
            st.subheader("Does training on more floods help? Leave one flood out")
            st.markdown(
                "Each flood in turn is predicted by a model trained on **all the other floods**, and compared "
                "with the old model trained on **2011 only**. This asks the question a planner cares about: how "
                "well does the model predict a flood year it has never seen?"
            )
            if {ML_ALL, ML_2011} <= set(loeo["method"]):
                w_old = wins(loeo, ML_ALL, "Greater Bangkok", vmetric, reference=ML_2011)
                w_bath = wins(loeo, ML_ALL, "Greater Bangkok", vmetric)
                w_def = wins(loeo, ML_ALL, "Greater Bangkok", vmetric, reference="Bathtub + defences")
                l1, l2, l3, l4 = st.columns(4)
                l1.metric(f"Beats the 2011-only model ({metric_name})", f"{w_old['better']} of {w_old['passes']} floods",
                          help="Radar seasons only: the 2011-only model can't be tested on 2011 without a hold-out.")
                if ev_all is not None and {ML_ALL, ML_2011} <= set(ev_all["method"]):
                    w_new = wins(ev_all, ML_ALL, vscope, vmetric, reference=ML_2011)
                    l2.metric("...and on the recent passes", f"{w_new['better']} of {w_new['passes']} passes",
                              help="The recent radar passes, which neither model was trained on.")
                l3.metric("Beats the bathtub", f"{w_bath['better']} of {w_bath['passes']} floods",
                          help=f"Clearly better on {w_bath['clearly_better']}, clearly worse on "
                               f"{w_bath['clearly_worse']} ({CI}% interval of the difference).")
                l4.metric("Beats bathtub + defences", f"{w_def['better']} of {w_def['passes']} floods")
                old_2011 = vt[(vt["test"] == TEST_CV) & vt["event"].str.startswith("2011")].set_index("method")
                if {ML_ALL, ML_2011} <= set(old_2011.index):
                    a, b = old_2011.loc[ML_ALL, vmetric], old_2011.loc[ML_2011, vmetric]
                    if a < b:
                        st.caption(
                            f"**The trade-off:** on the extreme 2011 flood the 2011-only model is still better "
                            f"({metric_name} {b:.3f} vs {a:.3f} in spatial CV). Nine of the ten training floods are "
                            "ordinary wet seasons, where 2–5% of the land floods; in 2011 about 21% did, including "
                            "places that rarely flood. The new model is the better guide to a typical year, and "
                            "the 2011 map (\"Observed flood (2011)\" layer) remains the guide to a rare, extreme one.")
            st.plotly_chart(score_over_time_chart(loeo, "Greater Bangkok", vmetric, intervals=True),
                            use_container_width=True)
            st.caption(f"One point per left-out flood (Greater Bangkok), shading = {CI}% block bootstrap interval. "
                       "The model is tested on every cell of the left-out year, including places it saw flood in "
                       "other years: features describe the place, not the weather, so this measures how well "
                       "past floods predict the next one.")

        st.subheader("All results")

        def with_interval(v, lo, hi):
            return f"{v:.3f}" if pd.isna(lo) else f"{v:.3f} ({lo:.3f}–{hi:.3f})"

        shown = vt[(vt["scope"] == vscope) | (vt["test"] != TEST_RADAR)]
        st.dataframe(
            pd.DataFrame({
                "Test": shown["test"], "Flood event": shown["event"], "Area": shown["scope"], "Method": shown["method"],
                "ROC-AUC": [with_interval(*r) for r in shown[["roc_auc", "roc_auc_lo", "roc_auc_hi"]].values],
                "PR-AUC": [with_interval(*r) for r in shown[["pr_auc", "pr_auc_lo", "pr_auc_hi"]].values],
                "Random PR-AUC": shown["random_pr_auc"], "Cells tested": shown["n_cells"],
                "Flooded": shown["flooded_share"] * 100, "Range shown": shown["interval"],
            }),
            hide_index=True, use_container_width=True, height=320,
            column_config={
                "Random PR-AUC": st.column_config.NumberColumn(format="%.4f"),
                "Cells tested": st.column_config.NumberColumn(format="%d"),
                "Flooded": st.column_config.NumberColumn("Flooded (%)", format="%.2f%%"),
            },
        )
        st.download_button("Download all results (CSV)", vt.to_csv(index=False).encode(),
                           "bangkok_flood_validation.csv", "text/csv")

        with st.expander("How the scores are worked out"):
            st.markdown(f"""
**Labels.** A ~65 m grid cell counts as *flooded* when at least {FLOODED_PCT}% of it is flooded on the radar map,
and as *dry* when under {DRY_PCT}% is, radar saw it, and it isn't permanent water (JRC Global Surface Water).
Cells in between are left out. The 2011 labels come from the Global Flood Database's MODIS map (250 m). Each
radar wet season (August–November, 2017 on) combines all its passes: flooded if flooded on at least 2 passes,
dry if seen and dry on every pass.

**Three kinds of test.**
- *Spatial CV:* the floods the ML model trains on, scored on 4 km blocks held out of training in every year.
- *Left-out flood:* each training flood predicted by a model trained on all the others.
- *Recent radar passes:* floods from the current year, which no model was trained on.

**Scores.** Each method gives every cell a score: the ML model's flood probability, how low a water level
floods it in the bathtub simulation, or the risk index (default weights).
- **ROC-AUC:** the chance that a flooded cell scores higher than a dry one. 0.5 = random, 1.0 = perfect.
- **PR-AUC (average precision):** how much of the top-scored land actually flooded. A random guess scores the
  share of cells flooded, so compare with that column.

**Intervals.** Neighbouring cells flood together, so they aren't independent tests. The radar intervals come
from a *spatial block bootstrap*: the map is cut into 4 km blocks, which are resampled 200 times, and every
method is re-scored on each resample. The {CI}% interval is the middle {CI}% of those scores. The difference
from the bathtub uses the same resamples, so it has its own interval. The left-out floods get intervals the
same way. The spatial CV scores pool the five folds, so every cell is scored once by the model that never saw
its block, and get their intervals from the same bootstrap.

**Known biases.**
- Radar and MODIS both miss much of the water between buildings, so the dense city looks drier than it was.
- Most detected flooding is in the eastern and western farmland, so the scores mostly reflect the rural fringe.
  The "Bangkok (50 districts)" area includes less of it.
- Flooded rice fields can be normal farming rather than a flood. Comparing each pass with the same weeks last
  year removes most of this, but not all.
- Passes a few days apart see the same flood, so they aren't independent tests either. Read "beats the
  bathtub on N passes" as a trend, not N separate trials.
""")
with tab_elev:
    below_km2 = area.cell_area_km2[bkk & (area.dem <= level)].sum()
    st.markdown(
        f"How much of Bangkok's land sits at each height. **{below_km2:,.0f} km² "
        f"({below_km2 / area.cell_area_km2[bkk].sum():.0%})** of the city's 50 districts is at or below "
        f"{level:.1f} m. Not all of it floods in the simulation, because some low ground isn't connected "
        f"to a river or canal. Turn on **Ground elevation** in the sidebar to see this on the map."
    )
    st.plotly_chart(
        elevation_histogram(area.dem[bkk], area.cell_area_km2[bkk], level), use_container_width=True
    )
    lowest = table.nsmallest(5, "elevation")
    st.caption("Lowest districts on average: " + ", ".join(
        f"{r.district} ({r.elevation:.1f} m)" for r in lowest.itertuples()))
with tab_top:
    if result.area_km2 == 0:
        st.write("No flooding at this level.")
    else:
        st.plotly_chart(top_districts_chart(table), use_container_width=True)
with tab_curve:
    st.markdown(
        "How fast flooding grows as the water rises. The sharp steps at 0.5, 1.0, 1.5 and 2.0 m come from "
        "the elevation data: Copernicus rounds many flat areas to 0.5 m heights, so large patches "
        "switch to flooded all at once. Read the overall slope, not the individual steps."
    )
    band = dem_unc.curve(connected, use_defences) if dem_unc is not None else None
    st.plotly_chart(level_curve_chart(get_curve(connected, use_defences), level, band), use_container_width=True)
    if band is not None:
        u = dem_unc.meta
        here = dem_unc.at(connected, use_defences, level)
        st.markdown(
            f"**How sure is this?** The elevation data has errors, and in a city this flat a few tens of "
            f"centimetres decide whether a street floods. The shaded bands show the middle 90% of "
            f"**{u['runs']} simulations**, each with random errors added to the elevation (standard deviation "
            f"{u['sigma_m']:.1f} m, similar over about {u['corr_m']:.0f} m, so neighbouring cells are wrong "
            f"together). At {level:.1f} m: **{here['area_km2_p5']:,.0f}–{here['area_km2_p95']:,.0f} km²** "
            f"flooded and **{fmt_people(here['people_p5'])}–{fmt_people(here['people_p95'])} people**, against "
            f"{result.area_km2:,.0f} km² and {fmt_people(result.people)} on the elevation as it is."
        )
        pess = get_dem_uncertainty(C.DEM_UNCERTAINTY_PESSIMISTIC_FILE,
                                   file_version(C.DEM_UNCERTAINTY_PESSIMISTIC_FILE))
        if pess is not None and (p := pess.at(connected, use_defences, level)) is not None:
            st.markdown(
                f"**The size of the error matters more than the spread.** The band is narrow because errors average "
                f"out over thousands of cells. But if the elevation error were {pess.meta['sigma_m']:.1f} m instead "
                f"of {u['sigma_m']:.1f} m, the range at {level:.1f} m would be "
                f"**{p['area_km2_p5']:,.0f}–{p['area_km2_p95']:,.0f} km²** and "
                f"**{fmt_people(p['people_p5'])}–{fmt_people(p['people_p95'])} people**. Treat the numbers as an "
                "order of magnitude, and the map as a district-scale picture, not a street-scale one."
            )
        st.caption(
            "The single simulation can sit near the edge of the band, or outside it: the elevation data rounds "
            "flat areas to 0.5 m steps, so at exactly 0.5, 1.0, 1.5 m… a whole plateau sits at the water line and "
            "floods together. Add a little error and only part of it does. The error size is an estimate, not a "
            "measured value (see Method & limitations). Tick **Chance flooded** in the sidebar to see where the "
            "simulations disagree."
        )
with tab_table:
    st.dataframe(
        table.drop(columns="district_id").sort_values("risk", ascending=False),
        hide_index=True, use_container_width=True,
        column_config={
            "district": "District",
            "elevation": st.column_config.NumberColumn("Avg. elevation (m)", format="%.1f"),
            "ml_score": st.column_config.ProgressColumn("ML susceptibility", min_value=0, max_value=1, format="%.2f"),
            "observed_2011_pct": st.column_config.NumberColumn("Flooded in 2011 (%)", format="%.0f%%"),
            "flooded_now_pct": st.column_config.NumberColumn(f"Flooded now (%), {radar_date}", format="%.1f%%"),
            "flooded_now_km2": None,
            "risk": st.column_config.ProgressColumn("Risk index", min_value=0, max_value=1, format="%.2f"),
            "pct_flooded": st.column_config.ProgressColumn("Area flooded", min_value=0, max_value=100, format="%.1f%%"),
            "flooded_km2": st.column_config.NumberColumn("Flooded km²", format="%.1f"),
            "people_affected": st.column_config.NumberColumn("People affected", format="%d"),
        },
    )
with tab_method:
    st.markdown(f"""
**Flood simulation ("bathtub" model).** Every grid cell (~65 m) whose ground elevation is at or below
the chosen water level floods. When *water must flow from rivers/canals* is on, a low area floods only
if it is connected to a river or canal cell. Depth = water level − ground elevation.

**Flood defences (optional).** With *Include flood defences* on, the Chao Phraya river walls (crest 2.5–3.0 m,
by section), the King's Dike around eastern Bangkok (2.5 m) and the Suvarnabhumi Airport dike (3.5 m) block water
until the level rises above their crest. Canals inside the walls don't flood the land around them, because pumps
and gates keep them low. The routes follow published descriptions traced along OpenStreetMap roads; the crest
heights are the lower end of published ranges. Hover a line on the map for its source. The northern closure
between Phahon Yothin Road and the river is approximate.

**Risk index.** A weighted blend of three factors, each scaled 0–1:
low elevation (relative to the area), distance to the nearest river/canal (decays over ~1 km),
and log population density. Change the weights in the sidebar to see how the ranking shifts.

**Data.** Copernicus GLO-30 DEM · WorldPop 2020 constrained population · OpenStreetMap waterways ·
geoBoundaries district boundaries. The study area covers greater Bangkok
(≈ {fmt_people(area.population.sum())} people); stats in the table cover only Bangkok's 50 districts
(≈ {fmt_people(area.population[bkk].sum())} people).

**Uncertainty.** Two kinds are shown.
- *Elevation error* (`scripts/dem_uncertainty.py`): the bathtub simulation is rerun 50 times, each time on the
  elevation map plus a random error field with a standard deviation of {dem_unc.meta['sigma_m'] if dem_unc else 0.7:.1f} m
  that is correlated over about {dem_unc.meta['corr_m'] if dem_unc else 300:.0f} m. Simulating plausible versions of
  a global DEM this way follows Hawker et al. (2018, *Frontiers in Earth Science* 6:233). The size is an
  estimate: Copernicus GLO-30 is off by 1.6 m on average in built-up areas (Hawker et al. 2022, FABDEM, *Environ.
  Res. Lett.* 17:024016), mostly because it includes buildings. Taking the lowest value in each cell removes much of
  that, leaving an assumed 0.7 m. Ranges are the 5th to 95th percentile over the runs. Defence crest heights are
  kept fixed.
- *ML model disagreement:* the standard deviation of the flood probability across the five cross-validation
  models, each trained with a different fifth of the map held out. Every model score in the **Validation** tab
  has a 95% interval from a spatial block bootstrap.

These ranges cover the elevation error and the model's sensitivity to its training data. They don't cover the
other limits listed below, such as static water, missing drainage or old population data, so the real
uncertainty is larger.

**Limitations — this is an exploratory model, not a forecast:**
- Flood walls and dikes are modelled only when *Include flood defences* is on, with one crest height per
  section; real walls vary along their length and have gaps. Pumping stations, drainage tunnels and the
  capacity of the gates are not modelled, and the west bank's (Thonburi) polder dikes aren't included.
- Water is static: no flow speed, rainfall timing, tides or duration.
- The elevation data is a *surface* model (includes buildings). Taking the lowest value in each cell
  reduces this, but some errors remain (the ranges above show their effect), and Bangkok's ground is
  sinking a few cm per year.
- The elevation data rounds many flat areas to 0.5 m steps, which is why flooded area jumps at
  0.5, 1.0, 1.5 and 2.0 m.
- Population figures are 2020 modelled estimates.
""")
