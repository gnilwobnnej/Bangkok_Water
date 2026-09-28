"""Map layers, animation frames and charts."""
from __future__ import annotations

import base64
import io

import branca.colormap as bcm
import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import colormaps
from matplotlib.colors import LightSource, LinearSegmentedColormap
from PIL import Image

from src import config as C
from src.data_loader import StudyArea

MAX_DEPTH_M = 3.0
WATER_CMAP = LinearSegmentedColormap.from_list("water", ["#8fd3ff", "#2b7bd6", "#0a2f6b"])
RISK_CMAP = colormaps["YlOrRd"]
# Magma without its near-black end, so low ground stays visible on a dark basemap and doesn't read as water.
ELEV_COLORS = ["#3b0f70", "#8c2981", "#de4968", "#fe9f6d", "#fcfdbf"]
ELEV_CMAP = LinearSegmentedColormap.from_list("elevation", ELEV_COLORS)
ML_CMAP = colormaps["RdPu"]
OBSERVED_COLOR = (255, 176, 0)  # amber, so it stands out against the simulated-flood blues
RADAR_COLOR = (255, 64, 64)     # red: flooding seen by radar on the selected date
BOUNDS = [[C.SOUTH, C.WEST], [C.NORTH, C.EAST]]
CENTER = [(C.SOUTH + C.NORTH) / 2, (C.WEST + C.EAST) / 2]
# Esri canvas basemaps need no API key (CARTO's tiles are now watermarked without one).
ESRI_CANVAS = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
ESRI_ATTR = "Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors"


def fmt_people(n: float) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.2f} M"
    if n >= 1e3:
        return f"{n / 1e3:.0f} K"
    return f"{n:.0f}"


# ---------- raster -> image ----------

def depth_rgba(depth: np.ndarray, alpha: int = 210) -> np.ndarray:
    rgba = (WATER_CMAP(np.clip(depth / MAX_DEPTH_M, 0, 1)) * 255).astype("uint8")
    rgba[..., 3] = np.where(depth > 0, alpha, 0)
    return rgba


def risk_rgba(risk: np.ndarray, inside: np.ndarray, alpha: int = 150) -> np.ndarray:
    rgba = (RISK_CMAP(risk) * 255).astype("uint8")
    rgba[..., 3] = np.where(inside, alpha, 0)
    return rgba


def elevation_rgba(dem: np.ndarray, vmin: float, vmax: float, alpha: int = 225) -> np.ndarray:
    """Colour elevation stretched to [vmin, vmax], multiplied by hillshade so relief stands out."""
    norm = np.clip((dem - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    rgb = ELEV_CMAP(norm)[..., :3]
    shade = LightSource(azdeg=315, altdeg=40).hillshade(np.clip(dem, vmin - 1, vmax + 5), vert_exag=30, dx=65, dy=65)
    rgb = rgb * (0.6 + 0.4 * shade[..., None])
    rgba = np.empty(dem.shape + (4,), dtype="uint8")
    rgba[..., :3] = (np.clip(rgb, 0, 1) * 255).astype("uint8")
    rgba[..., 3] = alpha
    return rgba


def probability_rgba(prob: np.ndarray, alpha: int = 170) -> np.ndarray:
    rgba = (ML_CMAP(np.clip(prob, 0, 1)) * 255).astype("uint8")
    rgba[..., 3] = alpha
    return rgba


def observed_rgba(observed: np.ndarray, alpha: int = 170) -> np.ndarray:
    """Cells flooded in 2011 in amber; dry and unknown cells transparent."""
    rgba = np.zeros(observed.shape + (4,), dtype="uint8")
    rgba[..., :3] = OBSERVED_COLOR
    rgba[..., 3] = np.where(observed == 1, alpha, 0)
    return rgba


def radar_rgba(flood_pct: np.ndarray, max_alpha: int = 235) -> np.ndarray:
    """Radar-detected flooding in red, more opaque where more of the cell is flooded."""
    rgba = np.zeros(flood_pct.shape + (4,), dtype="uint8")
    rgba[..., :3] = RADAR_COLOR
    frac = np.clip(flood_pct / 100, 0, 1)
    rgba[..., 3] = np.where(frac >= 0.1, (80 + frac * (max_alpha - 80)).astype("uint8"), 0)
    return rgba


def to_data_url(rgba: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def basemap_rgb(area: StudyArea) -> np.ndarray:
    """Dark shaded-relief backdrop with district borders, used for the animation frames."""
    ls = LightSource(azdeg=315, altdeg=40)
    shade = ls.hillshade(np.clip(area.dem, -2, 15), vert_exag=40, dx=65, dy=65)
    elev = np.clip((area.dem + 1) / 8, 0, 1)
    base = 0.10 + 0.22 * shade[..., None] * np.array([0.9, 0.95, 1.0]) + 0.08 * elev[..., None]
    ids = area.district_ids
    edge = (ids != np.roll(ids, 1, 0)) | (ids != np.roll(ids, 1, 1))
    base[edge] = [0.55, 0.55, 0.6]
    base[area.water & (area.dem <= 0.5)] = [0.15, 0.25, 0.40]
    return np.clip(base, 0, 1)


def flood_frame(base: np.ndarray, depth: np.ndarray) -> np.ndarray:
    rgba = depth_rgba(depth).astype("float32") / 255
    a = rgba[..., 3:4]
    return ((base * (1 - a) + rgba[..., :3] * a) * 255).astype("uint8")


# ---------- interactive map ----------

def build_map(
    area: StudyArea, table: pd.DataFrame, depth: np.ndarray | None, risk: np.ndarray | None,
    show_waterways: bool, dark: bool, elevation_range: tuple[float, float] | None = None,
    ml_prob: np.ndarray | None = None, observed: np.ndarray | None = None,
    radar: np.ndarray | None = None, radar_label: str = "Radar flood",
) -> folium.Map:
    m = folium.Map(
        location=CENTER, zoom_start=11, control_scale=True,
        tiles=ESRI_CANVAS.format("World_Dark_Gray_Base" if dark else "World_Light_Gray_Base"),
        attr=ESRI_ATTR,
    )
    if elevation_range is not None:
        vmin, vmax = elevation_range
        folium.raster_layers.ImageOverlay(
            to_data_url(elevation_rgba(area.dem, vmin, vmax)), bounds=BOUNDS,
            name="Ground elevation", interactive=False, zindex=1,
        ).add_to(m)
        legend = bcm.LinearColormap(ELEV_COLORS, vmin=vmin, vmax=vmax)
        legend.caption = "Ground elevation (m)"
        legend.add_to(m)
        # White card behind the legend so it stays readable on the dark basemap.
        m.get_root().header.add_child(folium.Element(
            "<style>.legend.leaflet-control{background:rgba(255,255,255,.88);"
            "padding:4px 8px 2px;border-radius:6px}</style>"
        ))
    if risk is not None:
        folium.raster_layers.ImageOverlay(
            to_data_url(risk_rgba(risk, area.district_ids > 0)), bounds=BOUNDS,
            name="Flood risk index", interactive=False, zindex=2,
        ).add_to(m)
    if ml_prob is not None:
        folium.raster_layers.ImageOverlay(
            to_data_url(probability_rgba(ml_prob)), bounds=BOUNDS,
            name="ML flood susceptibility", interactive=False, zindex=2,
        ).add_to(m)
    if observed is not None:
        folium.raster_layers.ImageOverlay(
            to_data_url(observed_rgba(observed)), bounds=BOUNDS,
            name="Observed flood (2011)", interactive=False, zindex=4,
        ).add_to(m)
    if radar is not None:
        folium.raster_layers.ImageOverlay(
            to_data_url(radar_rgba(radar)), bounds=BOUNDS,
            name=radar_label, interactive=False, zindex=5,
        ).add_to(m)
    if depth is not None:
        folium.raster_layers.ImageOverlay(
            to_data_url(depth_rgba(depth)), bounds=BOUNDS,
            name="Flood depth", interactive=False, zindex=3,
        ).add_to(m)
    if show_waterways:
        folium.GeoJson(
            area.waterways[["kind", "geometry"]].to_json(),
            name="Rivers & canals",
            style_function=lambda f: {
                # Thin and faint so channels aren't mistaken for floodwater.
                "color": "#9be7ff",
                "weight": 1.2 if f["properties"]["kind"] == "river" else 0.6,
                "opacity": 0.55 if f["properties"]["kind"] == "river" else 0.35,
            },
        ).add_to(m)

    districts = area.districts.merge(table.drop(columns="district"), on="district_id")
    districts["people_label"] = districts["people_affected"].map(fmt_people)
    districts["risk_label"] = districts["risk"].map("{:.2f}".format)
    districts["pct_label"] = districts["pct_flooded"].map("{:.1f}%".format)
    districts["elev_label"] = districts["elevation"].map("{:.1f} m".format)
    fields = ["district", "elev_label", "risk_label", "pct_label", "people_label"]
    aliases = ["District", "Avg. elevation", "Risk index", "Area flooded", "People affected"]
    if "ml_score" in districts:
        districts["ml_label"] = districts["ml_score"].map("{:.2f}".format)
        districts["obs_label"] = districts["observed_2011_pct"].map(
            lambda v: "no data" if pd.isna(v) else f"{v:.0f}%")
        fields += ["ml_label", "obs_label"]
        aliases += ["ML susceptibility", "Flooded in 2011"]
    if "flooded_now_pct" in districts:
        districts["now_label"] = districts["flooded_now_pct"].map("{:.1f}%".format)
        fields.append("now_label")
        aliases.append("Flooded now (radar)")
    folium.GeoJson(
        districts[fields + ["geometry"]].to_json(),
        name="Districts",
        style_function=lambda f: {"color": "#bbbbbb", "weight": 1, "fillOpacity": 0},
        highlight_function=lambda f: {"color": "#ffffff", "weight": 3, "fillOpacity": 0.1},
        tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases),
    ).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    return m


# ---------- charts ----------

def top_districts_chart(table: pd.DataFrame, n: int = 10) -> go.Figure:
    top = table.nlargest(n, "pct_flooded").iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top["pct_flooded"], y=top["district"], orientation="h",
        marker_color="#2b7bd6",
        customdata=np.stack([top["people_affected"].map(fmt_people)], axis=-1),
        hovertemplate="%{y}<br>%{x:.1f}% flooded<br>%{customdata[0]} people<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Area flooded (%)", xaxis_range=[0, 100], height=420,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def level_curve_chart(curve: pd.DataFrame, level: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve["level"], y=curve["area_km2"], name="Flooded area (km²)",
        line=dict(color="#2b7bd6", width=3),
        hovertemplate="%{x:.1f} m → %{y:,.0f} km²<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=curve["level"], y=curve["people"] / 1e6, name="People affected (M)", yaxis="y2",
        line=dict(color="#f28e2b", width=3, dash="dot"),
        hovertemplate="%{x:.1f} m → %{y:.2f} M people<extra></extra>",
    ))
    fig.add_vline(x=level, line_dash="dash", line_color="gray",
                  annotation_text=f"{level:.1f} m", annotation_position="top left")
    fig.update_layout(
        xaxis_title="Water level above mean sea level (m)",
        yaxis=dict(title="Flooded area (km²)"),
        yaxis2=dict(title="People affected (millions)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.12), height=420, margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def elevation_histogram(dem: np.ndarray, cell_area_km2: np.ndarray, level: float) -> go.Figure:
    """Land area per 0.25 m elevation band, split into below / above the water level."""
    edges = np.arange(-2, 10.01, 0.25)
    clipped = np.clip(dem, edges[0], edges[-1] - 1e-3)
    area, _ = np.histogram(clipped, bins=edges, weights=cell_area_km2)
    centers = (edges[:-1] + edges[1:]) / 2
    below = edges[1:] <= level + 1e-9
    fig = go.Figure()
    for mask, name, color in [(below, "At or below water level", "#2b7bd6"), (~below, "Above water level", "#de4968")]:
        fig.add_trace(go.Bar(
            x=centers[mask], y=area[mask], width=0.24, name=name, marker_color=color,
            hovertemplate="%{x:.2f} m ± 0.125 → %{y:,.0f} km²<extra></extra>",
        ))
    fig.add_vline(x=level, line_dash="dash", line_color="gray",
                  annotation_text=f"water {level:.1f} m", annotation_position="top right")
    fig.update_layout(
        xaxis_title="Ground elevation (m above sea level; ends include everything beyond −2 / 10 m)",
        yaxis_title="Land area (km²)", barmode="overlay", height=420,
        legend=dict(orientation="h", y=1.12), margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def model_comparison_chart(cv: pd.DataFrame) -> go.Figure:
    """Spatial cross-validation scores of the ML model vs the existing methods."""
    order = ["ML model (LightGBM)", "ML model (trained on 2011)", "Bathtub simulation",
             "Risk index (default weights)"]
    known = [m for m in order if m in set(cv["method"])]
    cv = cv.set_index("method").reindex(known + [m for m in cv["method"] if m not in known]).reset_index()
    fig = go.Figure()
    for metric, name, color in [("roc_auc", "ROC-AUC", "#2b7bd6"), ("pr_auc", "PR-AUC", "#de4968")]:
        std = f"{metric}_std"
        fig.add_trace(go.Bar(
            x=cv["method"], y=cv[metric], name=name, marker_color=color,
            error_y=dict(type="data", array=cv[std], visible=True) if std in cv else None,
            text=cv[metric].map("{:.2f}".format), textposition="inside", insidetextanchor="middle",
            hovertemplate="%{x}<br>" + name + " %{y:.3f}<extra></extra>",
        ))
    fig.update_layout(
        barmode="group", yaxis=dict(title="Score (1.0 = perfect)", range=[0, 1.1]), height=380,
        legend=dict(orientation="h", y=1.12), margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def feature_importance_chart(features: pd.DataFrame, n: int = 10) -> go.Figure:
    """Mean |SHAP| per feature, coloured by whether higher values raise or lower flood chance."""
    top = features.nlargest(n, "mean_abs_shap").iloc[::-1]
    raises = top["direction"] > 0
    fig = go.Figure(go.Bar(
        x=top["mean_abs_shap"], y=top["label"], orientation="h",
        marker_color=np.where(raises, "#de4968", "#2b7bd6"),
        customdata=np.where(raises, "higher value → more likely to flood", "higher value → less likely to flood"),
        hovertemplate="%{y}<br>impact %{x:.3f}<br>%{customdata}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Average impact on the prediction (mean |SHAP|)", height=420,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def flood_timeline_chart(passes: pd.DataFrame, rain: pd.DataFrame, selected: str | None = None) -> go.Figure:
    """Daily rainfall (bars) and radar-detected flooded area per pass (lines)."""
    fig = go.Figure()
    if len(rain):
        fig.add_trace(go.Bar(
            x=rain["date"], y=rain["mm"], name="Daily rainfall (mm)", marker_color="rgba(43,123,214,0.35)",
            yaxis="y2", hovertemplate="%{x}<br>%{y:.0f} mm rain<extra></extra>",
        ))
    for col, name, color in [("flooded_km2", "Flooded, greater Bangkok (km²)", "#de4968"),
                             ("flooded_km2_bangkok", "Flooded, Bangkok 50 districts (km²)", "#8c2981")]:
        fig.add_trace(go.Scatter(
            x=passes["date"], y=passes[col], name=name, mode="lines+markers",
            line=dict(color=color, width=3),
            customdata=np.stack([passes["direction"].str.title(), passes["orbit"]], axis=-1),
            hovertemplate="%{x}<br>%{y:.1f} km²<br>%{customdata[0]} pass, orbit %{customdata[1]}<extra></extra>",
        ))
    if selected:
        fig.add_vline(x=selected, line_dash="dash", line_color="gray")
    fig.update_layout(
        yaxis=dict(title="Flooded area (km²)", rangemode="tozero"),
        yaxis2=dict(title="Rainfall (mm/day)", overlaying="y", side="right", rangemode="tozero", showgrid=False),
        legend=dict(orientation="h", y=1.15), height=400, margin=dict(l=10, r=10, t=40, b=10),
        barmode="overlay",
    )
    return fig


def score_over_time_chart(ev: pd.DataFrame, scope: str) -> go.Figure:
    """ROC-AUC of each method against each radar pass."""
    colors = {"ML model (trained on 2011)": "#2b7bd6", "Bathtub simulation": "#f28e2b",
              "Risk index (default weights)": "#8a94a3"}
    fig = go.Figure()
    for method, grp in ev[ev["scope"] == scope].groupby("method", sort=False):
        fig.add_trace(go.Scatter(
            x=grp["date"], y=grp["roc_auc"], name=method, mode="lines+markers",
            line=dict(color=colors.get(method), width=3),
            hovertemplate="%{x}<br>ROC-AUC %{y:.2f}<extra>" + method + "</extra>",
        ))
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray", annotation_text="random",
                  annotation_position="bottom right")
    fig.update_layout(
        yaxis=dict(title="ROC-AUC (1.0 = perfect)", range=[0.3, 1.0]), height=360,
        legend=dict(orientation="h", y=1.15), margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig
