"""Capture screenshots of the running app for the presenter guide.

Requires Playwright (`pip install playwright`) and Google Chrome. Start the app first:
    streamlit run app.py --server.port 8599 --server.headless true
    python scripts/capture_screenshots.py
Screenshots are written to docs/screenshots/.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

URL = "http://localhost:8599"
OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
MAP = 'iframe[title="streamlit_folium.st_folium"]'
MAP_HEIGHT = 620  # the component iframe reports a taller box than the visible map

# Sidebar slider order in app.py
LEVEL, W_ELEV, W_WATER, W_POP = 0, 1, 2, 3


def trim_bottom(path, pad=20):
    """Remove trailing blank rows (tab panels are as tall as the tallest tab)."""
    im = Image.open(path)
    rows = np.where((np.asarray(im.convert("L")) < 245).any(axis=1))[0]
    if len(rows):
        im.crop((0, 0, im.width, min(im.height, rows[-1] + pad))).save(path)


class App:
    def __init__(self, page):
        self.pg = page

    def settle(self, ms=4500):
        """Wait for Streamlit to finish rerunning and for map tiles to load."""
        self.pg.wait_for_timeout(600)
        self.pg.wait_for_function(
            "() => !document.querySelector('[data-testid=\"stStatusWidget\"]')"
            "      || !document.querySelector('[data-testid=\"stStatusWidget\"]').innerText.includes('Running')",
            timeout=60000,
        )
        # During a rerun the old map iframe lingers next to the new one; wait until only one is left.
        self.pg.wait_for_function(
            "() => document.querySelectorAll('iframe[title=\"streamlit_folium.st_folium\"]').length <= 1",
            timeout=120000,
        )
        self.pg.wait_for_timeout(ms)

    def slider(self, index, value, lo, step):
        thumb = self.pg.locator('[data-testid="stSlider"]').nth(index).locator('[role="slider"]').first
        thumb.focus()
        for _ in range(70):  # Home isn't supported, so walk down to the minimum
            thumb.press("ArrowLeft")
        for _ in range(round((value - lo) / step)):
            thumb.press("ArrowRight")
        self.settle()

    def level(self, value):
        self.slider(LEVEL, value, 0.0, 0.1)

    def weights(self, elev, water, pop):
        self.slider(W_ELEV, elev, 0.0, 0.05)
        self.slider(W_WATER, water, 0.0, 0.05)
        self.slider(W_POP, pop, 0.0, 0.05)

    def checkbox(self, label, on):
        box = self.pg.locator('[data-testid="stSidebar"] label').filter(has_text=label).first
        checked = box.locator("input").is_checked()
        if checked != on:
            box.click()
            self.settle()

    def map_box(self):
        self.pg.wait_for_function(  # a stale map iframe can linger briefly after a rerun
            "() => document.querySelectorAll('iframe[title=\"streamlit_folium.st_folium\"]').length <= 1",
            timeout=120000,
        )
        self.pg.locator(MAP).scroll_into_view_if_needed()
        box = self.pg.locator(MAP).bounding_box()
        return {**box, "height": MAP_HEIGHT}

    def shot_map(self, name):
        self.pg.screenshot(path=OUT / f"{name}.png", clip=self.map_box())
        print("saved", name)

    def shot_page(self, name, height=1100):
        self.pg.screenshot(path=OUT / f"{name}.png", clip={"x": 0, "y": 0, "width": 1600, "height": height})
        print("saved", name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        # WebGL (software-rendered) is needed for the 3D buildings tab.
        browser = p.chromium.launch(channel="chrome", args=["--enable-webgl", "--ignore-gpu-blocklist", "--use-angle=swiftshader"])
        pg = browser.new_page(viewport={"width": 1600, "height": 1100}, device_scale_factor=1.5)
        app = App(pg)
        pg.goto(URL, wait_until="networkidle")
        pg.wait_for_selector(MAP, timeout=90000)
        # Hide Streamlit's fixed top bar so it doesn't cover the map or clutter screenshots.
        pg.add_style_tag(content='[data-testid="stHeader"], [data-testid="stDecoration"] {display: none !important;}')
        app.settle(6000)

        # 1. Overview and sidebar at the default 1.0 m
        app.shot_page("01_overview")
        app.shot_map("01b_map_level_1_0")
        pg.set_viewport_size({"width": 1600, "height": 2100})
        app.settle(1500)
        pg.locator('[data-testid="stSidebar"]').screenshot(path=OUT / "02_sidebar.png")
        print("saved 02_sidebar")
        pg.set_viewport_size({"width": 1600, "height": 1100})
        app.settle(2500)

        # 2. District tooltip: hover over a point east of the map centre (eastern districts)
        box = app.map_box()
        pg.mouse.move(box["x"] + box["width"] * 0.60, box["y"] + 250)
        pg.mouse.move(box["x"] + box["width"] * 0.66, box["y"] + 290, steps=15)
        pg.frame_locator(MAP).locator(".leaflet-tooltip").first.wait_for(timeout=15000)
        pg.wait_for_timeout(500)
        pg.screenshot(path=OUT / "03_tooltip.png", clip=box)
        print("saved 03_tooltip")
        pg.mouse.move(5, 5)

        # 3. Rising water: compare levels
        for lv, name in [(0.5, "04_level_0_5"), (2.0, "05_level_2_0")]:
            app.level(lv)
            app.shot_map(name)

        # 4. Rain-ponding scenario (connectivity off) at 1.0 m
        app.level(1.0)
        app.checkbox("Water must flow from rivers/canals", False)
        app.shot_map("06_connectivity_off")
        app.checkbox("Water must flow from rivers/canals", True)

        # 4b. Flood defences at 2.0 m (only once scripts/prepare_defences.py has run)
        if pg.locator('[data-testid="stSidebar"] label').filter(has_text="Include flood defences").count():
            app.level(2.0)
            app.checkbox("Include flood defences", True)
            app.shot_map("29_defences_2_0")
            app.checkbox("Include flood defences", False)
            app.level(1.0)

        # 5. Animation, captured mid-run
        pg.get_by_role("button", name="Animate rising water").click()
        pg.wait_for_timeout(3500)
        app.shot_page("07_animation", height=1100)
        pg.wait_for_selector("text=Animation finished", timeout=120000)

        # 6. Elevation layer
        app.checkbox("Flood depth", False)
        app.checkbox("Ground elevation", True)
        app.shot_map("08_elevation")
        app.checkbox("Flood depth", True)
        app.level(1.5)
        app.shot_map("09_elevation_flood")
        app.checkbox("Ground elevation", False)

        # 7. Risk index under different weightings
        app.checkbox("Flood depth", False)
        app.checkbox("Risk index", True)
        for (e, w, pop), name in [
            ((0.5, 0.3, 0.2), "10_risk_default"),
            ((0.3, 0.6, 0.1), "11_risk_river"),
            ((0.6, 0.0, 0.4), "12_risk_rain"),
            ((0.25, 0.15, 0.6), "13_risk_people"),
            ((0.6, 0.4, 0.0), "14_risk_hazard_only"),
        ]:
            app.weights(e, w, pop)
            app.shot_map(name)
        app.weights(0.5, 0.3, 0.2)
        app.checkbox("Risk index", False)

        # 8. Machine-learning model layers (only present once scripts/train_model.py has run)
        has_ml = pg.locator('[data-testid="stSidebar"] label').filter(has_text="ML flood susceptibility").count() > 0
        if has_ml:
            app.checkbox("ML flood susceptibility", True)
            app.shot_map("19_ml_susceptibility")
            app.checkbox("ML flood susceptibility", False)
            app.checkbox("Observed flood (2011)", True)
            app.shot_map("20_observed_2011")
            app.checkbox("Flood depth", True)  # simulated 1.5 m under the observed 2011 extent
            app.shot_map("21_observed_vs_simulated")
            app.checkbox("Observed flood (2011)", False)

        # 9. Current flood from Sentinel-1 radar (only once scripts/fetch_current_flood.py has run)
        has_radar = pg.locator('[data-testid="stSidebar"] label').filter(has_text="Radar flood").count() > 0
        if has_radar:
            app.checkbox("Flood depth", False)
            app.checkbox("Radar flood", True)  # latest pass is selected by default
            app.shot_map("23_radar_flood")
            app.checkbox("Radar flood", False)
        app.checkbox("Flood depth", True)

        # 10. Analysis tabs at 1.5 m
        tabs = pg.locator('[data-testid="stTabs"]')
        for label, name in [
            ("Most affected districts", "15_tab_top"),
            ("Elevation profile", "16_tab_elevation"),
            ("Flood curve", "17_tab_curve"),
            ("District table", "18_tab_table"),
        ] + ([("ML model", "22_tab_ml")] if has_ml else []) + (
            [("Current flood", "24_tab_current")] if has_radar else []) + (
            [("Validation", "28_tab_validation")] if has_ml or has_radar else []):
            pg.get_by_role("tab").filter(has_text=label).click()
            pg.wait_for_timeout(2500)
            if name in ("24_tab_current", "28_tab_validation"):
                # Taller than the window: show the tab down to the end of its first table
                pg.set_viewport_size({"width": 1600, "height": 2100})
                app.settle(2500)
                tabs.evaluate("e => e.scrollIntoView({block: 'start'})")
                pg.wait_for_timeout(1500)
                top = tabs.bounding_box()
                table = pg.locator('[data-testid="stDataFrame"]:visible').first.bounding_box()
                pg.screenshot(path=OUT / f"{name}.png", clip={
                    "x": top["x"], "y": top["y"], "width": top["width"],
                    "height": table["y"] + table["height"] - top["y"] + 10})
                pg.set_viewport_size({"width": 1600, "height": 1100})
                app.settle(2500)
            else:
                tabs.screenshot(path=OUT / f"{name}.png")
                trim_bottom(OUT / f"{name}.png")
            print("saved", name)

        # 11. Buildings 3D (only once scripts/prepare_buildings.py has run); WebGL needs extra render time
        pg.get_by_role("tab").filter(has_text="Buildings 3D").click()
        pg.wait_for_timeout(2000)
        deck = pg.locator('[data-testid="stDeckGlJsonChart"]')
        if deck.count():
            def shot_deck(name):
                # Streamlit scrolls inside its own container, so scroll by script and crop the page shot.
                deck.first.evaluate("e => e.scrollIntoView({block: 'center'})")
                pg.wait_for_timeout(2000)
                pg.screenshot(path=OUT / f"{name}.png", clip=deck.first.bounding_box())
                print("saved", name)
            app.settle(10000)
            shot_deck("25_buildings_district")
            pg.locator("label").filter(has_text="Whole city").first.click()
            app.settle(10000)
            shot_deck("26_buildings_city")
            tabs.screenshot(path=OUT / "27_tab_buildings.png")
            trim_bottom(OUT / "27_tab_buildings.png")
            im = Image.open(OUT / "27_tab_buildings.png")  # the tab row is scrolled half out of view; drop it
            im.crop((0, 70, im.width, im.height)).save(OUT / "27_tab_buildings.png")
            print("saved 27_tab_buildings")
        browser.close()


if __name__ == "__main__":
    main()
