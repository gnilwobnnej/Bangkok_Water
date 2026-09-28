"""Render docs/presenter_guide.html to a PDF with headless Chrome (via Playwright).

    python scripts/build_guide_pdf.py

Screenshots are embedded as compressed JPEG copies to keep the PDF small; the PNGs are untouched.
"""
import re
import sys
import tempfile
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

DOCS = Path(__file__).resolve().parent.parent / "docs"
SRC = DOCS / "presenter_guide.html"
# Optional first argument: another output path (e.g. when the PDF is open in a viewer and locked)
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else DOCS / "Bangkok_Flood_Simulator_Presenter_Guide.pdf"
MAX_WIDTH_PX = 1800  # plenty for a full-width A4 image

FOOTER = (
    '<div style="width:100%;font-size:8px;color:#8a94a3;padding:0 15mm;display:flex;'
    'justify-content:space-between;font-family:Segoe UI,Arial,sans-serif">'
    '<span>Bangkok Flood Simulator: Presenter Guide</span>'
    '<span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>'
)


def compressed_copy(html: str, tmp: Path) -> Path:
    def to_jpeg(match):
        png = DOCS / match.group(1)
        im = Image.open(png).convert("RGB")
        if im.width > MAX_WIDTH_PX:
            im = im.resize((MAX_WIDTH_PX, round(im.height * MAX_WIDTH_PX / im.width)), Image.LANCZOS)
        jpg = tmp / (png.stem + ".jpg")
        im.save(jpg, quality=85, optimize=True)
        return f'src="{jpg.name}"'

    out = tmp / SRC.name
    out.write_text(re.sub(r'src="(screenshots/[^"]+\.png)"', to_jpeg, html), encoding="utf-8")
    return out


with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
    page_file = compressed_copy(SRC.read_text(encoding="utf-8"), Path(tmp))
    browser = p.chromium.launch(channel="chrome")
    page = browser.new_page()
    page.goto(page_file.as_uri(), wait_until="networkidle")
    page.pdf(
        path=str(OUT), format="A4", print_background=True, prefer_css_page_size=True,
        display_header_footer=True, header_template="<div></div>", footer_template=FOOTER,
    )
    browser.close()
print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
