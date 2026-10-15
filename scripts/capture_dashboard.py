"""Capture full-page screenshots of every dashboard page into docs/dashboard/ (used by the report).

    streamlit run dashboard/app.py        # in another terminal
    python scripts/capture_dashboard.py
"""
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PAGES = [("", "1_overview"), ("content", "2_content"), ("audience", "3_audience"),
         ("recommendations", "4_recommendations"), ("model-quality", "5_model_quality"), ("data-pipeline", "6_data_pipeline")]
OUT = Path(__file__).resolve().parent.parent / "docs" / "dashboard"
URL = "http://localhost:8501"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")  # uses the installed Chrome, no browser download
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1.5)
        for path, name in PAGES:
            page.goto(f"{URL}/{path}")
            page.wait_for_selector(".pt", timeout=60000)
            page.wait_for_function("!document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=120000)
            time.sleep(3)  # let Plotly and data grids finish drawing
            height = page.evaluate("document.querySelector('[data-testid=\"stMainBlockContainer\"]').scrollHeight")
            page.set_viewport_size({"width": 1440, "height": int(height) + 80})
            time.sleep(2)
            page.screenshot(path=str(OUT / f"{name}.png"))
            page.set_viewport_size({"width": 1440, "height": 1000})
            print("captured", name, height)
        browser.close()


if __name__ == "__main__":
    main()
