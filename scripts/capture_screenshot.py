"""Capture a genuine screenshot of the running app (no mockups).

Usage: python scripts/capture_screenshot.py [--url http://127.0.0.1:5057] [--train 11014] [--out PATH]
If a live provider is configured the live source is used; otherwise the recorded-replay
source is used and the image is labelled as NOT LIVE in the README.
"""
import argparse
import json
import sys
import urllib.request

from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:5057")
ap.add_argument("--train", default="")
ap.add_argument("--out", default="docs/screenshots/railmind-live-dashboard.png")
a = ap.parse_args()

health = json.load(urllib.request.urlopen(a.url + "/api/health"))
live = health["provider"]["configured"]
with sync_playwright() as p:
    for ch in ("msedge", "chrome", None):
        try:
            b = p.chromium.launch(channel=ch) if ch else p.chromium.launch()
            break
        except Exception as e:  # noqa: BLE001
            err = e
    else:
        sys.exit(f"no browser available: {err}")
    pg = b.new_page(viewport={"width": 1700, "height": 1250})
    pg.goto(a.url)
    pg.wait_for_selector("#health")
    pg.wait_for_timeout(1500)
    pg.select_option("#source", "live" if live else "replay")
    if a.train:
        pg.fill("#trainNo", a.train)
    pg.click("#go")
    pg.wait_for_selector("#compare table", timeout=120000)
    pg.wait_for_timeout(500)
    pg.evaluate("state.t = Math.round(state.tmax * 0.6); draw();")
    pg.wait_for_timeout(1000)
    pg.screenshot(path=a.out, full_page=True)
    b.close()
print("live provider configured:", live, "->", a.out)
