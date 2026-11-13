"""Download the free-licensed Pakistani house photos listed in data/pk_photos.json.

Photos come from Wikimedia Commons (CC BY, CC BY-SA or public domain). Each one is saved once
to realestate_agent/static/homes/pk/<slug>.jpg (at most 1280 px wide) plus a 640 px wide
<slug>-sm.jpg thumbnail. Resizing is the only change. Files that already exist are skipped.

Run: python scripts/fetch_photos.py
"""
import io
import json
import time
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "pk_photos.json"
OUT = ROOT / "realestate_agent" / "static" / "homes" / "pk"
HEADERS = {"User-Agent": "DeedlyStudentProject/1.0 (university coursework)"}


def save(img: Image.Image, path: Path, width: int) -> None:
    img = img.copy()
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    img.save(path, "JPEG", quality=82, optimize=True, progressive=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for photo in json.loads(MANIFEST.read_text(encoding="utf-8")):
        if photo.get("dir"):  # Already in the repo under another folder.
            continue
        big, small = OUT / f"{photo['slug']}.jpg", OUT / f"{photo['slug']}-sm.jpg"
        if big.exists() and small.exists():
            continue
        req = urllib.request.Request(photo["thumb_url"], headers=HEADERS)
        img = Image.open(io.BytesIO(urllib.request.urlopen(req, timeout=60).read())).convert("RGB")
        save(img, big, 1280)
        save(img, small, 640)
        print("saved", photo["slug"], img.size)
        time.sleep(0.5)  # Be polite to Wikimedia.


if __name__ == "__main__":
    main()
