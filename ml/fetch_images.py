"""Download the invoice images for the rows we can score, into ml/data/images/.

The HuggingFace viewer serves images from signed URLs that expire, so they are
copied locally once - the demo then works offline (and during a presentation
without wifi). Only rows that have a ground-truth summary are fetched, since
those are the only ones the demo shows.

Usage:  python ml/fetch_images.py [split]      (default: test)
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

DATASET = "mychen76/invoices-and-receipts_ocr_v1"
API = "https://datasets-server.huggingface.co/rows"
OUT_DIR = Path(__file__).resolve().parent / "data" / "images"


def fetch(split="test", batch=100, max_rows=400):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    for offset in range(0, max_rows, batch):
        url = f"{API}?dataset={DATASET}&config=default&split={split}&offset={offset}&length={batch}"
        result = subprocess.run(["curl", "-s", url], capture_output=True, text=True, timeout=180)
        rows = json.loads(result.stdout or "{}").get("rows", [])
        if not rows:
            break
        for item in rows:
            row = item["row"]
            # only rows with an answer key are ever shown in the demo
            try:
                parsed = json.loads(row["parsed_data"])["json"]
                # the annotation is stored as a dict OR its string repr
                summary = ast.literal_eval(parsed) if isinstance(parsed, str) else parsed
            except Exception:
                continue
            if not (summary.get("summary") or {}).get("total_gross_worth"):
                continue
            src = (row.get("image") or {}).get("src")
            if not src:
                continue
            dest = OUT_DIR / f"{row['id']}.jpg"
            if dest.exists():
                saved += 1
                continue
            subprocess.run(["curl", "-s", "-L", "-o", str(dest), src], timeout=180)
            if dest.exists() and dest.stat().st_size > 0:
                saved += 1
        print(f"  through offset {offset}: {saved} images")
    print(f"saved {saved} images -> {OUT_DIR}")


if __name__ == "__main__":
    fetch(sys.argv[1] if len(sys.argv) > 1 else "test")
