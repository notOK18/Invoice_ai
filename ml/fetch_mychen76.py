"""Fetch a sample of the mychen76 invoices dataset into ml/data/ as JSONL.

Pulls rows from the HuggingFace datasets-server REST API (no `datasets` library
needed) in batches of 100, and writes one compact JSON object per line with just
the two things we need per invoice: the OCR text and the ground-truth summary.

Usage:  python ml/fetch_mychen76.py [n_rows] [split]
        (defaults: 300 rows from the 'train' split)
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

DATASET = "mychen76/invoices-and-receipts_ocr_v1"
API = "https://datasets-server.huggingface.co/rows"
OUT_DIR = Path(__file__).resolve().parent / "data"


def _fetch_batch(split, offset, length):
    """Return the raw rows list for one API page, via curl (handles TLS certs)."""
    url = f"{API}?dataset={DATASET}&config=default&split={split}&offset={offset}&length={length}"
    result = subprocess.run(["curl", "-s", url], capture_output=True, text=True, timeout=120)
    return json.loads(result.stdout).get("rows", [])


def _ocr_text(raw_data):
    """Decode the OCR word list stored (double-encoded) in raw_data -> plain text."""
    words = ast.literal_eval(json.loads(raw_data)["ocr_words"])
    return "\n".join(words)


def _parsed_json(parsed_data):
    """The structured annotation lives under parsed_data['json'] (a dict or its repr)."""
    obj = json.loads(parsed_data)["json"]
    return ast.literal_eval(obj) if isinstance(obj, str) else obj


def fetch(n_rows=300, split="train"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"mychen76_{split}.jsonl"
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for offset in range(0, n_rows, 100):
            batch = _fetch_batch(split, offset, min(100, n_rows - offset))
            if not batch:
                break
            for item in batch:
                row = item["row"]
                try:
                    parsed = _parsed_json(row["parsed_data"])
                    record = {
                        "id": row.get("id"),
                        "text": _ocr_text(row["raw_data"]),
                        "summary": parsed.get("summary", {}),
                        "header": parsed.get("header", {}),
                        "items": parsed.get("items", []),
                    }
                except Exception as exc:
                    print(f"  skip row {row.get('id')}: {exc}")
                    continue
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            print(f"  fetched through offset {offset} ({written} rows)")
    print(f"wrote {written} rows -> {out_path}")
    return out_path


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    sp = sys.argv[2] if len(sys.argv) > 2 else "train"
    fetch(n, sp)
