"""Test the trained total_amount reliability model on real inputs.

Two modes:
  * no args      -> run over the held-out test split and print a table of
                    predictions vs. the true answer (build intuition, see misses)
  * <file.txt>   -> run on one invoice's text file and print the confidence

The model outputs P(wrong); the reported confidence is 1 - P(wrong). Reuses the
same feature computation as training, and the extractor read-only.

Usage:  python ml/predict.py               # demo on the test split
        python ml/predict.py invoice.txt   # score one invoice
"""

import json
import sys
from pathlib import Path

import joblib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "src"))
sys.path.insert(0, str(HERE))
from invoice_ai.pipeline import InvoiceProcessor          # noqa: E402
from build_dataset import (USE_SUMMARY_HEURISTIC, compute_features,  # noqa: E402
                           euro_to_float)

BUNDLE = joblib.load(HERE / "models" / "reliability_total.joblib")
MODEL, FEATURES = BUNDLE["model"], BUNDLE["features"]
# Must match the configuration the model was trained on.
PROC = InvoiceProcessor(prefer_summary_total=USE_SUMMARY_HEURISTIC)


def score_text(text):
    """Return (invoice, features, confidence) for one invoice's OCR text."""
    inv = PROC.process_invoice_text(text)
    feats = compute_features(PROC, text, inv)
    x = [[float(feats[f]) for f in FEATURES]]
    p_wrong = MODEL.predict_proba(x)[0][1]
    return inv, feats, 1.0 - p_wrong


def demo_test_split(limit=20):
    path = HERE / "data" / "mychen76_test.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    print(f"{'extracted':>12} {'true(gross)':>12} {'conf':>6} {'model':>8} {'actual':>8}  ok?")
    print("-" * 60)
    hits = 0
    shown = 0
    shown_total = 0
    for r in rows:
        gross = euro_to_float(r["summary"].get("total_gross_worth"))
        if gross is None:
            continue  # receipts with no annotation: no answer key to score against
        net = euro_to_float(r["summary"].get("total_net_worth"))
        inv, _, conf = score_text(r["text"])
        ext = euro_to_float(inv.total_amount) if inv.total_amount else None
        actual_ok = (ext is not None and gross is not None and abs(ext - gross) <= 0.02) or \
                    (ext is not None and net is not None and abs(ext - net) <= 0.02)
        model_says_ok = conf >= 0.5
        agree = model_says_ok == actual_ok
        shown_total += 1
        hits += agree
        if shown < limit:
            shown += 1
            print(f"{str(inv.total_amount):>12} {str(gross):>12} {conf:6.2f} "
                  f"{'OK' if model_says_ok else 'FLAG':>8} {'OK' if actual_ok else 'BAD':>8}  "
                  f"{'✓' if agree else '✗'}")
    n = shown_total
    print("-" * 60)
    print(f"model agreed with reality on {hits}/{n} ({100*hits//n}%) of the test invoices")


def score_file(file_path):
    text = Path(file_path).read_text(encoding="utf-8")
    inv, feats, conf = score_text(text)
    print(f"file           : {file_path}")
    print(f"extracted total: {inv.total_amount}")
    print(f"features       : {feats}")
    print(f"confidence     : {conf:.3f}   ->  {'TRUST' if conf >= 0.5 else 'REVIEW (likely wrong)'}")


def main():
    if len(sys.argv) > 1:
        score_file(sys.argv[1])
    else:
        demo_test_split()


if __name__ == "__main__":
    main()
