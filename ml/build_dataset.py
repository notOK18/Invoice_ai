"""Build the features+label training table for the total_amount reliability model.

For each invoice: run the existing extractor over the OCR text, compute a small
set of numeric features describing the extracted total, and label it 0 (correct)
if the extracted value matches a legitimate summary total (net or gross), else 1.

Writes ml/data/total_amount.csv and prints, per label, the mean of each feature
so we can see which features actually carry signal before training. Reuses the
pipeline read-only; nothing in it is modified.

Usage:  python ml/build_dataset.py
"""

import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from invoice_ai.pipeline import InvoiceProcessor  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
SPLITS = ["train", "test"]  # build a features CSV for each split that exists

# How "correct" is defined:
#   "strict"    - the extraction must equal the gross total actually billed
#   "plausible" - any legitimate summary total (net or gross) counts
# Set with:  python ml/build_dataset.py [strict|plausible]
LABEL_MODE = "plausible"

# The reliability model exists to catch bad parses, so it is trained on the
# extractor's raw pattern matching (prefer_summary_total=False). With the summary
# heuristic on, extraction is ~91% correct and there are too few failures to learn
# from - see the comparison in the README notes.
USE_SUMMARY_HEURISTIC = False

# Only the two features that carry signal on this data. Measured, not assumed:
#   * a raw digit-count feature conflated "small amount" with "garbage" (a real
#     $8.50 invoice was flagged); dropping it RAISED held-out AUC 0.905 -> 0.940.
#   * has_value / is_numeric / near_keyword / has_currency were ~constant here, so
#     training drove their weights to 0; removing them left every metric identical
#     (AUC 0.940, balanced accuracy 0.929) with a far simpler model.
# Re-test those if the data changes - on noisier scans they could matter again.
FEATURES = [
    "has_decimal",      # value has a 2-digit cents part (garbage tokens usually don't)
    "reconcile_ok",     # line items (if parsed) reconcile with the value
    "summary_gap",      # CONTINUOUS: relative distance to the summary's own total
    "no_summary",       # 1 when there is no summary block to cross-check against
]

# HONEST CAVEAT: on this dataset `summary_gap` alone scores ROC-AUC 1.000 - the
# logistic regression adds nothing on top of it. That is not a great model so
# much as a great cross-check: these invoices always carry a summary whose
# arithmetic pins the true total, so a rule suffices and ML is not needed here.
# Expect the model to earn its keep only on documents where the cross-check is
# unavailable or noisy (no summary block, bad OCR) - which is what `no_summary`
# is there to flag.


def euro_to_float(value):
    """Parse the dataset's European money format ('$8,25' -> 8.25)."""
    if value is None:
        return None
    s = str(value).replace("$", "").replace(" ", "").strip().replace(".", "").replace(",", ".")
    s = re.sub(r"[^\d.]", "", s)
    try:
        return float(s)
    except ValueError:
        return None


_KEYWORDS = ("total", "gross", "net", "amount", "summary", "balance", "due")


def near_keyword(text, value):
    """1 if the extracted value appears on a line near a total-ish keyword."""
    if not value:
        return 0
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if value in line:
            window = " ".join(lines[max(0, i - 2): i + 1]).lower()
            if any(k in window for k in _KEYWORDS):
                return 1
    return 0


def summary_gap(proc, text, raw):
    """How far the extracted total sits from the summary block's own gross total.

    An invoice's summary states net + VAT = gross, so the largest value there is
    the amount actually billed. Comparing the extraction against it is an
    independent cross-check computed purely from the document - when the two
    disagree, the extraction is suspect. Returns (gap, no_summary):
      gap        - relative distance, 0.0 (agrees) .. 1.0 (completely different)
      no_summary - 1 when there is nothing to cross-check against
    """
    # Parse the extraction with the SAME dual-format parser the cross-check uses,
    # or '550.00' (US) and '889,20' (euro) would not be comparable.
    value = proc.money_value(raw) if raw else None
    # The invoice's own arithmetic (subtotal + tax = total) confirms the amount
    # billed, whatever the wording - so this works on ordinary invoices too, not
    # only the "SUMMARY / Net worth / Gross worth" layout of the training set.
    gross = proc.verified_total(text)
    if gross is None:
        return 1.0, 1          # nothing to check against: maximally uncertain
    if value is None:
        return 1.0, 0
    return min(abs(value - gross) / gross, 1.0), 0


def compute_features(proc, text, inv):
    raw = inv.total_amount
    gap, no_summary = summary_gap(proc, text, raw)
    return {
        "has_decimal": int(bool(raw) and bool(re.search(r"[.,]\d{2}(?!\d)", str(raw)))),
        "reconcile_ok": int(bool(inv.line_items) and proc._is_valid_number(raw)
                            and proc._totals_reconcile(inv)),
        "summary_gap": round(gap, 4),
        "no_summary": no_summary,
    }


def build_split(split, proc):
    in_path = DATA_DIR / f"mychen76_{split}.jsonl"
    if not in_path.exists():
        return None
    out_path = DATA_DIR / f"total_amount_{split}.csv"
    rows = [json.loads(line) for line in in_path.open(encoding="utf-8")]

    table = []
    skipped = 0
    for r in rows:
        gross = euro_to_float(r["summary"].get("total_gross_worth"))
        # ~79% of this dataset are receipts with an EMPTY annotation. Without a
        # true total there is no answer key, so they cannot be labelled - keeping
        # them would silently mark every one "wrong" and poison the training set.
        if gross is None:
            skipped += 1
            continue

        net = euro_to_float(r["summary"].get("total_net_worth"))
        inv = proc.process_invoice_text(r["text"])
        ext = euro_to_float(inv.total_amount) if inv.total_amount else None

        def eq(a, b):
            return a is not None and b is not None and abs(a - b) <= 0.02

        if LABEL_MODE == "strict":
            # Must equal the gross total actually billed.
            label = 0 if eq(ext, gross) else 1
        else:
            # "plausible": any legitimate summary total (net or gross) counts, so
            # the model learns to spot garbage parses rather than column choice.
            label = 0 if (eq(ext, gross) or eq(ext, net)) else 1
        feats = compute_features(proc, r["text"], inv)
        feats["label"] = label
        table.append(feats)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURES + ["label"])
        writer.writeheader()
        writer.writerows(table)

    n = len(table)
    pos = sum(r["label"] for r in table)
    print(f"[{split}] wrote {n} rows -> {out_path.name}  (skipped {skipped} with no ground truth)")
    print(f"[{split}] label balance: wrong(1)={pos} ({100*pos//n}%)  correct(0)={n-pos} ({100*(n-pos)//n}%)")
    if split == "train":
        print(f"\n{'feature':14s} {'mean|correct':>13s} {'mean|wrong':>12s}  (gap = signal)")
        for feat in FEATURES:
            c = [r[feat] for r in table if r["label"] == 0]
            w = [r[feat] for r in table if r["label"] == 1]
            mc, mw = sum(c) / len(c), sum(w) / max(len(w), 1)
            flag = "  <-- separates" if abs(mc - mw) >= 0.25 else ""
            print(f"{feat:14s} {mc:13.2f} {mw:12.2f}{flag}")
        print()
    return out_path


def main():
    global LABEL_MODE
    if len(sys.argv) > 1 and sys.argv[1] in ("strict", "plausible"):
        LABEL_MODE = sys.argv[1]
    print(f"label mode: {LABEL_MODE}\n")
    proc = InvoiceProcessor(prefer_summary_total=USE_SUMMARY_HEURISTIC)
    for split in SPLITS:
        build_split(split, proc)


if __name__ == "__main__":
    main()
