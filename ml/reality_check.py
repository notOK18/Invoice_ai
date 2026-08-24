"""Reality-check: run the existing extractor over the mychen76 sample and report
how often it finds the total, and the resulting label balance.

This tells us BEFORE building the full pipeline whether the data gives usable
signal: if the extractor never finds a total, or every row gets the same label,
there is nothing for a model to learn. Nothing here touches the pipeline.

Usage:  python ml/reality_check.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from invoice_ai.pipeline import InvoiceProcessor  # noqa: E402

DATA = Path(__file__).resolve().parent / "data" / "mychen76_train.jsonl"


def euro_to_float(value):
    """Parse the dataset's European-formatted money ('$8,25' -> 8.25)."""
    if value is None:
        return None
    s = str(value).replace("$", "").replace(" ", "").strip()
    s = s.replace(".", "").replace(",", ".")  # '.' thousands, ',' decimal
    s = re.sub(r"[^\d.]", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def main():
    rows = [json.loads(line) for line in DATA.open(encoding="utf-8")]
    proc = InvoiceProcessor()

    found = correct = wrong = no_total = 0
    examples = []
    for r in rows:
        true_total = euro_to_float(r["summary"].get("total_gross_worth"))
        inv = proc.process_invoice_text(r["text"])
        extracted = euro_to_float(inv.total_amount) if inv.total_amount else None

        if inv.total_amount:
            found += 1
        if extracted is None:
            no_total += 1
            label = 1
        elif true_total is not None and abs(extracted - true_total) <= 0.01:
            correct += 1
            label = 0
        else:
            wrong += 1
            label = 1
        if len(examples) < 10:
            examples.append((r["id"], inv.total_amount, r["summary"].get("total_gross_worth"), label))

    n = len(rows)
    print(f"rows                         : {n}")
    print(f"extractor produced a total   : {found}/{n}  ({100*found//n}%)")
    print(f"  correct  (label 0)         : {correct}")
    print(f"  wrong    (label 1)         : {wrong}")
    print(f"  no total (label 1)         : {no_total}")
    pos = wrong + no_total
    print(f"LABEL BALANCE -> wrong/missing={pos} ({100*pos//n}%)   correct={correct} ({100*correct//n}%)")
    print("\nexamples (id, extracted, true, label):")
    for e in examples:
        print("  ", e)


if __name__ == "__main__":
    main()
