"""Bootstrap the confidence models from the annotated invoice dataset.

Runs the extractor over every invoice that has ground-truth annotations, compares
each field against the truth, and writes the result into the database as if a
reviewer had looked at it:

    extraction matches the annotation  ->  "accepted"
    it does not                        ->  "edited"

That gives the per-field models a starting point instead of waiting for the first
20 real reviews. These are SIMULATED reviews derived from annotations, not real
reviewer behaviour - they are marked with a `dataset:` source so they can be told
apart, and real reviews should be expected to shift the models afterwards.

Usage:
    python ml/seed_from_dataset.py            # seed + retrain
    python ml/seed_from_dataset.py --dry-run  # report only, write nothing
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ml"))

from invoice_ai.confidence import ConfidenceScorer  # noqa: E402
from invoice_ai.db import Database  # noqa: E402
from invoice_ai.pipeline import InvoiceProcessor  # noqa: E402
from build_dataset import euro_to_float  # noqa: E402

DATA = ROOT / "ml" / "data" / "mychen76_train.jsonl"
# Only fields the dataset actually annotates; currency has no ground truth here.
SEEDED_FIELDS = ("invoice_number", "invoice_date", "total_amount", "supplier")


def _iso(value):
    """The annotation writes MM/DD/YYYY; the extractor normalises to ISO."""
    match = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", str(value or ""))
    if not match:
        return None
    return f"{match.group(3)}-{int(match.group(1)):02d}-{int(match.group(2)):02d}"


def truth_for(row):
    header = row.get("header", {})
    return {
        "invoice_number": header.get("invoice_no"),
        "invoice_date": _iso(header.get("invoice_date")),
        "total_amount": euro_to_float(row.get("summary", {}).get("total_gross_worth")),
        "supplier": header.get("seller"),
    }


def matches(proc, field, extracted, truth):
    """Did the extractor get this field right, by the annotation's reckoning?"""
    if extracted is None or truth is None:
        return False
    if field == "total_amount":
        value = proc.money_value(extracted)
        return value is not None and abs(value - truth) <= 0.02
    if field == "supplier":
        # The annotation appends the address to the company name; compare on the
        # leading part so an address the extractor never saw is not counted wrong.
        return str(truth).lower().startswith(str(extracted).strip().lower()[:18])
    return str(extracted).strip().lower() == str(truth).strip().lower()


def seed(dry_run=False):
    rows = [json.loads(line) for line in DATA.open(encoding="utf-8")]
    labelled = [r for r in rows if r.get("summary", {}).get("total_gross_worth")]
    print(f"{len(labelled)} annotated invoices of {len(rows)}\n")

    proc = InvoiceProcessor()
    db = Database()
    scorer = ConfidenceScorer(db, proc)
    tally = {f: {"accepted": 0, "edited": 0} for f in SEEDED_FIELDS}

    for row in labelled:
        text = row["text"]
        invoice = proc.process_invoice_text(text)
        truth = truth_for(row)
        invoice_id = None if dry_run else db.add_invoice(f"dataset:{row['id']}", text)

        for field in SEEDED_FIELDS:
            extracted = getattr(invoice, field)
            action = "accepted" if matches(proc, field, extracted, truth[field]) else "edited"
            tally[field][action] += 1
            if dry_run:
                continue
            feats = scorer.features(field, invoice, text)
            conf, version = scorer.score(field, feats,
                                         heuristic=invoice.field_confidence.get(field))
            extraction_id = db.add_extraction(invoice_id, field, extracted, conf,
                                              feats, model_version=version)
            db.record_review(extraction_id, action, extracted,
                             extracted if action == "accepted" else truth[field])

    print(f"{'field':16s} {'accepted':>9} {'edited':>8}   trainable?")
    for field, counts in tally.items():
        ok = min(counts["accepted"], counts["edited"]) >= 5 and sum(counts.values()) >= 20
        print(f"{field:16s} {counts['accepted']:9d} {counts['edited']:8d}   "
              f"{'yes' if ok else 'NO - one class too small'}")

    if dry_run:
        print("\n(dry run - nothing written)")
        return

    print("\nretraining…")
    for result in scorer.retrain_all():
        if result.get("trained"):
            print(f"  {result['field']:16s} v{result['version']}  "
                  f"{result['n_samples']} reviews, cv balanced-acc {result['quality']}")
        else:
            print(f"  {result['field']:16s} skipped - {result['reason']}")
    print(f"\ndatabase: {db.stats()}")


if __name__ == "__main__":
    seed(dry_run="--dry-run" in sys.argv)
