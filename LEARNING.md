# Learning confidence loop

Extract fields → score each one → auto-approve at 90% or flag for review →
learn from what the reviewer does → score the next invoice better.

## Run it

```bash
.venv/bin/python review_ui.py
```

Put invoices in `data/invoices/`. Each field shows its own confidence:
green = auto-approved (≥ 90%), amber = flagged. Click **Save & Learn** on every
invoice — *including ones you did not change*, because "accepted unchanged" is
half the training signal.

## How it works

```
invoice ──► extractor ──► per-field features ──► confidence
                                                    │
                                        ≥ 0.90 ─────┴───── < 0.90
                                    auto-approve         review
                                                            │
                                          reviewer accepts or edits
                                                            │
                                      logged to SQLite as a label
                                        accepted → 0   edited → 1
                                                            │
                                            retrain that field's model
                                                            │
                                        next invoice scored better
```

## The pieces

| File | Role |
|---|---|
| `src/invoice_ai/db.py` | SQLite store: invoices, extractions, reviews, model versions |
| `src/invoice_ai/confidence.py` | Per-field features, scoring, retraining |
| `review_ui.py` | Review screen; logs every accept/edit and retrains on save |
| `data/invoice_ai.db` | The database |
| `data/models/<field>.joblib` | The trained model for each field |

## Cold start

A field is scored by the pipeline's hand-tuned heuristic until it has
**20 reviews with at least 3 of each outcome** (`MIN_SAMPLES`, `MIN_PER_CLASS`
in `confidence.py`). Before that there is not enough evidence to separate good
extractions from bad ones, and a model fitted on a handful of rows would be
worse than the heuristic. Each field crosses that line independently — the UI
shows `heuristic` or `model vN` next to every score.

Fields a human has already corrected are marked `human-verified` at 1.0 and are
never sent back for review.

## Inspecting what it has learned

```python
from invoice_ai.db import Database
from invoice_ai.pipeline import InvoiceProcessor
from invoice_ai.confidence import ConfidenceScorer

db = Database()
sc = ConfidenceScorer(db, InvoiceProcessor())
print(db.stats())          # row counts
print(db.review_counts())  # accepted vs edited
print(sc.status())         # per field: scored_by, reviews, edited
```

## Tuning

* `AUTO_APPROVE_AT` (`confidence.py`) — the 0.90 bar. Raise it to review more,
  lower it to review less.
* `MIN_SAMPLES` / `MIN_PER_CLASS` — how much evidence before a model replaces
  the heuristic.
* `FEATURE_NAMES` — the features each field is judged on. `total_amount` is the
  strongest because an invoice's own arithmetic (subtotal + tax = total) can
  verify it; the other fields have no such cross-check, which is exactly why a
  learned score matters more for them.

## Honest limits

* Training accuracy is reported on the rows the model was fitted to, not a
  held-out split. With few reviews it will look better than it is.
* Retraining on every save means early models swing with each new review. That
  settles as reviews accumulate.
* The model learns *your reviewers' behaviour*. If they rubber-stamp bad
  extractions, it learns to trust them too.
