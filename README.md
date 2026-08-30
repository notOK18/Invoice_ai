# Invoice AI

Extracts the key fields from an invoice, scores how much to trust each one, and
learns from what reviewers correct.

Fields at or above **90% confidence** are auto-approved; anything below is
flagged for review. Every review — whether you changed a field or left it
alone — becomes a training example, so the scores improve as you use it.

## Run it

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python review_ui.py
```

The page opens empty. Upload an invoice (JPG/PNG/TXT) and it is OCR'd,
extracted, scored per field and shown next to the document. Correct anything
wrong and click **Save & Learn**.

```bash
python review_ui.py            # start empty, upload from the page
python review_ui.py --folder   # also load unreviewed files in data/invoices
python review_ui.py --all      # load every file in data/invoices
```

## How it works

```
invoice ─► OCR ─► extract fields ─► score each field ─► ≥90%  auto-approve
 (easyocr)        (patterns)        (heuristic or           <90%  review
                                     learned model)                 │
                                                                    ▼
                                              reviewer accepts or edits
                                                                    │
                    ┌───────────────────────────────────────────────┤
                    ▼                                               ▼
        label learned from the correction            accepted/edited stored
        (so a new layout is handled next time)       (retrains the scorer)
```

Three things are learned from a correction:

| what | effect |
|---|---|
| the corrected **value** | reapplied to that invoice, matched by content fingerprint |
| the **label** beside it | a new wording (e.g. `Gesamtbetrag`) joins the patterns after 3 sightings |
| **accepted vs edited** | trains the per-field confidence model |

A field uses the hand-written heuristic until it has 20+ reviews with at least
5 of each outcome, and only switches to a learned model if that model beats 0.65
balanced accuracy on held-out folds. The UI shows which is in use per field.

## Models and technology

This project uses two per-field logistic-regression classifiers (scikit-learn)
stored in `data/models/` — `invoice_date.joblib` and `supplier.joblib` — each
predicting the probability that an extracted field will need correcting, trained
on accept/edit outcomes recorded in `data/invoice_ai.db`, while
`invoice_number`, `total_amount` and `currency` still fall back to hand-written
heuristic scores for lack of enough failure examples, and the OCR itself is
EasyOCR's pretrained CRAFT + CRNN neural network (downloaded to `~/.EasyOCR/`)
rather than anything trained here.

The stack is Python 3.13 with EasyOCR/PyTorch for text recognition, OpenCV and
Pillow for image preprocessing, scikit-learn for the confidence models, SQLite
for the learning store, regular expressions for field extraction, and a
standard-library HTTP server with plain HTML/JavaScript for the review UI.

## Layout

| path | role |
|---|---|
| `src/invoice_ai/pipeline.py` | OCR, extraction patterns, heuristic scoring |
| `src/invoice_ai/preprocess.py` | image cleanup before OCR (also a standalone tool) |
| `src/invoice_ai/confidence.py` | per-field features, learned scoring, retraining |
| `src/invoice_ai/patterns.py` | inferring labels from corrections |
| `src/invoice_ai/db.py` | SQLite store |
| `review_ui.py` | the review app |
| `ml/` | dataset fetch, seeding, offline training and a demo page |
| `data/invoice_ai.db` | invoices, extractions, reviews, models, learned labels |
| `data/models/` | trained model per field (belongs with the database) |

`LEARNING.md` covers the learning loop in more detail.

## Image quality

Resolution is the single biggest factor in whether extraction works. Measured
across the sample invoices:

| source image | OCR confidence | fields extracted |
|---|---|---|
| 640x640 (0.4 MP) | 0.68 | 2.0 / 3 |
| 1432x2048 (2.9 MP) | 0.86 | **3.0 / 3** |

Scanning at 300 DPI, or photographing with a phone, comfortably clears the
second row. Preprocessing raises the confidence of a small scan but cannot
recover detail that was never captured, so it is no substitute.

The cleanup used before OCR is also a standalone tool:

```bash
export PYTHONPATH=src
python -m invoice_ai.preprocess report  invoice.jpg   # size, skew, contrast
python -m invoice_ai.preprocess clean   invoice.jpg out.png
python -m invoice_ai.preprocess compare invoice.jpg   # OCR score per rendering
```

`compare` is the useful one when a page reads badly: it OCRs the original and
each cleaned rendering and prints the confidence of each, so the choice is
measured rather than guessed.

## Notes

* **The database and `data/models/` are a pair.** Back them up or delete them
  together; a model without its history cannot be explained or retrained.
* To start clean: delete both, then `python ml/seed_from_dataset.py` to
  bootstrap from the annotated dataset (fetch it first with
  `python ml/fetch_mychen76.py`).
* Handling **real** invoices means the database will hold real IBANs and tax
  ids. Use a private repository for that.
* Tests: `python -m pytest`
