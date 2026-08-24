"""Per-field confidence scoring that learns from review outcomes.

For each extracted field this computes a small set of numeric features, then
scores it with a logistic regression trained on what reviewers actually did:

    accepted unchanged -> label 0 (the extraction was right)
    edited             -> label 1 (the extraction was wrong)

Confidence is 1 - P(wrong). Fields at or above AUTO_APPROVE_AT need no review.

Until a field has enough reviewed examples to train on, its score comes from the
pipeline's existing hand-tuned heuristic - so the system is useful from the first
invoice and improves as reviews accumulate, rather than being useless until some
training set exists.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .db import Database
from .pipeline import CORRECTABLE_FIELDS, InvoiceData, InvoiceProcessor

# A field is auto-approved at or above this confidence; below it goes to review.
AUTO_APPROVE_AT = 0.90

# A learned model replaces the heuristic only once there is enough evidence:
# enough reviews overall, and enough of BOTH outcomes to separate them.
MIN_SAMPLES = 20
MIN_PER_CLASS = 5

# ...and only if it demonstrably works. Balanced accuracy is 0.5 for a model that
# ignores the input, so a model must clear this on held-out folds to be adopted.
# Without this gate a handful of minority examples produces a model that flags
# almost everything - measurably worse than the heuristic it would replace.
MIN_QUALITY = 0.65


# The feature each field is described by. Keeping them per-field (rather than one
# shared vector) lets each use the evidence that actually exists for it.
FEATURE_NAMES: Dict[str, List[str]] = {
    "invoice_number": ["has_value", "plausible_format", "n_chars", "has_digit",
                       "has_letter", "ocr_conf", "ocr_available"],
    "invoice_date":   ["has_value", "is_iso", "year_plausible", "ocr_conf", "ocr_available"],
    "total_amount":   ["has_value", "is_numeric", "has_decimal", "verified_gap",
                       "unverifiable", "reconcile_ok", "ocr_conf", "ocr_available"],
    # n_chars is a crude proxy (a bare label like "Client:" is short, a company
    # name is longer) and was the ONLY feature carrying signal here. The two
    # below describe what actually distinguishes them, so the model has real
    # evidence instead of length alone.
    "supplier":       ["has_value", "n_chars", "is_generic", "known_supplier",
                       "looks_like_label", "has_company_suffix",
                       "ocr_conf", "ocr_available"],
    "currency":       ["has_value", "ocr_conf", "ocr_available"],
}


class ConfidenceScorer:
    """Computes features, scores fields, and retrains from the review history."""

    def __init__(self, db: Database, processor: InvoiceProcessor,
                 model_dir: Optional[Path] = None):
        self.db = db
        self.proc = processor
        # Models live beside the database they were trained from, so pointing at
        # a different database (a test, a second dataset) cannot pick up models
        # trained from another one.
        self.model_dir = Path(model_dir) if model_dir else db.path.parent / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, Any] = {}
        for field in CORRECTABLE_FIELDS:
            self._load(field)

    # ------------------------------------------------------------------
    # features
    # ------------------------------------------------------------------
    def features(self, field: str, invoice: InvoiceData, text: str,
                 ocr_confidence: Optional[float] = None) -> Dict[str, float]:
        """Numeric description of one extracted field, for scoring or training."""
        if ocr_confidence is None:
            ocr_confidence = getattr(invoice, "ocr_confidence", None)
        raw = getattr(invoice, field, None)
        value = "" if raw is None else str(raw)
        common = {
            "has_value": float(bool(value.strip())),
            # 0 with the companion flag at 0 means "no OCR involved", which is a
            # different thing from "OCR was very unsure" - the flag disambiguates.
            "ocr_conf": float(ocr_confidence or 0.0),
            "ocr_available": float(ocr_confidence is not None),
        }

        if field == "invoice_number":
            common.update({
                "plausible_format": float(bool(re.match(r"^[A-Za-z0-9][A-Za-z0-9\-_/]{1,}$", value))),
                "n_chars": float(min(len(value), 30)),
                "has_digit": float(bool(re.search(r"\d", value))),
                "has_letter": float(bool(re.search(r"[A-Za-z]", value))),
            })
        elif field == "invoice_date":
            year = re.match(r"^(\d{4})-\d{2}-\d{2}$", value)
            common.update({
                "is_iso": float(bool(year)),
                "year_plausible": float(bool(year) and 1990 <= int(year.group(1)) <= 2100),
            })
        elif field == "total_amount":
            # The invoice's own arithmetic (subtotal + tax = total) is the
            # strongest evidence available for this field.
            verified = self.proc.verified_total(text)
            number = self.proc.money_value(value)
            if verified is None or number is None:
                gap, unverifiable = 1.0, float(verified is None)
            else:
                gap, unverifiable = min(abs(number - verified) / verified, 1.0), 0.0
            common.update({
                "is_numeric": float(number is not None),
                "has_decimal": float(bool(re.search(r"[.,]\d{2}(?!\d)", value))),
                "verified_gap": gap,
                "unverifiable": unverifiable,
                "reconcile_ok": float(bool(invoice.line_items)
                                      and self.proc._is_valid_number(raw)
                                      and self.proc._totals_reconcile(invoice)),
            })
        elif field == "supplier":
            stripped = value.strip()
            common.update({
                "n_chars": float(min(len(stripped), 60)),
                "is_generic": float(self.proc._is_generic_supplier(stripped)),
                "known_supplier": float(stripped in self.proc.known_suppliers),
                # A column header caught instead of a company name: one or two
                # words ending in a colon ("Client:", "Bill To:").
                "looks_like_label": float(bool(re.fullmatch(
                    r"[A-Za-z][A-Za-z ]{0,18}[:;]?", stripped)) and stripped.endswith(":")),
                # Legal-entity endings are strong evidence of a real company.
                "has_company_suffix": float(bool(re.search(
                    r"\b(?:ltd|llc|plc|inc|gmbh|ag|sa|bv|nv|co|corp|company|group|"
                    r"and sons|partners)\b\.?$", stripped, re.IGNORECASE))),
            })
        return {name: common.get(name, 0.0) for name in FEATURE_NAMES[field]}

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------
    def score(self, field: str, features: Dict[str, float],
              heuristic: Optional[float] = None) -> Tuple[float, Optional[int]]:
        """Confidence for one field, plus the model version used (None = heuristic)."""
        model = self._models.get(field)
        if model is None:
            return float(heuristic if heuristic is not None else 0.5), None
        x = [[features[name] for name in FEATURE_NAMES[field]]]
        p_wrong = float(model["pipeline"].predict_proba(x)[0][1])
        return round(1.0 - p_wrong, 3), model["version"]

    @staticmethod
    def needs_review(confidence: float) -> bool:
        return confidence < AUTO_APPROVE_AT

    # ------------------------------------------------------------------
    # learning
    # ------------------------------------------------------------------
    def retrain(self, field: str) -> Dict[str, Any]:
        """Retrain one field from every review logged so far.

        Returns a short status dict. Training is skipped (leaving the heuristic in
        place) until there are enough reviews, and enough of both outcomes.
        """
        rows = self.db.training_rows(field)
        n = len(rows)
        n_wrong = sum(r["label"] for r in rows)
        n_right = n - n_wrong
        if n < MIN_SAMPLES or n_wrong < MIN_PER_CLASS or n_right < MIN_PER_CLASS:
            return {"field": field, "trained": False, "n_samples": n,
                    "n_wrong": n_wrong, "reason": "not enough reviews yet"}

        names = FEATURE_NAMES[field]
        X = [[float(r["features"].get(name, 0.0)) for name in names] for r in rows]
        y = [r["label"] for r in rows]
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        # Judge it on data it did not fit, and on BOTH classes - plain accuracy
        # would look excellent for a model that simply follows the majority.
        folds = max(2, min(5, n_wrong, n_right))
        quality = float(cross_val_score(
            pipe, X, y, cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=0),
            scoring="balanced_accuracy").mean())
        if quality < MIN_QUALITY:
            return {"field": field, "trained": False, "n_samples": n, "n_wrong": n_wrong,
                    "quality": round(quality, 3),
                    "reason": f"model scored {quality:.2f}, below {MIN_QUALITY} - keeping heuristic"}

        pipe.fit(X, y)
        version = self.db.record_model(field, n, n_wrong, {"cv_balanced_accuracy": round(quality, 3)})
        bundle = {"pipeline": pipe, "features": names, "version": version}
        joblib.dump(bundle, self.model_dir / f"{field}.joblib")
        self._models[field] = bundle
        return {"field": field, "trained": True, "version": version,
                "n_samples": n, "n_wrong": n_wrong, "quality": round(quality, 3)}

    def retrain_all(self) -> List[Dict[str, Any]]:
        return [self.retrain(field) for field in CORRECTABLE_FIELDS]

    def status(self) -> Dict[str, Any]:
        """What each field is currently scored by, and how much evidence exists."""
        out = {}
        for field in CORRECTABLE_FIELDS:
            rows = self.db.training_rows(field)
            model = self._models.get(field)
            out[field] = {
                "scored_by": f"model v{model['version']}" if model else "heuristic",
                "reviews": len(rows),
                "edited": sum(r["label"] for r in rows),
            }
        return out

    def _load(self, field: str) -> None:
        path = self.model_dir / f"{field}.joblib"
        if path.exists():
            try:
                self._models[field] = joblib.load(path)
            except Exception:
                pass  # a corrupt model must not stop the app; heuristic still works
