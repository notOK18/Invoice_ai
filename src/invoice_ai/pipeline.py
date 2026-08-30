import hashlib
import json
import re
import ssl
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    import easyocr
except ImportError:
    easyocr = None
else:
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except Exception:
        pass

from dateutil import parser as date_parser

INVOICE_NUMBER_PATTERNS = [
    r"\binvoice\s*(?:#|no\.?|number|:)\s*[:#]?\s*(?P<value>[A-Z0-9\-_/]+)",
    r"\binv(?:oice)?\s*(?:#|no\.?|number|:)\s*[:#]?\s*(?P<value>[A-Z0-9\-_/]+)",
    # German: "Rechnung Nr. 2020-1010" / "Rechnungsnummer: ...". Kept away from
    # "Kundennr" (customer number), which is a different field.
    r"\brechnungs?\s*(?:nr\.?|nummer)\s*[:.]?\s*(?P<value>[A-Z0-9][A-Z0-9\-_/]+)",
]

# Dates written with a month name rather than digits: "7 April 2015",
# "April 7, 2015", "Apr 2015". The day is optional because some invoices give
# only month and year.
MONTH_NAMES = (r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
               r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
               r"nov(?:ember)?|dec(?:ember)?")
MONTH_NAME_DATE = (rf"(?:\d{{1,2}}\s+)?(?:{MONTH_NAMES})\.?,?\s+\d{{1,2}},?\s*\d{{4}}"
                   rf"|(?:\d{{1,2}}\s+)?(?:{MONTH_NAMES})\.?,?\s*\d{{4}}")

DATE_PATTERNS = [
    r"\binvoice\s*date\s*[:\-\s]+(?P<value>[0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})",
    # "Date of issue:" / "Issue date:" - the label may sit on its own line with the
    # date on the next, which OCR of a columnar layout produces routinely.
    r"\bdate\s+of\s+issue\s*[:\-]?\s*(?P<value>[0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})",
    r"\bissue\s*date\s*[:\-]?\s*(?P<value>[0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})",
    # German: "Rechnungsdatum:" is the invoice date. Listed before the generic
    # date pattern, and before Liefer-/Leistungs-/Falligkeitsdatum, so a delivery
    # or due date never wins. The value usually sits on the following line.
    r"\brechnungsdatum\s*[:\-]?\s*(?P<value>[0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})",
    r"\bdate\s*[:\-\s]+(?P<value>[0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})",
    # Month-name dates: "7 April 2015", "April 7, 2015", "Apr 2015". Tried after
    # the numeric forms so a plain date still wins, and last of all so a due date
    # is only used when nothing better was found.
    rf"\binvoice\s*date\s*[:\-\s]*(?P<value>{MONTH_NAME_DATE})",
    rf"\bdate\s*[:\-\s]*(?P<value>{MONTH_NAME_DATE})",
]

TOTAL_PATTERNS = [
    r"\btotal\s+amount\s*[:\-\s]+\$?\s*(?P<value>[0-9.,]+)",
    r"\bamount\s+due\s*[:\-\s]+\$?\s*(?P<value>[0-9.,]+)",
    r"\bbalance\s+due\s*[:\-\s]+\$?\s*(?P<value>[0-9.,]+)",
    # German gross totals. Before the generic "total" so "Gesamtbetrag" wins,
    # and excluding "Nettobetrag", which is the amount BEFORE tax.
    r"\b(?:gesamtbetrag|rechnungsbetrag|endbetrag|bruttobetrag)\s*[:\-]?\s*(?P<value>[0-9.,]+)",
    r"\btotal\s*[:\-\s]+\$?\s*(?P<value>[0-9.,]+)",
]

# An invoice with a summary table lists net, VAT and gross; the amount actually
# billed is the gross. It is found separately from TOTAL_PATTERNS because those
# take the first number after "Total", and the OCR column order varies - on many
# layouts that lands on the net instead.
SUMMARY_ANCHOR = re.compile(r"\bsummary\b", re.IGNORECASE)
GROSS_ANCHOR = re.compile(r"\bgross\s*worth\b", re.IGNORECASE)
SUMMARY_STOP = re.compile(r"\btotal\b", re.IGNORECASE)
# A money value; thousands may be separated by a space or non-breaking space.
TOTALS_TAIL_LINES = 40  # totals sit at the end of an invoice
MONEY_TOKEN = re.compile(r"\d[\d  ]*[.,]\d{2}(?!\d)")

SUPPLIER_PATTERNS = [
    # "Seller:" is the party issuing the invoice - the supplier. Its value often
    # sits on a later line (columnar OCR), so allow newlines before it but stop
    # at the first non-empty line.
    r"\bseller\s*[:\-]?\s*\n?\s*(?P<value>[^\n]{3,80})",
    r"\bbill\s*to\s*[:\-\s]*(?P<value>[^\n]{3,80})",
    r"\bfrom\s*[:\-\s]*(?P<value>[^\n]{3,80})",
    r"\bsupplier\s*[:\-\s]*(?P<value>[^\n]{3,80})",
    r"\bvendor\s*[:\-\s]*(?P<value>[^\n]{3,80})",
]

LINE_ITEM_PATTERN = re.compile(
    r"(?P<quantity>\d+)\s+(?P<description>[A-Za-z0-9 &\-]+?)\s+\$?(?P<unit_price>[0-9.,]+)\s+\$?(?P<line_total>[0-9.,]+)",
    re.IGNORECASE,
)

# A page scanned sideways or upside down still produces characters, but they are
# the wrong letters. Below this mean OCR confidence, re-read trying rotations.
ROTATION_RETRY_BELOW = 0.60
# Turns to try, and the mirror of each. 180 first: an upside-down page is the
# common case. Stop as soon as one reads clearly, so the full sweep is rare.
ORIENTATIONS_TRIED = [180, 90, 270, 0]
ORIENTATION_GOOD_ENOUGH = 0.75
# Below this, try harder renderings of the page (contrast, threshold).
PREPROCESS_RETRY_BELOW = 0.75

# Currency assumed when a document shows no symbol or wording at all. None means
# make no assumption: a currency is only reported when the document actually
# shows one, so the field is left empty and the invoice flagged rather than
# labelled with a guess that could be confidently wrong.
DEFAULT_CURRENCY = None

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".avif", ".webp"}

# Fields that must be present for an invoice to be auto-approved.
CRITICAL_FIELDS = ("invoice_number", "invoice_date", "total_amount")

# Business importance of each field, from the finance team's ranking. These are a
# statement of what it COSTS to get a field wrong - not how reliably it can be
# extracted - so they are set here, never learned.
IMPORTANCE_SCALE = {"Critical": 6.0, "Highest": 5.0, "Very High": 4.0,
                    "High": 3.0, "Medium": 2.0, "Low": 1.0}

# field -> (importance level, the source system's name for it)
FIELD_IMPORTANCE = {
    # The amount billed outranks everything: a wrong total is a wrong payment,
    # whereas a wrong invoice number is a filing error that can be corrected.
    "total_amount":   ("Critical",  "ClntTotal"),     # total charged to the customer
    "invoice_number": ("Highest",   "-"),             # not ranked in the list; set by finance
    "invoice_date":   ("Very High", "TransDate/ValueDate"),
    "currency":       ("Very High", "CurrID"),
    "supplier":       ("Very High", "ClientID"),      # the "Bill To" party
    "line_items":     ("High",      "Qty"),
    # ClntAccID / DepartmentID / AccountID are deliberately absent: they are
    # assigned by the accounting system, never printed on an invoice, so there is
    # nothing here to extract or score.
}


def _normalised_weights(importance):
    points = {f: IMPORTANCE_SCALE[level] for f, (level, _) in importance.items()}
    total = sum(points.values())
    return {f: round(p / total, 4) for f, p in points.items()}


# Relative importance of each field in the overall confidence score.
FIELD_WEIGHTS = _normalised_weights(FIELD_IMPORTANCE)

# Default routing thresholds on the overall confidence (0..1).
DEFAULT_AUTO_APPROVE_THRESHOLD = 0.85
DEFAULT_REVIEW_THRESHOLD = 0.60

# Routing outcomes.
ROUTE_AUTO_APPROVE = "auto_approve"
ROUTE_REVIEW = "review"
ROUTE_REJECT = "reject"

# Fields a human can correct (and that a saved correction can override).
CORRECTABLE_FIELDS = ("invoice_number", "invoice_date", "total_amount", "supplier", "currency")


@dataclass
class InvoiceData:
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    total_amount: Optional[str] = None
    supplier: Optional[str] = None
    currency: Optional[str] = None
    line_items: List[Dict[str, Any]] = field(default_factory=list)
    review_required: bool = False
    confidence: float = 0.0
    field_confidence: Dict[str, float] = field(default_factory=dict)
    route: str = ROUTE_REVIEW
    review_reasons: List[str] = field(default_factory=list)
    raw_text: Optional[str] = None
    source_path: Optional[str] = None
    corrections: Dict[str, Any] = field(default_factory=dict)
    learned: bool = False  # True when a saved human correction was applied
    # Mean OCR confidence for this document, or None when the text did not
    # come from OCR. Carried on the invoice so scoring does not have to reach
    # into processor state that the next document overwrites.
    ocr_confidence: Optional[float] = None
    # True when total_amount is a best-effort guess rather than a value a
    # pattern actually matched, so it can be scored (and shown) as such.
    total_is_guess: bool = False


class InvoiceProcessor:
    def __init__(
        self,
        corrections_dir: Optional[Path] = None,
        ocr_backend: str = "easyocr",
        ocr_languages: Optional[List[str]] = None,
        auto_approve_threshold: float = DEFAULT_AUTO_APPROVE_THRESHOLD,
        review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
        prefer_summary_total: bool = True,
        learned_patterns: Optional[Dict[str, List[str]]] = None,
        default_currency: Optional[str] = DEFAULT_CURRENCY,
    ):
        # Used when a document shows no currency at all. Set to None to go
        # back to leaving it empty and flagging the invoice.
        self.default_currency = default_currency
        # {field: [regex]} inferred from past corrections. Appended AFTER the
        # built-in patterns, so a learned rule extends the extractor but can
        # never override a hand-written one.
        self.learned_patterns = learned_patterns or {}
        # When True, an invoice's summary block decides the total (the gross,
        # i.e. the amount actually billed). Set False to fall back to the plain
        # TOTAL_PATTERNS on layouts where that heuristic does not apply.
        self.prefer_summary_total = prefer_summary_total
        self.corrections_dir = Path(corrections_dir) if corrections_dir else None
        self.corrections = self._load_corrections()
        # Supplier names learned from past corrections, used to recover the
        # supplier on brand-new invoices where extraction missed it.
        self.known_suppliers = self._collect_known_suppliers()
        self.ocr_backend = ocr_backend.lower()
        self.ocr_languages = ocr_languages or ["en"]
        self.auto_approve_threshold = auto_approve_threshold
        self.review_threshold = review_threshold
        self._ocr_reader = None
        # Mean OCR confidence (0..1) captured from the most recent OCR run,
        # or None when the text came from a source without OCR confidence.
        self._last_ocr_confidence: Optional[float] = None

    def _load_corrections(self) -> List[Dict[str, Any]]:
        if not self.corrections_dir or not self.corrections_dir.exists():
            return []

        corrections = []
        for path in sorted(self.corrections_dir.glob("*.json")):
            try:
                corrections.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return corrections

    def process_invoice_file(self, path: Path) -> InvoiceData:
        path = Path(path)
        self._last_ocr_confidence = None
        try:
            raw_text = self._load_text_from_file(path)
        except Exception as exc:
            fallback = InvoiceData(
                review_required=True,
                confidence=0.0,
                route=ROUTE_REJECT,
                review_reasons=[f"could not read file: {exc}"],
                raw_text="",
                source_path=str(path),
            )
            fallback.corrections = {}
            return fallback

        if raw_text is None:
            raise ValueError(f"Unsupported invoice file: {path}")

        normalized_text = self._normalize_text(raw_text)
        extracted = self.extract_fields(normalized_text)
        extracted.raw_text = raw_text
        extracted.source_path = str(path)
        extracted.ocr_confidence = self._last_ocr_confidence
        self._learn_and_apply(extracted, normalized_text)
        self._apply_confidence_and_routing(extracted, ocr_confidence=self._last_ocr_confidence)
        return extracted

    def process_invoice_text(self, text: str) -> InvoiceData:
        normalized_text = self._normalize_text(text)
        extracted = self.extract_fields(normalized_text)
        extracted.raw_text = text
        self._learn_and_apply(extracted, normalized_text)
        # Plain text has no OCR confidence signal.
        self._apply_confidence_and_routing(extracted, ocr_confidence=None)
        return extracted

    def _learn_and_apply(self, invoice: InvoiceData, normalized_text: str) -> None:
        """Apply a saved correction if one matches; otherwise use learned hints."""
        correction = self._match_corrections(invoice)
        invoice.corrections = correction
        if correction:
            self._apply_corrections(invoice, correction)
            invoice.learned = True
        elif not invoice.supplier:
            recovered = self._recover_known_supplier(normalized_text)
            if recovered:
                invoice.supplier = recovered

    def _load_text_from_file(self, path: Path) -> Optional[str]:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            return path.read_text(encoding="utf-8")
        if suffix in SUPPORTED_IMAGE_EXTENSIONS:
            if Image is None:
                raise RuntimeError("Pillow is required to load invoice image files.")
            return self._perform_ocr(str(path))
        return None

    def _get_ocr_reader(self) -> Any:
        if self.ocr_backend != "easyocr":
            return None
        if self._ocr_reader is None:
            if easyocr is None:
                raise RuntimeError("easyocr is required for OCR on invoice images. Install it with `pip install easyocr`.")
            self._ocr_reader = easyocr.Reader(self.ocr_languages)
        return self._ocr_reader

    @staticmethod
    def _read_results(results):
        """(text, mean_confidence) from easyocr's (bbox, text, confidence) items."""
        lines: List[str] = []
        confidences: List[float] = []
        for item in results:
            text = item[1] if len(item) > 1 else None
            conf = item[2] if len(item) > 2 else None
            if isinstance(text, str) and text.strip():
                lines.append(text.strip())
                if isinstance(conf, (int, float)):
                    confidences.append(float(conf))
        mean = sum(confidences) / len(confidences) if confidences else None
        return "\n".join(lines), mean

    def _prepared_image(self, image):
        """The page upscaled and straightened, or None if it cannot be prepared."""
        try:
            from . import preprocess
            return preprocess.prepare(image)
        except Exception:
            return None      # preprocessing is an optimisation, never a blocker

    def _best_variant(self, reader, image, text, confidence):
        """OCR several renderings of the page and keep whichever reads best."""
        try:
            from . import preprocess
            candidates = preprocess.variants(image)
        except Exception:
            return text, confidence
        best = (text, confidence)
        for _, rendering in candidates:
            try:
                candidate_text, candidate_conf = self._read_results(reader.readtext(rendering))
            except Exception:
                continue
            if candidate_conf and candidate_conf > (best[1] or 0):
                best = (candidate_text, candidate_conf)
        return best

    def _best_rotation(self, reader, image, text, confidence):
        """Re-read a badly-scoring page with the image itself reoriented.

        Asking OCR to handle rotated text recovers the characters but returns
        them bottom-to-top, so labels end up AFTER their values and nothing
        downstream matches. Reorienting the image keeps the reading order intact,
        which is what extraction and label-learning depend on.

        Mirrored variants are included because a page can be flipped as well as
        turned (a scanner fed face-up, a phone's front camera), and no amount of
        rotating will ever un-mirror it.
        """
        if Image is None or not isinstance(image, (str, Path)):
            return text, confidence
        try:
            from PIL import ImageOps
            original = Image.open(image)
        except Exception:
            return text, confidence

        best = (text, confidence)
        for mirrored in (False, True):
            base = ImageOps.mirror(original) if mirrored else original
            for angle in ORIENTATIONS_TRIED:
                try:
                    import numpy
                    turned = base.rotate(angle, expand=True).convert("RGB")
                    candidate_text, candidate_conf = self._read_results(
                        reader.readtext(numpy.array(turned)))
                except Exception:
                    continue
                if candidate_conf and candidate_conf > (best[1] or 0):
                    best = (candidate_text, candidate_conf)
                # Clearly readable: stop rather than trying every orientation.
                if best[1] and best[1] >= ORIENTATION_GOOD_ENOUGH:
                    return best
        return best

    def _perform_ocr(self, image: Any) -> str:
        if self.ocr_backend == "easyocr":
            try:
                reader = self._get_ocr_reader()
                # Enlarge a small page and straighten it before reading. Cheap,
                # needs no OCR to decide, and measurably lifts confidence on the
                # low-resolution scans that fail most often.
                prepared = self._prepared_image(image)
                text, confidence = self._read_results(
                    reader.readtext(prepared if prepared is not None else image))

                # Still poor: try harder renderings of the page and keep the one
                # that actually reads best. Which helps depends on the scan, so
                # it is measured rather than assumed.
                if confidence is not None and confidence < PREPROCESS_RETRY_BELOW:
                    text, confidence = self._best_variant(reader, image, text, confidence)

                # A rotated page still yields characters, but they are the wrong
                # letters ("Total" upside down reads as "Iexol"), so the text is
                # useless and confidence collapses. Retrying with rotations costs
                # about a second, so only do it when the first pass looks bad.
                if confidence is not None and confidence < ROTATION_RETRY_BELOW:
                    text, confidence = self._best_rotation(reader, image, text, confidence)

                self._last_ocr_confidence = confidence
                return text
            except Exception as exc:
                raise RuntimeError(f"EasyOCR failed: {exc}") from exc

        if pytesseract is None:
            raise RuntimeError("pytesseract is required for OCR on invoice images.")
        # pytesseract does not expose a simple aggregate confidence here.
        self._last_ocr_confidence = None
        return pytesseract.image_to_string(image)

    def _normalize_text(self, text: str) -> str:
        normalized = re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n").strip())
        # OCR routinely mistakes a currency symbol for another glyph: "$154.06"
        # comes back as "S154.06", "{13,715.52", "[99.00". Restoring them keeps
        # the money patterns matching instead of the whole total being lost.
        normalized = re.sub(r"(?<![A-Za-z])S(?=\d)", "$", normalized)
        # A bracket glyph before a number is a mangled currency symbol, but which
        # one is unknown - "{13,715.52" was a rupee sign, not a dollar. Drop it so
        # the amount patterns match, without inventing a currency that is not there.
        normalized = re.sub(r"[{\[(](?=\d)", " ", normalized)
        return normalized

    def _find_first(self, patterns: List[str], text: str) -> Optional[str]:
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group("value").strip()
                return value
        return None

    def verified_total(self, text: str) -> Optional[float]:
        """The total confirmed by the invoice's own arithmetic, if it can be.

        Invoices state subtotal + tax = total, whatever the wording ("Net worth /
        VAT / Gross worth", "Subtotal / Tax / Total", "Amount Due"...). Searching
        the closing section for a value that two others add up to identifies the
        amount actually billed, independently of any extraction pattern.

        Returns None when the arithmetic cannot be confirmed - a missing tax line,
        or a document that is not laid out this way.
        """
        # Totals live at the end; limit the search so unrelated figures earlier in
        # the document cannot form a coincidental sum.
        tail = "\n".join(text.splitlines()[-TOTALS_TAIL_LINES:])
        values = sorted({v for v in (self.money_value(t) for t in MONEY_TOKEN.findall(tail)) if v})
        if len(values) < 3:
            return None
        total = values[-1]
        others = values[:-1]
        # net + tax = total, but an invoice may split the tax across rates
        # (e.g. 350.00 + 12.50 + 16.00 = 378.50), so allow a third component.
        for i, a in enumerate(others):
            for j in range(i + 1, len(others)):
                b = others[j]
                if abs(a + b - total) <= 0.02:
                    return total
                for c in others[j + 1:]:
                    if abs(a + b + c - total) <= 0.02:
                        return total
        return None

    def summary_money_values(self, text: str) -> List[str]:
        """Every money value in the invoice's summary block, in order.

        The summary lists the net, VAT and gross amounts; returns them as the raw
        strings, or [] when the invoice has no summary block.
        """
        anchor = SUMMARY_ANCHOR.search(text)
        if anchor is None:
            # No "SUMMARY" heading: fall back to the LAST "Gross worth" label,
            # since the line-items table repeats that header above the summary.
            matches = list(GROSS_ANCHOR.finditer(text))
            if not matches:
                return []
            anchor = matches[-1]

        block = text[anchor.end():]
        stop = SUMMARY_STOP.search(block)
        if stop:
            block = block[:stop.start()]
        return [v.strip() for v in MONEY_TOKEN.findall(block)]

    @staticmethod
    def money_value(token: Any) -> Optional[float]:
        """Numeric value of a money token, in either '1,234.56' or '1 606,67' form.

        The last separator followed by exactly two digits is the decimal point;
        any other separator is a thousands separator.
        """
        text = re.sub(r"[^\d.,]", "", str(token or ""))
        if not text:
            return None
        match = re.search(r"[.,](\d{2})$", text)
        if match:
            whole = re.sub(r"[^\d]", "", text[: match.start()])
            return float(f"{whole or 0}.{match.group(1)}")
        try:
            return float(re.sub(r"[^\d]", "", text))
        except ValueError:
            return None

    def _find_summary_total(self, text: str) -> Optional[str]:
        """The gross total from an invoice's summary block, if it has one.

        The summary states net + VAT = gross, so the gross is the LARGEST value
        there. (Taking the last value instead is wrong whenever the OCR column
        order differs - that scored 91% against ground truth where max scores
        100%.) Returns None when there is no summary block.
        """
        values = self.summary_money_values(text)
        if not values:
            return None
        scored = [(self.money_value(v), v) for v in values]
        scored = [(n, v) for n, v in scored if n is not None]
        return max(scored)[1] if scored else None

    def _patterns_for(self, field: str, builtin: List[str]) -> List[str]:
        """Built-in patterns first, then anything learned for this field."""
        return list(builtin) + list(self.learned_patterns.get(field, ()))

    def extract_fields(self, text: str) -> InvoiceData:
        invoice_number = self._find_first(
            self._patterns_for("invoice_number", INVOICE_NUMBER_PATTERNS), text)
        invoice_date = self._find_first(
            self._patterns_for("invoice_date", DATE_PATTERNS), text)
        # Prefer the summary block's gross total; fall back to the generic patterns.
        total_amount = None
        if self.prefer_summary_total:
            total_amount = self._find_summary_total(text)
        total_amount = total_amount or self._find_first(
            self._patterns_for("total_amount", TOTAL_PATTERNS), text)
        total_is_guess = False
        if not total_amount:
            # Nothing matched, but leaving the box empty helps no one: the
            # reviewer has to read the figure off the page and type it, and the
            # label-learner has no value to anchor on. Offering the most likely
            # candidate gives them something to confirm or correct instead, and
            # its low confidence still sends the invoice to review.
            total_amount = self._guess_total(text)
            total_is_guess = total_amount is not None
        supplier = self._find_first(
            self._patterns_for("supplier", SUPPLIER_PATTERNS), text)
        currency = self._infer_currency(text)
        line_items = self._extract_line_items(text)

        if invoice_date:
            invoice_date = self._normalize_date(invoice_date)
        if supplier and self._is_generic_supplier(supplier):
            supplier = None

        return InvoiceData(
            invoice_number=invoice_number,
            invoice_date=invoice_date,
            total_amount=total_amount,
            supplier=supplier,
            currency=currency,
            line_items=line_items,
            total_is_guess=total_is_guess,
        )

    def _guess_total(self, text: str) -> Optional[str]:
        """The most likely total when no pattern matched: a best-effort candidate.

        Prefers the figure the invoice's own arithmetic confirms; otherwise the
        largest money value near the end, which is where totals sit. Returned so
        the reviewer has something to correct rather than an empty box - it is
        scored as a guess, never as a confident reading.
        """
        verified = self.verified_total(text)
        if verified is not None:
            return f"{verified:.2f}"
        tail = "\n".join(text.splitlines()[-TOTALS_TAIL_LINES:])
        values = [(self.money_value(t), t) for t in MONEY_TOKEN.findall(tail)]
        values = [(n, t) for n, t in values if n is not None]
        return max(values)[1] if values else None

    def _normalize_date(self, value: str) -> str:
        try:
            # Anchor missing components to the 1st of January rather than today:
            # "Apr 2015" must normalise to 2015-04-01, not to whatever day it
            # happens to be parsed on - otherwise the same invoice yields a
            # different date tomorrow.
            from datetime import datetime
            parsed = date_parser.parse(value, dayfirst=False,
                                       default=datetime(2000, 1, 1))
            return parsed.strftime("%Y-%m-%d")
        except Exception:
            return value

    def _is_generic_supplier(self, supplier: Optional[str]) -> bool:
        if not supplier:
            return True
        # Strip trailing punctuation first: a column header comes through as
        # "Client:" and would otherwise slip past the check for "client".
        value = supplier.strip().strip(":;.,-").strip().lower()
        generic_values = {
            "bill to",
            "ship to",
            "seller",
            "invoice",
            "invoice number",
            "invoice date",
            "client",
            "customer",
            "due date",
        }
        return value in generic_values or value.startswith("invoice") or value.startswith("bill to") or value.startswith("ship to")

    def _infer_currency(self, text: str) -> Optional[str]:
        """The currency a document is in, from its symbols or wording.

        Falls back to `default_currency` when nothing is found: a scan often
        loses the symbol entirely (OCR drops a faint "$"), and an invoice that is
        simply missing a currency would otherwise be flagged forever even though
        the answer is never in doubt for a business trading in one currency.
        """
        lowered = text.lower()
        if "usd" in lowered or "$" in text:
            return "USD"
        if "eur" in lowered or "€" in text:
            return "EUR"
        if "gbp" in lowered or "£" in text:
            return "GBP"
        if "inr" in lowered or "₹" in text or re.search(r"\brs\.?\b|rupee", lowered):
            return "INR"
        # Wording and place names are a strong hint when the symbol did not
        # survive OCR - a scan often loses the glyph but keeps the address.
        if any(word in lowered for word in ("rechnung", "gesamtbetrag", "mwst", "netto")):
            return "EUR"
        if any(word in lowered for word in ("delhi", "mumbai", "bank india", "gst", "cst")):
            return "INR"
        return self.default_currency

    def _extract_line_items(self, text: str) -> List[Dict[str, Any]]:
        items = []
        for match in LINE_ITEM_PATTERN.finditer(text):
            items.append({
                "quantity": int(match.group("quantity")),
                "description": match.group("description").strip(),
                "unit_price": match.group("unit_price"),
                "line_total": match.group("line_total"),
            })
        return items

    # ------------------------------------------------------------------
    # Confidence scoring & routing
    # ------------------------------------------------------------------
    def _apply_confidence_and_routing(
        self, invoice: InvoiceData, ocr_confidence: Optional[float] = None
    ) -> InvoiceData:
        """Score each field, compute overall confidence, and choose a route."""
        field_conf, reasons = self._score_fields(invoice, ocr_confidence)
        overall = self._overall_confidence(field_conf)

        # A previously human-corrected invoice is a strong trust signal.
        if invoice.corrections:
            overall = max(overall, 0.9)
            reasons.append("matched a saved correction")

        route = self._route(invoice, overall)

        invoice.field_confidence = {key: round(value, 3) for key, value in field_conf.items()}
        invoice.confidence = round(overall, 3)
        invoice.route = route
        invoice.review_reasons = reasons
        invoice.review_required = route != ROUTE_AUTO_APPROVE
        return invoice

    def _score_fields(
        self, invoice: InvoiceData, ocr_confidence: Optional[float]
    ) -> "tuple[Dict[str, float], List[str]]":
        reasons: List[str] = []
        scores: Dict[str, float] = {}

        # invoice_number
        if not invoice.invoice_number:
            scores["invoice_number"] = 0.0
            reasons.append("invoice_number missing")
        elif self._is_plausible_invoice_number(invoice.invoice_number):
            scores["invoice_number"] = 0.95
        else:
            scores["invoice_number"] = 0.5
            reasons.append("invoice_number format unusual")

        # invoice_date
        if not invoice.invoice_date:
            scores["invoice_date"] = 0.0
            reasons.append("invoice_date missing")
        elif self._is_iso_date(invoice.invoice_date):
            scores["invoice_date"] = 0.95
        else:
            scores["invoice_date"] = 0.5
            reasons.append("invoice_date could not be normalized")

        # total_amount
        if invoice.total_amount and invoice.total_is_guess:
            scores["total_amount"] = 0.3
            reasons.append("total_amount is a guess - no pattern matched")
        elif not invoice.total_amount:
            scores["total_amount"] = 0.0
            reasons.append("total_amount missing")
        elif self._is_valid_number(invoice.total_amount):
            scores["total_amount"] = 0.9
        else:
            scores["total_amount"] = 0.4
            reasons.append("total_amount is not numeric")

        # supplier
        if invoice.supplier:
            scores["supplier"] = 0.75
        else:
            scores["supplier"] = 0.0
            reasons.append("supplier missing")

        # currency
        # A missing currency scores 0 like every other missing field. It used to
        # score 0.4, which quietly inflated the overall confidence and showed an
        # empty field as ~30% rather than as a failure.
        if invoice.currency:
            scores["currency"] = 0.9
        else:
            scores["currency"] = 0.0
            reasons.append("currency missing")

        # line_items
        if invoice.line_items:
            scores["line_items"] = min(1.0, 0.6 + 0.1 * len(invoice.line_items))
        else:
            scores["line_items"] = 0.3
            reasons.append("no line items detected")

        # Cross-check: do the line-item totals reconcile with the invoice total?
        if invoice.line_items and invoice.total_amount and self._is_valid_number(invoice.total_amount):
            if self._totals_reconcile(invoice):
                scores["total_amount"] = max(scores["total_amount"], 0.98)
            else:
                reasons.append("line-item totals do not match the invoice total")

        # Blend in OCR confidence when available (scanned images).
        if ocr_confidence is not None:
            factor = 0.65 + 0.35 * max(0.0, min(1.0, ocr_confidence))
            scores = {key: value * factor for key, value in scores.items()}
            if ocr_confidence < 0.5:
                reasons.append(f"low OCR confidence ({ocr_confidence:.2f})")

        return scores, reasons

    def _overall_confidence(self, field_conf: Dict[str, float]) -> float:
        """Importance-weighted average over the fields actually scored.

        Weights exist for fields this pipeline does not produce (the accounting
        codes are looked up elsewhere), so normalise over the fields present -
        otherwise a missing field would silently count as zero confidence.
        """
        weights = {key: FIELD_WEIGHTS[key] for key in field_conf if key in FIELD_WEIGHTS}
        total_weight = sum(weights.values())
        if not total_weight:
            return 0.0
        return sum(field_conf[key] * weight for key, weight in weights.items()) / total_weight

    def _route(self, invoice: InvoiceData, overall: float) -> str:
        critical_present = all(getattr(invoice, name) for name in CRITICAL_FIELDS)
        if overall >= self.auto_approve_threshold and critical_present:
            return ROUTE_AUTO_APPROVE
        if overall >= self.review_threshold:
            return ROUTE_REVIEW
        return ROUTE_REJECT

    def _totals_reconcile(self, invoice: InvoiceData, tolerance: float = 0.02) -> bool:
        if not invoice.line_items or not self._is_valid_number(invoice.total_amount):
            return False
        total = self._to_number(invoice.total_amount)
        if not total:
            return False
        running = 0.0
        for item in invoice.line_items:
            value = self._to_number(item.get("line_total"))
            if value is None:
                return False
            running += value
        return abs(running - total) / total <= tolerance

    @staticmethod
    def _to_number(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            return float(str(value).replace(",", "").replace("$", "").strip())
        except (TypeError, ValueError):
            return None

    def _is_valid_number(self, value: Any) -> bool:
        return self._to_number(value) is not None

    def _is_iso_date(self, value: Any) -> bool:
        return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", str(value or "")))

    def _is_plausible_invoice_number(self, value: Any) -> bool:
        return bool(re.match(r"^[A-Za-z0-9][A-Za-z0-9\-_/]{1,}$", str(value or "")))

    # ------------------------------------------------------------------
    # Learning: saved human corrections
    # ------------------------------------------------------------------
    def _correction_fields(self, correction: Dict[str, Any]) -> Dict[str, Any]:
        """Return the field overrides from a correction record.

        Supports both the structured form ({"fields": {...}}) and a flat form
        where the correctable fields sit at the top level.
        """
        if isinstance(correction.get("fields"), dict):
            return correction["fields"]
        return {k: v for k, v in correction.items() if k in CORRECTABLE_FIELDS}

    @staticmethod
    def text_fingerprint(text: Optional[str]) -> Optional[str]:
        """A stable id for a document's content, ignoring spacing and case.

        Lets the same invoice be recognised when it is uploaded again under a
        different file name - which happens routinely, because a duplicate upload
        is saved as "invoice-2.jpg" so it cannot overwrite the first.
        """
        if not text or not text.strip():
            return None
        condensed = re.sub(r"\s+", " ", text).strip().lower()
        return hashlib.sha1(condensed.encode("utf-8")).hexdigest()

    def _match_corrections(self, invoice: InvoiceData) -> Dict[str, Any]:
        """Find a saved correction for this invoice.

        Tries the source file name, then the document's content fingerprint, then
        the invoice number. The fingerprint matters because the other two both
        fail on a re-upload: the file is renamed to avoid a clash, and the number
        is often exactly what extraction could not read.
        """
        source_name = Path(invoice.source_path).name if invoice.source_path else None
        for correction in self.corrections:
            if source_name and correction.get("source") == source_name:
                return correction

        fingerprint = self.text_fingerprint(invoice.raw_text)
        if fingerprint:
            for correction in self.corrections:
                if correction.get("fingerprint") == fingerprint:
                    return correction

        for correction in self.corrections:
            number = self._correction_fields(correction).get("invoice_number") or correction.get("invoice_number")
            if number and invoice.invoice_number == number:
                return correction
        return {}

    def _apply_corrections(self, invoice: InvoiceData, correction: Dict[str, Any]) -> None:
        """Overwrite the invoice's fields with the human-corrected values."""
        for key, value in self._correction_fields(correction).items():
            if key in CORRECTABLE_FIELDS:
                setattr(invoice, key, value)

    def _collect_known_suppliers(self) -> set:
        suppliers = set()
        for correction in self.corrections:
            supplier = self._correction_fields(correction).get("supplier")
            if supplier and str(supplier).strip():
                suppliers.add(str(supplier).strip())
        return suppliers

    def _recover_known_supplier(self, text: str) -> Optional[str]:
        """If a previously-learned supplier name appears in the text, return it."""
        lowered = text.lower()
        for supplier in self.known_suppliers:
            if supplier.lower() in lowered:
                return supplier
        return None

    def save_correction(
        self, source_name: str, fields: Dict[str, Any], invoice_number: Optional[str] = None,
        fingerprint: Optional[str] = None
    ) -> Path:
        """Persist a human correction so it is applied to this invoice in future.

        Writes one JSON file per source into the corrections directory and
        refreshes the in-memory correction/known-supplier caches.
        """
        if not self.corrections_dir:
            raise ValueError("No corrections_dir configured; cannot save corrections.")
        self.corrections_dir.mkdir(parents=True, exist_ok=True)

        clean_fields = {k: v for k, v in fields.items() if k in CORRECTABLE_FIELDS}
        record = {
            "source": source_name,
            "invoice_number": invoice_number or clean_fields.get("invoice_number"),
            # Identifies the document by its content, so a re-upload under a new
            # file name still finds this correction.
            "fingerprint": fingerprint,
            "fields": clean_fields,
        }
        safe_stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(source_name).stem) or "correction"
        out_path = self.corrections_dir / f"{safe_stem}.json"
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

        # refresh caches so the correction takes effect immediately
        self.corrections = self._load_corrections()
        self.known_suppliers = self._collect_known_suppliers()
        return out_path

    def record_correction(self, invoice: InvoiceData, fields: Dict[str, Any]) -> InvoiceData:
        """Save a correction AND apply it to the given invoice in-place.

        Used by the review UI: persists the fix for the future and updates the
        invoice's fields, confidence and routing right away (no re-OCR needed).
        """
        source_name = Path(invoice.source_path).name if invoice.source_path else (
            invoice.invoice_number or "unknown"
        )
        self.save_correction(
            source_name,
            fields,
            invoice_number=fields.get("invoice_number") or invoice.invoice_number,
            fingerprint=self.text_fingerprint(invoice.raw_text),
        )
        self._apply_corrections(invoice, {"fields": fields})
        invoice.learned = True
        invoice.corrections = {"source": source_name, "fields": {k: v for k, v in fields.items() if k in CORRECTABLE_FIELDS}}
        # corrected values are human-verified → recompute (boosts to auto_approve)
        self._apply_confidence_and_routing(invoice, ocr_confidence=None)
        return invoice

    def process_directory(self, input_dir: Path, output_dir: Path) -> List[InvoiceData]:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results: List[InvoiceData] = []
        for path in sorted(input_dir.iterdir()):
            if not path.is_file():
                continue
            try:
                invoice = self.process_invoice_file(path)
            except Exception:
                continue
            output_path = output_dir / f"{path.stem}.json"
            self.save_result(invoice, output_path)
            results.append(invoice)
        return results

    def save_result(self, invoice: InvoiceData, output_path: Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        invoice_dict = asdict(invoice)
        output_path.write_text(json.dumps(invoice_dict, indent=2), encoding="utf-8")
