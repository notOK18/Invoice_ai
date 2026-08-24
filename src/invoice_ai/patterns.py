"""Learn extraction labels from human corrections.

When a reviewer fixes a field, the value they typed is known to be correct and
present in the document. The word next to it is therefore a good candidate for
the label that introduces that field - "Gesamtbetrag" before a total, "Rechnung
Nr" before an invoice number. Collecting those candidates lets the extractor pick
up new wordings (another language, another vendor's template) without anyone
editing the code.

Two safeguards keep a coincidence from becoming a rule:

  * a label must be confirmed on several different invoices before it is used
    (the caller applies the threshold via Database.trusted_labels)
  * learned labels only ever EXTEND the built-in patterns; they are appended
    after them, so a learned rule can never override a hand-written one.

This module is pure text handling - no database, no pipeline imports - so the
inference can be tested on its own.
"""

import re
from typing import Dict, List

# A label is a short piece of mostly-alphabetic text. Accented characters are
# included so non-English labels are recognised.
LABEL_RE = re.compile(r"[^\W\d_][\w .\-/]{2,29}", re.UNICODE)

# Words that are never field labels, however often they sit next to a value.
STOPWORDS = {"total", "eur", "usd", "gbp", "invoice", "rechnung"}

# How the value of each field looks, used when a learned label is turned into a
# regex. Fields absent here are not learned (currency is inferred, not labelled).
VALUE_PATTERNS: Dict[str, str] = {
    "invoice_number": r"[A-Za-z0-9][A-Za-z0-9\-_/]+",
    "invoice_date": r"[0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4}",
    "total_amount": r"[0-9][0-9.,]*",
    "supplier": r"[^\n]{3,80}",
}

LEARNABLE_FIELDS = tuple(VALUE_PATTERNS)


def _clean(text: str) -> str:
    return " ".join(str(text or "").split()).strip(" :.-\t")


def value_variants(field: str, value: str) -> List[str]:
    """The forms a confirmed value might take in the document.

    Dates are stored normalised (2020-09-19) but written many ways on the page
    (19.09.2020, 09/19/2020...), so searching for the stored form alone would
    never locate them - and no date label would ever be learned.
    """
    value = str(value or "").strip()
    if not value:
        return []
    variants = [value]
    if field == "invoice_date":
        iso = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value)
        if iso:
            year, month, day = iso.groups()
            for sep in (".", "/", "-"):
                variants += [
                    f"{day}{sep}{month}{sep}{year}",          # 19.09.2020
                    f"{month}{sep}{day}{sep}{year}",          # 09.19.2020
                    f"{int(day)}{sep}{int(month)}{sep}{year}",  # 19.9.2020
                ]
    # de-duplicate, keep order
    seen, out = set(), []
    for item in variants:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def candidate_labels(text: str, value: str, lines_above: int = 2) -> List[str]:
    """Labels that could introduce `value` in `text`.

    Looks on the value's own line first (a label to its left), then on the lines
    immediately above - OCR of a two-column layout usually puts the label on its
    own line with the value on the next.
    """
    value = str(value or "").strip()
    if len(value) < 2:
        return []

    lines = [line.strip() for line in str(text or "").split("\n")]
    found: List[str] = []
    for index, line in enumerate(lines):
        if value not in line:
            continue

        before = _clean(line.split(value)[0])
        if before and LABEL_RE.fullmatch(before):
            found.append(before)
            continue  # a same-line label wins; do not also look above

        for above in range(index - 1, max(-1, index - 1 - lines_above), -1):
            candidate = _clean(lines[above])
            if candidate and LABEL_RE.fullmatch(candidate):
                found.append(candidate)
                break

    # de-duplicate, keep order, drop anything meaningless
    seen, out = set(), []
    for label in found:
        key = label.lower()
        if key in seen or key in STOPWORDS:
            continue
        seen.add(key)
        out.append(label)
    return out


def label_to_pattern(field: str, label: str) -> str:
    """A regex matching `label` followed by a value of `field`'s shape."""
    value = VALUE_PATTERNS[field]
    # \s* spans newlines, which is what makes "label on its own line, value on
    # the next" work - the common OCR layout.
    return rf"\b{re.escape(label)}\s*[:\-.]?\s*(?P<value>{value})"


def learned_patterns(labels_by_field: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """{field: [regex, ...]} built from trusted labels."""
    out: Dict[str, List[str]] = {}
    for field, labels in (labels_by_field or {}).items():
        if field not in VALUE_PATTERNS:
            continue
        out[field] = [label_to_pattern(field, label) for label in labels]
    return out
