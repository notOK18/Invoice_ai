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
]

DATE_PATTERNS = [
    r"\binvoice\s*date\s*[:\-\s]+(?P<value>[0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})",
    r"\bdate\s*[:\-\s]+(?P<value>[0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})",
]

TOTAL_PATTERNS = [
    r"\btotal\s+amount\s*[:\-\s]+\$?\s*(?P<value>[0-9.,]+)",
    r"\bamount\s+due\s*[:\-\s]+\$?\s*(?P<value>[0-9.,]+)",
    r"\bbalance\s+due\s*[:\-\s]+\$?\s*(?P<value>[0-9.,]+)",
    r"\btotal\s*[:\-\s]+\$?\s*(?P<value>[0-9.,]+)",
]

SUPPLIER_PATTERNS = [
    r"\bbill\s*to\s*[:\-\s]*(?P<value>[^\n]{3,80})",
    r"\bfrom\s*[:\-\s]*(?P<value>[^\n]{3,80})",
    r"\bsupplier\s*[:\-\s]*(?P<value>[^\n]{3,80})",
    r"\bvendor\s*[:\-\s]*(?P<value>[^\n]{3,80})",
]

LINE_ITEM_PATTERN = re.compile(
    r"(?P<quantity>\d+)\s+(?P<description>[A-Za-z0-9 &\-]+?)\s+\$?(?P<unit_price>[0-9.,]+)\s+\$?(?P<line_total>[0-9.,]+)",
    re.IGNORECASE,
)

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".avif", ".webp"}


@dataclass
class InvoiceData:
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    total_amount: Optional[str] = None
    supplier: Optional[str] = None
    currency: Optional[str] = None
    line_items: List[Dict[str, Any]] = field(default_factory=list)
    review_required: bool = False
    raw_text: Optional[str] = None
    source_path: Optional[str] = None
    corrections: Dict[str, Any] = field(default_factory=dict)


class InvoiceProcessor:
    def __init__(
        self,
        corrections_dir: Optional[Path] = None,
        ocr_backend: str = "easyocr",
        ocr_languages: Optional[List[str]] = None,
    ):
        self.corrections_dir = Path(corrections_dir) if corrections_dir else None
        self.corrections = self._load_corrections()
        self.ocr_backend = ocr_backend.lower()
        self.ocr_languages = ocr_languages or ["en"]
        self._ocr_reader = None

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
        try:
            raw_text = self._load_text_from_file(path)
        except Exception as exc:
            fallback = InvoiceData(
                review_required=True,
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
        extracted.corrections = self._match_corrections(extracted)
        extracted.review_required = self._needs_review(extracted)
        return extracted

    def process_invoice_text(self, text: str) -> InvoiceData:
        normalized_text = self._normalize_text(text)
        extracted = self.extract_fields(normalized_text)
        extracted.raw_text = text
        extracted.corrections = self._match_corrections(extracted)
        extracted.review_required = self._needs_review(extracted)
        return extracted

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

    def _perform_ocr(self, image: Any) -> str:
        if self.ocr_backend == "easyocr":
            try:
                reader = self._get_ocr_reader()
                results = reader.readtext(image)
                return "\n".join(
                    text.strip()
                    for _, text, _ in results
                    if isinstance(text, str) and text.strip()
                )
            except Exception as exc:
                raise RuntimeError(f"EasyOCR failed: {exc}") from exc

        if pytesseract is None:
            raise RuntimeError("pytesseract is required for OCR on invoice images.")
        return pytesseract.image_to_string(image)

    def _normalize_text(self, text: str) -> str:
        normalized = re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n").strip())
        normalized = re.sub(r"(?<![A-Za-z])S(?=\d)", "$", normalized)
        return normalized

    def _find_first(self, patterns: List[str], text: str) -> Optional[str]:
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group("value").strip()
                return value
        return None

    def extract_fields(self, text: str) -> InvoiceData:
        invoice_number = self._find_first(INVOICE_NUMBER_PATTERNS, text)
        invoice_date = self._find_first(DATE_PATTERNS, text)
        total_amount = self._find_first(TOTAL_PATTERNS, text)
        supplier = self._find_first(SUPPLIER_PATTERNS, text)
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
        )

    def _normalize_date(self, value: str) -> str:
        try:
            parsed = date_parser.parse(value, dayfirst=False)
            return parsed.strftime("%Y-%m-%d")
        except Exception:
            return value

    def _is_generic_supplier(self, supplier: Optional[str]) -> bool:
        if not supplier:
            return True
        value = supplier.strip().lower()
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
        if "usd" in text.lower() or "$" in text:
            return "USD"
        if "eur" in text.lower() or "€" in text:
            return "EUR"
        if "gbp" in text.lower() or "£" in text:
            return "GBP"
        return None

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

    def _needs_review(self, invoice: InvoiceData) -> bool:
        if not invoice.invoice_number or not invoice.invoice_date or not invoice.total_amount:
            return True
        if invoice.review_required:
            return True
        return False

    def _match_corrections(self, invoice: InvoiceData) -> Dict[str, Any]:
        for correction in self.corrections:
            if correction.get("invoice_number") and invoice.invoice_number == correction["invoice_number"]:
                return correction
        return {}

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
