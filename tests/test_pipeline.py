import types
from pathlib import Path

from invoice_ai.pipeline import InvoiceProcessor


def test_extract_invoice_fields_from_text():
    sample_text = (
        "Invoice # INV-2026-0001\n"
        "Invoice Date: 07/09/2026\n"
        "Bill To: Acme Corporation\n"
        "Total Amount: $1,234.56\n"
        "1 Widget A $100.00 $100.00\n"
        "2 Widget B $150.00 $300.00\n"
    )

    processor = InvoiceProcessor(corrections_dir=Path("data/corrections"))
    invoice = processor.process_invoice_text(sample_text)

    assert invoice.invoice_number == "INV-2026-0001"
    assert invoice.invoice_date == "2026-07-09"
    assert invoice.supplier == "Acme Corporation"
    assert invoice.total_amount == "1,234.56"
    assert invoice.currency == "USD"
    assert len(invoice.line_items) == 2


def test_review_required_when_fields_missing():
    sample_text = "Invoice Date: 07/09/2026\nTotal: $100.00\n"
    processor = InvoiceProcessor()
    invoice = processor.process_invoice_text(sample_text)

    assert invoice.review_required is True


def test_extract_total_amount_with_ocr_noise():
    processor = InvoiceProcessor()
    invoice = processor.process_invoice_text(
        "Invoice # INV-100\nInvoice Date: 07/09/2026\nBill To: Acme Corp\nTotal\nS154.06"
    )

    assert invoice.total_amount == "154.06"


def test_ignore_generic_supplier_labels():
    processor = InvoiceProcessor()
    invoice = processor.process_invoice_text("Bill To\nInvoice Number:\n998\nInvoice Date: 03/22/2025\nTotal\n$209.00")

    assert invoice.supplier is None


def test_easyocr_backend_is_used_when_requested(monkeypatch):
    class DummyReader:
        def readtext(self, image):
            return [
                (None, "Invoice # INV-9001", 0.9),
                (None, "Total Amount: $42.00", 0.8),
            ]

    monkeypatch.setattr(
        "invoice_ai.pipeline.easyocr",
        types.SimpleNamespace(Reader=lambda languages: DummyReader()),
    )

    processor = InvoiceProcessor(ocr_backend="easyocr")
    extracted_text = processor._perform_ocr(object())

    assert "INV-9001" in extracted_text
    assert "$42.00" in extracted_text
