import types
from pathlib import Path

from invoice_ai.pipeline import InvoiceProcessor, InvoiceData


def test_saved_correction_overrides_fields_and_marks_learned(tmp_path):
    processor = InvoiceProcessor(corrections_dir=tmp_path)
    invoice = processor.process_invoice_text(
        "Invoice # X-1\nInvoice Date: 01/02/2026\nTotal Amount: $5.00\n"
    )
    invoice.source_path = "data/invoices/x.png"

    processor.record_correction(invoice, {"supplier": "Globex", "total_amount": "5.00"})

    assert invoice.supplier == "Globex"
    assert invoice.learned is True
    assert invoice.route == "auto_approve"

    # a fresh processor loads the correction from disk and matches by source file
    reloaded = InvoiceProcessor(corrections_dir=tmp_path)
    match = reloaded._match_corrections(InvoiceData(source_path="data/invoices/x.png"))
    assert reloaded._correction_fields(match)["supplier"] == "Globex"


def test_learned_supplier_recovered_on_a_new_invoice(tmp_path):
    processor = InvoiceProcessor(corrections_dir=tmp_path)
    first = processor.process_invoice_text(
        "Invoice # A\nInvoice Date: 01/01/2026\nTotal Amount: $1.00\n"
    )
    first.source_path = "data/invoices/a.png"
    processor.record_correction(first, {"supplier": "Initech"})

    # brand-new processor + new invoice that merely mentions the learned supplier
    reloaded = InvoiceProcessor(corrections_dir=tmp_path)
    new_invoice = reloaded.process_invoice_text(
        "Invoice # B\nInvoice Date: 02/02/2026\nFrom Initech\nTotal Amount: $2.00\n"
    )
    assert new_invoice.supplier == "Initech"


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


def test_high_confidence_invoice_is_auto_approved():
    sample_text = (
        "Invoice # INV-2026-0001\n"
        "Invoice Date: 07/09/2026\n"
        "Bill To: Acme Corporation\n"
        "Total Amount: $1,234.56\n"
        "1 Widget A $100.00 $100.00\n"
        "2 Widget B $150.00 $300.00\n"
    )
    processor = InvoiceProcessor()
    invoice = processor.process_invoice_text(sample_text)

    assert invoice.route == "auto_approve"
    assert invoice.confidence >= 0.85
    assert invoice.review_required is False
    # every scored field is present in the per-field breakdown
    for field_name in ("invoice_number", "invoice_date", "total_amount", "supplier", "currency", "line_items"):
        assert field_name in invoice.field_confidence


def test_low_confidence_invoice_is_rejected_with_reasons():
    processor = InvoiceProcessor()
    invoice = processor.process_invoice_text("Some random receipt text with no fields")

    assert invoice.route == "reject"
    assert invoice.confidence < 0.6
    assert invoice.review_required is True
    assert any("invoice_number" in reason for reason in invoice.review_reasons)


def test_missing_one_critical_field_routes_to_review_or_reject():
    # invoice number missing -> cannot be auto-approved
    sample_text = "Invoice Date: 07/09/2026\nBill To: Acme Corporation\nTotal Amount: $500.00\n"
    processor = InvoiceProcessor()
    invoice = processor.process_invoice_text(sample_text)

    assert invoice.route in {"review", "reject"}
    assert invoice.route != "auto_approve"


def test_low_ocr_confidence_lowers_score(monkeypatch):
    import types

    class DummyReader:
        def readtext(self, image):
            return [
                (None, "Invoice # INV-9001", 0.20),
                (None, "Invoice Date: 07/09/2026", 0.20),
                (None, "Total Amount: $42.00", 0.20),
            ]

    monkeypatch.setattr(
        "invoice_ai.pipeline.easyocr",
        types.SimpleNamespace(Reader=lambda languages: DummyReader()),
    )
    processor = InvoiceProcessor(ocr_backend="easyocr")
    text = processor._perform_ocr(object())
    invoice = processor.extract_fields(processor._normalize_text(text))
    processor._apply_confidence_and_routing(invoice, ocr_confidence=processor._last_ocr_confidence)

    assert abs(processor._last_ocr_confidence - 0.2) < 1e-6
    assert any("low OCR confidence" in reason for reason in invoice.review_reasons)
    assert invoice.confidence < 0.85


def test_thresholds_are_configurable():
    # A normally auto-approved invoice must fall back to review under a stricter bar.
    sample_text = (
        "Invoice # INV-2026-0001\n"
        "Invoice Date: 07/09/2026\n"
        "Bill To: Acme Corporation\n"
        "Total Amount: $1,234.56\n"
        "1 Widget A $100.00 $100.00\n"
    )
    default = InvoiceProcessor()
    strict = InvoiceProcessor(auto_approve_threshold=0.99)

    assert default.process_invoice_text(sample_text).route == "auto_approve"
    assert strict.process_invoice_text(sample_text).route == "review"


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
