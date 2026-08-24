import json
import sys
import threading
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for extra in (str(ROOT), str(ROOT / "src")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import review_ui  # noqa: E402
from invoice_ai.pipeline import InvoiceProcessor  # noqa: E402


def test_review_server_serves_page_image_and_saves(tmp_path, monkeypatch):
    # keep test output out of the real output/ folder
    monkeypatch.setattr(review_ui, "OUTPUT_DIR", tmp_path / "out")
    # ...and off the real database: the review loop writes to it, and any models
    # trained there would change the confidences this test asserts on.
    from invoice_ai.db import Database
    monkeypatch.setattr(review_ui, "DB", Database(tmp_path / "test.db"))
    monkeypatch.setattr(review_ui, "_SCORER", None)

    processor = InvoiceProcessor(corrections_dir=tmp_path / "corr")
    invoice = processor.process_invoice_text(
        "Invoice # US-001\nInvoice Date: 11/02/2019\nTotal Amount: $154.06\n"
    )
    invoice.source_path = "data/invoices/invoice1.png"
    invoices = {"invoice1.png": invoice}

    httpd, url = review_ui.create_server(processor, invoices, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        page = urllib.request.urlopen(url).read().decode()
        assert "Invoice Review" in page
        assert "US-001" in page  # invoice data embedded

        # serving the real invoice image
        with urllib.request.urlopen(url + "image?name=invoice1.png") as resp:
            assert resp.status == 200
            assert resp.headers.get("Content-Type") == "image/png"
            assert len(resp.read()) > 0

        # saving a correction learns it and updates the routing
        body = json.dumps({"source": "invoice1.png",
                           "fields": {"supplier": "East Repair Inc"}}).encode()
        req = urllib.request.Request(
            url + "save", data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        result = json.loads(urllib.request.urlopen(req).read())
        assert result["fields"]["supplier"] == "East Repair Inc"
        assert result["learned"] is True
        assert result["route"] == "auto_approve"
    finally:
        httpd.shutdown()
