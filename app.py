import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from invoice_ai.pipeline import InvoiceProcessor


def main() -> None:
    input_dir = Path("data/invoices")
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = InvoiceProcessor(corrections_dir=Path("data/corrections"), ocr_backend="easyocr")

    invoice_files = sorted(path for path in input_dir.iterdir() if path.is_file())
    if not invoice_files:
        print(f"No invoice files found in {input_dir}")
        return

    processed_count = 0
    for invoice_path in invoice_files:
        try:
            invoice = processor.process_invoice_file(invoice_path)
        except Exception as exc:
            print(f"Failed to process {invoice_path.name}: {exc}")
            continue

        output_path = output_dir / f"{invoice_path.stem}.json"
        processor.save_result(invoice, output_path)
        processed_count += 1
        print(f"Processed {invoice_path.name} -> {output_path}")

    print(f"Finished processing {processed_count} invoice(s)")


if __name__ == "__main__":
    main()
