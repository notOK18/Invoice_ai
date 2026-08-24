"""Copy a batch of invoice images into data/invoices/ for labelling.

Labelling IS reviewing: open review_ui.py, correct what the extractor got wrong,
and each save records accepted/edited per field - exactly the training labels the
confidence models learn from. This script only stages a manageable batch, since
OCR takes a few seconds per image and the review page processes everything it
finds at startup.

Images already imported are skipped, so repeated runs keep moving through the
folder rather than re-importing the same files.

Usage:
    python ml/import_images.py <source-folder> [count]

Example:
    python ml/import_images.py "Receipt or Invoice.v1i.folder/train/invoice" 25
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "invoices"
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
# Imported files keep this prefix so they are easy to spot and remove later.
PREFIX = "batch_"


def imported_already():
    """Original names of images imported by earlier runs."""
    return {p.name[len(PREFIX):] for p in DEST.glob(f"{PREFIX}*")} if DEST.exists() else set()


def import_batch(source, count=25):
    source = Path(source)
    if not source.is_dir():
        raise SystemExit(f"not a folder: {source}")

    DEST.mkdir(parents=True, exist_ok=True)
    done = imported_already()
    # Sorted, not random, so successive runs walk the folder in a stable order
    # and never re-offer the same images.
    candidates = [p for p in sorted(source.iterdir())
                  if p.suffix.lower() in IMAGE_TYPES and p.name not in done]

    if not candidates:
        print(f"nothing new to import from {source} ({len(done)} already done)")
        return []

    batch = candidates[:count]
    for path in batch:
        shutil.copy2(path, DEST / f"{PREFIX}{path.name}")

    print(f"imported {len(batch)} image(s) -> {DEST}")
    print(f"  already labelled/imported before: {len(done)}")
    print(f"  still waiting in the source folder: {len(candidates) - len(batch)}")
    print("\nnext: python review_ui.py   (first run per image takes a few seconds to OCR)")
    return batch


def clear_batches():
    """Remove imported images (their reviews stay in the database)."""
    removed = 0
    for path in DEST.glob(f"{PREFIX}*"):
        path.unlink()
        removed += 1
    print(f"removed {removed} imported image(s); recorded reviews are untouched")


if __name__ == "__main__":
    if "--clear" in sys.argv:
        clear_batches()
    elif len(sys.argv) < 2:
        print(__doc__)
    else:
        import_batch(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 25)
