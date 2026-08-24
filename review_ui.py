"""Side-by-side invoice review UI with a learning loop.

Opens a browser page that starts empty; upload invoices from it and each one is
OCR'd, extracted and scored per field. Fix anything wrong and click "Save &
Learn": the correction is stored, the label beside your value is learned so the
extractor handles that layout next time, and the confidence models retrain from
whether you accepted or edited each field.

Run:
    python review_ui.py            # start empty, upload from the page
    python review_ui.py --folder   # also load unreviewed files in data/invoices
    python review_ui.py --all      # load every file in data/invoices
"""

import json
import sys
import threading
import webbrowser
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, quote

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from invoice_ai.pipeline import (CORRECTABLE_FIELDS, FIELD_WEIGHTS,  # noqa: E402
                                 SUPPORTED_IMAGE_EXTENSIONS, InvoiceProcessor)
from invoice_ai.db import Database  # noqa: E402
from invoice_ai.confidence import AUTO_APPROVE_AT, ConfidenceScorer  # noqa: E402
from invoice_ai.patterns import (LEARNABLE_FIELDS, candidate_labels,  # noqa: E402
                                 learned_patterns, value_variants)

# How many different invoices must confirm a label before the extractor uses it.
# One sighting is a coincidence; three is a pattern.
LABEL_CONFIRMATIONS = 3

INVOICE_DIR = ROOT / "data" / "invoices"
CORRECTIONS_DIR = ROOT / "data" / "corrections"
OUTPUT_DIR = ROOT / "output"

# Set up once at import; the review loop writes every accept/edit here and the
# scorer retrains from them.
DB = Database()
_SCORER: ConfidenceScorer = None
# invoice source name -> {field: extraction_id}, so a review can be attributed
# to the exact extraction the reviewer was looking at.
EXTRACTION_IDS = {}


def scorer(processor: InvoiceProcessor = None) -> ConfidenceScorer:
    """The shared scorer, built on first use.

    Created lazily so the server works whether it was started by main() or wired
    up directly (as the tests do).
    """
    global _SCORER
    if _SCORER is None:
        _SCORER = ConfidenceScorer(DB, processor or InvoiceProcessor())
    return _SCORER

CONTENT_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp",
    ".tiff": "image/tiff", ".avif": "image/avif",
}


def refresh_learned_patterns(processor) -> dict:
    """Load labels confirmed often enough, and hand them to the extractor."""
    labels = DB.trusted_labels(min_seen=LABEL_CONFIRMATIONS)
    processor.learned_patterns = learned_patterns(labels)
    return labels


def learn_labels_from(invoice, edited_fields) -> list:
    """Infer labels from the values a reviewer just typed.

    The corrected value is known to be right and to appear in the document, so
    whatever sits beside it is a candidate label for that field.
    """
    text = invoice.raw_text or ""
    noted = []
    for field, value in edited_fields.items():
        if field not in LEARNABLE_FIELDS or not value:
            continue
        # A date is stored normalised but written differently on the page, so
        # try each form the value could take.
        for form in value_variants(field, value):
            for label in candidate_labels(text, form):
                times = DB.note_label(field, label)
                noted.append({"field": field, "label": label, "times_seen": times,
                              "in_use": times >= LABEL_CONFIRMATIONS})
            if any(n["field"] == field for n in noted):
                break
    return noted


def overall_confidence(scores: dict) -> float:
    """One score for the whole invoice: the field scores weighted by importance.

    FIELD_WEIGHTS is the business view of which fields matter (getting the total
    wrong costs more than getting the currency wrong), so a weighted average says
    "how much of what matters is trustworthy" - unlike a plain average, which
    would let a confident currency mask a doubtful total.
    """
    weights = {f: FIELD_WEIGHTS.get(f, 0.0) for f in scores}
    total = sum(weights.values())
    if not total:
        return round(sum(scores.values()) / len(scores), 3) if scores else 0.0
    return round(sum(scores[f] * w for f, w in weights.items()) / total, 3)


def invoice_payload(name: str, invoice) -> dict:
    """The JSON the page needs to render one invoice card.

    field_confidence comes from the learned scorer (falling back to the pipeline
    heuristic per field until that field has enough reviews to train on), and a
    field is flagged for review purely by the AUTO_APPROVE_AT threshold.
    """
    sc = scorer()
    scores, sources = {}, {}
    text = invoice.raw_text or ""
    # Fields a human has already corrected are verified by definition - never
    # send those back for review.
    corrected = set((invoice.corrections or {}).get("fields", {}))
    for field in CORRECTABLE_FIELDS:
        if field in corrected:
            scores[field], sources[field] = 1.0, "human-verified"
            continue
        feats = sc.features(field, invoice, text)
        conf, version = sc.score(field, feats,
                                 heuristic=invoice.field_confidence.get(field))
        scores[field] = conf
        sources[field] = f"model v{version}" if version else "heuristic"

    to_review = [f for f, c in scores.items() if sc.needs_review(c)]
    return {
        "source": name,
        "image_url": "/image?name=" + quote(name),
        "fields": {key: getattr(invoice, key) for key in CORRECTABLE_FIELDS},
        "overall": overall_confidence(scores),
        "weakest": round(min(scores.values()), 3) if scores else 0.0,
        "confidence": round(min(scores.values()), 3) if scores else 0.0,
        "field_confidence": scores,
        "score_source": sources,
        "needs_review": to_review,
        "route": "review" if to_review else "auto_approve",
        "threshold": AUTO_APPROVE_AT,
        "review_reasons": invoice.review_reasons,
        "learned": invoice.learned,
        "line_items": invoice.line_items,
        "learning": sc.status(),
        "learned_labels": DB.all_labels(),
        "label_confirmations": LABEL_CONFIRMATIONS,
        # What extraction actually ran on. Shown in the page because a blank
        # field is nearly always the OCR garbling a label, not the scorer failing
        # - and you cannot tell which without seeing the text.
        "ocr_text": text,
        "ocr_confidence": getattr(invoice, "ocr_confidence", None),
    }


def build_page(payloads: list) -> str:
    return PAGE_TEMPLATE.replace("__DATA__", json.dumps(payloads))


def make_handler(processor: InvoiceProcessor, invoices: dict):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the console quiet
            pass

        def _send(self, code, body, content_type="application/json"):
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                payloads = [invoice_payload(n, inv) for n, inv in invoices.items()]
                self._send(200, build_page(payloads), "text/html; charset=utf-8")
            elif parsed.path == "/image":
                name = parse_qs(parsed.query).get("name", [""])[0]
                self._serve_image(name)
            else:
                self._send(404, "not found", "text/plain")

        def _handle_upload(self):
            """Save an uploaded invoice, run it through the pipeline, score it.

            The file name arrives in a header so the body can stay the raw bytes.
            """
            raw_name = self.headers.get("X-Filename", "upload")
            # Keep only the base name: an uploaded "../../etc/passwd" must not
            # escape the invoice folder.
            safe = Path(raw_name).name or "upload"
            if Path(safe).suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS | {".txt"}:
                self._send(400, json.dumps({"error": f"unsupported file type: {safe}"}))
                return

            length = int(self.headers.get("Content-Length", 0))
            blob = self.rfile.read(length)
            if not blob:
                self._send(400, json.dumps({"error": "empty upload"}))
                return

            INVOICE_DIR.mkdir(parents=True, exist_ok=True)
            dest = INVOICE_DIR / safe
            stem, suffix, n = dest.stem, dest.suffix, 2
            while dest.exists():                      # never overwrite an existing invoice
                dest = INVOICE_DIR / f"{stem}-{n}{suffix}"
                n += 1
            dest.write_bytes(blob)

            try:
                invoice = processor.process_invoice_file(dest)
            except Exception as exc:
                dest.unlink(missing_ok=True)
                self._send(500, json.dumps({"error": f"could not read invoice: {exc}"}))
                return

            invoices[dest.name] = invoice
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            processor.save_result(invoice, OUTPUT_DIR / f"{dest.stem}.json")
            record_extractions(dest.name, invoice)
            self._send(200, json.dumps(invoice_payload(dest.name, invoice)))

        def _serve_image(self, name):
            path = (INVOICE_DIR / name).resolve()
            if INVOICE_DIR.resolve() not in path.parents or not path.is_file():
                self._send(404, "no image", "text/plain")
                return
            self._send(200, path.read_bytes(), CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream"))

        def do_POST(self):
            if urlparse(self.path).path == "/upload":
                self._handle_upload()
                return
            if urlparse(self.path).path != "/save":
                self._send(404, "not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
                name = data["source"]
                fields = {k: v for k, v in data.get("fields", {}).items() if k in CORRECTABLE_FIELDS}
                invoice = invoices[name]
            except (KeyError, ValueError):
                self._send(400, json.dumps({"error": "bad request"}))
                return

            # Log what the human did to EVERY field, not just the ones they
            # changed: "accepted unchanged" is as much a training label as an
            # edit, and without it the model only ever sees failures.
            before = {f: getattr(invoice, f) for f in CORRECTABLE_FIELDS}
            ids = EXTRACTION_IDS.get(name, {})
            for field, extraction_id in ids.items():
                old, new = before.get(field), fields.get(field)
                same = (old or "") == (new or "")
                DB.record_review(extraction_id, "accepted" if same else "edited",
                                 old, new)

            # Learn labels from what changed: the reviewer's value is correct and
            # present in the text, so its neighbour is a candidate label.
            edited = {f: v for f, v in fields.items()
                      if (before.get(f) or "") != (v or "") and v}
            noted = learn_labels_from(invoice, edited)
            refresh_learned_patterns(processor)

            processor.record_correction(invoice, fields)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            processor.save_result(invoice, OUTPUT_DIR / f"{Path(name).stem}.json")

            # Retrain immediately so the very next invoice is scored better.
            trained = [r for r in scorer(processor).retrain_all() if r.get("trained")]
            for r in trained:
                print(f"  ↻ retrained {r['field']} -> v{r['version']} "
                      f"({r['n_samples']} reviews)")

            payload = invoice_payload(name, invoice)
            payload["retrained"] = trained
            payload["labels_learned"] = noted
            record_extractions(name, invoice)  # re-score for any further edits
            self._send(200, json.dumps(payload))

    return Handler


def create_server(processor, invoices, port=0):
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(processor, invoices))
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    return httpd, url


def process_all(processor, include_reviewed=False) -> dict:
    """Process the invoices waiting in INVOICE_DIR.

    Already-reviewed invoices are skipped by default, so the page shows what
    still needs attention rather than everything ever processed. OCR is the slow
    part, so skipping them also makes startup much faster as the folder grows.
    """
    invoices = {}
    if not INVOICE_DIR.exists():
        print(f"  no invoice folder yet ({INVOICE_DIR}) - upload one from the page")
        return invoices

    done = set() if include_reviewed else DB.reviewed_sources()
    files = sorted(p for p in INVOICE_DIR.iterdir() if p.is_file() and not p.name.startswith("."))
    skipped = [p for p in files if p.name in done]
    files = [p for p in files if p.name not in done]
    if skipped:
        print(f"  skipping {len(skipped)} already-reviewed invoice(s)")
    for path in files:
        try:
            invoice = processor.process_invoice_file(path)
        except Exception as exc:
            print(f"  ! {path.name}: {exc}")
            continue
        invoices[path.name] = invoice
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        processor.save_result(invoice, OUTPUT_DIR / f"{path.stem}.json")
        record_extractions(path.name, invoice)
        print(f"  - {path.name}: {invoice.route} (confidence {invoice.confidence:.2f})")
    return invoices


def record_extractions(name: str, invoice) -> None:
    """Persist this invoice and the value+features+score of each of its fields.

    Storing the features as they were at scoring time is what makes the review
    that follows usable as a training example later.
    """
    sc = scorer()
    text = invoice.raw_text or ""
    invoice_id = DB.add_invoice(name, ocr_text=text)
    ids = {}
    for field in CORRECTABLE_FIELDS:
        feats = sc.features(field, invoice, text)
        conf, version = sc.score(field, feats,
                                 heuristic=invoice.field_confidence.get(field))
        ids[field] = DB.add_extraction(invoice_id, field, getattr(invoice, field),
                                       conf, feats, model_version=version)
    EXTRACTION_IDS[name] = ids


def main():
    print("Processing invoices (first run downloads the OCR model)…")
    processor = InvoiceProcessor(corrections_dir=CORRECTIONS_DIR, ocr_backend="easyocr")
    scorer(processor)
    labels = refresh_learned_patterns(processor)
    if labels:
        print(f"  learned labels in use: {labels}")
    # The page starts empty: it shows what you upload in this session, not a
    # backlog of everything ever processed. Startup is then instant, since OCR
    # only runs on what you actually hand it.
    if "--folder" in sys.argv or "--all" in sys.argv:
        invoices = process_all(processor, include_reviewed="--all" in sys.argv)
    else:
        invoices = {}
        print("  starting empty - upload invoices from the page"
              "\n  (--folder loads unreviewed files from data/invoices, --all loads every file)")
    httpd, url = create_server(processor, invoices)
    print(f"\nReview UI ready at {url}\nClose this window (Ctrl+C) when you're done.")
    threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
        httpd.shutdown()


PAGE_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Invoice Review</title>
<style>
  :root { --bg:#0f1720; --card:#182634; --line:#26384a; --ink:#e7eef6; --muted:#9db2c8;
          --green:#2ec16b; --amber:#e8a33d; --red:#e2555a; --accent:#3b9ae1; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:18px 24px; border-bottom:1px solid var(--line); display:flex; align-items:baseline; gap:12px; }
  header h1 { font-size:20px; margin:0; }
  header p { margin:0; color:var(--muted); font-size:13px; }
  .wrap { padding:24px; display:flex; flex-direction:column; gap:24px; max-width:1400px; margin:0 auto; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:14px; overflow:hidden; }
  .card-head { display:flex; justify-content:space-between; align-items:center; padding:12px 18px; border-bottom:1px solid var(--line); }
  .card-head .name { font-weight:600; }
  .cols { display:grid; grid-template-columns:1fr 1fr; gap:0; }
  @media (max-width:900px){ .cols { grid-template-columns:1fr; } }
  .imgpane { background:#0b1119; display:flex; align-items:center; justify-content:center; padding:16px; max-height:640px; overflow:auto; }
  .imgpane img { max-width:100%; border-radius:8px; box-shadow:0 6px 20px rgba(0,0,0,.4); }
  .form { padding:18px 20px; border-left:1px solid var(--line); }
  .field { margin-bottom:12px; }
  .field label { display:block; font-size:12px; color:var(--muted); margin-bottom:4px; text-transform:uppercase; letter-spacing:.4px; }
  .field input { width:100%; padding:9px 11px; border-radius:8px; border:1px solid var(--line); background:#0f1a25; color:var(--ink); font-size:14px; }
  .field input:focus { outline:none; border-color:var(--accent); }
  .conf { font-size:11px; color:var(--muted); margin-top:3px; }
  .badges { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  .badge { padding:4px 10px; border-radius:999px; font-size:12px; font-weight:600; }
  .route-auto_approve { background:rgba(46,193,107,.15); color:var(--green); }
  .route-review { background:rgba(232,163,61,.15); color:var(--amber); }
  .route-reject { background:rgba(226,85,90,.15); color:var(--red); }
  .learned { background:rgba(59,154,225,.15); color:var(--accent); }
  .meter { height:7px; border-radius:6px; background:#0b1119; overflow:hidden; margin-top:6px; }
  .meter > span { display:block; height:100%; background:linear-gradient(90deg,var(--red),var(--amber),var(--green)); }
  .reasons { font-size:12px; color:var(--muted); margin-top:10px; }
  .reasons li { margin:2px 0; }
  .items { width:100%; border-collapse:collapse; margin-top:10px; font-size:12px; }
  .items th, .items td { border-bottom:1px solid var(--line); padding:5px 6px; text-align:left; color:var(--muted); }
  .actions { margin-top:14px; display:flex; gap:10px; align-items:center; }
  button.save { background:var(--accent); color:#fff; border:none; border-radius:8px; padding:10px 16px; font-size:14px; font-weight:600; cursor:pointer; }
  button.save:hover { filter:brightness(1.08); }
  .saved-note { color:var(--green); font-size:13px; }
  .ocr { margin-top:12px; border:1px solid var(--line); border-radius:8px; padding:8px 10px; }
  .ocr summary { cursor:pointer; color:var(--muted); font-size:12.5px; }
  .ocr summary:hover { color:var(--ink); }
  .ocrmeta { color:var(--muted); font-size:11.5px; margin:6px 0 4px; }
  .ocr pre { margin:0; max-height:260px; overflow:auto; white-space:pre-wrap;
             word-break:break-word; background:#0b1119; border-radius:6px;
             padding:9px 11px; font-size:11.5px; line-height:1.5; color:var(--ink); }
  .uploadbar { display:flex; align-items:center; gap:14px; padding:14px 24px;
               border-bottom:1px solid var(--line); background:var(--card); }
  .uploadbtn { background:var(--accent); color:#fff; border-radius:8px; padding:9px 16px;
               font-size:13.5px; font-weight:600; cursor:pointer; }
  .uploadbtn:hover { filter:brightness(1.08); }
  .uploadnote { color:var(--muted); font-size:12.5px; }
  .overall { display:flex; align-items:center; gap:10px; margin:2px 0 10px; }
  .overall .num { font-size:26px; font-weight:650; letter-spacing:-.5px; }
  .overall .lbl { color:var(--muted); font-size:12px; }
  .obar { flex:1; height:8px; border-radius:5px; background:#0b1119; overflow:hidden; }
  .obar > span { display:block; height:100%; border-radius:5px; }
</style>
</head>
<body>
<header>
  <h1>Invoice Review</h1>
  <p>Fields at or above the confidence bar are auto-approved; the rest are flagged. Every save — edited <i>or</i> untouched — trains the scorer for the next invoice.</p>
</header>
<div class="uploadbar">
  <label class="uploadbtn" for="newInvoice">＋ Upload invoice</label>
  <input type="file" id="newInvoice" accept="image/*,.txt" multiple hidden />
  <span class="uploadnote" id="uploadNote">JPG, PNG or TXT — it is scored as soon as it is read.</span>
</div>
<div class="wrap" id="wrap"></div>

<script>
const DATA = __DATA__;
const FIELDS = ["invoice_number","invoice_date","total_amount","supplier","currency"];
const wrap = document.getElementById('wrap');

function pct(x){ return Math.round((x||0)*100); }

function card(inv){
  const el = document.createElement('div');
  el.className = 'card';
  el.innerHTML = `
    <div class="card-head">
      <span class="name">📄 ${inv.source}</span>
      <span class="badges" data-badges></span>
    </div>
    <div class="cols">
      <div class="imgpane"><img src="${inv.image_url}" alt="${inv.source}" /></div>
      <div class="form">
        <div class="overall">
          <span class="num" data-overall></span>
          <span class="lbl" data-overall-label></span>
          <span class="obar"><span data-obar></span></span>
        </div>
        ${FIELDS.map(f=>`
          <div class="field">
            <label>${f.replace(/_/g,' ')}</label>
            <input data-field="${f}" value="${inv.fields[f] ?? ''}" />
            <div class="conf" data-conf="${f}"></div>
          </div>`).join('')}
        <div class="meter"><span data-meter></span></div>
        <ul class="reasons" data-reasons></ul>
        ${inv.line_items && inv.line_items.length ? `
          <table class="items"><thead><tr><th>Qty</th><th>Description</th><th>Unit</th><th>Total</th></tr></thead>
          <tbody>${inv.line_items.map(li=>`<tr><td>${li.quantity??''}</td><td>${li.description??''}</td><td>${li.unit_price??''}</td><td>${li.line_total??''}</td></tr>`).join('')}</tbody></table>`:''}
        <details class="ocr">
          <summary>OCR text — what extraction actually read</summary>
          <div class="ocrmeta" data-ocrmeta></div>
          <pre data-ocrtext></pre>
        </details>
        <div class="actions">
          <button class="save" data-save>Save &amp; Learn</button>
          <span class="saved-note" data-note></span>
        </div>
      </div>
    </div>`;
  renderState(el, inv);
  el.querySelector('[data-save]').addEventListener('click', ()=>save(el, inv));
  return el;
}

function renderState(el, inv){
  const badges = el.querySelector('[data-badges]');
  const n = (inv.needs_review||[]).length;
  badges.innerHTML = `<span class="badge route-${inv.route}">${inv.route.replace('_',' ')}</span>
    <span class="badge">lowest field ${pct(inv.confidence)}% (bar ${pct(inv.threshold??0.9)}%)</span>
    ${n?`<span class="badge route-review">${n} field${n>1?'s':''} to check</span>`
       :'<span class="badge route-auto_approve">all fields clear</span>'}
    ${inv.learned?'<span class="badge learned">✓ learned</span>':''}`;
  el.querySelector('[data-meter]').style.width = pct(inv.confidence)+'%';
  const ov = inv.overall ?? inv.confidence, okAll = !(inv.needs_review||[]).length;
  const col = okAll ? 'var(--green)' : (ov >= 0.75 ? 'var(--amber)' : 'var(--red)');
  const o = el.querySelector('[data-overall]');
  o.textContent = pct(ov)+'%'; o.style.color = col;
  el.querySelector('[data-overall-label]').textContent =
    'overall confidence · weakest field ' + pct(inv.weakest ?? ov) + '%';
  const ob = el.querySelector('[data-obar]');
  ob.style.width = pct(ov)+'%'; ob.style.background = col;
  const thr = inv.threshold ?? 0.9;
  FIELDS.forEach(f=>{
    const c = (inv.field_confidence||{})[f];
    const node = el.querySelector(`[data-conf="${f}"]`);
    if(c==null){ node.textContent=''; return; }
    const low = c < thr;
    const src = (inv.score_source||{})[f] || '';
    node.innerHTML = `<b style="color:${low?'var(--amber)':'var(--green)'}">${pct(c)}%</b>` +
      ` ${low?'· needs review':'· auto-approve'} <span style="opacity:.6">· ${src}</span>`;
    const input = el.querySelector(`[data-field="${f}"]`);
    if(input) input.style.borderColor = low ? 'var(--amber)' : 'var(--line)';
  });
  const pre = el.querySelector('[data-ocrtext]');
  if (pre) {
    pre.textContent = inv.ocr_text || '(no text — this invoice produced nothing readable)';
    const oc = inv.ocr_confidence;
    el.querySelector('[data-ocrmeta]').textContent =
      (oc == null ? 'source: plain text (no OCR)' : `OCR confidence ${pct(oc)}%`) +
      ` · ${(inv.ocr_text || '').length} characters`;
  }
  const reasons = el.querySelector('[data-reasons]');
  reasons.innerHTML = (inv.review_reasons||[]).map(r=>`<li>• ${r}</li>`).join('');
}

async function save(el, inv){
  const fields = {};
  el.querySelectorAll('[data-field]').forEach(i=>fields[i.dataset.field] = i.value.trim() || null);
  const note = el.querySelector('[data-note]');
  note.textContent = 'Saving…';
  const res = await fetch('/save', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({source: inv.source, fields})});
  if(!res.ok){ note.textContent = 'Error saving'; return; }
  const updated = await res.json();
  Object.assign(inv, updated);
  renderState(el, inv);
  const rt = (updated.retrained||[]);
  note.textContent = rt.length
    ? `Saved ✓ — retrained ${rt.map(r=>r.field+' v'+r.version).join(', ')}`
    : 'Saved ✓ — logged for learning';
  setTimeout(()=>note.textContent='', 2500);
}

if (!DATA.length) {
  wrap.innerHTML = `<div class="card" style="padding:34px;text-align:center;color:var(--muted)">
    <div style="font-size:15px;color:var(--ink);margin-bottom:6px">No invoices yet</div>
    Upload one above to extract, score and review it.</div>`;
}
DATA.forEach(inv => wrap.appendChild(card(inv)));

const note = document.getElementById('uploadNote');
document.getElementById('newInvoice').addEventListener('change', async e => {
  const files = [...e.target.files];
  e.target.value = '';
  for (const file of files) {
    note.textContent = `Reading ${file.name}… (OCR can take a moment)`;
    try {
      const res = await fetch('/upload', {method:'POST', body:file,
                                          headers:{'X-Filename': file.name}});
      const inv = await res.json();
      if (inv.error) { note.textContent = `${file.name}: ${inv.error}`; continue; }
      wrap.prepend(card(inv));
      note.textContent = `${inv.source} added — ${pct(inv.overall)}% overall`;
    } catch (err) {
      note.textContent = `${file.name}: ${err}`;
    }
  }
});
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
