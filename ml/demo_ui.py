"""Presentation UI for the total_amount reliability model.

Opens a browser page that demonstrates the trained model:
  * headline metrics (held-out ROC-AUC, balanced accuracy, data sizes)
  * a live scorer — pick a held-out invoice (or paste your own text) and watch
    the extractor run and the model return a confidence
  * the learned weights, so an audience can see what the model actually relies on
  * a sample of held-out predictions vs. what was really true

Run:
    .venv/bin/python ml/demo_ui.py
"""

import json
import sys
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import joblib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "src"))
sys.path.insert(0, str(HERE))

from invoice_ai.pipeline import InvoiceProcessor          # noqa: E402
from build_dataset import (USE_SUMMARY_HEURISTIC, compute_features,  # noqa: E402
                           euro_to_float)

BUNDLE = joblib.load(HERE / "models" / "reliability_total.joblib")
MODEL, FEATURES = BUNDLE["model"], BUNDLE["features"]
# Must match the configuration the model was trained on, or the features the
# model sees at demo time would not be the ones it learned from.
PROC = InvoiceProcessor(prefer_summary_total=USE_SUMMARY_HEURISTIC)

# Written by ml/train_model.py, so these can never go stale behind the model.
METRICS = json.loads((HERE / "models" / "metrics.json").read_text())
IMAGE_DIR = HERE / "data" / "images"


def learned_weights():
    """[(feature, weight)] sorted by magnitude — what the model leans on."""
    lr = MODEL.named_steps["logisticregression"]
    pairs = sorted(zip(FEATURES, lr.coef_[0]), key=lambda kv: -abs(kv[1]))
    return [{"name": n, "weight": round(float(w), 3)} for n, w in pairs]


def _score(text, invoice):
    """Features + confidence for an already-extracted invoice."""
    feats = compute_features(PROC, text, invoice)
    x = [[float(feats[f]) for f in FEATURES]]
    p_wrong = float(MODEL.predict_proba(x)[0][1])
    # What the cross-check compared against: the summary block's own gross total.
    # This is read from the document, so it exists for uploads too (unlike the
    # dataset's answer key) and explains where the confidence came from.
    verified = PROC.verified_total(text)
    return {
        "extracted": euro_to_float(invoice.total_amount) if invoice.total_amount else None,
        "extracted_raw": invoice.total_amount,   # as printed on the invoice
        "features": feats,
        "confidence": round(1.0 - p_wrong, 3),
        "summary_total": verified,
        "summary_values": [],
    }


def score_text(text):
    """Run the extractor + model on one invoice's text."""
    return _score(text, PROC.process_invoice_text(text))


def score_image(raw_bytes, suffix=".jpg"):
    """OCR an uploaded invoice photo, then run the extractor + model on it.

    Returns the same shape as score_text() plus the OCR text, so the page can
    show what was actually read off the photo.
    """
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = Path(tmp.name)
    try:
        invoice = PROC.process_invoice_file(tmp_path)
        text = invoice.raw_text or ""
        result = _score(text, invoice)
        result["ocr_text"] = text
        return result
    finally:
        tmp_path.unlink(missing_ok=True)


def load_samples(limit=40):
    """Held-out invoices with their true totals, for the picker and results table."""
    path = HERE / "data" / "mychen76_test.jsonl"
    if not path.exists():
        return []
    samples = []
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        gross = euro_to_float(row["summary"].get("total_gross_worth"))
        if gross is None:
            continue  # receipts with no annotation: no answer key, nothing to show
        net = euro_to_float(row["summary"].get("total_net_worth"))
        result = score_text(row["text"])
        extracted = result["extracted"]  # already a float; do not re-parse

        samples.append({
            "id": row["id"],
            "text": row["text"],
            # Show every money value the same way (parsed float), so the columns
            # are comparable; the dataset prints European format ('889,20').
            "extracted": extracted,
            "extracted_raw": result["extracted_raw"],
            "true_net": net,
            "true_gross": gross,
            "confidence": result["confidence"],
            # Matches the training labels: any legitimate summary total counts.
            "actually_ok": extracted is not None and (
                abs(extracted - gross) <= 0.02
                or (net is not None and abs(extracted - net) <= 0.02)),
            "has_image": (IMAGE_DIR / f"{row['id']}.jpg").is_file(),
            "matched": None,  # filled in below
        })
        s = samples[-1]
        s["matched"] = ("gross" if abs((extracted or -1) - gross) <= 0.02
                        else ("net" if net is not None and abs((extracted or -1) - net) <= 0.02
                              else None))
        if len(samples) >= limit:
            break
    return samples


def build_page(samples):
    data = {"metrics": METRICS, "weights": learned_weights(), "samples": samples}
    return PAGE_TEMPLATE.replace("__DATA__", json.dumps(data))


def make_handler(samples):
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
                self._send(200, build_page(samples), "text/html; charset=utf-8")
            elif parsed.path == "/image":
                self._serve_image(parse_qs(parsed.query).get("id", [""])[0])
            else:
                self._send(404, "not found", "text/plain")

        def _serve_image(self, image_id):
            path = (IMAGE_DIR / f"{image_id}.jpg").resolve()
            if IMAGE_DIR.resolve() not in path.parents or not path.is_file():
                self._send(404, "no image", "text/plain")
                return
            self._send(200, path.read_bytes(), "image/jpeg")

        def do_POST(self):
            if urlparse(self.path).path == "/score-image":
                length = int(self.headers.get("Content-Length", 0))
                blob = self.rfile.read(length)
                if not blob:
                    self._send(400, json.dumps({"error": "no image"}))
                    return
                try:
                    self._send(200, json.dumps(score_image(blob)))
                except Exception as exc:
                    self._send(500, json.dumps({"error": str(exc)}))
                return
            if urlparse(self.path).path != "/score":
                self._send(404, "not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
                text = payload.get("text", "")
            except ValueError:
                self._send(400, json.dumps({"error": "bad request"}))
                return
            self._send(200, json.dumps(score_text(text)))

    return Handler


def main():
    print("Loading held-out invoices and scoring them…")
    samples = load_samples()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(samples))
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"\nDemo ready at {url}\nPress Ctrl+C when you're done.")
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
<title>Invoice Confidence Model</title>
<style>
  :root { --bg:#0f1720; --card:#182634; --line:#26384a; --ink:#e7eef6; --muted:#9db2c8;
          --green:#2ec16b; --amber:#e8a33d; --red:#e2555a; --accent:#3b9ae1; --zero:#6b7f94; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font-family:-apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:20px 24px; border-bottom:1px solid var(--line); }
  header h1 { font-size:21px; margin:0 0 4px; }
  header p { margin:0; color:var(--muted); font-size:13px; }
  .wrap { padding:24px; max-width:1180px; margin:0 auto;
          display:flex; flex-direction:column; gap:22px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:20px; }
  .card h2 { font-size:15px; margin:0 0 4px; }
  .card .sub { color:var(--muted); font-size:12.5px; margin:0 0 16px; }

  /* headline numbers — a stat tile, not a chart */
  .tiles { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }
  @media (max-width:820px){ .tiles { grid-template-columns:repeat(2,1fr); } }
  .tile { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:16px 18px; }
  .tile .n { font-size:30px; font-weight:650; letter-spacing:-.5px; }
  .tile .l { color:var(--muted); font-size:12px; margin-top:2px; }

  /* live scorer */
  .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
  select, textarea, button { font-family:inherit; font-size:13.5px; }
  select, textarea { background:#0f1a25; color:var(--ink);
                     border:1px solid var(--line); border-radius:8px; padding:9px 11px; }
  select { min-width:260px; }
  textarea { width:100%; min-height:110px; margin-top:10px; resize:vertical; line-height:1.45; }
  button { background:var(--accent); color:#fff; border:none; border-radius:8px;
           padding:10px 16px; font-weight:600; cursor:pointer; }
  button:hover { filter:brightness(1.08); }
  button.ghost { background:transparent; border:1px solid var(--line); color:var(--muted); }
  .upload { background:var(--accent); color:#fff; border-radius:8px; padding:10px 16px;
            font-size:13.5px; font-weight:600; cursor:pointer; display:inline-block; }
  .upload:hover { filter:brightness(1.08); }

  .result { margin-top:16px; display:none; grid-template-columns:1fr 1fr; gap:18px; }
  .result.on { display:grid; grid-template-columns:0.9fr 1fr 1fr; }
  .imgpane { background:#0b1119; border:1px solid var(--line); border-radius:10px;
             padding:8px; max-height:430px; overflow:auto; display:none; }
  .imgpane.on { display:block; }
  .imgpane img { width:100%; border-radius:6px; display:block; }
  @media (max-width:980px){ .result.on { grid-template-columns:1fr; } }
  .kv { display:flex; justify-content:space-between; gap:10px; padding:6px 0;
        border-bottom:1px solid var(--line); font-size:13px; }
  .kv span:first-child { color:var(--muted); }
  .verdict { font-size:26px; font-weight:650; margin:2px 0 6px; }
  .meter { height:9px; border-radius:6px; background:#0b1119; overflow:hidden; }
  .meter > span { display:block; height:100%; border-radius:6px; transition:width .35s ease; }

  /* diverging weight bars: blue = pushes toward correct, red = pushes toward wrong */
  .wbar { display:grid; grid-template-columns:120px 1fr 62px; gap:10px;
          align-items:center; margin-bottom:7px; font-size:12.5px; }
  .wbar .nm { color:var(--muted); text-align:right; }
  .track { position:relative; height:19px; background:#0b1119; border-radius:5px; }
  .track .mid { position:absolute; left:50%; top:0; bottom:0; width:1px; background:var(--zero); }
  .track i { position:absolute; top:2px; bottom:2px; border-radius:4px; }
  .wbar .val { font-variant-numeric:tabular-nums; text-align:left; color:var(--ink); }
  .legend { display:flex; gap:16px; font-size:12px; color:var(--muted); margin:2px 0 14px; }
  .legend b { display:inline-block; width:10px; height:10px; border-radius:3px;
              margin-right:6px; vertical-align:-1px; }

  table { width:100%; border-collapse:collapse; font-size:12.5px; }
  th, td { text-align:left; padding:7px 8px; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:600; }
  td.num { font-variant-numeric:tabular-nums; }
  .pill { padding:3px 9px; border-radius:999px; font-size:11.5px; font-weight:600; }
  .ok { background:rgba(46,193,107,.15); color:var(--green); }
  .flag { background:rgba(226,85,90,.15); color:var(--red); }
  .agree { color:var(--green); } .miss { color:var(--amber); }
  .note { color:var(--muted); font-size:12.5px; line-height:1.6; }
  .caveat { margin:-8px 0 0; padding:11px 14px; border-radius:10px;
            background:rgba(232,163,61,.10); border:1px solid rgba(232,163,61,.35);
            color:var(--ink); font-size:12.5px; line-height:1.55; }
  .caveat b { color:var(--amber); }
</style>
</head>
<body>
<header>
  <h1>Invoice Confidence Model</h1>
  <p>A logistic-regression model that scores how much to trust the extracted <b>total</b> — learned from data, not hand-tuned.</p>
</header>

<div class="wrap">

  <div class="tiles" id="tiles"></div>
  <p class="caveat" id="caveat"></p>

  <div class="card">
    <h2>Live demo</h2>
    <p class="sub">Pick a held-out invoice the model has never seen — or paste your own text — and score it.</p>
    <div class="row">
      <select id="picker"></select>
      <button id="scoreBtn">Score it</button>
      <button class="ghost" id="clearBtn">Clear</button>
    </div>
    <div class="row" style="margin-top:10px">
      <label class="upload" for="photo">\U0001F4F7 Upload an invoice photo</label>
      <input type="file" id="photo" accept="image/*" hidden />
      <span class="note" id="photoNote">JPG/PNG. First run downloads the OCR model (~1 min).</span>
    </div>
    <textarea id="text" placeholder="…or paste invoice text here"></textarea>
    <div class="result" id="result">
      <div class="imgpane" id="imgpane"><img id="invimg" alt="invoice" /></div>
      <div>
        <div class="verdict" id="verdict"></div>
        <div class="meter"><span id="meter"></span></div>
        <div class="kv"><span>Extracted total</span><b id="extracted"></b></div>
        <div class="kv"><span id="lblCross">Cross-check found on the invoice</span><b id="crossRef"></b></div>
        <div class="kv" id="rowTruth"><span>Verified answer (dataset only)</span><b id="truthGross"></b></div>
      </div>
      <div>
        <div class="sub" style="margin:0 0 8px">Features the model saw</div>
        <div id="feats"></div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>What the model learned</h2>
    <p class="sub">Nobody typed these numbers — training found them. Longer bar = stronger influence.</p>
    <div class="legend">
      <span><b style="background:#3b9ae1"></b>pushes toward “trustworthy”</span>
      <span><b style="background:#e2555a"></b>pushes toward “wrong”</span>
    </div>
    <div id="weights"></div>
    <p class="note" style="margin-top:14px">
      The model is essentially a <b>“does this look like a real money value?”</b> detector: digit count,
      cents, and a currency dominate. Features that carried no signal were driven to ~0 automatically.
    </p>
  </div>

  <div class="card">
    <h2>Held-out predictions</h2>
    <p class="sub">Invoices excluded from training — the model's confidence next to what was actually true.</p>
    <table>
      <thead><tr>
        <th>Extracted</th><th>True net</th><th>True gross</th><th>Confidence</th>
        <th>Model says</th><th>Reality</th><th>Agree?</th>
      </tr></thead>
      <tbody id="rows"></tbody>
    </table>
  </div>

</div>

<script>
const DATA = __DATA__;
const $ = id => document.getElementById(id);
const pct = x => Math.round((x || 0) * 100);
const confColor = c => c >= 0.7 ? 'var(--green)' : c >= 0.4 ? 'var(--amber)' : 'var(--red)';
// one money format everywhere; the dataset itself prints European ('889,20')
const money = v => (v === null || v === undefined || v === '')
  ? '—' : '$' + Number(v).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});

/* headline numbers */
$('tiles').innerHTML = [
  [DATA.metrics.roc_auc.toFixed(3), 'ROC-AUC (held-out)'],
  [DATA.metrics.balanced_accuracy.toFixed(3), 'Balanced accuracy'],
  [`${DATA.metrics.train_rows} / ${DATA.metrics.train_wrong}`, 'Trained rows / failures'],
  [`${DATA.metrics.test_rows} / ${DATA.metrics.test_wrong}`, 'Held-out rows / failures'],
].map(([n, l]) => `<div class="tile"><div class="n">${n}</div><div class="l">${l}</div></div>`).join('');

/* honest caption: does one feature already explain the whole score? */
(() => {
  const m = DATA.metrics, solo = m.best_single_feature_auc, name = m.best_single_feature;
  if (solo == null) return;
  const collapses = Math.abs(solo - m.roc_auc) < 0.02;
  $('caveat').innerHTML = collapses
    ? `<b>Read this before quoting the number.</b> The feature <code>${name}</code> alone scores
       AUC ${solo.toFixed(3)} \u2014 matching the full model, so the arithmetic cross-check is doing
       the work, not the machine learning. On these invoices a rule is enough; the model would
       earn its keep where that cross-check is missing or noisy.`
    : `The strongest single feature, <code>${name}</code>, scores AUC ${solo.toFixed(3)} on its own,
       versus ${m.roc_auc.toFixed(3)} for the full model \u2014 so the model adds real signal beyond it.`;
})();

/* diverging weight bars, direct-labelled (identity never by color alone) */
const maxW = Math.max(...DATA.weights.map(w => Math.abs(w.weight)), 0.001);
$('weights').innerHTML = DATA.weights.map(w => {
  const frac = Math.abs(w.weight) / maxW * 50;           // half-width => diverging
  const toWrong = w.weight > 0;
  const style = toWrong
    ? `left:50%; width:${frac}%; background:#e2555a;`
    : `right:50%; width:${frac}%; background:#3b9ae1;`;
  return `<div class="wbar">
      <span class="nm">${w.name}</span>
      <span class="track"><i style="${style}"></i><span class="mid"></span></span>
      <span class="val">${w.weight > 0 ? '+' : ''}${w.weight.toFixed(2)}</span>
    </div>`;
}).join('');

/* held-out results table */
$('rows').innerHTML = DATA.samples.slice(0, 15).map(s => {
  const modelOk = s.confidence >= 0.5;
  const agree = modelOk === s.actually_ok;
  // highlight the gross total when the extraction matched it
  const hit = t => s.matched === t ? ' style="color:var(--green)"' : '';
  return `<tr>
    <td class="num">${money(s.extracted)}</td>
    <td class="num"${hit('net')}>${money(s.true_net)}</td>
    <td class="num"${hit('gross')}>${money(s.true_gross)}</td>
    <td class="num" style="color:${confColor(s.confidence)}">${pct(s.confidence)}%</td>
    <td><span class="pill ${modelOk ? 'ok' : 'flag'}">${modelOk ? 'trust' : 'review'}</span></td>
    <td>${s.actually_ok ? `matched ${s.matched}` : 'was wrong'}</td>
    <td class="${agree ? 'agree' : 'miss'}">${agree ? '✓ agreed' : '✗ missed'}</td>
  </tr>`;
}).join('');

/* live scorer */
$('picker').innerHTML = '<option value="">— choose a held-out invoice —</option>' +
  DATA.samples.map((s, i) => `<option value="${i}">#${s.id} · net ${money(s.true_net)} · gross ${money(s.true_gross)}</option>`).join('');

$('picker').addEventListener('change', e => {
  const s = DATA.samples[e.target.value];
  if (s) { $('text').value = s.text; show(s, {extracted: s.extracted, confidence: s.confidence, features: null}); }
});

$('clearBtn').addEventListener('click', () => {
  $('text').value = ''; $('picker').value = ''; $('result').classList.remove('on');
});

$('scoreBtn').addEventListener('click', async () => {
  const text = $('text').value.trim();
  if (!text) return;
  const res = await fetch('/score', {method:'POST', headers:{'Content-Type':'application/json'},
                                     body: JSON.stringify({text})});
  const out = await res.json();
  const picked = DATA.samples[$('picker').value];
  show(picked, out);
});

$('photo').addEventListener('change', async e => {
  const file = e.target.files[0];
  if (!file) return;
  $('photoNote').textContent = 'Reading the photo with OCR… this can take a moment.';
  const res = await fetch('/score-image', {method:'POST', body: file});
  const out = await res.json();
  if (out.error) { $('photoNote').textContent = 'Error: ' + out.error; return; }
  $('photoNote').textContent = 'OCR done — scored below.';
  $('text').value = out.ocr_text || '';
  $('picker').value = '';
  show(null, out, URL.createObjectURL(file));
});

function show(sample, out, uploadedUrl) {
  const c = out.confidence;
  $('verdict').textContent = c >= 0.5 ? 'Trust this total' : 'Send to review';
  $('verdict').style.color = confColor(c);
  $('meter').style.width = pct(c) + '%';
  $('meter').style.background = confColor(c);
  const raw = out.extracted ?? out.extracted_raw;
  $('extracted').textContent = raw == null ? '— none found —'
    : `${money(sample ? sample.extracted : raw)}  (read as “${out.extracted_raw ?? raw}”)`;
  // What the cross-check compared against — read from the document itself, so
  // it is available for uploads too. This is what drives summary_gap.
  const f = out.features || {};
  if (out.summary_total != null) {
    $('crossRef').textContent = money(out.summary_total)
      + (out.summary_values && out.summary_values.length > 1
         ? '  (from ' + out.summary_values.map(money).join(' / ') + ')' : '');
    $('crossRef').style.color = 'var(--ink)';
  } else {
    $('crossRef').textContent = 'no summary block on this invoice — nothing to check against';
    $('crossRef').style.color = 'var(--amber)';
  }
  // The dataset's answer key only exists for the held-out invoices.
  $('rowTruth').style.display = sample ? '' : 'none';
  if (sample) $('truthGross').textContent =
      money(sample.true_gross) + '  (net ' + money(sample.true_net) + ')';
  if (out.features) {
    $('feats').innerHTML = Object.entries(out.features)
      .map(([k, v]) => `<div class="kv"><span>${k}</span><b>${v}</b></div>`).join('');
  } else {
    $('feats').innerHTML = '<div class="note">Click <b>Score it</b> to compute the features.</div>';
  }
  const pane = $('imgpane');
  if (uploadedUrl) {
    $('invimg').src = uploadedUrl;
    pane.classList.add('on');
  } else if (sample && sample.has_image) {
    $('invimg').src = '/image?id=' + encodeURIComponent(sample.id);
    pane.classList.add('on');
  } else {
    pane.classList.remove('on');
  }
  $('result').classList.add('on');
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
