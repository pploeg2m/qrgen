#!/usr/bin/env python3
"""Precision-labeling pass for Schlussstrich detections (data4).

Shows one detected line at a time (red on the page + a zoomed strip) and you
mark it Real Schlussstrich vs Not (frame / underline / cancellation / grid).
Live precision = real / (real+not). Sample a few dozen for a precision estimate.

  python projects/qrgen/schlussstrich_label.py --port 7871

Labels -> data4/schlussstrich_labels.json  { "<set>/<nnn>#<i>": 1|0 }
"""
import argparse, glob, json, random
from pathlib import Path
import numpy as np
from PIL import Image
import gradio as gr

ROOT = Path("data4")
LABELS_PATH = ROOT / "schlussstrich_labels.json"
LABELS = json.loads(LABELS_PATH.read_text()) if LABELS_PATH.exists() else {}

# flat list of all detections
DETS = []
for p in glob.glob("data4/runs/WiMa_SS26_*/*/schlussstrich.json"):
    leaf = Path(p).parent
    ss = (json.loads(Path(p).read_text()).get("schlussstriche") or [])
    for i, d in enumerate(ss):
        DETS.append({"key": f"{leaf.parent.name.replace('WiMa_SS26_','')}/{leaf.name}#{i}",
                     "leaf": leaf, "i": i, "det": d, "all": ss})
random.seed(42); random.shuffle(DETS)
BY_KEY = {d["key"]: d for d in DETS}
_cur = {"key": None}


def _save():
    LABELS_PATH.write_text(json.dumps(LABELS, indent=0))


def _next():
    for d in DETS:
        if d["key"] not in LABELS:
            return d["key"]
    return None


def _imgs(key):
    d = BY_KEY[key]
    import cv2
    im = cv2.imread(str(d["leaf"] / "oriented.png")); H, W = im.shape[:2]
    for j, o in enumerate(d["all"]):                       # context: other lines faint yellow
        if j == d["i"]:
            continue
        yy = int(float(o.get("y_frac", 0)) * H)
        cv2.line(im, (0, yy), (W, yy), (0, 200, 200), 2)
    y = int(float(d["det"].get("y_frac", 0)) * H)
    cv2.line(im, (0, y), (W, y), (0, 0, 255), 5)           # target line red
    page = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); page.thumbnail((560, 780))
    y0, y1 = max(0, int(y-0.05*H)), min(H, int(y+0.05*H))
    strip = cv2.cvtColor(im[y0:y1, :], cv2.COLOR_BGR2RGB)
    strip = Image.fromarray(strip); strip.thumbnail((900, 200))
    return page, strip


def _status(key):
    d = BY_KEY[key]["det"]
    real = sum(LABELS.values()); tot = len(LABELS); notr = tot - real
    prec = f"{100*real/tot:.0f}%" if tot else "—"
    lines = [f"### {key}",
             f"det: y={d.get('y_frac')}, drawn={d.get('drawn')}, full_width={d.get('reaches_full_width')}, "
             f"label_above={d.get('label_above')}, conf={d.get('confidence')}",
             f"labeled: **{tot}**  (real={real}, not={notr})  ·  **precision so far = {prec}**  ·  detections total={len(DETS)}"]
    return "\n\n".join(lines)


def _advance(label):
    k = _cur["key"]
    if k is not None and label is not None:
        LABELS[k] = label; _save()
    nk = _next(); _cur["key"] = nk
    if nk is None:
        return None, None, "### ✅ All detections labeled."
    p, s = _imgs(nk)
    return p, s, _status(nk)


def build():
    with gr.Blocks(title="Schlussstrich precision") as demo:
        gr.Markdown("## Schlussstrich precision labeling — is the RED line a real Schlussstrich?")
        with gr.Row():
            with gr.Column(scale=3):
                page = gr.Image(type="pil", height=760, show_label=False)
            with gr.Column(scale=2):
                strip = gr.Image(type="pil", show_label=True, label="zoom around the line")
                status = gr.Markdown()
                b_real = gr.Button("✅  Real Schlussstrich", variant="primary")
                b_not = gr.Button("❌  Not (frame / underline / cancel / grid)")
                b_skip = gr.Button("⏭  Skip")
        b_real.click(lambda: _advance(1), None, [page, strip, status])
        b_not.click(lambda: _advance(0), None, [page, strip, status])
        b_skip.click(lambda: _advance(None), None, [page, strip, status])
        demo.load(lambda: _advance(None), None, [page, strip, status])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=7871)
    a = ap.parse_args()
    build().launch(server_name="127.0.0.1", server_port=a.port)
