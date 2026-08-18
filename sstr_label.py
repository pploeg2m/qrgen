#!/usr/bin/env python3
"""SSTR label tool. Three jobs: display data, store/correct labels, switch mode.

Cursor walks ORDER. Display shows image + label:
  - labeled page  -> shows stored label
  - unlabeled     -> shows suggestion: "0" (LABEL mode) or classifier count (APPLY mode)
Input:
  n / ->  next        p / <-  prev      (navigate only, no store)
  number  set label + advance
  c       accept the shown suggestion as label + advance
  Apply   toggle LABEL <-> APPLY (classifier suggests; confirm/correct builds train set)
  Finish & train  -> data4/sstr_model.json
  python projects/qrgen/sstr_label.py --port 7872
"""
import argparse, glob, json, sys
from collections import defaultdict
from pathlib import Path
from PIL import Image
import gradio as gr
import cv2
sys.path.insert(0, "projects/qrgen")
import sstr_detect as D

ROOT = Path("data4")
CPATH = ROOT / "sstr_counts.json"
COUNTS = json.loads(CPATH.read_text()) if CPATH.exists() else {}
MODEL = json.loads((ROOT/"sstr_model.json").read_text())["params"] if (ROOT/"sstr_model.json").exists() else {}
import numpy as np
from sstr_detect import PX_PER_CM
PPATH = ROOT / "sstr_positions.json"
POS = json.loads(PPATH.read_text()) if PPATH.exists() else {}
RPATH = ROOT / "sstr_reviewed.json"                       # sweep: pages confirmed done
REVIEWED = json.loads(RPATH.read_text()) if RPATH.exists() else []
IGPATH = ROOT / "sstr_ignore.json"                        # Schmierblatt / no-mans-land pages
IGNORE = json.loads(IGPATH.read_text()) if IGPATH.exists() else []
SWEEP = "--sweep" in sys.argv                             # per-student combined pass
DISP_W = 700


def _saver():
    RPATH.write_text(json.dumps(REVIEWED))


def _savei():
    IGPATH.write_text(json.dumps(IGNORE))


def _mark_reviewed(key):
    if key not in REVIEWED:
        REVIEWED.append(key); _saver()


import re, pytesseract


def _savep():
    PPATH.write_text(json.dumps(POS))


def _next_label(last):
    if not last:
        return "a"
    m = re.match(r"^(.*?)([a-f])$", last)          # trailing letter -> a->b
    if m and m.group(2) < "f":
        return m.group(1) + chr(ord(m.group(2)) + 1)
    m = re.match(r"^(\d+)$", last)                  # pure number -> +1
    if m:
        return str(int(m.group(1)) + 1)
    return last


def ocr_label(g, yline, fx0, icd):
    """OCR the left region (frame..ICD+bit) around a line for a subtask label."""
    x0 = max(0, fx0 - 5); x1 = min(g.shape[1], icd + int(0.4 * (icd - fx0)) + 8)
    h = int(0.7 * PX_PER_CM); y0 = max(0, yline - h); y1 = min(g.shape[0], yline + h)
    crop = g[y0:y1, x0:x1]
    if crop.size == 0 or (x1 - x0) > 6 * (y1 - y0):
        return None
    up = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    try:
        txt = pytesseract.image_to_string(up, config="--psm 6 -c tessedit_char_whitelist=Aufgabe0123456789.)abcdef")
    except Exception:
        return None
    t = txt.replace("Aufgabe", "").replace(" ", "").replace("\n", "")
    m = re.search(r"(\d+(?:\.\d+)?)?\)?([a-f])\)?", t) or re.search(r"(\d+(?:\.\d+)?)\)", t)
    if not m:
        return None
    num = m.group(1) or ""; let = (m.group(2) if m.lastindex and m.re.groups >= 2 else "") or ""
    return (num + let) or None


def find_line(g, xf, yf, fx0, fx1):
    """Click -> nearby ink -> CC -> LMS-fit line. If CC spans >=65% of the frame
    width, extend the drawn line left-frame..right-frame; else keep CC extent."""
    H, W = g.shape
    # 1) seed y from the click
    b0 = int(0.5 * PX_PER_CM); R = int(0.18 * PX_PER_CM)
    y0 = max(0, yf - b0); y1 = min(H, yf + b0)
    seed = (g[y0:y1, :] < 160).astype(np.uint8) * 255
    seed = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(0.08*W)), 3)))
    nl, lab = cv2.connectedComponents(seed)
    sy = yf - y0
    lbl = lab[sy, xf] if (0 <= sy < seed.shape[0] and 0 <= xf < W) else 0
    if lbl == 0:
        wn = lab[max(0, sy-R):sy+R, max(0, xf-R):min(W, xf+R)]; nz = wn[wn > 0]
        lbl = int(nz[0]) if nz.size else 0
    if lbl == 0:
        return None
    yc = int(np.where(lab == lbl)[0].mean()) + y0
    # 2) thin horizontal stripe around yc  + 3) aggressive dilation
    sb = int(0.30 * PX_PER_CM); sy0 = max(0, yc - sb); sy1 = min(H, yc + sb)
    strip = (g[sy0:sy1, :] < 165).astype(np.uint8) * 255
    strip = cv2.dilate(strip, cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, int(0.20*W)), 3)))
    strip = cv2.morphologyEx(strip, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.20*W), 1)))
    # 4) longest left->right CC
    n, lab2, stats, _ = cv2.connectedComponentsWithStats(strip, connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax([stats[k, cv2.CC_STAT_WIDTH] for k in range(1, n)]))
    if stats[i, cv2.CC_STAT_WIDTH] < 0.3 * W:
        return None
    ys, xs = np.where(lab2 == i); ysf = ys + sy0
    # 5) LMS-approximate that CC
    m, b = np.polyfit(xs.astype(float), ysf.astype(float), 1)
    ccw = xs.max() - xs.min(); span = max(1, fx1 - fx0)
    x0e, x1e = (int(fx0), int(fx1)) if ccw >= 0.65 * span else (int(xs.min()), int(xs.max()))
    return {"y_frac": round(float(ysf.mean())/H, 4), "x0": x0e, "x1": x1e, "m": float(m), "b": float(b)}

_ba = defaultdict(list)                                   # author -> [(sheet, side_rank, key)]
for p in sorted(glob.glob("data4/runs/WiMa_SS26_*/*/oriented.png")):
    lf = Path(p).parent
    eg = lf / "empty_gate.json"
    if eg.exists() and json.loads(eg.read_text()).get("is_empty"):
        continue
    q = lf / "qr.json"
    qd = json.loads(q.read_text()) if q.exists() else {}
    a = qd.get("booklet_id") or "?"
    srank = 0 if qd.get("side") == "front" else 1
    key = f"{lf.parent.name.replace('WiMa_SS26_','')}/{lf.name}"
    _ba[a].append((qd.get("sheet") or 99, srank, key))
if SWEEP:                                                 # per-student booklet order
    ORDER = [k for a in sorted(_ba) for (_, _, k) in sorted(_ba[a])]
else:                                                     # round-robin across authors
    for a in _ba:
        _ba[a].sort()
    ORDER, _i = [], {a: 0 for a in _ba}
    while any(_i[a] < len(_ba[a]) for a in _ba):
        for a in sorted(_ba):
            if _i[a] < len(_ba[a]):
                ORDER.append(_ba[a][_i[a]][2]); _i[a] += 1

# sweep resumes at the first page not yet reviewed
_start = next((j for j, k in enumerate(ORDER) if k not in REVIEWED), 0) if SWEEP else 0
S = {"pos": _start, "apply": False, "sug": 0}


def _save():
    CPATH.write_text(json.dumps(COUNTS, indent=0))


def _compute():
    """Load+deskew+geometry for the current page; cache the clean base (once)."""
    key = ORDER[S["pos"]]; s, n = key.split("/")
    g = cv2.imread(str(ROOT/"runs"/f"WiMa_SS26_{s}"/n/"oriented.png"), cv2.IMREAD_GRAYSCALE)
    p = D.prepare(g); gd = p["g"]; S["g"] = gd; H, W = gd.shape; S["H"], S["W"] = H, W
    S["scale"] = W / DISP_W; S["fx0"] = p["frame"][0]; S["fx1"] = p["frame"][2]; S["icd"] = p["icd_x"]
    S["base"] = cv2.cvtColor(gd, cv2.COLOR_GRAY2BGR)
    S["applylines"] = ([(p["icd_x"], ln["y_px"], p["frame"][2]) for ln in D.count_lines(p, **MODEL)]
                       if S["apply"] and MODEL else [])


def _render():
    """Fast redraw from cached base — overlays only, no file read / deskew."""
    if S["pos"] >= len(ORDER) or "base" not in S:
        return None, "### end of set"
    key = ORDER[S["pos"]]; H, W = S["H"], S["W"]
    vis = S["base"].copy()
    for (x0, y, x1) in S.get("applylines", []):
        cv2.line(vis, (x0, y), (x1, y), (0, 180, 0), 4)
    for L in POS.get(key, []):
        yy = int(L["m"]*L["x0"]+L["b"]) if "m" in L else int(L["y_frac"]*H)
        yy2 = int(L["m"]*L["x1"]+L["b"]) if "m" in L else yy
        cv2.line(vis, (L["x0"], yy), (L["x1"], yy2), (0, 0, 255), 6)
        if L.get("label"):
            cv2.putText(vis, L["label"], (L["x0"]+8, yy-12), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4)
    labeled = COUNTS.get(key)
    rev = key in REVIEWED; ign = key in IGNORE
    S["sug"] = labeled if labeled is not None else (len(S.get("applylines", [])) if (S["apply"] and MODEL) else 0)
    npk = len(POS.get(key, []))
    if SWEEP:
        if ign:
            banner = "IGNORE — Schmierblatt (not graded)"; col = (150, 0, 150)
        else:
            banner = (("✓ REVIEWED " if rev else "• unreviewed ")) + f"picked:{npk}"
            col = (0, 130, 0) if rev else (200, 0, 0)
    else:
        banner = (f"LABEL: {labeled}" if labeled is not None else f"suggest: {S['sug']}") + f"  picked:{npk}"
        col = (0, 130, 0) if labeled is not None else (200, 0, 0)
    cv2.rectangle(vis, (0, 0), (1200, 130), (255, 255, 255), -1)
    cv2.putText(vis, banner, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.6, col, 7)
    im = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)).resize((DISP_W, int(H * DISP_W / W)))
    if SWEEP:
        state = "🚫 IGNORE (Schmierblatt)" if ign else ("✓ done" if rev else "press space when done")
        st = (f"### [{S['pos']+1}/{len(ORDER)}] {key} — SWEEP\n"
              f"{npk} SSTR picked  ·  reviewed **{len(REVIEWED)}/{len(ORDER)}**  "
              f"·  ignored {len(IGNORE)}  ·  {state}")
    else:
        mode = "APPLY" if S["apply"] else "LABEL"
        tag = f"label = **{labeled}**" if labeled is not None else f"suggested = **{S['sug']}** (press c to accept)"
        st = f"### [{S['pos']+1}/{len(ORDER)}] {key} — {mode}\n{tag}  ·  stored: {len(COUNTS)}"
    return im, st


def _display():
    if S["pos"] >= len(ORDER):
        return None, "### end of set"
    _compute(); return _render()


def _store(v):
    if S["pos"] < len(ORDER):
        k = ORDER[S["pos"]]; COUNTS[k] = v; _save(); _mark_reviewed(k); S["pos"] += 1
    return _display()


def _done():
    """Sweep: confirm this page (picks are correct / none needed) + advance."""
    if S["pos"] < len(ORDER):
        _mark_reviewed(ORDER[S["pos"]])
    S["pos"] = min(len(ORDER) - 1, S["pos"] + 1)
    return _display()


def _next_unreviewed():
    for j in range(S["pos"] + 1, len(ORDER)):
        if ORDER[j] not in REVIEWED:
            S["pos"] = j; break
    return _display()


def _ignore():
    """Sweep: mark this page Schmierblatt / no-mans-land (not graded) + advance.
    Toggles: press again on an ignored page to un-ignore it."""
    if S["pos"] < len(ORDER):
        k = ORDER[S["pos"]]
        if k in IGNORE:
            IGNORE.remove(k); _savei(); return _display()   # un-ignore, stay
        IGNORE.append(k); _savei(); _mark_reviewed(k)
        S["pos"] = min(len(ORDER) - 1, S["pos"] + 1)
    return _display()


def _set(v):
    return _store(v)


def _confirm():
    return _store(S["sug"])


def _next():
    S["pos"] = min(len(ORDER) - 1, S["pos"] + 1); return _display()


def _prev():
    S["pos"] = max(0, S["pos"] - 1); return _display()


def _click(evt: gr.SelectData):
    if S["pos"] >= len(ORDER) or "g" not in S:
        im, st = _display(); return im, st, ""
    x, y = evt.index; xf = int(x * S["scale"]); yf = int(y * S["scale"])
    key = ORDER[S["pos"]]; lst = POS.setdefault(key, [])
    tol = int(0.22 * PX_PER_CM)
    for j, L in enumerate(lst):                         # click on a line = remove it
        if abs(int(L["y_frac"] * S["H"]) - yf) < tol and L["x0"] - tol <= xf <= L["x1"] + tol:
            lst.pop(j); _savep(); _mark_reviewed(key); im, st = _render(); return im, st, ""
    L = find_line(S["g"], xf, yf, S["fx0"], S["fx1"])    # else add
    if L:
        lab = ocr_label(S["g"], int(L["y_frac"]*S["H"]), S["fx0"], S["icd"]) or \
              _next_label(lst[-1].get("label") if lst else S.get("last_label", ""))
        L["label"] = lab; S["last_label"] = lab
        lst.append(L); _savep(); _mark_reviewed(key)
        im, st = _render(); return im, st, lab
    im, st = _render(); return im, st, ""


def _setlabel(v):
    key = ORDER[S["pos"]] if S["pos"] < len(ORDER) else None
    if key and POS.get(key):
        POS[key][-1]["label"] = (v or "").strip(); _savep()
    return _render()


def _jump(v):
    S["pos"] = max(0, min(len(ORDER) - 1, int(v) - 1)); return _display()


def _apply():
    if not MODEL:
        im, _ = _display(); return im, "### train a classifier first (no sstr_model.json)", gr.update()
    S["apply"] = not S["apply"]; im, st = _display()
    return im, st, gr.update(value=("🟢 Apply: ON" if S["apply"] else "🔁 Apply: OFF"),
                             variant=("primary" if S["apply"] else "secondary"))


def _train():
    global MODEL
    lab = {k: (6 if str(v) == ">5" else int(v)) for k, v in COUNTS.items()}
    if len(lab) < 5:
        yield "### need more labels"; return
    yield f"### training… (grid-search over {len(lab)} labels, ~15s)"
    preps = {}
    for k in lab:
        s, n = k.split("/")
        preps[k] = D.prepare(cv2.imread(str(ROOT/"runs"/f"WiMa_SS26_{s}"/n/"oriented.png"), cv2.IMREAD_GRAYSCALE))
    best = None
    for td in (120, 140, 160, 180):
        for sp in (0.6, 0.7, 0.8, 0.9):
            for cf in (0.04, 0.08):
                e = x = 0
                for k, y in lab.items():
                    c = len(D.count_lines(preps[k], T_DARK=td, SPAN_MIN=sp, CLOSE_FRAC=cf))
                    e += abs(c-y); x += (c == y)
                mae = e/len(lab)
                if best is None or (mae, -x) < (best[0], -best[1]):
                    best = (mae, x, {"T_DARK": td, "SPAN_MIN": sp, "CLOSE_FRAC": cf})
    mae, x, params = best
    (ROOT/"sstr_model.json").write_text(json.dumps({"params": params, "mae": round(mae, 3), "exact": x, "n": len(lab)}, indent=2))
    MODEL = params
    yield f"### ✅ trained (done)  MAE={mae:.2f}  exact={x}/{len(lab)}  {params}"


HEAD = ("<script>document.addEventListener('keydown',function(e){"
        "if(e.target&&(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA'||e.target.isContentEditable))return;"
        "var c=function(id){var el=document.getElementById(id);if(!el)return;"
        "var b=el.tagName==='BUTTON'?el:el.querySelector('button');(b||el).click();};"
        "var k=e.key,hit=true;"
        "if(k==='n'||k==='ArrowRight')c('nextb');"
        "else if(k==='p'||k==='ArrowLeft')c('prevb');"
        "else if(k===' '||k==='d')c('doneb');"
        "else if(k==='x')c('ignb');"
        "else if(k==='u')c('unrevb');"
        "else if(k==='c')c('confirmb');"
        "else if('012345'.indexOf(k)>=0)c('c'+k);"
        "else hit=false;"
        "if(hit){e.preventDefault();e.stopPropagation();"
        "if(document.activeElement&&document.activeElement.blur)document.activeElement.blur();}"
        "},true);</script>")


def build():
    with gr.Blocks(title="SSTR labeler") as demo:
        gr.Markdown("## SSTR sweep — add/move lines + label (space=done, u=next unreviewed)"
                    if SWEEP else "## SSTR count labeler")
        with gr.Row():
            with gr.Column(scale=3):
                img = gr.Image(type="pil", height=900, show_label=False)
            with gr.Column(scale=2):
                st = gr.Markdown()
                lblbox = gr.Textbox(label="last line label (edit + Enter)", scale=1)
                img.select(_click, None, [img, st, lblbox])
                lblbox.submit(_setlabel, lblbox, [img, st])
                with gr.Row():
                    for v in [0, 1, 2, 3, 4, 5, ">5"]:
                        gr.Button(str(v), elem_id="c"+(str(v) if v != ">5" else "m")).click(
                            (lambda v=v: _set(v)), None, [img, st])
                with gr.Row():
                    gr.Button("⬅ prev (p)", elem_id="prevb").click(_prev, None, [img, st])
                    gr.Button("next ➡ (n)", elem_id="nextb").click(_next, None, [img, st])
                    gr.Button("✓ confirm (c)", elem_id="confirmb", variant="primary").click(_confirm, None, [img, st])
                with gr.Row():
                    gr.Button("✓ done + next (space)", elem_id="doneb", variant="primary").click(_done, None, [img, st])
                    gr.Button("🚫 ignore/Schmierblatt (x)", elem_id="ignb").click(_ignore, None, [img, st])
                    gr.Button("↷ next unreviewed (u)", elem_id="unrevb").click(_next_unreviewed, None, [img, st])
                tout = gr.Markdown()
                with gr.Row():
                    ab = gr.Button("🔁 Apply: OFF")
                    ab.click(_apply, None, [img, st, ab])
                    gr.Button("🎓 Finish & train", variant="stop").click(_train, None, tout)
                sl = gr.Slider(1, len(ORDER), value=1, step=1, label="jump to page")
                sl.release(_jump, sl, [img, st])
        demo.load(_display, None, [img, st])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=7872)
    ap.add_argument("--sweep", action="store_true", help="per-student combined add/move+label pass")
    build().launch(server_name="127.0.0.1", server_port=ap.parse_args().port, head=HEAD)
