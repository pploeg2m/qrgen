#!/usr/bin/env python3
"""Active-learning labeler for EMPTY (grid-only) pages — data4.

ASYMMETRIC COST: skipping a page with student ink is unacceptable; sending a
blank page to OCR is merely wasteful. So this is a ONE-SIDED, high-precision
gate, not balanced accuracy:
  * cost-weighted LogisticRegression (ink misclassification penalised),
  * a CONSERVATIVE threshold pinned so ZERO labeled ink-page is predicted empty
    (+ margin) → guaranteed 100% ink-recall on the labeled set,
  * active learning AUDITS THE SKIP ZONE (samples the pages it would skip)
    interleaved with boundary cases.

  python projects/qrgen/label_empty_al.py --port 7870

Labels persist to data4/empty_labels.json  { "<set>/<nnn>": 0|1 }  (1 = empty).
"""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image
import gradio as gr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

ROOT = Path("data4")
FEATS = json.loads((ROOT / "empty_features.json").read_text())
KEYS = sorted(FEATS)
LABELS_PATH = ROOT / "empty_labels.json"
LABELS = json.loads(LABELS_PATH.read_text()) if LABELS_PATH.exists() else {}
FNAMES = ["resid_ink%", "n_comp", "maxcomp‰", "band_cv", "dark%"]

INK_WEIGHT = 8.0          # penalise calling an ink page "empty"
SAFETY_MARGIN = 0.02      # extra p-margin above the worst ink page
_MODEL = {"scaler": None, "clf": None, "thr": None, "ink_recall": None, "acc": None}
_SEED_FRACS = [0.0, 1.0, 0.5, 0.25, 0.75, 0.12, 0.62, 0.37, 0.87, 0.06, 0.94]
_cur = {"key": None}


def _save():
    LABELS_PATH.write_text(json.dumps(LABELS, indent=0))


def _refit():
    X = np.array([FEATS[k] for k in LABELS]); y = np.array([LABELS[k] for k in LABELS])
    if len(y) < 6 or len(set(y.tolist())) < 2:
        _MODEL.update(scaler=None, clf=None, thr=None, ink_recall=None, acc=None); return
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000, class_weight={0: INK_WEIGHT, 1: 1.0}).fit(sc.transform(X), y)
    p = clf.predict_proba(sc.transform(X))[:, 1]          # p(empty)
    ink_p = p[y == 0]
    # conservative cutoff: strictly above the worst (highest-p-empty) ink page
    thr = min(0.999, float(ink_p.max()) + SAFETY_MARGIN) if len(ink_p) else 0.5
    ink_recall = float((ink_p < thr).mean()) if len(ink_p) else 1.0   # =1.0 by construction
    try:
        k = min(5, int(np.bincount(y).min()))
        acc = float(cross_val_score(clf, sc.transform(X), y, cv=max(2, k)).mean()) if k >= 2 else None
    except Exception:
        acc = None
    _MODEL.update(scaler=sc, clf=clf, thr=thr, ink_recall=ink_recall, acc=acc)


def _proba(keys):
    if _MODEL["clf"] is None:
        return None
    return _MODEL["clf"].predict_proba(_MODEL["scaler"].transform([FEATS[k] for k in keys]))[:, 1]


def _safe_skip_count():
    """Pages predicted empty at the conservative threshold (labeled empties + safe unlabeled)."""
    if _MODEL["clf"] is None:
        return None
    thr = _MODEL["thr"]
    unl = [k for k in KEYS if k not in LABELS]
    pe = int((_proba(unl) >= thr).sum()) if unl else 0
    return pe + sum(LABELS.values())


def _next_key():
    unl = [k for k in KEYS if k not in LABELS]
    if not unl:
        return None
    if _MODEL["clf"] is not None:
        p = _proba(unl)
        # interleave: audit the skip zone (highest p_empty) / refine boundary (|p-thr|)
        if len(LABELS) % 2 == 0:
            return unl[int(np.argmax(p))]
        return unl[int(np.argmin(np.abs(p - _MODEL["thr"])))]
    unl_sorted = sorted(unl, key=lambda k: FEATS[k][0])   # seed: spread over resid_ink
    frac = _SEED_FRACS[len(LABELS) % len(_SEED_FRACS)]
    return unl_sorted[min(len(unl_sorted) - 1, int(frac * (len(unl_sorted) - 1)))]


def _img(key):
    s, n = key.split("/")
    im = Image.open(ROOT / "runs" / s / n / "oriented.png").convert("RGB")
    im.thumbnail((720, 1000))
    return im


def _status(key):
    n = len(LABELS); ne = sum(LABELS.values()); ni = n - ne
    fv = ", ".join(f"{a}={b}" for a, b in zip(FNAMES, FEATS[key]))
    lines = [f"### {key}", f"labeled: **{n}**  (empty={ne}, ink={ni})  ·  remaining={len(KEYS)-n}",
             f"features: {fv}"]
    if _MODEL["clf"] is not None:
        p = float(_proba([key])[0]); thr = _MODEL["thr"]
        flag = "🔒 would SKIP (≥thr)" if p >= thr else "→ would send to OCR"
        acc = f"{_MODEL['acc']:.0%}" if _MODEL["acc"] is not None else "n/a"
        lines += [f"p(empty)={p:.2f}   thr={thr:.2f}   {flag}",
                  f"**safe-skip ≈ {_safe_skip_count()}/{len(KEYS)}**  ·  ink-recall={_MODEL['ink_recall']:.0%}  ·  CV acc={acc}"]
    else:
        lines.append("_need ≥6 labels (both classes) before the gate trains_")
    return "\n\n".join(lines)


def _advance(label):
    k = _cur["key"]
    if k is not None and label is not None:
        LABELS[k] = label; _save(); _refit()
    nk = _next_key(); _cur["key"] = nk
    if nk is None:
        return None, "### ✅ All pages labeled."
    return _img(nk), _status(nk)


def _report():
    _refit()
    if _MODEL["clf"] is None:
        return "Need ≥6 labels (both classes) first."
    ss = _safe_skip_count()
    coef = dict(zip(FNAMES, np.round(_MODEL["clf"].coef_[0], 2).tolist()))
    acc = f"{_MODEL['acc']:.0%}" if _MODEL["acc"] is not None else "n/a"
    return (f"**Safe-skip (grid-only) = {ss}/{len(KEYS)}** → ~{100*ss/len(KEYS):.0f}% of pages skip cloud OCR.\n\n"
            f"threshold p(empty) ≥ {_MODEL['thr']:.2f} (pinned above worst labeled ink page + {SAFETY_MARGIN} margin)\n\n"
            f"guaranteed ink-recall on labels = {_MODEL['ink_recall']:.0%}  ·  CV acc = {acc}  ·  labels = {len(LABELS)}\n\n"
            f"coef = {coef}")


def build():
    with gr.Blocks(title="Empty-page gate labeler") as demo:
        gr.Markdown("## Empty (grid-only) gate — active learning (high-precision / never-skip-ink)")
        with gr.Row():
            with gr.Column(scale=3):
                img = gr.Image(type="pil", height=760, show_label=False)
            with gr.Column(scale=2):
                status = gr.Markdown()
                b_empty = gr.Button("✅  Yes — EMPTY (grid only)", variant="primary")
                b_ink = gr.Button("✏️  Has ink (not empty)")
                b_skip = gr.Button("⏭  Skip")
                b_rep = gr.Button("📊  Fit & report")
                rep = gr.Markdown()
        b_empty.click(lambda: _advance(1), None, [img, status])
        b_ink.click(lambda: _advance(0), None, [img, status])
        b_skip.click(lambda: _advance(None), None, [img, status])
        b_rep.click(_report, None, rep)
        demo.load(lambda: _advance(None), None, [img, status])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=7870)
    a = ap.parse_args()
    _refit()
    build().launch(server_name="127.0.0.1", server_port=a.port)
