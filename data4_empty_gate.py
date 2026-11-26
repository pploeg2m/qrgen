#!/usr/bin/env python3
"""Batch empty(grid-only) gate — the OCR cost gate for the QR template.

Replaces the shipped ensemble (which scores 0% recall on this template) with a
cost-weighted logistic gate on the masked OpenCV features (see
empty_features_v2.py). Decision: SKIP cloud OCR iff p(empty) >= threshold,
where threshold is pinned above the worst labeled ink page (never-skip-ink).

  train  : fit on empty_labels.json + empty_features_v2.json -> empty_gate_model.json
  apply  : write <leaf>/empty_gate.json {is_empty, p_empty} for every page
  page   : classify_page(gray) for live pipeline use (does masking+features)

Usage:  python projects/qrgen/data4_empty_gate.py [train|apply|all] [--batch data5]
(a batch may point its "empty_gate_model" at another batch's trained model
to reuse it — apply then gates the new batch with the shared model)
"""
import json, glob, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scan2e2x.batch import load_batch, pop_batch_arg  # noqa: E402

CFG = load_batch(pop_batch_arg(sys.argv))
ROOT = CFG.root
MODEL_PATH = CFG.empty_gate_model_path
INK_WEIGHT = 8.0
MARGIN = 0.02


def train():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    LAB = json.loads((ROOT / "empty_labels.json").read_text())
    F = json.loads((ROOT / "empty_features_v2.json").read_text())
    keys = [k for k in LAB if k in F]
    X = np.array([F[k] for k in keys], float); y = np.array([LAB[k] for k in keys])
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000, class_weight={0: INK_WEIGHT, 1: 1.0}).fit(sc.transform(X), y)
    p = clf.predict_proba(sc.transform(X))[:, 1]
    thr = min(0.999, float(p[y == 0].max()) + MARGIN)
    kf = min(5, int(np.bincount(y).min()))
    acc = float(cross_val_score(clf, sc.transform(X), y, cv=max(2, kf)).mean())
    model = {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
             "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
             "threshold": thr, "cv_acc": acc, "n_labels": len(y),
             "ink_recall_labels": float((p[y == 0] < thr).mean())}
    MODEL_PATH.write_text(json.dumps(model, indent=2))
    print(f"trained: labels={len(y)} thr={thr:.2f} CVacc={acc:.0%} ink-recall={model['ink_recall_labels']:.0%}")
    return model


def _load():
    return json.loads(MODEL_PATH.read_text())


def p_empty(feat, m=None):
    m = m or _load()
    z = (np.array(feat, float) - m["mean"]) / m["scale"]
    return float(1 / (1 + np.exp(-(np.dot(z, m["coef"]) + m["intercept"]))))


def is_empty_feat(feat, m=None):
    m = m or _load()
    return p_empty(feat, m) >= m["threshold"]


def classify_page(gray, m=None):
    """Live use: mask + features + gate on a grayscale page image."""
    from empty_features_v2 import features
    feat = features(gray); m = m or _load()
    p = p_empty(feat, m)
    return (p >= m["threshold"]), p


def apply_all():
    m = _load()
    F = json.loads((ROOT / "empty_features_v2.json").read_text())
    n_empty = 0
    for leaf in CFG.leaf_glob("oriented.png"):
        key = f"{leaf.parent.name}/{leaf.name}"
        if key not in F:
            continue
        pe = p_empty(F[key], m); emp = pe >= m["threshold"]
        (leaf / "empty_gate.json").write_text(json.dumps(
            {"is_empty": bool(emp), "p_empty": round(pe, 3), "threshold": m["threshold"]}))
        n_empty += int(emp)
    tot = len(F)
    print(f"applied: {tot} pages, empty(skip OCR)={n_empty} ({100*n_empty/tot:.0f}%), "
          f"non-empty(to OCR)={tot-n_empty}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("train", "all"):
        train()
    if cmd in ("apply", "all"):
        apply_all()
