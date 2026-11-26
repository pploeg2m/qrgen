#!/usr/bin/env python3
"""v2 empty-detection features with proper masking for the data4 QR template.

Pipeline per page (oriented.png, gray):
  1. deskew (median angle of long near-horizontal dark lines = frame/grid)
  2. detect the outer thick black frame -> its inner rectangle = answer area
  3. skip the ~1cm vertical left margin (Korrekturrand) line
     (QR + ASCII sit BELOW the frame, so they're excluded by the frame bbox)
  4. inside the answer area: adaptive-threshold -> remove long H/V grid lines
     -> residual = student ink
  5. features from the residual.

Cache: <batch>/empty_features_v2.json  { "<set>/<nnn>": [f1..f5] }
"""
import glob, json, sys
from pathlib import Path
import cv2, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scan2e2x.batch import load_batch, pop_batch_arg  # noqa: E402

FEATNAMES = ["resid_ink_pct", "n_components", "max_comp_permil", "band_cv", "dark_pct"]
PX_PER_CM = 300 / 2.54   # oriented.png is 300 DPI


def deskew(gray):
    edges = cv2.Canny(gray, 30, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi/360, threshold=150,
                            minLineLength=int(0.30*gray.shape[1]), maxLineGap=30)
    cand = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            a = np.degrees(np.arctan2(y2-y1, x2-x1))
            if abs(a) < 20:                       # near-horizontal
                cand.append(a)
            elif abs(abs(a) - 90) < 20:           # near-vertical -> same skew
                cand.append(a - 90 if a > 0 else a + 90)
    ang = float(np.median(cand)) if cand else 0.0
    ang = max(-15.0, min(15.0, ang))
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), ang, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255), ang


def frame_bbox(gray):
    """Outer thick-black-frame extent via long dark H/V lines."""
    h, w = gray.shape
    dark = (gray < 110).astype(np.uint8) * 255
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w*0.25), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h*0.25)))
    frame = cv2.bitwise_or(cv2.morphologyEx(dark, cv2.MORPH_OPEN, hk),
                           cv2.morphologyEx(dark, cv2.MORPH_OPEN, vk))
    ys, xs = np.where(frame > 0)
    fallback = (int(0.10*w), int(0.05*h), int(0.97*w), int(0.95*h))
    if len(xs) < 200:
        return fallback, False
    bb = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    # sanity: frame must cover most of the page and be roughly A4-portrait
    fw, fh = bb[2]-bb[0], bb[3]-bb[1]
    if fw < 0.6*w or fh < 0.6*h or not (1.2 < fh/max(fw, 1) < 1.7):
        return fallback, False
    return bb, True


def answer_crop(gray):
    g, ang = deskew(gray)
    (x0, y0, x1, y1), found = frame_bbox(g)
    left  = int(x0 + 1.3*PX_PER_CM)   # skip frame + ~1cm left margin line
    top   = int(y0 + 0.15*PX_PER_CM)
    right = int(x1 - 0.15*PX_PER_CM)
    bot   = int(y1 - 0.15*PX_PER_CM)
    if right-left < 100 or bot-top < 100:      # frame detection failed -> safe fallback
        h, w = g.shape
        left, top, right, bot = int(0.16*w), int(0.10*h), int(0.97*w), int(0.90*h)
        found = False
    return g[top:bot, left:right], (left, top, right, bot), ang, found


def residual_ink(crop):
    bw = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 10)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    lines = cv2.bitwise_or(cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk),
                           cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk))
    return cv2.morphologyEx(cv2.subtract(bw, lines), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def features(gray):
    crop, bbox, ang, found = answer_crop(gray)
    resid = residual_ink(crop)
    size = resid.size
    resid_ratio = float(resid.sum()) / 255.0 / size
    n, _, stats, _ = cv2.connectedComponentsWithStats(resid, connectivity=8)
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 20]
    max_area = (max(areas) / size) if areas else 0.0
    rows = resid.sum(1).astype(np.float32) / 255.0
    nb = 20; bh = max(1, len(rows)//nb)
    bm = np.array([rows[i*bh:(i+1)*bh].mean() for i in range(nb)])
    band_cv = float(bm.std()/bm.mean()) if bm.mean() > 0 else 0.0
    dark = float((crop < 128).mean())
    return [round(resid_ratio*100, 4), len(areas), round(max_area*1000, 3),
            round(band_cv, 4), round(dark*100, 4)]


def viz(keys, out, runs_dir):
    tw, th = 300, 400
    cols = len(keys); sheet = np.full((th, cols*tw, 3), 255, np.uint8)
    for i, k in enumerate(keys):
        s, n = k.split("/")
        g = cv2.imread(str(runs_dir / s / n / "oriented.png"), cv2.IMREAD_GRAYSCALE)
        gd, ang = deskew(g)
        (x0, y0, x1, y1), found = frame_bbox(gd)
        crop, bb, a, fnd = answer_crop(g)
        vis = cv2.cvtColor(gd, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 0, 0), 6)      # frame (blue)
        cv2.rectangle(vis, (bb[0], bb[1]), (bb[2], bb[3]), (0, 160, 0), 6)  # answer area (green)
        f = features(g)
        vis = cv2.resize(vis, (tw, th))
        cv2.putText(vis, f"{n} a={ang:.1f} r={f[0]}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
        sheet[:, i*tw:(i+1)*tw] = vis
    cv2.imwrite(out, sheet); print("viz ->", out)


def main():
    cfg = load_batch(pop_batch_arg(sys.argv))
    if len(sys.argv) > 1 and sys.argv[1] == "--viz":
        viz(sys.argv[2:], str(cfg.root / "_mask_viz.png"), cfg.runs_dir)
        return
    out = cfg.root / "empty_features_v2.json"
    cache = {}
    leaves = cfg.leaf_glob("oriented.png")
    for i, leaf in enumerate(leaves):
        p = str(leaf / "oriented.png")
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        cache[f"{leaf.parent.name}/{leaf.name}"] = features(g)
        if (i+1) % 200 == 0:
            out.write_text(json.dumps(cache)); print(f"  ...{i+1}", flush=True)
    out.write_text(json.dumps(cache))
    print(f"v2 features cached: {len(cache)} -> {out}")


if __name__ == "__main__":
    main()
