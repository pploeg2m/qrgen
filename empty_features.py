#!/usr/bin/env python3
"""Pure-OpenCV features for empty(grid-only)-page detection on data4.

Cache: data4/empty_features.json  { "<set>/<nnn>": [f1..f5] }
Features (interior crop, printed grid/frame removed):
  0 resid_ink_pct   residual ink % after grid/frame line removal
  1 n_components     # non-grid connected components (area>=20)
  2 max_comp_permil  largest component area, per-mille of crop
  3 band_cv          coefficient of variation of per-band ink (spatial concentration)
  4 dark_pct         % of interior pixels darker than 128
"""
import glob, json, sys
from pathlib import Path
import cv2, numpy as np

FEATNAMES = ["resid_ink_pct", "n_components", "max_comp_permil", "band_cv", "dark_pct"]


def features(gray):
    h, w = gray.shape
    crop = gray[int(0.12*h):int(0.92*h), int(0.12*w):int(0.96*w)]
    bw = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 10)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    lines = cv2.bitwise_or(cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk),
                           cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk))
    resid = cv2.morphologyEx(cv2.subtract(bw, lines), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    size = resid.size
    resid_ratio = float(resid.sum()) / 255.0 / size
    n, lab, stats, _ = cv2.connectedComponentsWithStats(resid, connectivity=8)
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 20]
    n_comp = len(areas)
    max_area = (max(areas) / size) if areas else 0.0
    rows = resid.sum(1).astype(np.float32) / 255.0
    nb = 20; bh = max(1, len(rows) // nb)
    bm = np.array([rows[i*bh:(i+1)*bh].mean() for i in range(nb)])
    band_cv = float(bm.std() / bm.mean()) if bm.mean() > 0 else 0.0
    dark = float((crop < 128).mean())
    return [round(resid_ratio*100, 4), n_comp, round(max_area*1000, 3),
            round(band_cv, 4), round(dark*100, 4)]


def main():
    root = Path("data4/runs")
    out = Path("data4/empty_features.json")
    cache = json.loads(out.read_text()) if out.exists() else {}
    leaves = sorted(glob.glob(str(root / "WiMa_SS26_*" / "*" / "oriented.png")))
    n = 0
    for p in leaves:
        leaf = Path(p).parent
        key = f"{leaf.parent.name}/{leaf.name}"
        if key in cache:
            continue
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        cache[key] = features(g)
        n += 1
        if n % 200 == 0:
            out.write_text(json.dumps(cache)); print(f"  ...{n}", flush=True)
    out.write_text(json.dumps(cache))
    print(f"features cached: {len(cache)} pages -> {out}")


if __name__ == "__main__":
    main()
