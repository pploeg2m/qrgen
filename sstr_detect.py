#!/usr/bin/env python3
"""Classical Schlussstrich detector (data4). prepare() once, count_lines() with
tunable params (for calibration). Deskew is always applied in prepare()."""
import cv2, numpy as np
from empty_features_v2 import deskew, frame_bbox, PX_PER_CM

T_DARK, SPAN_MIN, CLOSE_FRAC = 140, 0.80, 0.04
EXCL_CM, MAX_THICK_CM, MERGE_GAP = 0.55, 0.55, 8


def find_icd(gray, fb):
    x0, y0, x1, y1 = fb
    lo, hi = int(x0 + 0.4*PX_PER_CM), int(x0 + 1.8*PX_PER_CM)
    if hi <= lo or y1 <= y0:
        return int(x0 + 1.1*PX_PER_CM)
    cd = (gray[y0:y1, lo:hi] < T_DARK).sum(0).astype(float) / max(1, y1-y0)
    j = int(np.argmax(cd))
    return lo + j if cd[j] > 0.5 else int(x0 + 1.1*PX_PER_CM)


def prepare(gray):
    g, ang = deskew(gray); fb, _ = frame_bbox(g); x0, y0, x1, y1 = fb
    icd = find_icd(g, fb)
    ay0, ay1 = int(y0 + EXCL_CM*PX_PER_CM), int(y1 - EXCL_CM*PX_PER_CM)
    ok = (x1 - icd >= 100 and ay1 - ay0 >= 100)
    return {"roi": g[ay0:ay1, icd:x1] if ok else None, "icd_x": int(icd),
            "frame": [int(v) for v in fb], "angle": round(ang, 2),
            "ay0": ay0, "H": g.shape[0], "ok": ok, "g": g}


def count_lines(p, T_DARK=T_DARK, SPAN_MIN=SPAN_MIN, CLOSE_FRAC=CLOSE_FRAC,
                MAX_THICK_CM=MAX_THICK_CM, MERGE_GAP=MERGE_GAP,
                ISO_CM=0.40, ISO_MAX=0.06):
    if not p["ok"]:
        return []
    roi, ay0, H = p["roi"], p["ay0"], p["H"]
    W = roi.shape[1]
    dark = (roi < T_DARK).astype(np.uint8) * 255
    kw = max(5, int(CLOSE_FRAC * W))
    closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1)))
    # keep only horizontal structures (drop text/grid speckle)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(0.25 * W)), 1))
    horiz = cv2.morphologyEx(closed, cv2.MORPH_OPEN, hk)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(horiz, connectivity=8)
    mt = int(MAX_THICK_CM * PX_PER_CM)
    gap = int(ISO_CM * PX_PER_CM); off = max(2, int(0.06 * PX_PER_CM)); hR = roi.shape[0]
    darkf = dark.astype(float) / 255.0
    cand = []
    for i in range(1, n):
        w = stats[i, cv2.CC_STAT_WIDTH]; h = stats[i, cv2.CC_STAT_HEIGHT]
        if not (w >= SPAN_MIN * W and h <= mt and w >= 6 * h):
            continue
        yy = int(cent[i][1])
        ab = darkf[max(0, yy-off-gap):max(0, yy-off), :]      # band above
        be = darkf[min(hR, yy+off):min(hR, yy+off+gap), :]    # band below
        if ab.size and be.size and ab.mean() < ISO_MAX and be.mean() < ISO_MAX:
            cand.append(yy)                                   # isolated line
    cand.sort()
    out, last = [], -10**9
    for yy in cand:                       # merge near-duplicate lines
        if yy - last <= MERGE_GAP:
            continue
        last = yy; yc = ay0 + yy
        out.append({"y_frac": round(yc/H, 4), "y_px": int(yc), "thick_px": 0})
    return out


def detect(gray, **kw):
    p = prepare(gray); lines = count_lines(p, **kw)
    return {"count": len(lines), "lines": lines, "icd_x": p["icd_x"],
            "frame": p["frame"], "angle": p["angle"]}
