#!/usr/bin/env python3
"""Batch orientation: rotate every ingested page so its QR sits LOWER-LEFT.

Non-destructive: reads <leaf>/original.png, writes <leaf>/oriented.png and
<leaf>/orient.json. original.png is never modified.

Robust: uses cv2.QRCodeDetector().detect() (stable localization) to find the
QR centroid -> quadrant -> rotation, then RE-DETECTS on the result to verify
the QR really landed lower-left (else the page is flagged, left unrotated).

Quadrant(before) -> np.rot90 CCW count k (CCW cycle TL->BL->BR->TR->TL):
  lower-left=0  upper-left=1(90)  upper-right=2(180)  lower-right=3(270)
"""
import glob, json, sys
from pathlib import Path
import cv2, numpy as np

DET = cv2.QRCodeDetector()
MAP = {"lower-left": 0, "upper-left": 1, "upper-right": 2, "lower-right": 3}


def quad(pts, shape):
    h, w = shape[:2]
    c = pts.reshape(-1, 2).mean(0)
    return f"{'lower' if c[1]/h > 0.5 else 'upper'}-{'left' if c[0]/w < 0.5 else 'right'}"


def detect_quad(gray):
    ok, pts = DET.detect(gray)
    if ok and pts is not None:
        return quad(pts, gray.shape)
    return None


def process(leaf: Path):
    img = cv2.imread(str(leaf / "original.png"))
    if img is None:
        return "no_png"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    q0 = detect_quad(gray)
    if q0 is None:
        # QR not localized as-is; try each rotation to find one that localizes lower-left
        k = None
        for kk in range(4):
            if detect_quad(np.ascontiguousarray(np.rot90(gray, kk))) == "lower-left":
                k = kk; break
        if k is None:
            # give up: copy original unrotated, flag
            cv2.imwrite(str(leaf / "oriented.png"), img)
            _write(leaf, None, None, k=0, verified=False, reason="qr_not_localized")
            return "flag_noqr"
    else:
        k = MAP[q0]

    out = np.ascontiguousarray(np.rot90(img, k))
    # self-verify
    q1 = detect_quad(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY))
    verified = (q1 == "lower-left")
    val = DET.detectAndDecode(out)[0] or None
    cv2.imwrite(str(leaf / "oriented.png"), out)
    _write(leaf, q0, val, k=k, verified=verified,
           reason=None if verified else f"post_quad={q1}")
    return "ok" if verified else "flag_unverified"


def _write(leaf, q0, val, k, verified, reason):
    (leaf / "orient.json").write_text(json.dumps({
        "rotation_ccw_deg": k * 90,
        "quadrant_before": q0,
        "verified_lower_left": verified,
        "qr_raw": val,
        "reason": reason,
    }, ensure_ascii=False, indent=2))


def main():
    runs_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data4/runs")
    from collections import Counter
    leaves = sorted(Path(p).parent for p in glob.glob(str(runs_root / "*" / "*" / "original.png")))
    if not leaves:
        leaves = sorted(Path(p).parent for p in glob.glob(str(runs_root / "*" / "original.png")))
    tally = Counter(); rots = Counter()
    for i, leaf in enumerate(leaves):
        r = process(leaf)
        tally[r] += 1
        try:
            rots[json.loads((leaf / "orient.json").read_text())["rotation_ccw_deg"]] += 1
        except Exception:
            pass
        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{len(leaves)}", flush=True)
    print(f"\nleaves: {len(leaves)}")
    print(f"outcomes: {dict(tally)}")
    print(f"rotations applied (deg CCW): {dict(rots)}")
    flagged = tally.get("flag_unverified", 0) + tally.get("flag_noqr", 0)
    print(f"FLAGGED for review: {flagged}")


if __name__ == "__main__":
    main()
