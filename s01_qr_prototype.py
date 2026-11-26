#!/usr/bin/env python3
"""s01_qr — PROTOTYPE.

For each PDF page:
  1. Render to grayscale.
  2. Find the page rotation (0/90/180/270) that decodes the QR AND places it
     in the LOWER-LEFT quadrant (the authoritative "correct orientation" rule).
  3. Parse the QR payload  ID<booklet>-P<page>.
  4. Resolve booklet-id -> Matrikel-Nr via an authoritative bijective map
     (CSV: qr_id,matr_nr). Stubbed to None when no --map is given.

Emits one dict per page (the future page.json/qr.json identity block):
  {page_index, qr_raw, decoded_ok, booklet_id, page_no,
   rotation_ccw_deg, qr_quadrant_before, matr_nr}

Usage:
  python projects/qrgen/s01_qr_prototype.py projects/qrgen/out.pdf [--dpi 200]
         [--map qr_matr.csv] [--limit N] [--out results.json]
"""
from __future__ import annotations
import argparse, csv, json, re, sys
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np

# Real scanned format:  HRW-A-V1-P000479-S01-F   (sys-paperid-sheet-side)
# Synthetic out.pdf:     ID001-P01
PAYLOAD_RE = re.compile(
    r"^(?P<sys>HRW-[A-Z]-V\d+)-P(?P<id>\d+)-S(?P<sheet>\d+)-(?P<side>[FB])$"
    r"|^ID(?P<id2>[0-9A-Za-z]+)-P(?P<page>\d+)$"
)
_DET = cv2.QRCodeDetector()


def _decode(gray: np.ndarray):
    """Return (value, points) or (None, None)."""
    val, pts, _ = _DET.detectAndDecode(gray)
    if val and pts is not None:
        return val, pts.reshape(-1, 2)
    return None, None


def _quadrant(pts: np.ndarray, shape) -> str:
    h, w = shape[:2]
    cx, cy = pts[:, 0].mean() / w, pts[:, 1].mean() / h
    vert = "lower" if cy > 0.5 else "upper"
    horiz = "left" if cx < 0.5 else "right"
    return f"{vert}-{horiz}"


def orient_and_decode(gray: np.ndarray):
    """Try 4 rotations; pick the one whose QR is lower-left. np.rot90(k) is CCW.

    Returns (value, rotation_ccw_deg, quadrant_at_k0) or (None, None, quad0).
    """
    quad0 = None
    first_hit = None  # (value, deg) for any decode, as fallback
    for k in range(4):
        rot = np.ascontiguousarray(np.rot90(gray, k))
        val, pts = _decode(rot)
        if val is None:
            continue
        q = _quadrant(pts, rot.shape)
        if k == 0:
            quad0 = q
        if first_hit is None:
            first_hit = (val, k * 90)
        if q == "lower-left":
            return val, k * 90, (quad0 if quad0 is not None else q)
    # QR decoded in some orientation but never landed lower-left (rare/skew):
    if first_hit is not None:
        return first_hit[0], first_hit[1], quad0
    return None, None, quad0


def load_map(path: Path | None) -> dict:
    if not path:
        return {}
    m = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f, delimiter=";" if ";" in open(path).readline() else ","):
            if len(row) >= 2 and row[0].strip().lower() not in ("qr_id", "id"):
                m[row[0].strip()] = row[1].strip()
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--map", type=Path, default=None, help="CSV qr_id,matr_nr")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    qr2matr = load_map(args.map)
    doc = fitz.open(str(args.pdf))
    n = doc.page_count if args.limit is None else min(args.limit, doc.page_count)

    results = []
    for pno in range(n):
        pix = doc[pno].get_pixmap(dpi=args.dpi)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if pix.n >= 3 else img
        val, deg, quad0 = orient_and_decode(gray)
        rec = {
            "page_index": pno + 1,
            "qr_raw": val,
            "decoded_ok": val is not None,
            "booklet_id": None,
            "page_no": None,
            "side": None,
            "sys": None,
            "rotation_ccw_deg": deg,
            "qr_quadrant_before": quad0,
            "matr_nr": None,
        }
        if val:
            m = PAYLOAD_RE.match(val)
            if m and m.group("id"):            # real HRW format
                rec["booklet_id"] = m.group("id")
                rec["page_no"] = int(m.group("sheet"))
                rec["side"] = {"F": "front", "B": "back"}[m.group("side")]
                rec["sys"] = m.group("sys")
                rec["matr_nr"] = qr2matr.get(m.group("id"))
            elif m and m.group("id2"):         # synthetic out.pdf format
                rec["booklet_id"] = m.group("id2")
                rec["page_no"] = int(m.group("page"))
                rec["matr_nr"] = qr2matr.get(m.group("id2"))
        results.append(rec)

    # ── summary ───────────────────────────────────────────────────────────
    ok = sum(r["decoded_ok"] for r in results)
    parsed = sum(r["booklet_id"] is not None for r in results)
    from collections import Counter
    booklets = Counter(r["booklet_id"] for r in results if r["booklet_id"])
    rots = Counter(r["rotation_ccw_deg"] for r in results)
    quads = Counter(r["qr_quadrant_before"] for r in results)
    print(f"pages: {len(results)}")
    print(f"QR decoded: {ok}/{len(results)}   payload parsed (ID-P): {parsed}/{len(results)}")
    print(f"booklets (id -> #pages): {dict(booklets)}")
    print(f"rotation applied to reach QR-lower-left (deg CCW): {dict(rots)}")
    print(f"QR quadrant BEFORE rotation: {dict(quads)}")
    # page-sequence sanity per booklet
    for bid in booklets:
        pages = sorted(r["page_no"] for r in results if r["booklet_id"] == bid and r["page_no"])
        gaps = [p for p in range(1, (pages[-1] if pages else 0) + 1) if p not in pages]
        dups = [p for p in set(pages) if pages.count(p) > 1]
        print(f"  {bid}: pages {pages[0]}..{pages[-1]} ({len(pages)})  gaps={gaps or 'none'}  dups={dups or 'none'}")
    print("\nfirst 6 records:")
    for r in results[:6]:
        print("  ", r)
    if not qr2matr:
        print("\nNOTE: no --map given -> matr_nr is null (needs authoritative QR-id->Matrikel table).")

    if args.out:
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
