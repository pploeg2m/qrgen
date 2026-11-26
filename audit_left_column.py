#!/usr/bin/env python3
"""Audit: task markers in the left column, Schlussstrich habits, and
page breaks used as implicit Schlussstriche (data4).

Per non-empty page:
  * LEFT COLUMN — cut the narrow marker column of the printed frame
    (between the frame's left border and the first inner vertical rule,
    template V positions 0.048..0.096 projected via register.json) and
    cluster the handwritten ink into MARKER instances ("1) a)" at a task
    start). Width heuristic: a cluster spanning >= 55 % of the usable
    column width counts as "task + subtask letter", narrower as bare
    "task NO".
  * SCHLUSSSTRICH — drawn separator lines from the schlussstrich.json
    probe (note: probed on pre-deskew geometry; counts/fractions only).

Per student (booklet pages in reading order S01-F, S01-B, ...):
  * marker-column usage category (always/mostly/sometimes/never),
  * Schlussstrich totals,
  * IMPLICIT page-break separators: a page whose FIRST marker sits in the
    top 15 % of the frame (a new task starts at the very top) while the
    previous page shows no Schlussstrich in its bottom quarter — the page
    break itself was used as the separator.

Run from the repo root:  python projects/qrgen/audit_left_column.py
"""
import csv, glob, json, re, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("data4")
CSV = ROOT / "priorExam" / "anwesenheit_gefiltert_final.csv"

EDGE_PAD = 14        # px: mask the printed vertical rules at the strip edges
INK_THR = 150        # gray value below which a pixel counts as ink
MIN_BLOB = 40        # px area for a component to count
CLUSTER_GAP = 120    # px: vertical gap that separates two marker instances
WIDE_FRAC = 0.55     # cluster width share => "task + subtask letter"
TOP_FRAC = 0.15      # first marker above this frame fraction = task at page top
BOTTOM_SSTR = 0.75   # schlussstrich y_frac >= this = "closes the page"


def page_metrics(leaf: Path):
    """(n_markers, n_wide, first_marker_frac|None, sstr_count, sstr_bottom)"""
    reg = json.loads((leaf / "register.json").read_text()) \
        if (leaf / "register.json").exists() else {}
    sstr = json.loads((leaf / "schlussstrich.json").read_text()) \
        if (leaf / "schlussstrich.json").exists() else {}
    strokes = sstr.get("schlussstriche") or []
    sstr_count = len(strokes)
    sstr_bottom = any((s.get("y_frac") or 0) >= BOTTOM_SSTR for s in strokes)

    if not reg.get("ok"):
        return None
    img = cv2.imread(str(leaf / "oriented.png"), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    fx, fb = reg["x"], reg["frame_bbox_px"]
    x0 = int((fx["s"] * 0.048 + fx["o"]) * w) + EDGE_PAD
    x1 = int((fx["s"] * 0.096 + fx["o"]) * w) - EDGE_PAD
    strip = img[fb["y0"] + EDGE_PAD:fb["y1"] - EDGE_PAD, x0:x1]
    if strip.size == 0 or x1 <= x0:
        return None

    ink = (strip < INK_THR).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(ink)
    blobs = [(int(cent[i][1]), stats[i, 0], stats[i, 0] + stats[i, 2])
             for i in range(1, n) if stats[i, 4] >= MIN_BLOB]
    if not blobs:
        return 0, 0, None, sstr_count, sstr_bottom
    blobs.sort()
    clusters = [[blobs[0]]]
    for b in blobs[1:]:
        if b[0] - clusters[-1][-1][0] <= CLUSTER_GAP:
            clusters[-1].append(b)
        else:
            clusters.append([b])
    usable = x1 - x0
    n_wide = sum(
        1 for c in clusters
        if (max(e for _, _, e in c) - min(s for _, s, _ in c)) >= WIDE_FRAC * usable
    )
    first_frac = clusters[0][0][0] / max(strip.shape[0], 1)
    return len(clusters), n_wide, first_frac, sstr_count, sstr_bottom


def is_empty(leaf: Path) -> bool:
    p = leaf / "empty_gate.json"
    return p.exists() and json.load(open(p)).get("is_empty", False)


def main():
    pkg2row = {}
    for r in csv.DictReader(CSV.read_text(encoding="utf-8-sig").splitlines(),
                            delimiter=";"):
        pkg = (r.get("Packagenr.") or "").strip()
        if pkg.isdigit():
            pkg2row[int(pkg)] = r
    z2pkg = {}
    for pkg, r in pkg2row.items():
        zm = re.search(r"\d+", (r.get("Zusatzblattnr.") or ""))
        if zm:
            z2pkg[int(zm.group())] = pkg

    # booklet pages in reading order (extras appended at the end)
    booklets = defaultdict(list)
    for qp in sorted(glob.glob("data4/runs/WiMa_*/*/qr.json")):
        leaf = Path(qp).parent
        q = json.load(open(qp))
        raw = q.get("qr_raw") or ""
        srank = 0 if q.get("side") == "front" else 1
        bid = q.get("booklet_id")
        if not bid:
            m = re.search(r"[PX](\d+)-", raw)
            bid = m.group(1) if m else None
        if bid is None:
            continue
        pkg = int(bid)
        sheet = q.get("sheet") or 99          # extras sort last
        owner = z2pkg.get(pkg, pkg) if (q.get("is_extra_sheet")
                                        or pkg in z2pkg) else pkg
        booklets[owner].append((sheet, srank, leaf))

    per_student = {}
    for pkg, pages in sorted(booklets.items()):
        pages = [lf for _, _, lf in sorted(pages, key=lambda t: (t[0], t[1]))
                 if not is_empty(lf)]
        st = {"pages": 0, "marked": 0, "markers": 0, "wide": 0,
              "sstr_pages": 0, "sstr_total": 0,
              "implicit": 0, "explicit": 0}
        prev_bottom_sstr = None
        for leaf in pages:
            res = page_metrics(leaf)
            if res is None:
                continue
            n_mark, n_wide, first_frac, sstr_count, sstr_bottom = res
            st["pages"] += 1
            st["markers"] += n_mark
            st["wide"] += n_wide
            if n_mark:
                st["marked"] += 1
            if sstr_count:
                st["sstr_pages"] += 1
                st["sstr_total"] += sstr_count
            # task starting at the very top of this page = separator at the
            # preceding page break: explicit if that page closed with a
            # bottom Schlussstrich, implicit otherwise.
            if (prev_bottom_sstr is not None and n_mark
                    and first_frac is not None and first_frac <= TOP_FRAC):
                st["explicit" if prev_bottom_sstr else "implicit"] += 1
            prev_bottom_sstr = sstr_bottom
        if st["pages"]:
            per_student[pkg] = st

    cats = defaultdict(list)
    for pkg, st in sorted(per_student.items()):
        share = st["marked"] / st["pages"]
        cat = ("always" if share >= 0.8 else "mostly" if share >= 0.5
               else "sometimes" if share > 0 else "never")
        r = pkg2row.get(pkg)
        name = f"{r['Nachname']}, {r['Vorname']}" if r else "?"
        matr = r.get("Matrikelnr.", "?") if r else "?"
        cats[cat].append((pkg, matr, name, st, share))

    n_students = len(per_student)
    tot = lambda k: sum(st[k] for st in per_student.values())
    n_pages, n_marked = tot("pages"), tot("marked")
    n_markers, n_wide = tot("markers"), tot("wide")
    n_sstr_students = sum(1 for st in per_student.values() if st["sstr_total"])
    n_impl_students = sum(1 for st in per_student.values() if st["implicit"])

    print("═" * 74)
    print("LEFT-COLUMN MARKER + SCHLUSSSTRICH AUDIT (data4)".center(74))
    print("═" * 74)
    print(f"\n{n_students} students · {n_pages} non-empty pages · "
          f"{n_marked} pages with column markers ({n_marked/n_pages:.0%})")
    print(f"markers: {n_markers} instances, {n_wide} "
          f"({n_wide/max(n_markers,1):.0%}) look like 'task + subtask letter' "
          f"(width heuristic)")
    print(f"schlussstrich: {tot('sstr_total')} drawn on {tot('sstr_pages')} "
          f"pages by {n_sstr_students} students "
          f"({n_sstr_students/n_students:.0%})")
    print(f"page-break separators: {tot('implicit')} implicit "
          f"(no closing line, by {n_impl_students} students) vs "
          f"{tot('explicit')} explicit (bottom Schlussstrich before break)\n")
    hdr = (f"   {'pkg':>7s} {'matrikel':>9s} {'name':<28s} "
           f"{'marked':>8s} {'markers':>7s} {'sstr':>4s} {'impl':>4s} {'expl':>4s}")
    for cat in ("always", "mostly", "sometimes", "never"):
        rows = cats.get(cat, [])
        if not rows:
            continue
        print(f"── column markers {cat} ({len(rows)} students, "
              f"{len(rows)/n_students:.0%}) " + "─" * 25)
        print(hdr)
        for pkg, matr, name, st, share in rows:
            print(f"   P{pkg:06d} {matr:>9s} {name:<28.28s} "
                  f"{st['marked']:3d}/{st['pages']:<3d} "
                  f"{st['markers']:4d}({st['wide']}) "
                  f"{st['sstr_total']:4d} {st['implicit']:4d} {st['explicit']:4d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
