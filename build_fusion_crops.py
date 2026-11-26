#!/usr/bin/env python3
"""Fusion subtask-boundary detector → automatic per-subtask crops.

Fuses, per student (booklet pages in reading order):
  1. VERIFIED SCHLUSSSTRICHE  — per-leaf schlussstrich.json (y_frac,
     label_above) from the CV+VLM verifier: in-page segment cuts, each
     carrying the label of the segment ABOVE it,
  2. REGISTERED MARKER COLUMN — ink clusters at the left edge of the frame
     (register.json geometry, band widened past the printed column since
     students spill "2)a)" over the rule): a marker near a segment start
     confirms a NEW subtask (used for implicit page-break separators),
  3. GRADE-MAP SEQUENCE       — the exam's canonical subtask order
     (1=Sparplan a-c, 2=LinAlg a-c, 3=Marktanteile a-d, 4=Gewinnfunktion
     a-e, 5=Marktgleichgewicht a-e, 6=Bonus a-c): label evidence is
     decoded monotonically along it; unlabeled segments continue the
     current subtask, unless marker-confirmed new → advance by one.

Output (mirrors the GT tree for the viewer's compare tab):
  <batch>/osiris_fusion/<matr>/<assignment>/<grade_id>.jpg
  <batch>/osiris_fusion/<matr>/fusion_partition.json   (audit trail)
  <batch>/osiris_fusion/_fusion_report.json            (corpus stats)

Run from the repo root:  python projects/qrgen/build_fusion_crops.py [--batch data5]
"""
import glob, json, os, re, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scan2e2x.batch import load_batch, pop_batch_arg  # noqa: E402

# resolve the batch BEFORE importing make_submissions, so its module-level
# _init() binds to the same batch (it reads $S2E_BATCH)
_batch = pop_batch_arg(sys.argv)
if _batch:
    os.environ["S2E_BATCH"] = _batch
CFG = load_batch()

sys.path.insert(0, "projects/qrgen")
from make_submissions import load_csv, scan, is_empty  # noqa: E402

ROOT = CFG.root
OUT = CFG.osiris_fusion
ASSIGN = CFG.assignment
W = 1000                      # stitched crop width (same as submissions)
PAD = 0.012                   # y overlap pad: sstr y_frac is pre-deskew
MARKER_TOP = 0.14             # marker within top 14% of a piece = start marker

# grade map (exam order): batch.json grade_map [[task, family, subs], ...]
FAMS = CFG.grade_map
SEQ = [(t, s) for t, fam, subs in FAMS for s in subs]
GRADE_ID = {(t, s): f"{fam}_{s.upper()}" for t, fam, subs in FAMS for s in subs}
IDX = {ts: i for i, ts in enumerate(SEQ)}

_LAB = re.compile(r"([1-6])?\s*[).\]]?\s*([a-g])\s*[).\]]?")


def parse_label(la):
    """'4) b)' / '3a)' / 'b)' / 'a' -> (task|None, sub) or None."""
    if not la:
        return None
    m = _LAB.search(str(la))
    if not m:
        return None
    return m.group(1), m.group(2).lower()


def label_index(ev, cur):
    """Grade-map index for evidence (task?, sub), resolved near *cur*."""
    t, s = ev
    if t and (t, s) in IDX:
        return IDX[(t, s)]
    # letter only: pick the next occurrence of this sub-letter at/after cur,
    # preferring the current task's own letter slot
    for i in range(max(cur, 0), len(SEQ)):
        if SEQ[i][1] == s:
            return i
    return None


def marker_ys(leaf, reg, gray):
    """y positions (page fracs) of ink clusters in the widened marker band."""
    h, w = gray.shape
    fx, fb = reg["x"], reg["frame_bbox_px"]
    x0 = int((fx["s"] * 0.048 + fx["o"]) * w) + 14
    x1 = int((fx["s"] * 0.135 + fx["o"]) * w)
    strip = gray[fb["y0"] + 10:fb["y1"] - 10, x0:x1]
    if strip.size == 0:
        return []
    ink = (strip < 150).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(ink)
    ys = sorted(int(cent[i][1]) for i in range(1, n) if stats[i, 4] >= 40)
    out = []
    for y in ys:
        if not out or y - out[-1][1] > 120:
            out.append([y, y])
        else:
            out[-1][1] = y
    return [(fb["y0"] + 10 + (a + b) / 2) / h for a, b in out]


def student_segments(pages):
    """Cut the ordered page stream into segments with label evidence."""
    segments = []          # {pieces: [(leaf, y0f, y1f)], ev, marker}
    cur = None

    def close(ev=None):
        nonlocal cur
        if cur is not None:
            cur["ev"] = ev
            segments.append(cur)
        cur = None

    for leaf in pages:
        reg = json.loads((leaf / "register.json").read_text()) \
            if (leaf / "register.json").exists() else {}
        if not reg.get("ok"):
            continue
        img = cv2.imread(str(leaf / "oriented.png"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h = img.shape[0]
        fb = reg["frame_bbox_px"]
        fy0, fy1 = fb["y0"] / h, fb["y1"] / h

        ss = json.loads((leaf / "schlussstrich.json").read_text()) \
            if (leaf / "schlussstrich.json").exists() else {}
        strokes = sorted(
            (s for s in (ss.get("schlussstriche") or [])
             if fy0 + 0.02 < (s.get("y_frac") or 0) < fy1 - 0.01),
            key=lambda s: s["y_frac"])
        marks = marker_ys(leaf, reg, img)

        # PASS 2: cut events = Schlussstriche AND mid-page markers. A stroke
        # ENDS the segment above it (carrying its label_above); a marker
        # STARTS a new segment just above the marker ink. Markers near the
        # frame top (page-break continuation logic) or near a stroke (start
        # confirmation of the stroke's cut) are not separate cuts.
        events = [{"y": s["y_frac"], "kind": "sstr",
                   "lab": s.get("label_above")} for s in strokes]
        for m in marks:
            y = m - 0.008
            if y <= fy0 + 0.03:
                continue
            if any(abs(y - e["y"]) < 0.035 for e in events):
                continue
            events.append({"y": y, "kind": "marker"})
        events.sort(key=lambda e: e["y"])
        dedup = []
        for e in events:
            if dedup and e["y"] - dedup[-1]["y"] < 0.02:
                if dedup[-1]["kind"] == "marker" and e["kind"] == "sstr":
                    dedup[-1] = e
                continue
            dedup.append(e)
        events = dedup

        bounds = [fy0] + [e["y"] for e in events] + [fy1]
        for i in range(len(bounds) - 1):
            y0, y1 = bounds[i], bounds[i + 1]
            if y1 - y0 < 0.015:
                continue
            started = any(y0 - 0.01 <= m <= y0 + min(0.12, (y1 - y0) * 0.5)
                          for m in marks)
            if i == 0:
                # first piece: continues previous segment unless a marker
                # sits at its very top (implicit page-break separator)
                if cur is None or started:
                    close()
                    cur = {"pieces": [], "marker": started}
            else:
                ev = events[i - 1]
                if ev["kind"] == "sstr":
                    close(parse_label(ev.get("lab")))
                else:               # mid-page marker cut: unlabeled end
                    close(None)
                    started = True
                cur = {"pieces": [], "marker": started}
            cur["pieces"].append((leaf, y0, y1))
    close()
    return segments


def decode(segments):
    """Monotone grade-map walk: evidence jumps forward, markers advance."""
    cur = -1
    for seg in segments:
        ev = seg.get("ev")
        idx = label_index(ev, cur) if ev else None
        if idx is not None and idx >= cur:
            cur = idx
        elif cur < 0:
            cur = 0
        elif seg.get("marker") and cur < len(SEQ) - 1:
            cur += 1
        seg["ts"] = SEQ[cur]
    return segments


def render(matr, segments):
    """Stitch each grade's pieces into <grade_id>.jpg, return grade set."""
    bygrade = defaultdict(list)
    for seg in segments:
        bygrade[GRADE_ID[seg["ts"]]].extend(seg["pieces"])
    outdir = OUT / matr / ASSIGN
    if outdir.is_dir():                 # drop stale crops from a prior pass
        import shutil
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    imcache = {}
    for gid, pieces in bygrade.items():
        parts = []
        for leaf, y0, y1 in pieces:
            if leaf not in imcache:
                im = Image.open(leaf / "oriented.png").convert("RGB")
                reg = json.loads((leaf / "register.json").read_text())
                imcache[leaf] = (im, reg["frame_bbox_px"])
            im, fb = imcache[leaf]
            a = max(0, int((y0 - PAD) * im.height))
            b = min(im.height, int((y1 + PAD) * im.height))
            piece = im.crop((fb["x0"], a, fb["x1"], b))
            parts.append(piece.resize((W, max(1, round(piece.height * W / piece.width)))))
        canvas = Image.new("RGB", (W, sum(p.height for p in parts)), "white")
        y = 0
        for p in parts:
            canvas.paste(p, (0, y)); y += p.height
        canvas.save(outdir / f"{gid}.jpg", quality=88)
    return set(bygrade)


def main():
    pkg2row, z2pkg = load_csv()
    booklets, extras, _bad = scan()
    z_as_booklet = {z for z in z2pkg if z in booklets}
    report = {"students": 0, "segments": 0, "labeled": 0, "marker_new": 0,
              "grades_fusion": 0, "grades_gt": 0, "both": 0,
              "only_gt": 0, "only_fusion": 0}

    for pkg in sorted(set(booklets) & set(pkg2row)):
        if pkg in z_as_booklet:
            continue
        matr = (pkg2row[pkg].get("Matrikelnr.") or "").strip()
        if not matr:
            continue

        def seqpages(pk):
            return [lf for _, _, lf in sorted(booklets[pk], key=lambda t: (t[0], t[1]))]
        ordered = seqpages(pkg)
        for z, owner in z2pkg.items():
            if owner == pkg and z in z_as_booklet:
                ordered += seqpages(z)
        for z, lst in extras.items():
            if z2pkg.get(z) == pkg:
                ordered += [lf for _, lf in sorted(lst)]
        pages = [Path(lf) for lf in ordered if not is_empty(Path(lf))]

        segs = decode(student_segments(pages))
        grades = render(matr, segs)

        (OUT / matr / "fusion_partition.json").write_text(json.dumps({
            "matr_nr": matr, "n_segments": len(segs),
            "segments": [{"grade": GRADE_ID[s["ts"]],
                          "ev": s.get("ev"), "marker": s.get("marker", False),
                          "pieces": [[str(l), round(a, 4), round(b, 4)]
                                     for l, a, b in s["pieces"]]}
                         for s in segs]}, ensure_ascii=False, indent=1))

        gt = {p.stem for p in
              (ROOT / "osiris_submitted" / matr / ASSIGN).glob("*.jpg")}
        report["students"] += 1
        report["segments"] += len(segs)
        report["labeled"] += sum(1 for s in segs if s.get("ev"))
        report["marker_new"] += sum(1 for s in segs if s.get("marker"))
        report["grades_fusion"] += len(grades)
        report["grades_gt"] += len(gt)
        report["both"] += len(grades & gt)
        report["only_gt"] += len(gt - grades)
        report["only_fusion"] += len(grades - gt)
        print(f"  {matr}: {len(pages)} pages -> {len(segs)} segments -> "
              f"{len(grades)} grades (GT {len(gt)}, overlap {len(grades & gt)})",
              flush=True)

    (OUT / "_fusion_report.json").write_text(json.dumps(report, indent=1))
    print("\n== fusion report ==")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print("report ->", OUT / "_fusion_report.json")


if __name__ == "__main__":
    main()
