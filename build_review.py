#!/usr/bin/env python3
"""Build clean-slate review_server data for a batch (--batch <root>) ON TOP
of the framed submissions from `make_submissions --abgabe`.

Prerequisite: <batch>/runs/abgabe/HS*/<matr>/submission.png (frame-cropped:
logos, QR labels, printed headers/page numbers are already masked out by the
registered template frame). This script does NO image work of its own — it
reuses those PNGs and derives everything else from per-leaf metadata:

  <batch>/abgabe/<HS>/<matr>/
    submission.png    - symlink to ../../runs/abgabe/<HS>/<matr>/submission.png
    partition.json    - canonical (via partition_submissions.process_student);
                        image_layout reconstructed from register.json frame
                        geometry (same crop the stitcher applied)
    corrections.json  - _clean_slate:true + seeded page_splits:
                          * TCL #1 + SCL #1 at the top (frame top = 0)
                          * every VLM Schlussstrich -> next consecutive SCL,
                            y_frac remapped from full page into the frame
  <batch>/runs/<HS>/students.json - per-set manifest for review_server

Run:
  python projects/qrgen/build_review.py --batch data5 --all
  python projects/qrgen/build_review.py --batch data5 --set HS7
then:
  python tools/review_server.py --runs-root <batch>/runs \
      --abgabe-root <batch>/abgabe --assignment <assignment> \
      --source-nb <batch>/<notebook>
"""
import argparse, json, os, sys
from pathlib import Path
from collections import defaultdict
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scan2e2x.batch import load_batch, pop_batch_arg  # noqa: E402

# bind make_submissions to the same batch BEFORE importing it
_batch = pop_batch_arg(sys.argv)
if _batch:
    os.environ["S2E_BATCH"] = str(_batch)
CFG = load_batch()

sys.path.insert(0, "projects/qrgen")
sys.path.insert(0, "tools")
from make_submissions import (  # noqa: E402
    load_csv, scan, is_empty, load_hs_map, W, FRAME_INSET, LEFT_OUTSET)
from partition_submissions import process_student     # noqa: E402

RUNS = CFG.runs_dir
SUB_ROOT = CFG.abgabe_out          # runs/abgabe — the framed submissions
ABGABE = CFG.review_dir            # <batch>/abgabe — the review tree


def frame_box(leaf, size):
    """The crop box the stitcher applied: register.json frame_bbox_px inset
    by FRAME_INSET (mirrors make_submissions.frame_crop exactly), else the
    full page (same fallback as frame_crop)."""
    p = leaf / "register.json"
    if p.exists():
        r = json.load(open(p))
        b = r.get("frame_bbox_px")
        if r.get("ok") and b:
            m = FRAME_INSET
            return max(0, b["x0"] - LEFT_OUTSET), b["y0"] + m, b["x1"] - m, b["y1"] - m
    w, h = size
    return 0, 0, w, h


def _letter(i):
    """0->a, 1->b, ... 25->z, 26->aa, ..."""
    s = ""
    i += 1
    while i > 0:
        i, r = divmod(i - 1, 26)
        s = chr(97 + r) + s
    return s


SEARCH_WIN = 0.12     # +- window (page frac) around a VLM y to hunt the line
LINE_MIN_COVER = 0.30  # 3-row rolling dark cover a line row must reach (at W=1000)


def _refine_ys(leaf, vlm_ys):
    """Refine each VLM Strich y to the actual drawn line.

    The VLM (schlussstrich.json) is trustworthy on EXISTENCE but poor on
    position (median y error ~6-7 % of page height, measured 2026-09-15;
    a global CV detector with isolation checks fires on only ~11 % of these
    pages). So: within +-SEARCH_WIN of each VLM y, find the row with the
    strongest long horizontal dark run inside the frame; keep the VLM y
    only when no such row exists. Cached as <leaf>/sstr_refined.json."""
    cache = leaf / "sstr_refined.json"
    if cache.exists():
        c = json.load(open(cache))
        if c.get("vlm") == vlm_ys:
            return c["refined"]
    import cv2
    import numpy as np
    g = cv2.imread(str(leaf / "oriented.png"), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return vlm_ys
    Hf, Wf = g.shape
    bx0, by0, bx1, by1 = frame_box(leaf, (Wf, Hf))
    # analyze at the stitched W=1000 scale: at full 300 dpi a wavy freehand
    # line spreads its ink over ~30 rows and per-row coverage collapses;
    # downscaled, the wave compresses and the line shows as a dark band
    sc = W / Wf
    g = cv2.resize(g, (W, round(Hf * sc)), interpolation=cv2.INTER_AREA)
    H0 = g.shape[0]
    x0, y0b, x1, y1b = (int(v * sc) for v in (bx0, by0, bx1, by1))
    m = 10                                 # skip the frame's own border rules
    roi = (g[:, x0 + m:x1 - m] < 150).astype(np.uint8)
    roi = cv2.dilate(roi, np.ones((1, 25), np.uint8))   # bridge small gaps
    cover = np.convolve(roi.mean(axis=1), np.ones(3) / 3, mode="same")
    refined = []
    for yf in vlm_ys:
        yc = yf * H0
        lo = max(int(y0b + m), int(yc - SEARCH_WIN * H0))
        hi = min(int(y1b - m), int(yc + SEARCH_WIN * H0))
        best = None
        if hi > lo:
            win = cover[lo:hi]
            j = int(win.argmax())
            if win[j] >= LINE_MIN_COVER:
                # center of the contiguous dark band around the peak
                a = b = j
                while a > 0 and win[a - 1] >= LINE_MIN_COVER * 0.8:
                    a -= 1
                while b < len(win) - 1 and win[b + 1] >= LINE_MIN_COVER * 0.8:
                    b += 1
                best = (lo + (a + b) / 2) / H0
        refined.append(round(best, 4) if best is not None else yf)
    cache.write_text(json.dumps({"vlm": vlm_ys, "refined": refined}))
    return refined


def sstr_lines(leaf):
    """VLM-confirmed Schlussstriche with CV-refined positions (page frac)."""
    p = leaf / "schlussstrich.json"
    if not p.exists():
        return []
    vlm = json.load(open(p)).get("schlussstriche") or []
    if not vlm:
        return []
    vlm_ys = sorted(round(float(s.get("y_frac") or 0), 4) for s in vlm)
    refined = _refine_ys(leaf, vlm_ys)
    # two VLM striche may refine onto the same drawn line — deduplicate
    out, last = [], None
    for yf in sorted(refined):
        if last is not None and yf - last < 0.01:
            continue
        out.append({"y_frac": yf}); last = yf
    return out


def ordered_leaves(pkg, booklets, extras, z2pkg, z_as_booklet):
    """EXACTLY make_submissions.main's page order for one student."""
    def _seq(pk):
        return [lf for _, _, lf in sorted(booklets[pk], key=lambda t: (t[0], t[1]))]
    ordered = _seq(pkg)
    for z, owner in z2pkg.items():
        if owner == pkg and z in z_as_booklet:
            ordered += _seq(z)
    for z, lst in extras.items():
        if z2pkg.get(z) == pkg:
            for _, leaf in sorted(lst):
                ordered.append(leaf)
    return ordered


def seed_one(matr, hs, ordered):
    """-> (layout, page_splits, pages_manifest, warn) from metadata only."""
    sub_png = SUB_ROOT / hs / matr / "submission.png"
    if not sub_png.exists():
        return None, None, None, f"no framed submission: {sub_png}"

    layout_pages, splits_src = [], []
    y = 0
    for leaf in ordered:
        p = leaf / "oriented.png"
        if not p.exists() or is_empty(leaf):
            continue                       # same filter as stitch()
        with Image.open(p) as im:          # lazy: header only
            size = im.size
        x0, y0b, x1, y1b = frame_box(leaf, size)
        h = round((y1b - y0b) * W / (x1 - x0))
        # unique page key: a student's pages span multiple run PDFs, and
        # bare leaf names collide across them (HS1_1/003 vs HS1_2/003)
        pkey = f"{leaf.parent.name.removeprefix(CFG.run_prefix)}_{leaf.name}"
        layout_pages.append({"page_dir": pkey, "y0": y, "y1": y + h})
        # remap full-page Schlussstrich y_frac into the frame crop
        H0 = size[1]
        for ln in sorted(sstr_lines(leaf), key=lambda x: float(x.get("y_frac") or 0)):
            yf = float(ln.get("y_frac") or 0)
            yf2 = (yf * H0 - y0b) / (y1b - y0b) if y1b > y0b else 0
            if 0 < yf2 < 1:
                splits_src.append((pkey, round(yf2, 4)))
        y += h
    if not layout_pages:
        return None, None, None, "no kept pages"

    with Image.open(sub_png) as im:
        real_h = im.height
    warn = None
    if abs(real_h - y) > 2:
        warn = f"layout height {y} != submission.png height {real_h}"

    layout = {"width_px": W, "height_px": y, "pages": layout_pages}
    page_splits = defaultdict(list)
    first_pd = layout_pages[0]["page_dir"]
    page_splits[first_pd].append({"y_frac": 0.0, "task_id": "1", "subtask_id": None})
    page_splits[first_pd].append({"y_frac": 0.004, "task_id": None, "subtask_id": "a"})
    for n, (leaf_name, yf) in enumerate(splits_src, start=1):
        page_splits[leaf_name].append(
            {"y_frac": yf, "task_id": None, "subtask_id": _letter(n)})

    pages_manifest = [{"page_dir": lp["page_dir"], "page_index": i,
                       "side": "unknown", "task_id": None, "ocr_confidence": None}
                      for i, lp in enumerate(layout_pages)]
    return layout, dict(page_splits), pages_manifest, warn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", help="single Hörsaal set, e.g. HS7")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if not (a.set or a.all):
        ap.error("pass --set <name> or --all")

    pkg2row, z2pkg = load_csv()
    booklets, extras, _bad = scan()
    z_as_booklet = {z for z in z2pkg if z in booklets}
    hs_map = load_hs_map()
    ABGABE.mkdir(parents=True, exist_ok=True)

    by_set = defaultdict(dict)
    n_done = n_sstr = 0
    for pkg in sorted(set(booklets)):
        if pkg in z_as_booklet:
            continue
        r = pkg2row.get(pkg)
        if not r:
            continue
        matr = (r.get("Matrikelnr.") or "").strip()
        hs = hs_map.get(matr) or "HS_unknown"
        if a.set and hs != a.set:
            continue
        ordered = ordered_leaves(pkg, booklets, extras, z2pkg, z_as_booklet)
        layout, page_splits, pages_manifest, warn = seed_one(matr, hs, ordered)
        if warn:
            print(f"  ! {hs}/{matr}: {warn}")
        if layout is None:
            continue
        sd = ABGABE / hs / matr
        sd.mkdir(parents=True, exist_ok=True)
        link = sd / "submission.png"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.relpath(SUB_ROOT / hs / matr / "submission.png", sd))
        (sd / "partition.json").write_text(json.dumps({"image_layout": layout}))
        n_lines = sum(len(v) for v in page_splits.values()) - 2
        (sd / "corrections.json").write_text(json.dumps({
            "_clean_slate": True, "task_overrides": {}, "subtask_overrides": {},
            "page_splits": page_splits}, ensure_ascii=False))
        by_set[hs][matr] = {"matr_nr": matr,
                            "name": f"{r['Vorname']} {r['Nachname']}".strip(),
                            "modul": None, "fachsemester": None,
                            "pages": pages_manifest}
        n_done += 1; n_sstr += n_lines

    for hs, studs in by_set.items():
        (RUNS / hs).mkdir(parents=True, exist_ok=True)
        (RUNS / hs / "students.json").write_text(json.dumps(
            {"exam_id": hs, "students": studs, "unassigned_pages": []},
            ensure_ascii=False))

    print(f"seeded {n_done} students, {n_sstr} SSTR->SCL, {len(by_set)} sets")
    print("canonicalizing partition.json via process_student ...")
    ok = 0
    for hs, studs in by_set.items():
        for matr, sdata in studs.items():
            try:
                process_student(exam_root=RUNS / hs, abgabe_root=ABGABE, matr=matr,
                                 student_data=sdata, use_auto_scl=False,
                                 use_auto_task=False, no_render=True)
                ok += 1
            except Exception as e:
                print(f"  ! {hs}/{matr}: {e}")
    print(f"partition.json canonicalized for {ok}/{n_done}")


if __name__ == "__main__":
    main()
