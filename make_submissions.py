#!/usr/bin/env python3
"""Build one submission.png per student from data4 scans.

Identity: QR 'HRW-A-V1-P<pkg>-S<sheet>-<F|B>'  -> pkg = Packagenr in CSV.
Extra sheets: 'HRW-X-V1-X<z>-<F|B>'            -> z   = Zusatzblattnr in CSV.
Pages of one student are stitched top-to-bottom by (sheet, front<back);
extra sheets (X) are appended at the very end.

Also reports:
  * scan pkg / zusatz numbers with no CSV owner
  * CSV students never hit by a scan
"""
import csv, glob, json, re, sys
from pathlib import Path
from collections import defaultdict
import cv2, numpy as np
from PIL import Image
from empty_features_v2 import deskew

ROOT = Path("data4")
CSV = ROOT / "priorExam" / "anwesenheit_gefiltert_final.csv"
OUT = ROOT / "submissions"
W = 1000  # per-page width in stitched output


def load_csv():
    """pkg->row, zusatz->pkg. zusatz cells may be like '174','x239'."""
    pkg2row, z2pkg = {}, {}
    with open(CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter=";"):
            pkg = (r.get("Packagenr.") or "").strip()
            if not pkg.isdigit():
                continue
            pkg = int(pkg)
            pkg2row[pkg] = r
            z = (r.get("Zusatzblattnr.") or "").strip()
            m = re.search(r"\d+", z)
            if m:
                z2pkg[int(m.group())] = pkg
    return pkg2row, z2pkg


# human-verified OCR corrections: misread pkg -> true pkg
FIXUPS = {900503: 503}   # HS1_2/062 raw 'P900503-S07-B' = booklet 503's missing S07-B


def scan():
    """booklets: pkg -> list of (sheet, side_rank, leaf). extras: znum -> [(rank,leaf)]."""
    booklets = defaultdict(list)
    extras = defaultdict(list)
    bad = []
    for qp in glob.glob("data4/runs/*/*/qr.json"):
        d = json.load(open(qp))
        leaf = Path(qp).parent
        raw = d.get("qr_raw") or ""
        mx = re.search(r"X(\d+)-[FB]", raw)
        mn = re.search(r"P(\d+)-S(\d+)-[FB]", raw)
        # prefer structured fields (already recovered/human-reviewed in s01_qr)
        if d.get("is_extra_sheet") and d.get("booklet_id"):
            extras[int(d["booklet_id"])].append((_srank(d), leaf))
        elif d.get("booklet_id") and d.get("sheet"):
            booklets[int(d["booklet_id"])].append((int(d["sheet"]), _srank(d), leaf))
        elif mx:
            extras[int(mx.group(1))].append((_srank(d, raw), leaf))
        elif mn:
            pkg = FIXUPS.get(int(mn.group(1)), int(mn.group(1)))
            booklets[pkg].append((int(mn.group(2)), _srank(d, raw), leaf))
        else:
            bad.append((str(leaf), raw))
    return booklets, extras, bad


def _srank(d, raw=""):
    s = d.get("side")
    if s == "front":
        return 0
    if s == "back":
        return 1
    return 1 if raw.endswith("-B") else 0


def is_empty(leaf):
    p = leaf / "empty_gate.json"
    return p.exists() and json.load(open(p)).get("is_empty", False)


def deskew_color(path):
    """Deskew the color page using the angle from its gray frame/grid lines."""
    bgr = cv2.imread(str(path))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, ang = deskew(gray)
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), ang, 1.0)
    out = cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_CUBIC,
                         borderValue=(255, 255, 255))
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


def stitch(pages, dest):
    """pages: ordered list of leaf dirs -> vertical submission.png at width W.
    Empty pages are dropped; every kept page is deskewed first."""
    ims = []
    for leaf in pages:
        p = leaf / "oriented.png"
        if not p.exists() or is_empty(leaf):
            continue
        im = deskew_color(p)
        h = round(im.height * W / im.width)
        ims.append(im.resize((W, h)))
    if not ims:
        return 0
    total = sum(i.height for i in ims)
    canvas = Image.new("RGB", (W, total), "white")
    y = 0
    for i in ims:
        canvas.paste(i, (0, y)); y += i.height
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest)
    return len(ims)


SIDE = {0: "F", 1: "B"}


def audit_order(booklets, extras, z2pkg, pkg2row):
    """Every booklet's QR pages must form the consecutive run
    S01-F,S01-B,S02-F,...,SNN-B with no gaps, no duplicates."""
    print("\n===== QR-ORDER AUDIT =====")
    bad = []
    for pkg in sorted(booklets):
        pages = sorted(booklets[pkg], key=lambda t: (t[0], t[1]))
        seq = [(s, r) for s, r, _ in pages]              # (sheet, side_rank)
        dupes = sorted({x for x in seq if seq.count(x) > 1})
        uniq = sorted(set(seq))
        maxsheet = max(s for s, _ in uniq)
        expected = [(s, r) for s in range(1, maxsheet + 1) for r in (0, 1)]
        gaps = [x for x in expected if x not in uniq]
        n_empty = sum(1 for _, _, lf in pages if is_empty(lf))
        if dupes or gaps:
            bad.append(pkg)
            nm = pkg2row.get(pkg, {})
            who = f"{nm.get('Nachname','?')},{nm.get('Vorname','?')}" if nm else "NOT-IN-CSV"
            print(f"\nP{pkg:06d} {who}  ({len(pages)} pages, {n_empty} empty)")
            if gaps:
                print("   MISSING: " + " ".join(f"S{s:02d}-{SIDE[r]}" for s, r in gaps))
            if dupes:
                print("   DUPLICATE: " + " ".join(f"S{s:02d}-{SIDE[r]}" for s, r in dupes))
    # extra sheets ownership
    print("\n----- extra (X) sheets -----")
    for z in sorted(extras):
        owner = z2pkg.get(z)
        who = ""
        if owner and owner in pkg2row:
            who = f"-> pkg {owner} {pkg2row[owner]['Nachname']}"
        elif owner:
            who = f"-> pkg {owner} (not in CSV)"
        else:
            who = "-> NO CSV owner (dropped)"
        print(f"   X{z:06d} ({len(extras[z])} pg) {who}")
    print(f"\n{len(bad)} of {len(booklets)} booklets bail out of consecutive QR order.")


def main():
    build = "--build" in sys.argv
    do_audit = "--audit" in sys.argv
    pkg2row, z2pkg = load_csv()
    booklets, extras, bad = scan()

    if do_audit:
        audit_order(booklets, extras, z2pkg, pkg2row)
        return

    scan_pkgs = set(booklets)
    csv_pkgs = set(pkg2row)
    # a Zusatzblattnr may itself be a full scanned package booklet (student has 2)
    z_as_booklet = {z for z in z2pkg if z in booklets}     # e.g. Mhamad: pkg578 + 577
    z_hit = set(extras) | z_as_booklet                     # zusatz satisfied by X-sheet OR 2nd booklet

    print(f"CSV students: {len(csv_pkgs)}   scanned booklets: {len(scan_pkgs)}   "
          f"extra sheets: {len(extras)}   undecoded pages: {len(bad)}\n")

    # --- errors ---
    orphan = sorted(scan_pkgs - csv_pkgs - set(z2pkg))     # exclude 2nd-booklet Zusatz pkgs
    print(f"[E1] scan pkg NOT in CSV ({len(orphan)}):")
    for p in orphan:
        print(f"      P{p:06d}  ({len(booklets[p])} pages)")

    dbl = sorted(z_as_booklet)
    print(f"\n[D] students with TWO package booklets (main + Zusatz) ({len(dbl)}):")
    for z in dbl:
        p = z2pkg[z]; r = pkg2row[p]
        print(f"      {r['Nachname']}, {r['Vorname']}: main P{p:06d} + Zusatz P{z:06d}")

    missing = sorted(csv_pkgs - scan_pkgs)
    print(f"\n[E2] CSV student NEVER scanned ({len(missing)}):")
    for p in missing:
        r = pkg2row[p]
        print(f"      pkg {p}  {r['Nachname']}, {r['Vorname']}  (Matr {r['Matrikelnr.']})")

    ex_orphan = sorted(z for z in extras if z not in z2pkg)
    print(f"\n[E3] extra-sheet X-num with NO CSV Zusatz owner ({len(ex_orphan)}):")
    for z in ex_orphan:
        print(f"      X{z:06d}")

    z_unhit = sorted(set(z2pkg) - z_hit)
    print(f"\n[E4] CSV Zusatzblattnr with no scanned sheet/booklet ({len(z_unhit)}):")
    for z in z_unhit:
        p = z2pkg[z]; r = pkg2row[p]
        print(f"      Zusatz {z} -> pkg {p} {r['Nachname']}, {r['Vorname']}")

    if bad:
        print(f"\n[E5] undecoded QR pages ({len(bad)}):")
        for leaf, raw in bad:
            print(f"      {leaf}  raw={raw!r}")

    if not build:
        print("\n(analysis only — pass --build to write submission.png files)")
        return

    # --- build submissions ---
    OUT.mkdir(parents=True, exist_ok=True)
    print("\nbuilding submission.png ...")
    z_as_booklet = {z for z in z2pkg if z in booklets}
    n_ok = 0
    for pkg in sorted(scan_pkgs):
        if pkg in z_as_booklet:
            continue                                 # 2nd booklet: merged into its owner
        def _seq(pk):
            ps = sorted(booklets[pk], key=lambda t: (t[0], t[1]))
            return [lf for _, _, lf in ps]
        ordered = _seq(pkg)
        # append this student's Zusatz: 2nd full booklet(s), then X extra sheet(s)
        for z, owner in z2pkg.items():
            if owner == pkg and z in z_as_booklet:
                ordered += _seq(z)
        for z, lst in extras.items():
            if z2pkg.get(z) == pkg:
                for _, leaf in sorted(lst):
                    ordered.append(leaf)
        r = pkg2row.get(pkg)
        if r:
            slug = f"P{pkg:06d}_{r['Nachname']}_{r['Vorname']}"
        else:
            slug = f"P{pkg:06d}_UNKNOWN"
        slug = re.sub(r"[^\w.-]", "_", slug)
        n = stitch(ordered, OUT / slug / "submission.png")
        n_ok += 1
        print(f"  {slug}: {n} pages")
    print(f"\n{n_ok} submission.png written under {OUT}/")


if __name__ == "__main__":
    main()
