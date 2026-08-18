#!/usr/bin/env python3
"""Build clean-slate review_server data for data4 (fork A).

For every student it writes, under <abgabe>/<SET>/<matr>/:
  submission.png    - deskewed, empty-dropped, QR-ordered full pages stacked
  partition.json    - canonical (via partition_submissions.process_student),
                      image_layout built here and preserved across saves
  corrections.json  - _clean_slate:true + seeded page_splits:
                        * TCL #1 + SCL #1 at the very top (task 1, section 1)
                        * every human SSTR -> the next consecutive SCL (2,3,..)
and, under <runs_root>/<SET>/students.json, the per-set manifest review_server
reads on save.

SSTR positions come from data4/sstr_positions.json (within-page y_frac, which
maps straight through image_layout to the glued submission).

  python projects/qrgen/build_review_data4.py --set WiMa_SS26_HS7   # one set (smoke)
  python projects/qrgen/build_review_data4.py --all
then:
  python tools/review_server.py --port 8766 --runs-root data4/runs --abgabe-root data4/abgabe
"""
import argparse, csv, glob, json, re, sys
from pathlib import Path
from collections import defaultdict
import cv2
from PIL import Image
sys.path.insert(0, "projects/qrgen")
sys.path.insert(0, "tools")
from empty_features_v2 import deskew, frame_bbox
from partition_submissions import process_student

ROOT = Path("data4")
CSV = ROOT / "priorExam" / "anwesenheit_gefiltert_final.csv"
POS = json.loads((ROOT / "sstr_positions.json").read_text())
RUNS = ROOT / "runs"
ABGABE = ROOT / "abgabe"
W = 1000
FIXUPS = {900503: 503}


def load_csv():
    pkg2info, z2pkg = {}, {}
    with open(CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter=";"):
            pkg = (r.get("Packagenr.") or "").strip()
            if not pkg.isdigit():
                continue
            pkg = int(pkg)
            pkg2info[pkg] = (r["Matrikelnr."].strip(),
                             f"{r['Vorname']} {r['Nachname']}".strip())
            m = re.search(r"\d+", (r.get("Zusatzblattnr.") or ""))
            if m:
                z2pkg[int(m.group())] = pkg
    return pkg2info, z2pkg


def scan():
    """booklets pkg->[(sheet,srank,setshort,leaf)]; extras z->[(srank,setshort,leaf)]."""
    booklets, extras = defaultdict(list), defaultdict(list)
    for qp in glob.glob("data4/runs/WiMa_SS26_*/*/qr.json"):
        d = json.load(open(qp))
        leaf = Path(qp).parent
        setshort = leaf.parent.name.replace("WiMa_SS26_", "")
        raw = d.get("qr_raw") or ""
        srank = 0 if d.get("side") == "front" else 1
        mx = re.search(r"X(\d+)-[FB]", raw)
        if d.get("is_extra_sheet") and d.get("booklet_id"):
            extras[int(d["booklet_id"])].append((srank, setshort, leaf.name, leaf))
        elif d.get("booklet_id") and d.get("sheet"):
            booklets[int(d["booklet_id"])].append(
                (int(d["sheet"]), srank, setshort, leaf.name, leaf))
        elif mx:
            extras[int(mx.group(1))].append((srank, setshort, leaf.name, leaf))
        else:
            m = re.search(r"P(\d+)-S(\d+)-[FB]", raw)
            if m:
                pkg = FIXUPS.get(int(m.group(1)), int(m.group(1)))
                booklets[pkg].append((int(m.group(2)), srank, setshort, leaf.name, leaf))
    return booklets, extras


def is_empty(leaf):
    p = leaf / "empty_gate.json"
    return p.exists() and json.load(open(p)).get("is_empty", False)


def _letter(i):
    """0->a, 1->b, ... 25->z, 26->aa, ..."""
    s = ""
    i += 1
    while i > 0:
        i, r = divmod(i - 1, 26)
        s = chr(97 + r) + s
    return s


def _answer_box_top(gd, frame):
    """y_frac of the answer box's TOP edge (the inner rectangle below the
    Name/Matr/Modul header), NOT the outer page border. The box top is the
    first full-width solid horizontal rule below the header band; fall back
    to the outer frame y0 if none is found."""
    x0, y0, x1, y1 = frame
    H = gd.shape[0]
    dark = (gd[:, x0:x1] < 110).mean(1)                 # dark fraction per row
    for y in range(int(0.12 * H), int(0.55 * H)):       # skip header band
        if dark[y] > 0.55:                              # solid box rule
            return round(y / H, 4)
    return round(y0 / H, 4)


def prep_page(leaf):
    """Deskew the full page (NO crop — keeps sstr y_frac mapping intact).
    Returns (full-page PIL, box_top_frac) where box_top_frac is the answer
    box's top edge, used only to place the leading TCL/SCL there."""
    bgr = cv2.imread(str(leaf / "oriented.png"))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gd, ang = deskew(gray)
    H, w = gd.shape
    fb, _ = frame_bbox(gd)
    top_frac = _answer_box_top(gd, fb)
    M = cv2.getRotationMatrix2D((w / 2, H / 2), ang, 1.0)
    out = cv2.warpAffine(bgr, M, (w, H), flags=cv2.INTER_CUBIC,
                         borderValue=(255, 255, 255))
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)), top_frac


def build_students():
    """matr -> {matr, name, setfull, ordered:[(setshort,leaf_name,leafpath)]}."""
    pkg2info, z2pkg = load_csv()
    booklets, extras = scan()
    z_as_booklet = {z for z in z2pkg if z in booklets}
    students = {}
    for pkg in sorted(booklets):
        if pkg in z_as_booklet:
            continue                                  # 2nd booklet: merged into owner
        info = pkg2info.get(pkg)
        if not info:
            continue                                  # orphan pkg (not in CSV) — skip
        matr, name = info
        pages = sorted(booklets[pkg], key=lambda t: (t[0], t[1]))
        ordered = [(s, ln, lp) for _, _, s, ln, lp in pages]
        for z, owner in z2pkg.items():                # 2nd full booklet
            if owner == pkg and z in z_as_booklet:
                for _, _, s, ln, lp in sorted(booklets[z]):
                    ordered.append((s, ln, lp))
        for z, lst in extras.items():                 # X extra sheets
            if z2pkg.get(z) == pkg:
                for _, s, ln, lp in sorted(lst):
                    ordered.append((s, ln, lp))
        setfull = f"WiMa_SS26_{ordered[0][0]}"
        students[matr] = {"matr": matr, "name": name, "setfull": setfull,
                          "ordered": ordered}
    return students


def seed_one(stu):
    """-> (submission Image, image_layout, page_splits, students_pages)."""
    ims, layout_pages, kept = [], [], []
    y = 0
    top_frac = 0.0
    for setshort, leaf_name, leafpath in stu["ordered"]:
        if is_empty(leafpath):
            continue
        sp = leafpath / "solution.png"                # existence gate for compute_partition
        if not sp.exists():
            try:
                sp.symlink_to("oriented.png")
            except FileExistsError:
                pass
        im, box_top = prep_page(leafpath)
        if not kept:
            top_frac = box_top                        # box top of the first kept page
        h = round(im.height * W / im.width)
        im = im.resize((W, h))
        ims.append(im)
        layout_pages.append({"page_dir": leaf_name, "y0": y, "y1": y + h})
        kept.append((setshort, leaf_name))
        y += h
    if not ims:
        return None, None, None, None
    canvas = Image.new("RGB", (W, y), "white")
    yy = 0
    for im in ims:
        canvas.paste(im, (0, yy)); yy += im.height
    layout = {"width_px": W, "height_px": y, "pages": layout_pages}

    # page_splits: TCL#1 + SCL "a" at the TOP OF THE BOX (frame), then each
    # human SSTR -> the next consecutive SCL letter (b, c, d, ...).
    page_splits = defaultdict(list)
    first_pd = kept[0][1]
    page_splits[first_pd].append({"y_frac": top_frac, "task_id": "1", "subtask_id": None})
    page_splits[first_pd].append({"y_frac": round(top_frac + 0.004, 4),
                                  "task_id": None, "subtask_id": "a"})
    n = 1
    for setshort, leaf_name in kept:
        lines = POS.get(f"{setshort}/{leaf_name}") or []
        for ln in sorted(lines, key=lambda x: float(x.get("y_frac") or 0)):
            yf = float(ln.get("y_frac") or 0)
            if not (0 < yf < 1):
                continue
            page_splits[leaf_name].append(
                {"y_frac": round(yf, 4), "task_id": None, "subtask_id": _letter(n)})
            n += 1

    pages_manifest = [{"page_dir": ln, "page_index": i, "side": "unknown",
                       "task_id": None, "ocr_confidence": None}
                      for i, (_, ln) in enumerate(kept)]
    return canvas, layout, dict(page_splits), pages_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", help="single set dir name, e.g. WiMa_SS26_HS7")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if not (a.set or a.all):
        ap.error("pass --set <name> or --all")

    students = build_students()
    by_set = defaultdict(dict)                         # setfull -> {matr: student_data}
    ABGABE.mkdir(parents=True, exist_ok=True)

    n_done = n_sstr = 0
    for matr, stu in sorted(students.items()):
        if a.set and stu["setfull"] != a.set:
            continue
        canvas, layout, page_splits, pages_manifest = seed_one(stu)
        if canvas is None:
            continue
        setfull = stu["setfull"]
        sd = ABGABE / setfull / matr
        sd.mkdir(parents=True, exist_ok=True)
        canvas.save(sd / "submission.png")
        (sd / "partition.json").write_text(json.dumps({"image_layout": layout}))
        n_lines = sum(len(v) for k, v in page_splits.items()) - 2  # minus TCL1+SCL1
        (sd / "corrections.json").write_text(json.dumps({
            "_clean_slate": True, "task_overrides": {}, "subtask_overrides": {},
            "page_splits": page_splits}, ensure_ascii=False))
        by_set[setfull][matr] = {"matr_nr": matr, "name": stu["name"],
                                 "modul": None, "fachsemester": None,
                                 "pages": pages_manifest}
        n_done += 1; n_sstr += n_lines

    for setfull, studs in by_set.items():
        (RUNS / setfull / "students.json").write_text(json.dumps(
            {"exam_id": setfull, "students": studs, "unassigned_pages": []},
            ensure_ascii=False))

    print(f"seeded {n_done} students, {n_sstr} SSTR->SCL, {len(by_set)} sets")
    print("canonicalizing partition.json via process_student ...")
    ok = 0
    for setfull, studs in by_set.items():
        exam_root = RUNS / setfull
        for matr, sdata in studs.items():
            try:
                process_student(exam_root=exam_root, abgabe_root=ABGABE, matr=matr,
                                 student_data=sdata, use_auto_scl=False,
                                 use_auto_task=False, no_render=True)
                ok += 1
            except Exception as e:
                print(f"  ! {setfull}/{matr}: {e}")
    print(f"partition.json canonicalized for {ok}/{n_done}")


if __name__ == "__main__":
    main()
