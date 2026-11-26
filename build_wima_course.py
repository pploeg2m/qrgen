#!/usr/bin/env python3
"""Assemble the batch's complete nbgrader course tree locally, ready to
upload to osiris. Patches each student's notebook (embeds crop refs)
so no terminal 'apply' step is needed.

Layout built under <batch>/osiris_course/<course_dir>/:
  nbgrader_config.py
  students.csv                       (matr,name,batch,filled_cells)
  source/<assignment>/<notebook>
  submitted/<matr>/<assignment>/
      <notebook>                     (source copy, answer cells → ![..](gid.jpg))
      timestamp.txt
      <grade_id>.jpg                 (crops)

Run from the repo root:  python projects/qrgen/build_wima_course.py [--batch data5]
"""
import csv, json, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scan2e2x.batch import load_batch, pop_batch_arg  # noqa: E402

CFG = load_batch(pop_batch_arg(sys.argv))
SRC_NB = CFG.notebook_path
CROPS = CFG.osiris_submitted
ABGABE = CFG.review_dir
ROSTER = CFG.csv_path
OUT = CFG.course_out
ASSIGN = CFG.assignment
NBNAME = CFG.notebook
COURSE_ID = CFG.course_id
TS = CFG.submission_timestamp


def roster():
    m2n = {}
    with open(ROSTER, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter=";"):
            mt = (r.get("Matrikelnr.") or "").strip()
            if mt.isdigit():
                m2n[mt] = f"{r['Vorname']} {r['Nachname']}".strip()
    return m2n


def matr_set():
    """matr -> HS set short (batch)."""
    out = {}
    for pj in ABGABE.glob("*/*/partition.json"):
        out[pj.parent.name] = pj.parent.parent.name.replace(CFG.run_prefix, "")
    return out


def patch(nb, crop_dir):
    filled = 0
    for c in nb["cells"]:
        md = (c.get("metadata") or {}).get("nbgrader") or {}
        if not (md.get("solution") and md.get("grade")):
            continue
        gid = md.get("grade_id")
        if (crop_dir / f"{gid}.jpg").exists():
            c["source"] = [f"![Antwort {gid}]({gid}.jpg)\n"]; filled += 1
        else:
            c["source"] = ["*Keine Antwort eingereicht.*\n"]
        c.pop("attachments", None); c["outputs"] = c.get("outputs", [])
        if c["cell_type"] == "markdown":
            c.pop("outputs", None)
    return filled


def main():
    if not SRC_NB.exists():
        raise SystemExit(f"source notebook missing: {SRC_NB} — "
                         "generate it first (build_wima_source_nb.py)")
    m2n = roster(); m2s = matr_set()
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "source" / ASSIGN).mkdir(parents=True)
    shutil.copy(SRC_NB, OUT / "source" / ASSIGN / NBNAME)
    (OUT / "nbgrader_config.py").write_text(
        f"# nbgrader_config.py — {CFG.name}, isolated course root\n"
        "c = get_config()  # noqa: F821\n"
        f"c.CourseDirectory.course_id = '{COURSE_ID}'\n"
        "c.CourseDirectory.root = '.'\n")

    rows = []
    src_nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
    for cd in sorted(CROPS.glob("*/" + ASSIGN)):
        matr = cd.parent.name
        dst = OUT / "submitted" / matr / ASSIGN
        dst.mkdir(parents=True)
        n = 0
        for jpg in cd.glob("*.jpg"):
            shutil.copy(jpg, dst / jpg.name); n += 1
        nb = json.loads(json.dumps(src_nb))          # deep copy
        filled = patch(nb, dst)
        (dst / NBNAME).write_text(json.dumps(nb, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        (dst / "timestamp.txt").write_text(TS + "\n", encoding="utf-8")
        rows.append((matr, m2n.get(matr, "?"), m2s.get(matr, "?"), filled))

    with open(OUT / "students.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["matr", "name", "batch", "filled_cells"])
        for r in sorted(rows):
            w.writerow(r)

    tot_files = sum(1 for _ in OUT.rglob("*") if _.is_file())
    print(f"students: {len(rows)}   files: {tot_files}")
    print(f"course tree at {OUT}")


if __name__ == "__main__":
    main()
