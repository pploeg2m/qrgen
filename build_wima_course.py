#!/usr/bin/env python3
"""Assemble the complete OCR-Klausur-WiMa-SS26 nbgrader course tree locally,
ready to upload to osiris. Patches each student's notebook (embeds crop refs)
so no terminal 'apply' step is needed.

Layout built under data4/osiris_course/OCR-Klausur-WiMa-SS26/:
  nbgrader_config.py
  students.csv                       (matr,name,batch,filled_cells)
  source/HRW-Ang-Stat-Exam/HRW-WiMa-SS26-Exam.ipynb
  submitted/<matr>/HRW-Ang-Stat-Exam/
      HRW-WiMa-SS26-Exam.ipynb       (source copy, answer cells → ![..](gid.jpg))
      timestamp.txt
      <grade_id>.jpg                 (crops)
"""
import csv, json, shutil, datetime
from pathlib import Path

SP = Path("/private/tmp/claude-501/-Users-paul-scan2e2x-projects-qrgen/"
          "bc20c979-5f9d-4ed3-afe8-7ebde5e849cf/scratchpad")
SRC_NB = SP / "HRW-WiMa-SS26-Exam.ipynb"
CROPS = Path("data4/osiris_submitted")
ABGABE = Path("data4/abgabe")
ROSTER = Path("data4/priorExam/anwesenheit_gefiltert_final.csv")
OUT = Path("data4/osiris_course/OCR-Klausur-WiMa-SS26")
ASSIGN = "HRW-Ang-Stat-Exam"
NBNAME = "HRW-WiMa-SS26-Exam.ipynb"
COURSE_ID = "wima_ss26"
TS = "2026-08-03 08:00:00.000000 UTC"


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
        out[pj.parent.name] = pj.parent.parent.name.replace("WiMa_SS26_", "")
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
    m2n = roster(); m2s = matr_set()
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "source" / ASSIGN).mkdir(parents=True)
    shutil.copy(SRC_NB, OUT / "source" / ASSIGN / NBNAME)
    (OUT / "nbgrader_config.py").write_text(
        "# nbgrader_config.py — WiMath SS26, isolated course root\n"
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
