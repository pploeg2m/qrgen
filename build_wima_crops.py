#!/usr/bin/env python3
"""Batch-generate per-subtask answer crops (<grade_id>.jpg) for every data4
student, using review_server's OWN crop logic + the WiMath grade-map, so the
output is identical to what the review UI writes on save.

Output: data4/osiris_submitted/<matr>/HRW-Ang-Stat-Exam/<grade_id>.jpg
Flags each student review-complete (partition has the real 6-task structure)
vs seed-only (still one task → crops would be wrong; skip upload until reviewed).
"""
import json, sys
from pathlib import Path
sys.path.insert(0, "tools"); sys.path.insert(0, "projects/qrgen")
import exam_layout
import review_server as rs

NB = Path("/private/tmp/claude-501/-Users-paul-scan2e2x-projects-qrgen/"
          "bc20c979-5f9d-4ed3-afe8-7ebde5e849cf/scratchpad/HRW-WiMa-SS26-Exam.ipynb")
ABGABE = Path("data4/abgabe")
OUT = Path("data4/osiris_submitted")
ASSIGN = "HRW-Ang-Stat-Exam"

# install the WiMath layout into review_server's module global
rs.LAYOUT = rs._Layout(exam_layout.grade_map(NB))
print(f"layout: {len(rs.LAYOUT)} graded cells, tasks {sorted(rs.LAYOUT.csubs)}")

rows = []
for pj in sorted(ABGABE.glob("*/*/partition.json")):
    sd = pj.parent
    matr = sd.name
    part = json.loads(pj.read_text())
    sub_png = sd / "submission.png"
    if not sub_png.exists():
        continue
    corr = json.loads((sd / "corrections.json").read_text())
    reviewed = "_last_modified" in corr
    n_tasks = part.get("n_tasks", 0)
    outdir = OUT / matr / ASSIGN
    try:
        res = rs._regen_jpegs(sub_png, part, outdir)
        n = res.get("filled", 0)
    except Exception as e:
        rows.append((matr, "ERR", str(e)[:40], reviewed, n_tasks)); continue
    rows.append((matr, n, "", reviewed, n_tasks))

complete = [r for r in rows if r[3] and r[4] >= 5]
seed = [r for r in rows if not (r[3] and r[4] >= 5)]
tot_crops = sum(r[1] for r in rows if isinstance(r[1], int))
print(f"\nstudents: {len(rows)}   crops written: {tot_crops}")
print(f"review-complete (reviewed & >=5 tasks): {len(complete)}")
print(f"seed-only / partial (crops not upload-ready yet): {len(seed)}")
print("\nreview-complete students:")
for m, n, _, rev, nt in complete:
    print(f"  {m}: {n} crops, {nt} tasks")
