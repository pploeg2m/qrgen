#!/usr/bin/env python3
"""Batch-generate per-subtask answer crops (<grade_id>.jpg) for every
student of a batch, using review_server's OWN crop logic + the exam's
grade-map (from the batch's source notebook), so the output is identical
to what the review UI writes on save.

Output: <batch>/osiris_submitted/<matr>/<assignment>/<grade_id>.jpg
Flags each student review-complete (partition has the real task structure)
vs seed-only (still one task → crops would be wrong; skip upload until reviewed).

Run from the repo root:  python projects/qrgen/build_wima_crops.py [--batch data5]
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, "tools"); sys.path.insert(0, "projects/qrgen")
from scan2e2x.batch import load_batch, pop_batch_arg  # noqa: E402
import exam_layout  # noqa: E402
import review_server as rs  # noqa: E402


def main():
    cfg = load_batch(pop_batch_arg(sys.argv))
    nb = cfg.notebook_path
    abgabe = cfg.review_dir
    out = cfg.osiris_submitted
    assign = cfg.assignment

    # install the exam layout into review_server's module global
    rs.LAYOUT = rs._Layout(exam_layout.grade_map(nb))
    print(f"layout: {len(rs.LAYOUT)} graded cells, tasks {sorted(rs.LAYOUT.csubs)}")

    rows = []
    for pj in sorted(abgabe.glob("*/*/partition.json")):
        sd = pj.parent
        matr = sd.name
        part = json.loads(pj.read_text())
        sub_png = sd / "submission.png"
        if not sub_png.exists():
            continue
        corr = json.loads((sd / "corrections.json").read_text())
        reviewed = "_last_modified" in corr
        n_tasks = part.get("n_tasks", 0)
        outdir = out / matr / assign
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


if __name__ == "__main__":
    main()
