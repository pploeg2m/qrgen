#!/usr/bin/env python3
"""Seed review_server corrections.json from the SURE human SSTR positions.

DRAFT.  Converts data4/sstr_positions.json (human-verified Schlussstrich
picks) into the review_server correction model, so a review pass opens
with your cuts already in place instead of a blank clean slate:

    corrections.json = {
        "_clean_slate": true,                 # auto-detection stays OFF
        "page_splits": {                      # your SURE SCLs (the SSTRs)
            "<page_dir>": [ {"y_frac": .., "subtask_id": "1a"}, ... ]
        },
        "subtask_overrides": { "<page_dir>": "1a" }   # first label on the page
    }

Identity/order come from qr.json + the roster CSV (same join as
make_submissions.py): booklet_id -> Packagenr -> Matrikelnr.

Keyed by <page_dir> = the scan leaf name ("001").  sstr_positions.json
keys are "<SET>/<leaf>"; the SET half is dropped because corrections
live under one exam (=one HS set) at a time.

  python projects/qrgen/seed_sstr_corrections.py --out data4/runs/abgabe [--dry-run]

CAVEATS (must be settled before review_server will actually render):
  * y_frac reference: sstr_positions y_frac is within the *full deskewed
    page*.  review_server displays a submission glued from per-answer
    solution.png crops, whose y_frac is within-crop.  If data4 is wired
    to render full pages instead, y_frac passes through unchanged (what
    this draft assumes); otherwise a crop-relative remap is required.
  * data4 leaves lack solution.png/aufgabe.json, so partition_submissions
    cannot build the abgabe tree yet.  This seeder only writes the
    corrections sidecar + a students.json manifest.
"""
import argparse, csv, glob, json, re
from pathlib import Path
from collections import defaultdict

ROOT = Path("data4")
CSV = ROOT / "priorExam" / "anwesenheit_gefiltert_final.csv"
POS = ROOT / "sstr_positions.json"


def load_csv():
    """Packagenr -> (matr, name). Zusatzblattnr folds into its owner."""
    pkg2info, z2pkg = {}, {}
    with open(CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter=";"):
            pkg = (r.get("Packagenr.") or "").strip()
            if not pkg.isdigit():
                continue
            pkg = int(pkg)
            name = f"{r['Vorname']} {r['Nachname']}".strip()
            pkg2info[pkg] = (r["Matrikelnr."].strip(), name)
            m = re.search(r"\d+", (r.get("Zusatzblattnr.") or ""))
            if m:
                z2pkg[int(m.group())] = pkg
    return pkg2info, z2pkg


def leaf_index():
    """(SET, leaf) -> {pkg, matr, name, sheet, side, is_extra}."""
    pkg2info, z2pkg = load_csv()
    idx = {}
    for qp in glob.glob("data4/runs/*/*/qr.json"):
        d = json.load(open(qp))
        leaf = Path(qp).parent
        st = leaf.parent.name.replace("WiMa_SS26_", "")
        bid = d.get("booklet_id")
        if not bid:
            continue
        bid = int(bid)
        is_x = bool(d.get("is_extra_sheet"))
        pkg = z2pkg.get(bid, bid) if is_x else bid          # extras -> owner pkg
        info = pkg2info.get(pkg)
        if not info:
            continue
        idx[(st, leaf.name)] = {
            "pkg": pkg, "matr": info[0], "name": info[1],
            "sheet": d.get("sheet") or 99, "side": d.get("side"),
            "is_extra": is_x,
        }
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "abgabe",
                    help="abgabe root: writes <out>/<EXAM>/<MATR>/corrections.json")
    ap.add_argument("--exam", default="WiMa_SS26",
                    help="exam id used as the <EXAM> dir segment")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    idx = leaf_index()
    pos = json.loads(POS.read_text())

    # matr -> { page_dir -> [splits], first-label per page }
    per_student = defaultdict(lambda: {"page_splits": {}, "subtask_overrides": {},
                                       "name": None, "n_lines": 0})
    n_pages = n_lines = n_skipped = 0
    for key, lines in pos.items():
        st, leaf = key.split("/", 1)
        meta = idx.get((st, leaf))
        if not meta:
            n_skipped += 1
            continue
        matr = meta["matr"]
        S = per_student[matr]
        S["name"] = meta["name"]
        splits = []
        for ln in sorted(lines, key=lambda x: float(x.get("y_frac") or 0)):
            yf = float(ln.get("y_frac") or 0)
            if not (0 < yf < 1):
                continue
            e = {"y_frac": round(yf, 4)}
            lab = ln.get("label")
            if lab:
                e["subtask_id"] = str(lab)
            splits.append(e)
            n_lines += 1
        if not splits:
            continue
        S["page_splits"][leaf] = splits
        first_lab = next((s["subtask_id"] for s in splits if "subtask_id" in s), None)
        if first_lab:
            S["subtask_overrides"][leaf] = first_lab
        S["n_lines"] += len(splits)
        n_pages += 1

    print(f"students seeded: {len(per_student)}   pages: {n_pages}   "
          f"SSTR lines: {n_lines}   pages skipped (no roster join): {n_skipped}")

    if a.dry_run:
        # show one example
        for matr, S in list(per_student.items())[:1]:
            print(f"\nexample {matr} ({S['name']}): {S['n_lines']} lines")
            print(json.dumps({"_clean_slate": True,
                              "page_splits": S["page_splits"],
                              "subtask_overrides": S["subtask_overrides"]},
                             indent=2, ensure_ascii=False)[:800])
        print("\n(dry-run — nothing written)")
        return

    base = a.out / a.exam
    for matr, S in per_student.items():
        d = base / matr
        d.mkdir(parents=True, exist_ok=True)
        corr = {"_clean_slate": True,
                "page_splits": S["page_splits"],
                "subtask_overrides": S["subtask_overrides"]}
        (d / "corrections.json").write_text(
            json.dumps(corr, indent=1, ensure_ascii=False))
    print(f"wrote {len(per_student)} corrections.json under {base}/")


if __name__ == "__main__":
    main()
