#!/usr/bin/env python3
"""SCL-indicator audit over the Qwen3.8 transcriptions of the GT crops.

Counts the usual boundary suspects in every <crop>.qwen38-27b.md:
  - "Aufgabe N" / "Nr. N"        (task headers)
  - "N) a)" / "Na)" combos       (task+subtask markers)
  - bare "N)" and bare "a)"      (numeric / letter markers)
  - markdown horizontal rules    (--- : the transcribed Schlussstriche)

and validates them against the GT the crops embody: each crop IS one
subtask (…/<Family>_<L>.jpg), so an indicator on the FIRST line should
carry the crop's own subtask letter, and an hr on the LAST line matches
a GT subtask END (Schlussstrich closing the answer). Task numbers are
validated against a name->number map derived by majority vote from the
transcriptions themselves. Run from the repo root.
"""
import glob, json, re
from collections import Counter, defaultdict

AUF = re.compile(r"^\s*(?:aufgabe|nr\.?)[\s:]*([1-6])\b", re.I)
COMBO = re.compile(r"^\s*([1-6])\s*[).\]]?\s*([a-g])\s*[.)\]]")
NUM = re.compile(r"^\s*([1-6])\s*[).\]]")
LETTER = re.compile(r"^\s*([a-g])\s*\.?\)")
HR = re.compile(r"^\s*(?:-{3,}|_{3,}|\*{3,})\s*$")

files = sorted(glob.glob("data4/osiris_submitted/*/*/*.qwen38-27b.md"))
per_crop = []
fam_votes = defaultdict(Counter)

for f in files:
    stem = f.rsplit("/", 1)[1].split(".")[0]
    fam, letter = stem.rsplit("_", 1)
    lines = [l for l in open(f, encoding="utf-8").read().splitlines()]
    nonempty = [(i, l) for i, l in enumerate(lines) if l.strip()]
    rec = {"fam": fam, "gt_letter": letter.lower(), "first": None,
           "mid": [], "hr_last": False, "hr_mid": 0}
    for pos, (i, l) in enumerate(nonempty):
        hit = None
        m = COMBO.match(l)
        if m: hit = ("combo", m.group(1), m.group(2).lower())
        else:
            m = AUF.match(l)
            if m: hit = ("aufgabe", m.group(1), None)
            else:
                m = NUM.match(l)
                if m: hit = ("num", m.group(1), None)
                else:
                    m = LETTER.match(l)
                    if m: hit = ("letter", None, m.group(1).lower())
        if hit:
            if hit[1]: fam_votes[fam][hit[1]] += 1
            if pos == 0: rec["first"] = hit
            else: rec["mid"].append(hit)
        if HR.match(l):
            if pos == len(nonempty) - 1: rec["hr_last"] = True
            else: rec["hr_mid"] += 1
    per_crop.append(rec)

fam_num = {f: v.most_common(1)[0][0] for f, v in fam_votes.items() if v}
n = len(per_crop)
first_any = [r for r in per_crop if r["first"]]
first_letter = [r for r in per_crop if r["first"] and r["first"][2]]
letter_ok = sum(1 for r in first_letter if r["first"][2] == r["gt_letter"])
first_num = [r for r in per_crop if r["first"] and r["first"][1]]
num_ok = sum(1 for r in first_num if r["first"][1] == fam_num.get(r["fam"]))
mid_total = sum(len(r["mid"]) for r in per_crop)
mid_crops = sum(1 for r in per_crop if r["mid"])
kinds = Counter(r["first"][0] for r in first_any)
hr_last = sum(1 for r in per_crop if r["hr_last"])
hr_mid = sum(r["hr_mid"] for r in per_crop)
hr_any = sum(1 for r in per_crop if r["hr_last"] or r["hr_mid"])

print(f"crops: {n}   task-number map (majority vote): {fam_num}")
print(f"\n[SCL at crop START = GT subtask start]")
print(f"  {len(first_any)} crops ({len(first_any)/n:.0%}) open with an indicator: {dict(kinds)}")
print(f"  letter validation: {letter_ok}/{len(first_letter)} "
      f"({letter_ok/max(len(first_letter),1):.0%}) match the crop's GT letter")
print(f"  number validation: {num_ok}/{len(first_num)} "
      f"({num_ok/max(len(first_num),1):.0%}) match the family's task number")
print(f"\n[SCL mid-crop = would-be FALSE boundary]")
print(f"  {mid_total} indicators inside {mid_crops} crops ({mid_crops/n:.0%})")
print(f"\n[Schlussstrich (markdown hr) in transcriptions]")
print(f"  {hr_any} crops ({hr_any/n:.0%}) contain an hr; "
      f"{hr_last} as LAST line (= GT subtask end), {hr_mid} mid-text")

# persist results next to the manifest
out = {
    "n_crops": n, "task_number_map": fam_num,
    "start": {"any": len(first_any), "kinds": dict(kinds),
              "letter_checked": len(first_letter), "letter_ok": letter_ok,
              "number_checked": len(first_num), "number_ok": num_ok},
    "mid": {"indicators": mid_total, "crops": mid_crops},
    "hr": {"crops": hr_any, "last_line": hr_last, "mid_text": hr_mid},
    "per_crop": [
        {"fam": r["fam"], "gt_letter": r["gt_letter"], "first": r["first"],
         "n_mid": len(r["mid"]), "hr_last": r["hr_last"], "hr_mid": r["hr_mid"]}
        for r in per_crop
    ],
}
import json as _json
from pathlib import Path
Path("data4/osiris_submitted/_scl_audit.json").write_text(
    _json.dumps(out, ensure_ascii=False, indent=1))
print("\njson -> data4/osiris_submitted/_scl_audit.json")
