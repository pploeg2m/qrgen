#!/usr/bin/env python3
"""Schlussstrich SAMPLE: 2 booklets per set (~26 students) -> FSLab, then
aggregate per-student prevalence. Resumable (reuses schlussstrich.json)."""
import glob, json, re, sys
from collections import defaultdict, Counter
from pathlib import Path
import cv2
sys.path.insert(0, "/Users/paul/scan2e2x")
from scan2e2x.ocr.openai_vision_client import OpenAIVisionClient
from schlussstrich_all import PROMPT, parse

BOOKLETS_PER_SET = 2


def leaf_info(leaf):
    q = json.loads((leaf / "qr.json").read_text()) if (leaf / "qr.json").exists() else {}
    eg = json.loads((leaf / "empty_gate.json").read_text()) if (leaf / "empty_gate.json").exists() else {}
    return q.get("booklet_id"), bool(eg.get("is_empty"))


def pick_sample():
    by_set = defaultdict(lambda: defaultdict(list))   # set -> booklet -> [leaves]
    for p in sorted(glob.glob("data4/runs/WiMa_SS26_*/*/oriented.png")):
        leaf = Path(p).parent; s = leaf.parent.name
        bid, empty = leaf_info(leaf)
        if bid and not empty:
            by_set[s][bid].append(leaf)
    sample = []
    for s, books in by_set.items():
        for bid in sorted(books)[:BOOKLETS_PER_SET]:
            sample += [(s, bid, lf) for lf in books[bid]]
    return sample


def main():
    sample = pick_sample()
    c = OpenAIVisionClient(model="Qwen2.5-VL-32B-Instruct-Q4_K_M",
                           base_url="https://llm.fslab.de/v1", api_key="fslab")
    todo = [(s, b, lf) for s, b, lf in sample if not (lf / "schlussstrich.json").exists()]
    print(f"sample: {len(sample)} pages, {len(todo)} to fetch", flush=True)
    for i, (s, b, lf) in enumerate(todo, 1):
        try:
            r = parse(c.ask(cv2.imread(str(lf / "oriented.png")), PROMPT)) or {"schlussstriche": [], "count": 0}
            (lf / "schlussstrich.json").write_text(json.dumps(r, ensure_ascii=False))
        except Exception as e:
            print(f"  ERR {s}/{lf.name}: {e}", flush=True)
        if i % 25 == 0:
            print(f"  ...{i}/{len(todo)}", flush=True)

    # aggregate per student (booklet)
    per = defaultdict(lambda: {"pages": 0, "ss": 0, "labels": set(), "ruler": 0, "freehand": 0})
    for s, b, lf in sample:
        j = lf / "schlussstrich.json"
        if not j.exists():
            continue
        r = json.loads(j.read_text()); ss = r.get("schlussstriche", []) or []
        key = f"{s.replace('WiMa_SS26_','')}/{b}"
        per[key]["pages"] += 1; per[key]["ss"] += len(ss)
        for x in ss:
            if x.get("label_above"): per[key]["labels"].add(str(x["label_above"]))
            d = x.get("drawn"); per[key][d if d in ("ruler", "freehand") else "freehand"] += 1
    studs = list(per); users = [k for k in studs if per[k]["ss"] > 0]
    tot_ss = sum(per[k]["ss"] for k in studs)
    print("\n=== SCHLUSSSTRICH SAMPLE (per student/booklet) ===")
    print(f"students(booklets)={len(studs)}  using Schlussstrich={len(users)} "
          f"({100*len(users)/max(len(studs),1):.0f}%)  total_SS={tot_ss}")
    ru = sum(per[k]['ruler'] for k in studs); fh = sum(per[k]['freehand'] for k in studs)
    print(f"ruler={ru} freehand={fh}")
    print("id                pages  #SS  #tasks(labels)")
    for k in sorted(studs):
        d = per[k]
        print(f"  {k:16s}  {d['pages']:3d}   {d['ss']:3d}   {len(d['labels'])}")


if __name__ == "__main__":
    main()
