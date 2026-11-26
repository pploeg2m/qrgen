#!/usr/bin/env python3
"""Schlussstrich detection on a sample via FSLab Qwen2.5-VL."""
import argparse, glob, json, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, "/Users/paul/scan2e2x")
from scan2e2x.ocr.openai_vision_client import OpenAIVisionClient

PROMPT = """You are analyzing ONE page of a handwritten German exam written on squared (grid) paper inside a printed black rectangular frame.

TASK: Find every "Schlussstrich" — a horizontal line the student draws to mark the END of a (sub)task answer. It runs roughly left-to-right across most of the writing area.

A Schlussstrich:
- is HANDWRITTEN in pen (usually blue or black), separate from any printed line;
- is roughly horizontal - may be slightly wavy or tilted if freehand, or straight if drawn with a ruler;
- may be a single stroke OR 2-3 collinear segments with small gaps;
- spans most of the width, but length varies: it may overhang past the frame or stop short of it;
- stands alone, with a finished answer above it and blank space or the next answer below.

Do NOT report:
- the printed page frame/border or the faint printed grid squares;
- the printed vertical correction-margin line on the left edge;
- an underline beneath a heading or a task label (e.g. under "4) b)");
- fraction bars, equals-signs, or underlines inside a calculation;
- horizontal lines that are part of a TABLE of numbers / a bordered box (a line separating table cells is NOT a Schlussstrich);
- diagonal, wavy-scribble, or X-shaped strike-through / cancellation marks (those are cancellations, not Schlussstriche).

Output STRICT JSON only, nothing else:
{"schlussstriche":[{"y_frac": <0..1 vertical position from top of page>, "drawn": "ruler"|"freehand", "reaches_full_width": true|false, "label_above": "<subtask label just above this line, e.g. '4) b)', or null>", "confidence": <0..1>}], "count": <int>}
If there are none, return {"schlussstriche": [], "count": 0}. No commentary."""


def parse(txt):
    t = re.sub(r"```[a-z]*\n?", "", txt or "").strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    for c in ([m.group(0)] if m else [t]):
        try:
            return json.loads(c)
        except Exception:
            continue
    return None


def nonempty(gray):
    h, w = gray.shape
    crop = gray[int(0.12*h):int(0.92*h), int(0.12*w):int(0.96*w)]
    bw = cv2.adaptiveThreshold(crop,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,10)
    hk=cv2.getStructuringElement(cv2.MORPH_RECT,(25,1)); vk=cv2.getStructuringElement(cv2.MORPH_RECT,(1,25))
    lines=cv2.bitwise_or(cv2.morphologyEx(bw,cv2.MORPH_OPEN,hk),cv2.morphologyEx(bw,cv2.MORPH_OPEN,vk))
    resid=cv2.morphologyEx(cv2.subtract(bw,lines),cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    return float(resid.sum())/255.0/resid.size > 0.002


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="data4/runs/WiMa_SS26_HS7")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    c = OpenAIVisionClient(model="Qwen2.5-VL-32B-Instruct-Q4_K_M",
                           base_url="https://llm.fslab.de/v1", api_key="fslab")
    leaves = sorted(Path(p).parent for p in glob.glob(f"{args.run}/*/oriented.png"))
    if args.limit:
        leaves = leaves[:args.limit]
    per_student = defaultdict(lambda: {"pages":0, "ss":0, "labels":set(), "ruler":0, "freehand":0})
    n_pages = n_nonempty = n_with_ss = total_ss = 0
    for i, leaf in enumerate(leaves, 1):
        img = cv2.imread(str(leaf/"oriented.png"))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        q = json.loads((leaf/"qr.json").read_text()) if (leaf/"qr.json").exists() else {}
        sid = q.get("booklet_id") or "?"
        n_pages += 1
        if not nonempty(gray):
            (leaf/"schlussstrich.json").write_text(json.dumps({"count":0,"empty":True}))
            continue
        n_nonempty += 1; per_student[sid]["pages"] += 1
        try:
            r = parse(c.ask(img, PROMPT)) or {"schlussstriche":[], "count":0}
        except Exception as e:
            print(f"  {leaf.name}: ERR {e}", flush=True); continue
        ss = r.get("schlussstriche", []) or []
        (leaf/"schlussstrich.json").write_text(json.dumps(r, ensure_ascii=False))
        total_ss += len(ss)
        if ss: n_with_ss += 1
        per_student[sid]["ss"] += len(ss)
        for s in ss:
            if s.get("label_above"): per_student[sid]["labels"].add(str(s["label_above"]))
            per_student[sid][s.get("drawn","freehand") if s.get("drawn") in ("ruler","freehand") else "freehand"] += 1
        if i % 20 == 0: print(f"  ...{i}/{len(leaves)}", flush=True)

    print(f"\n=== HS7 Schlussstrich sample ===")
    print(f"pages={n_pages} nonempty={n_nonempty} pages_with_SS={n_with_ss} total_SS={total_ss}")
    studs = list(per_student)
    users = [s for s in studs if per_student[s]["ss"] > 0]
    print(f"students={len(studs)}  using Schlussstrich={len(users)} ({100*len(users)/max(len(studs),1):.0f}%)")
    print("per student: id  pages  #SS  #distinct_labels  ruler/freehand")
    for s in sorted(studs):
        d = per_student[s]
        print(f"  {s}  pages={d['pages']:2d}  SS={d['ss']:2d}  labels={len(d['labels']):2d}  ruler={d['ruler']} freehand={d['freehand']}")


if __name__ == "__main__":
    main()
