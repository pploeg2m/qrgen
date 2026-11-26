#!/usr/bin/env python3
"""Schlussstrich detection on ALL of a batch's non-empty pages via FSLab
Qwen2.5-VL (--batch <root>).

Uses the empty-gate (empty_gate.json) to skip grid-only pages. Resumable:
skips leaves that already have schlussstrich.json. Writes <leaf>/schlussstrich.json.
"""
import glob, json, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scan2e2x.batch import load_batch, pop_batch_arg
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
        # second candidate: LaTeX like \( x \) in label_above is an illegal
        # JSON escape — double any backslash not starting a valid escape
        for fix in (c, re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", c)):
            try:
                return json.loads(fix)
            except Exception:
                continue
    return None


def main():
    c = OpenAIVisionClient(model="Qwen2.5-VL-32B-Instruct-Q4_K_M",
                           base_url="https://llm.fslab.de/v1", api_key="fslab")
    leaves = load_batch(pop_batch_arg(sys.argv)).leaf_glob("oriented.png")
    todo = []
    for leaf in leaves:
        eg = leaf / "empty_gate.json"
        if eg.exists() and json.loads(eg.read_text()).get("is_empty"):
            continue                      # skip grid-only (cost gate)
        if (leaf / "schlussstrich.json").exists():
            continue                      # resume
        todo.append(leaf)
    print(f"non-empty to process: {len(todo)}", flush=True)
    done = err = 0
    for i, leaf in enumerate(todo, 1):
        img = cv2.imread(str(leaf / "oriented.png"))
        try:
            r = parse(c.ask(img, PROMPT))
        except Exception as e:
            err += 1; print(f"  ERR {leaf.parent.name}/{leaf.name}: {e}", flush=True); continue
        if r is None:
            # no/unparseable VLM reply (endpoint down, model unloaded, ...):
            # the empty default would fake a "no Strich" verdict AND block
            # the resume from retrying this leaf — count it as an error.
            err += 1; print(f"  ERR {leaf.parent.name}/{leaf.name}: no parseable VLM reply", flush=True); continue
        (leaf / "schlussstrich.json").write_text(json.dumps(r, ensure_ascii=False))
        done += 1
        if i % 50 == 0:
            print(f"  ...{i}/{len(todo)} (done={done} err={err})", flush=True)
    print(f"COMPLETE: processed={done} err={err}", flush=True)


if __name__ == "__main__":
    main()
