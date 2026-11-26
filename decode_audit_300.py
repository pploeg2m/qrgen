#!/usr/bin/env python3
"""True QR decode audit at native 300 DPI (reads ingested original.png)."""
import glob, re
from collections import Counter
from pathlib import Path
import cv2, numpy as np

DET = cv2.QRCodeDetector()
# booklet: HRW-A-V1-P000479-S01-F ; extra sheet: HRW-X-V1-X000174-F
BOOK = re.compile(r"HRW-[A-Z]-V\d+-P(\d+)-S(\w+)-([FB])")
XTRA = re.compile(r"HRW-[A-Z]-V\d+-X(\d+)-([FB])")


def decode(gray):
    for k in range(4):
        v = DET.detectAndDecode(np.ascontiguousarray(np.rot90(gray, k)))[0]
        if v:
            return v
    return None


def localized(gray):
    for k in range(4):
        ok, _ = DET.detect(np.ascontiguousarray(np.rot90(gray, k)))
        if ok:
            return True
    return False


def main():
    leaves = sorted(Path(p).parent for p in glob.glob("data4/runs/*/*/original.png"))
    tot = dec = xsheets = 0
    students = set()
    cat = Counter(); persetfail = Counter()
    fails = []
    for i, leaf in enumerate(leaves):
        g = cv2.imread(str(leaf / "original.png"), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        tot += 1
        v = decode(g)
        if v:
            dec += 1; cat["decoded"] += 1
            if BOOK.search(v):
                students.add(BOOK.search(v).group(1))
            elif XTRA.search(v):
                xsheets += 1
        else:
            setn = leaf.parent.name
            persetfail[setn] += 1
            if localized(g):
                cat["localized_decode_failed"] += 1
            else:
                cat["not_localized"] += 1
            fails.append((setn, leaf.name, "loc" if localized(g) else "noloc"))
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{len(leaves)}", flush=True)

    print(f"\nTOTAL pages:            {tot}")
    print(f"QR decoded @300DPI:     {dec}/{tot}  ({100*dec/tot:.1f}%)")
    print(f"DISTINCT students (QR): {len(students)}")
    print(f"Extra sheets (X):       {xsheets}")
    print(f"failure split:          {dict(cat)}")
    print(f"failures/set:           {dict(persetfail)}")
    print(f"total failures:         {len(fails)}")
    for x in fails:
        print("  ", x)


if __name__ == "__main__":
    main()
