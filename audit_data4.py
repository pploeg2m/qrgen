#!/usr/bin/env python3
"""Audit data4 WiMath scans: students (QR), roster compare, non-empty pages,
extra sheets (X in QR)."""
import glob, re, sys
from collections import Counter, defaultdict
import cv2, fitz, numpy as np

DET = cv2.QRCodeDetector()
# payload e.g. HRW-A-V1-P000479-S01-F ; extra sheets flagged by an 'X'
PAT = re.compile(r"^(?P<sys>HRW-[A-Z]-V\d+)-P(?P<id>[0-9X]+)-S(?P<sheet>[0-9X]+)-(?P<side>[A-Z]+)$")


def decode_any(gray):
    for k in range(4):
        v = DET.detectAndDecode(np.ascontiguousarray(np.rot90(gray, k)))[0]
        if v:
            return v
    return None


def interior_nonempty(gray):
    h, w = gray.shape
    crop = gray[int(0.12 * h):int(0.92 * h), int(0.12 * w):int(0.96 * w)]
    bw = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 10)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    lines = cv2.bitwise_or(cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk),
                           cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk))
    resid = cv2.subtract(bw, lines)
    resid = cv2.morphologyEx(resid, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    frac = float(resid.sum()) / 255.0 / resid.size
    return frac > 0.002, frac


def roster_count(path):
    try:
        sys.path.insert(0, ".")
        from scan2e2x.grading.pruefungsliste import PruefungslisteRoster
        return len(PruefungslisteRoster.from_pdf(path))
    except Exception as e:
        return f"ERR {e}"


def main():
    pdfs = sorted(glob.glob("data4/originals/*.pdf"))
    students = set()
    per_set = {}
    tot = dec = nonempty = xsheets = 0
    sides = Counter()
    x_examples = []
    for pdf in pdfs:
        doc = fitz.open(pdf)
        s_students, s_pages, s_dec, s_ne, s_x = set(), 0, 0, 0, 0
        for pno in range(doc.page_count):
            pix = doc[pno].get_pixmap(dpi=200)
            img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if pix.n >= 3 else img
            s_pages += 1; tot += 1
            ne, _ = interior_nonempty(gray)
            if ne: s_ne += 1; nonempty += 1
            v = decode_any(gray)
            if not v:
                continue
            s_dec += 1; dec += 1
            if "X" in v:
                s_x += 1; xsheets += 1
                if len(x_examples) < 15: x_examples.append((pdf.split("/")[-1], pno + 1, v))
            m = PAT.match(v)
            if m:
                students.add(m.group("id")); s_students.add(m.group("id"))
                sides[m.group("side")] += 1
        per_set[pdf.split("/")[-1]] = dict(students=len(s_students), pages=s_pages,
                                           decoded=s_dec, nonempty=s_ne, xsheets=s_x)
        print(f"{pdf.split('/')[-1]:24s} students={len(s_students):3d} pages={s_pages:3d} "
              f"decoded={s_dec:3d} nonempty={s_ne:3d} X={s_x}", flush=True)

    rc = roster_count("data4/priorExam/Prüfungsliste_SS26.pdf")
    print("\n" + "=" * 60)
    print(f"TOTAL pages:            {tot}")
    print(f"QR decoded:             {dec}/{tot}  ({100*dec/tot:.1f}%)")
    print(f"DISTINCT students (QR): {len(students)}")
    print(f"Roster (Prüfungsliste): {rc}")
    print(f"Non-empty pages (heur): {nonempty}/{tot}  ({100*nonempty/tot:.1f}%)")
    print(f"Extra sheets (X in QR): {xsheets}")
    print(f"side distribution:      {dict(sides)}")
    if x_examples:
        print("X examples:")
        for e in x_examples: print("  ", e)


if __name__ == "__main__":
    main()
