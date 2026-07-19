#!/usr/bin/env python3
"""Build a per-student exam PDF from lean seed page templates, using cpdf.

Pipeline:

  1. Take one or more lean raw PDFs, NO page numbers. In order they are the
     templates A, B, C, ... (1st positional = A, 2nd = B, ...).
  2. Build one student booklet from those templates:
       - with --pattern ABBBBBBB: page 1 = A, then 7x B  -> 8 pages (the pattern
         IS the per-student page sequence; -x is ignored).
       - without --pattern (original behaviour): exactly two templates FRONT BACK
         replicated -x times -> [A B] * X.
  3. Stamp page numbers 1..N onto that booklet -> numbering RESTARTS per student
     (every student gets pages 1, 2, 3, ...).
  4. Replicate the numbered booklet for S students.
  5. Stamp a unique 2 cm QR code on every page
     (one per page, from the id scheme: <idsymbol><student:03d>-P<page:02d>).
  6. If a Matrikel file is given (--matrikel <file>, one number per line), print
     each student's Matrikel number to the right of the QR on every one of their pages.

cpdf cannot find-and-replace baked-in page numbers, which is exactly why the
templates must start numberless: we *add* numbers in step 3 rather than replace.

QR placement uses cpdf's -draw mode (origin = page bottom-left, unit = PostScript
point; 1 cm = 28.3465 pt). The image is drawn into a 1x1 unit square, so scaling
by size_cm-in-points yields the exact physical size.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

# QR rendering is shared with the CSV generator living next to this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_qrcode_numbers import make_qr  # noqa: E402

PT_PER_CM = 72.0 / 2.54
CPDF = os.environ.get("CPDF", "cpdf")


def run(args):
    subprocess.run([CPDF, *args], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def page_count(pdf):
    out = subprocess.run([CPDF, "-pages", pdf], check=True, capture_output=True, text=True)
    return int(out.stdout.strip())


def stamp_qr(pdf, page, png, x_cm, y_cm, size_cm):
    """Draw `png` onto a single 1-based `page` of `pdf` in place."""
    x = x_cm * PT_PER_CM
    y = y_cm * PT_PER_CM
    s = size_cm * PT_PER_CM
    tmp = pdf + ".tmp"
    run([pdf, str(page), "-draw",
         "-draw-png", f"QR={png}",
         "-push",
         "-mtrans", f"{x:.3f} {y:.3f}",
         "-mscale", f"0 0 {s:.3f} {s:.3f}",
         "-image", "QR",
         "-pop",
         "-o", tmp])
    os.replace(tmp, pdf)


def add_text_all(pdf, text, x_pt, y_pt, font, size):
    """Superimpose `text` at (x_pt, y_pt) from page bottom-left on ALL pages."""
    tmp = pdf + ".tmp"
    run(["-add-text", text,
         "-font", font, "-font-size", str(size),
         "-bottomleft", f"{x_pt:.3f} {y_pt:.3f}",
         pdf, "-o", tmp])
    os.replace(tmp, pdf)


def ensure_qr(qr_dir, idsymbol, student, page, size_cm, dpi, border):
    combined = f"{idsymbol}{student:03d}-P{page:02d}"
    path = os.path.join(qr_dir, f"{combined}.png")
    if not os.path.exists(path):
        make_qr(combined, path, size_cm, dpi, border)
    return path


def resolve_matrikels(args, parser):
    """Read Matrikel numbers from --matrikel: either one number per line, or a
    German CSV (';' separated) whose FIRST column holds the Matrikel number."""
    if not args.matrikel:
        return None
    if not os.path.isfile(args.matrikel):
        parser.error(f"--matrikel file not found: {args.matrikel}")
    nums = []
    with open(args.matrikel, encoding="utf-8-sig") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            first = ln.split(";", 1)[0].strip().strip('"').strip()
            if first:
                nums.append(first)
    if not nums:
        parser.error(f"--matrikel file is empty: {args.matrikel}")
    return nums


def build_sequence(args, parser):
    """Return the ordered list of template paths making up one student booklet."""
    templates = args.templates
    if args.pattern:
        seq = []
        for ch in args.pattern.strip().upper():
            if not ("A" <= ch <= "Z"):
                parser.error(f"--pattern letter {ch!r} is not A-Z")
            idx = ord(ch) - ord("A")
            if idx >= len(templates):
                parser.error(f"--pattern uses '{ch}' but only {len(templates)} "
                             f"template(s) were given")
            seq.append(templates[idx])
        return seq, f"pattern {args.pattern.strip().upper()}"
    # original behaviour: FRONT BACK replicated X times
    if len(templates) != 2:
        parser.error("without --pattern you must pass exactly two templates: FRONT BACK")
    if not args.replicate:
        parser.error("-x/--replicate is required when --pattern is not given")
    return [templates[0], templates[1]] * args.replicate, f"front+back x{args.replicate}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("templates", nargs="+",
                   help="lean seed PDFs with NO page numbers, in order A B C ... "
                        "(without --pattern: exactly two = FRONT BACK)")
    p.add_argument("--pattern",
                   help="page sequence over template letters, e.g. ABBBBBBB "
                        "(A=1st template, B=2nd, ...). If omitted: FRONT BACK x -x.")
    p.add_argument("-s", "--students", type=int,
                   help="number of students (optional if --matrikel file given)")
    p.add_argument("-x", "--replicate", type=int,
                   help="sheet replications in default FRONT/BACK mode")
    p.add_argument("-o", "--output", required=True, help="final merged PDF path")
    p.add_argument("--idsymbol", required=True,
                   help="QR id prefix string, e.g. 'ID' -> 'ID001-P01' (required)")

    p.add_argument("--matrikel",
                   help="path to a Matrikel file: either one number per line, or a "
                        "German CSV (';' separated) whose first column is the Matrikel "
                        "number; sets the student count and is printed right of each QR")
    p.add_argument("--mat-font", default="Helvetica", help="Matrikel font")
    p.add_argument("--mat-size", type=int, default=11, help="Matrikel font size")
    p.add_argument("--mat-gap-cm", type=float, default=0.3,
                   help="gap between QR right edge and Matrikel text, cm (default: 0.3)")

    p.add_argument("--qr-dir", default="qrcodes", help="dir for QR PNGs (default: ./qrcodes)")
    # Defaults place the QR lower-left, right below the frame, flush to its left
    # border (HRW template: frame left = 0.95 cm, frame bottom = 2.22 cm from
    # page bottom; a 2 cm QR with a 0.1 cm gap sits at y = 0.12 cm).
    p.add_argument("--qr-x-cm", type=float, default=0.95,
                   help="QR bottom-left X from page-left, cm (default: 0.95 = frame left border)")
    p.add_argument("--qr-y-cm", type=float, default=0.12,
                   help="QR bottom-left Y from page-bottom, cm (default: 0.12 = below frame bottom)")
    p.add_argument("--qr-size-cm", type=float, default=2.0, help="QR edge length, cm")
    p.add_argument("--dpi", type=int, default=300, help="QR PNG resolution")
    p.add_argument("--border", type=int, default=4, help="QR quiet-zone modules")

    p.add_argument("--num-font", default="Helvetica", help="page-number font")
    p.add_argument("--num-size", type=int, default=12, help="page-number font size")
    p.add_argument("--num-dx", type=float, default=36.0,
                   help="page-number offset from bottom-right corner, X pt (default: 36)")
    p.add_argument("--num-dy", type=float, default=28.0,
                   help="page-number offset from bottom-right corner, Y pt (default: 28)")
    p.add_argument("--no-numbers", action="store_true", help="skip page numbering")

    p.add_argument("--keep-temp", action="store_true", help="keep the working directory")
    args = p.parse_args(argv)

    # cpdf ignores input files whose name does not end in .pdf, so normalise the
    # output name (otherwise reading it back for the page count would fail).
    if os.path.splitext(args.output)[1].lower() != ".pdf":
        args.output += ".pdf"
        print(f"note: output has no .pdf extension, writing to {args.output}")

    # --- resolve students / matrikels ---
    matrikels = resolve_matrikels(args, p)
    if matrikels is not None:
        if args.students is not None and args.students != len(matrikels):
            p.error(f"--students ({args.students}) != number of Matrikel entries "
                    f"({len(matrikels)})")
        students = len(matrikels)
    else:
        if args.students is None:
            p.error("either -s/--students or --matrikel <file> is required")
        students = args.students
    if students < 1:
        p.error("need at least one student")

    # --- resolve the booklet page sequence ---
    seq, mode_desc = build_sequence(args, p)
    pages_per_student = sum(page_count(t) for t in seq)
    print(f"booklet: {mode_desc} -> {pages_per_student} pages/student"
          f"  x {students} students = {pages_per_student * students} pages total"
          f"{'  (+Matrikel)' if matrikels else ''}")

    os.makedirs(args.qr_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="klausur_")
    try:
        # 1. build the raw booklet from the template sequence (no numbers)
        booklet_raw = os.path.join(work, "booklet_raw.pdf")
        run([*seq, "-o", booklet_raw])

        # 2. stamp page numbers 1..pages_per_student -> restarts per student
        numbered = os.path.join(work, "numbered.pdf")
        if args.no_numbers:
            shutil.copy(booklet_raw, numbered)
        else:
            run(["-add-text", "%Page",
                 "-font", args.num_font, "-font-size", str(args.num_size),
                 "-bottomright", f"{args.num_dx} {args.num_dy}",
                 booklet_raw, "-o", numbered])

        # Matrikel text position: just right of the QR, vertically centred on it.
        mat_x = (args.qr_x_cm + args.qr_size_cm + args.mat_gap_cm) * PT_PER_CM
        mat_y = (args.qr_y_cm + args.qr_size_cm / 2) * PT_PER_CM - 0.35 * args.mat_size

        # 3. per student: copy the numbered booklet, stamp a unique QR on each page
        student_pdfs = []
        for s in range(1, students + 1):
            sp = os.path.join(work, f"student_{s:03d}.pdf")
            shutil.copy(numbered, sp)
            for page in range(1, pages_per_student + 1):
                png = ensure_qr(args.qr_dir, args.idsymbol, s, page,
                                args.qr_size_cm, args.dpi, args.border)
                stamp_qr(sp, page, png, args.qr_x_cm, args.qr_y_cm, args.qr_size_cm)
            if matrikels:
                add_text_all(sp, matrikels[s - 1], mat_x, mat_y,
                             args.mat_font, args.mat_size)
            student_pdfs.append(sp)
            print(f"  student {s}/{students} done", end="\r", flush=True)
        print()

        # 4. merge all students -> final
        run([*student_pdfs, "-o", args.output])
        print(f"wrote {args.output}  ({page_count(args.output)} pages)")
    finally:
        if args.keep_temp:
            print(f"work dir kept: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
