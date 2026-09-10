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
  5. Stamp a unique 2 cm QR code on every page. The payload for a regular sheet is
        <institution>-<doctype>-<template_version>-<package_prefix><pkg:06d>-S<sheet:02d>-<side>
     e.g. HRW-A-V1-P000427-S03-F, where the package id is per-student and the
     sheet/side are derived from the page assuming double-sided printing
     (2 pages per physical sheet: odd page = side F/front, even = side B/back).
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


def add_text_page(pdf, text, x_pt, y_pt, font, size, page):
    """Superimpose `text` at (x_pt, y_pt) from bottom-left on ONE 1-based page."""
    tmp = pdf + ".tmp"
    run(["-add-text", text,
         "-font", font, "-font-size", str(size),
         "-bottomleft", f"{x_pt:.3f} {y_pt:.3f}",
         pdf, str(page), "-o", tmp])
    os.replace(tmp, pdf)


def add_text_cover(pdf, text, font, size, dx_pt, dy_pt):
    """Superimpose `text` on the cover (page 1), offset (dx, dy) from the top-right."""
    tmp = pdf + ".tmp"
    run(["-add-text", text,
         "-font", font, "-font-size", str(size),
         "-topright", f"{dx_pt:.3f} {dy_pt:.3f}",
         pdf, "1", "-o", tmp])
    os.replace(tmp, pdf)


def sheet_side(page):
    """Map a 1-based booklet page to (sheet_number, side).

    Assumes double-sided printing: 2 pages per physical sheet, odd page = side F
    (front), even page = side B (back). For single-sided templates instead, use
    `return page, "F"` here (each page becomes its own sheet, always side F)."""
    sheet = (page - 1) // 2 + 1
    side = "F" if page % 2 == 1 else "B"
    return sheet, side


def regular_payload(args, pkg_number, page):
    """Build the QR payload string for a regular sheet, e.g. HRW-A-V1-P000427-S03-F."""
    sheet, side = sheet_side(page)
    return (f"{args.institution}-{args.doctype}-{args.template_version}-"
            f"{args.package_prefix}{pkg_number:06d}-S{sheet:02d}-{side}")


def ensure_qr(qr_dir, payload, size_cm, dpi, border):
    """Return the path to the QR PNG for `payload`, generating it if missing."""
    path = os.path.join(qr_dir, f"{payload}.png")
    if not os.path.exists(path):
        make_qr(payload, path, size_cm, dpi, border)
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
    # --- QR payload components (regular sheet) ---
    # payload = <institution>-<doctype>-<template_version>-<package_prefix><pkg:06d>-S<sheet:02d>-<side>
    #           e.g. HRW-A-V1-P000427-S03-F
    p.add_argument("--institution", default="HRW",
                   help="institution code, 1st QR field (default: HRW)")
    p.add_argument("--doctype", default="A",
                   help="document-type code, 2nd QR field (default: A)")
    p.add_argument("--template-version", default="V1",
                   help="template-version code, 3rd QR field (default: V1)")
    p.add_argument("--package-prefix", default="P",
                   help="prefix letter for the unique package id, 4th QR field (default: P)")
    p.add_argument("--package-start", type=int, default=1,
                   help="package number for the first student; increments by 1 per "
                        "student (e.g. --package-start 427 -> P000427, P000428, ...)")
    p.add_argument("--idsymbol",
                   help="(deprecated, ignored) old QR id prefix; kept so existing "
                        "command lines still parse")

    p.add_argument("--matrikel",
                   help="path to a Matrikel file: either one number per line, or a "
                        "German CSV (';' separated) whose first column is the Matrikel "
                        "number; sets the student count and is printed right of each QR")
    p.add_argument("--mat-font", default="Helvetica", help="Matrikel font")
    p.add_argument("--mat-size", type=int, default=11, help="Matrikel font size")
    p.add_argument("--mat-gap-cm", type=float, default=0.3,
                   help="gap between QR right edge and Matrikel text, cm (default: 0.3)")

    # --- human-readable QR payload printed next to the QR on every side ---
    p.add_argument("--no-qr-label", action="store_true",
                   help="do NOT print the QR payload as visible text next to the QR "
                        "(the label is printed by default as a fallback if the QR is "
                        "unreadable)")
    p.add_argument("--label-font", default="Helvetica", help="QR-label font")
    p.add_argument("--label-size", type=int, default=9, help="QR-label font size")

    # --- package number printed on each cover page (page 1 of every booklet) ---
    p.add_argument("--no-cover-number", action="store_true",
                   help="do NOT print the package number on the cover page")
    p.add_argument("--cover-num-prefix", default="",
                   help="text placed before the package number on the cover, "
                        "e.g. 'Paket ' -> 'Paket P000427' (default: none)")
    p.add_argument("--cover-num-font", default="Helvetica", help="cover package-number font")
    p.add_argument("--cover-num-size", type=int, default=14, help="cover package-number font size")
    p.add_argument("--cover-num-dx", type=float, default=36.0,
                   help="cover package-number offset from top-right corner, X pt (default: 36)")
    p.add_argument("--cover-num-dy", type=float, default=36.0,
                   help="cover package-number offset from top-right corner, Y pt (default: 36)")

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

        # Text placed to the right of the QR, vertically centred on it. When both
        # the Matrikel and the QR label are printed, stack them on two lines.
        text_x = (args.qr_x_cm + args.qr_size_cm + args.mat_gap_cm) * PT_PER_CM
        center_y = (args.qr_y_cm + args.qr_size_cm / 2) * PT_PER_CM
        want_label = not args.no_qr_label
        want_matrikel = bool(matrikels)
        if want_matrikel and want_label:
            line_h = max(args.mat_size, args.label_size) * 1.15
            mat_y = center_y + line_h / 2 - 0.35 * args.mat_size
            label_y = center_y - line_h / 2 - 0.35 * args.label_size
        else:
            mat_y = center_y - 0.35 * args.mat_size
            label_y = center_y - 0.35 * args.label_size

        # 3. per student: copy the numbered booklet, stamp a unique QR on each page,
        #    plus (by default) the human-readable payload next to it as a fallback.
        student_pdfs = []
        for s in range(1, students + 1):
            pkg_number = args.package_start + (s - 1)
            sp = os.path.join(work, f"student_{s:03d}.pdf")
            shutil.copy(numbered, sp)
            for page in range(1, pages_per_student + 1):
                payload = regular_payload(args, pkg_number, page)
                png = ensure_qr(args.qr_dir, payload,
                                args.qr_size_cm, args.dpi, args.border)
                stamp_qr(sp, page, png, args.qr_x_cm, args.qr_y_cm, args.qr_size_cm)
                if want_label:
                    add_text_page(sp, payload, text_x, label_y,
                                  args.label_font, args.label_size, page)
            if want_matrikel:
                add_text_all(sp, matrikels[s - 1], text_x, mat_y,
                             args.mat_font, args.mat_size)
            if not args.no_cover_number:
                cover_text = f"{args.cover_num_prefix}{args.package_prefix}{pkg_number:06d}"
                add_text_cover(sp, cover_text, args.cover_num_font,
                               args.cover_num_size, args.cover_num_dx, args.cover_num_dy)
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
