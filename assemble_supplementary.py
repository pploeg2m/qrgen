#!/usr/bin/env python3
"""Build a stack of supplementary answer sheets, each with its own unique QR code.

Unlike the per-student exam booklets from assemble_PDF4exam.py, a supplementary
sheet is a STANDALONE physical sheet with two sides: a cover (front) and a back.
It has NO sheet-number field -- its unique id lives in the package-id field -- so
every side of every sheet carries its OWN QR code:

    <institution>-<doctype>-<template_version>-<prefix><id:06d>-<side>

e.g. HRW-X-V1-X000123-F (cover/front) and HRW-X-V1-X000123-B (back), both sharing
the same sheet id X000123 and differing only in the side letter.

The sheet is built front-then-back, exactly like `--pattern AB` in the main
assembler: pass a FRONT template and a BACK template (each 1 page), or a single
2-page template. Page 1 -> front side (default F), page 2 -> back side (default B).
The side letters are configurable via --front-side / --back-side.

The QR renderer and all cpdf helpers are shared with assemble_PDF4exam.py, which
must sit next to this file (as must gen_qrcode_numbers.py).

    assemble_supplementary.py FRONT [BACK] -n N -o OUT.pdf [options]
"""
import argparse
import os
import shutil
import sys
import tempfile

# Reuse the cpdf helpers and QR renderer from the main assembler next to us.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from assemble_PDF4exam import (  # noqa: E402
    run, page_count, stamp_qr, add_text_page, add_text_cover, ensure_qr, PT_PER_CM,
)


def sup_payload(args, sheet_id, side):
    """Build the QR payload for one side of a supplementary sheet.

    e.g. HRW-X-V1-X000123-F (front) or HRW-X-V1-X000123-B (back)."""
    return (f"{args.institution}-{args.doctype}-{args.template_version}-"
            f"{args.sup_prefix}{sheet_id:06d}-{side}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("templates", nargs="+",
                   help="supplementary sheet template(s): either FRONT BACK (two "
                        "1-page PDFs, = pattern AB) or a single 1- or 2-page PDF. "
                        "The combined sheet must be 1 or 2 pages.")
    p.add_argument("-n", "--count", type=int, required=True,
                   help="number of supplementary sheets to generate")
    p.add_argument("-o", "--output", required=True, help="final merged PDF path")

    # --- QR payload fields (see module docstring) ---
    p.add_argument("--institution", default="HRW", help="1st QR field (default: HRW)")
    p.add_argument("--doctype", default="X", help="2nd QR field (default: X)")
    p.add_argument("--template-version", default="V1", help="3rd QR field (default: V1)")
    p.add_argument("--sup-prefix", default="X",
                   help="prefix letter for the unique sheet id (default: X)")
    p.add_argument("--sup-start", type=int, default=1,
                   help="id for the first sheet; increments by 1 "
                        "(e.g. --sup-start 123 -> X000123, X000124, ...)")
    p.add_argument("--front-side", default="F",
                   help="side letter for the cover/front (page 1) (default: F)")
    p.add_argument("--back-side", default="B",
                   help="side letter for the back (page 2) (default: B)")

    # --- QR placement / rendering (same defaults as assemble_PDF4exam.py) ---
    p.add_argument("--qr-dir", default="qrcodes", help="dir for QR PNGs (default: ./qrcodes)")
    p.add_argument("--qr-x-cm", type=float, default=0.95,
                   help="QR bottom-left X from page-left, cm (default: 0.95)")
    p.add_argument("--qr-y-cm", type=float, default=0.12,
                   help="QR bottom-left Y from page-bottom, cm (default: 0.12)")
    p.add_argument("--qr-size-cm", type=float, default=2.0, help="QR edge length, cm")
    p.add_argument("--dpi", type=int, default=300, help="QR PNG resolution")
    p.add_argument("--border", type=int, default=4, help="QR quiet-zone modules")

    # --- readable payload label printed next to the QR (fallback if QR unreadable) ---
    p.add_argument("--no-qr-label", action="store_true",
                   help="do NOT print the QR payload as readable text next to the QR")
    p.add_argument("--label-font", default="Helvetica", help="QR-label font")
    p.add_argument("--label-size", type=int, default=9, help="QR-label font size")
    p.add_argument("--mat-gap-cm", type=float, default=0.3,
                   help="gap between QR right edge and label text, cm (default: 0.3)")

    # --- unique sheet id printed on the front (cover) page ---
    p.add_argument("--no-sup-number", action="store_true",
                   help="do NOT print the sheet id on the cover page")
    p.add_argument("--sup-num-prefix", default="",
                   help="text before the sheet id on the cover, e.g. 'Zusatz ' "
                        "-> 'Zusatz X000123' (default: none)")
    p.add_argument("--sup-num-font", default="Helvetica", help="cover sheet-id font")
    p.add_argument("--sup-num-size", type=int, default=14, help="cover sheet-id font size")
    p.add_argument("--sup-num-dx", type=float, default=36.0,
                   help="cover sheet-id offset from top-right corner, X pt (default: 36)")
    p.add_argument("--sup-num-dy", type=float, default=36.0,
                   help="cover sheet-id offset from top-right corner, Y pt (default: 36)")

    p.add_argument("--keep-temp", action="store_true", help="keep the working directory")
    args = p.parse_args(argv)

    # cpdf ignores inputs not ending in .pdf; normalise the output name.
    if os.path.splitext(args.output)[1].lower() != ".pdf":
        args.output += ".pdf"
        print(f"note: output has no .pdf extension, writing to {args.output}")

    if args.count < 1:
        p.error("need at least one supplementary sheet (-n >= 1)")
    if len(args.templates) > 2:
        p.error("give at most two templates: FRONT BACK (or a single 1-/2-page PDF)")
    for t in args.templates:
        if not os.path.isfile(t):
            p.error(f"template not found: {t}")

    total_pages = sum(page_count(t) for t in args.templates)
    if total_pages not in (1, 2):
        p.error(f"combined template is {total_pages} pages; a supplementary sheet must "
                f"be 1 (front only) or 2 (front+back). The payload has no sheet-number "
                f"field, so extra pages would collide -- give each sheet its own id.")

    # Side letter per page: page 1 = front, page 2 = back.
    side_of = {1: args.front_side, 2: args.back_side}

    os.makedirs(args.qr_dir, exist_ok=True)

    # Label position: to the right of the QR, vertically centred on it.
    text_x = (args.qr_x_cm + args.qr_size_cm + args.mat_gap_cm) * PT_PER_CM
    label_y = (args.qr_y_cm + args.qr_size_cm / 2) * PT_PER_CM - 0.35 * args.label_size
    want_label = not args.no_qr_label

    sides_desc = "+".join(side_of[i] for i in range(1, total_pages + 1))
    print(f"supplementary: {total_pages}-side sheet ({sides_desc}) x {args.count} "
          f"sheets = {total_pages * args.count} pages total  "
          f"(ids {args.sup_prefix}{args.sup_start:06d}.."
          f"{args.sup_prefix}{args.sup_start + args.count - 1:06d})")

    work = tempfile.mkdtemp(prefix="supp_")
    sheet_pdfs = []
    try:
        # Build the base sheet once (front then back), then copy it per sheet.
        base = os.path.join(work, "sheet_base.pdf")
        run([*args.templates, "-o", base])

        for i in range(args.count):
            sheet_id = args.sup_start + i
            sp = os.path.join(work, f"supp_{sheet_id:06d}.pdf")
            shutil.copy(base, sp)
            for page in range(1, total_pages + 1):
                payload = sup_payload(args, sheet_id, side_of[page])
                png = ensure_qr(args.qr_dir, payload,
                                args.qr_size_cm, args.dpi, args.border)
                stamp_qr(sp, page, png, args.qr_x_cm, args.qr_y_cm, args.qr_size_cm)
                if want_label:
                    add_text_page(sp, payload, text_x, label_y,
                                  args.label_font, args.label_size, page)
            if not args.no_sup_number:
                num_text = f"{args.sup_num_prefix}{args.sup_prefix}{sheet_id:06d}"
                add_text_cover(sp, num_text, args.sup_num_font, args.sup_num_size,
                               args.sup_num_dx, args.sup_num_dy)
            sheet_pdfs.append(sp)
            print(f"  sheet {i + 1}/{args.count} "
                  f"(id {args.sup_prefix}{sheet_id:06d}) done", end="\r", flush=True)
        print()

        run([*sheet_pdfs, "-o", args.output])
        print(f"wrote {args.output}  ({page_count(args.output)} pages)")
    finally:
        if args.keep_temp:
            print(f"work dir kept: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
