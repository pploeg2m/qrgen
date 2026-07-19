#!/usr/bin/env python3
"""Generate a QR-code index CSV, one row per (student, page) pair.

Produces a semicolon-delimited, UTF-8 (BOM) file:

    StudentID;PageNum;Combined;QRcode
    1;1;ID001-P01;/abs/path/qrcodes/ID001-P01.png
    ...

The ``Combined`` field is ``<idsymbol><student:03d>-P<page:02d>``.

For each row a square QR code that encodes ``Combined`` is rendered to a PNG and
its absolute file path is written to the ``QRcode`` column. The PNG carries a DPI
tag so its natural print size is exactly the requested edge length (default 2 cm).

The semicolon separator and BOM make the CSV directly consumable by German-locale
spreadsheet tools; the ``QRcode`` column is the per-page image path. See README.md
for how this index relates to the main PDF generator.
"""
import argparse
import csv
import os
import sys

CM_PER_INCH = 2.54


def make_qr(data, path, size_cm, dpi, border):
    """Render ``data`` as a square QR PNG of ``size_cm`` edge length at ``dpi``."""
    import qrcode
    from PIL import Image

    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        border=border,
        box_size=10,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")

    px = max(1, round(size_cm / CM_PER_INCH * dpi))
    # NEAREST keeps the modules crisp when scaling to the exact pixel size.
    img = img.resize((px, px), Image.NEAREST)
    img.save(path, format="PNG", dpi=(dpi, dpi))


def generate_rows(studinum, pagenum, idsymbol, qr_dir, size_cm, dpi, border, make_images):
    header = ("StudentID", "PageNum", "Combined", "QRcode")
    if not make_images:
        header = header[:3]
    yield header

    for student in range(1, studinum + 1):
        for page in range(1, pagenum + 1):
            combined = f"{idsymbol}{student:03d}-P{page:02d}"
            if not make_images:
                yield (student, page, combined)
                continue
            filename = f"{combined}.png"
            path = os.path.join(qr_dir, filename)
            make_qr(combined, path, size_cm, dpi, border)
            yield (student, page, combined, os.path.abspath(path))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("studinum", type=int, help="number of students")
    parser.add_argument("pagenum", type=int, help="pages per student")
    parser.add_argument("idsymbol", help="prefix symbol for the Combined id, e.g. 'ID'")
    parser.add_argument("-o", "--output", default="-",
                        help="output CSV path (default: stdout)")
    parser.add_argument("--qr-dir", default="qrcodes",
                        help="directory for the generated QR PNGs (default: ./qrcodes)")
    parser.add_argument("--size-cm", type=float, default=2.0,
                        help="QR edge length in centimetres (default: 2.0)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="PNG resolution; sets the natural print size (default: 300)")
    parser.add_argument("--border", type=int, default=4,
                        help="quiet-zone width in QR modules (default: 4)")
    parser.add_argument("--no-qr", action="store_true",
                        help="skip QR generation and emit only the first 3 columns")
    args = parser.parse_args(argv)

    if args.studinum < 1 or args.pagenum < 1:
        parser.error("studinum and pagenum must be >= 1")
    if args.size_cm <= 0 or args.dpi < 1 or args.border < 0:
        parser.error("size-cm must be > 0, dpi >= 1, border >= 0")

    make_images = not args.no_qr
    if make_images:
        os.makedirs(args.qr_dir, exist_ok=True)

    # utf-8-sig writes a BOM, which German-locale spreadsheet tools need to detect UTF-8.
    out = sys.stdout if args.output == "-" else open(args.output, "w", newline="", encoding="utf-8-sig")
    try:
        writer = csv.writer(out, delimiter=";", lineterminator="\n")
        rows = generate_rows(args.studinum, args.pagenum, args.idsymbol,
                             args.qr_dir, args.size_cm, args.dpi, args.border, make_images)
        writer.writerows(rows)
    finally:
        if out is not sys.stdout:
            out.close()


if __name__ == "__main__":
    main()
