# qrgen — QR-coded exam booklet generator

Tools to build print-ready exam PDFs where **every page carries a unique QR code**
(for later scan/OCR reassembly via `scan2e2x`), plus optional page numbers and
per-student Matrikel numbers.

The overall flow is *inspired by* mail-merge (the MS-Word-style "one document ×
a list of recipients" idea) — but this generator is fully self-contained: **no
Microsoft Office / Word suite is needed to run it.** Everything is produced
headlessly by `cpdf` and Python.

Two scripts:

| Script | Purpose |
|--------|---------|
| [`gen_qrcode_numbers.py`](gen_qrcode_numbers.py) | Generate the QR PNGs and a semicolon-separated index CSV (one row per student/page). |
| [`assemble_PDF4exam.py`](assemble_PDF4exam.py) | Assemble the full multi-student exam PDF from lean page templates, stamping page numbers + QR codes + Matrikel with `cpdf`. |

`assemble_PDF4exam.py` is the main tool; `gen_qrcode_numbers.py` is only needed if you
want the standalone CSV index (it shares the QR renderer, so `assemble_PDF4exam.py`
generates any missing PNGs itself).

## Requirements

- **Python 3** with `qrcode[pil]` (Pillow). Install into the project venv:
  ```bash
  .venv/bin/python -m pip install "qrcode[pil]"
  ```
- **cpdf** (Coherent PDF, ≥ 2.9) on `PATH` — does the PDF replication, page
  numbering, and QR stamping. Override the binary with the `CPDF` env var.
- **Ghostscript** (`gs`) is only used for rendering/verification, not required to build.

The QR id scheme is `<idsymbol><student:03d>-P<page:02d>` → e.g. `ID001-P01`.

---

## Examples / use cases

**1. Distinct cover page + 7 identical follow-up pages, 15 students.**
Each exam has 8 pages: the first sheet (template **A** = `front.pdf`) carries the
header lines (name / Matrikel fields), followed by 7 identical grid pages
(template **B** = `back.pdf`). The pattern `ABBBBBBB` spells exactly that.
```bash
python3 assemble_PDF4exam.py front.pdf back.pdf --pattern ABBBBBBB --idsymbol ID -s 15 -o out.pdf
```
→ 15 × 8 = 120 pages; each student numbered 1…8; every page a unique QR `ID001-P01`…`ID015-P08`.

**2. Classic front/back booklet, 50 students.**
A two-page sheet (`front.pdf` + `back.pdf`) simply repeated 8× → 16 pages each.
No `--pattern`, so `-x` drives the repetition.
```bash
python3 assemble_PDF4exam.py front.pdf back.pdf -x 8 --idsymbol ID -s 50 -o out.pdf
```
→ 50 × 16 = 800 pages.

**3. Roster-driven: student count + printed Matrikel from a CSV.**
The number of students comes from the roster file, and each student's Matrikel is
printed to the right of the QR on every page.
```bash
python3 assemble_PDF4exam.py front.pdf back.pdf --pattern ABBBBBBB --idsymbol ID \
  --matrikel sample_roster.csv -o out.pdf
```
→ one 8-page exam per line in `sample_roster.csv` (German `;`-CSV, Matrikel = column 1).

**4. Three templates: cover, then two alternating answer layouts.**
Letters map to templates in order (A = 1st, B = 2nd, C = 3rd). `ABCBC` = cover +
two BC blocks = 5 pages.
```bash
python3 assemble_PDF4exam.py cover.pdf lined.pdf grid.pdf --pattern ABCBC --idsymbol EX -s 30 -o out.pdf
```

**5. Draft without page numbers, custom QR spot.**
```bash
python3 assemble_PDF4exam.py front.pdf back.pdf -x 4 --idsymbol ID -s 5 \
  --no-numbers --qr-x-cm 17.0 --qr-y-cm 0.5 --qr-size-cm 1.5 -o draft.pdf
```

---

## assemble_PDF4exam.py

```
assemble_PDF4exam.py TEMPLATE... --idsymbol ID -o OUT.pdf [options]
```

Pipeline:

1. Take one or more lean, **numberless** page-template PDFs, in order A, B, C, …
2. Build one student booklet from them (see `--pattern`).
3. Stamp page numbers `1..N` — **restarts per student** (every student gets 1, 2, 3, …).
4. Replicate the numbered booklet for S students.
5. Stamp a unique 2 cm QR on every page (lower-left, below the frame by default).
6. Optionally print each student's Matrikel number to the right of the QR.

### QR ids & `--idsymbol` (required)

`--idsymbol` is the **prefix of the string every QR code encodes**. Each page's QR
payload is built as:

```
<idsymbol><student:03d>-P<page:02d>
```

so `--idsymbol ID` yields `ID001-P01` (student 1, page 1), `ID015-P08`
(student 15, page 8), and so on. This is the value a scanner reads back, and it is
what `scan2e2x` uses to identify student + page during OCR reassembly. It also
names the PNG files in `--qr-dir` (e.g. `qrcodes/ID001-P01.png`).

It is **not** printed as visible text (that's the page number, and optionally the
Matrikel). Any string works, and there is no default — you must pass one.

```bash
# QR payloads: ID001-P01, ID001-P02, ...
assemble_PDF4exam.py front.pdf back.pdf -x 8 --idsymbol ID -s 50 -o out.pdf

# different prefix -> EX001-P01, EX001-P02, ...
assemble_PDF4exam.py front.pdf back.pdf -x 8 --idsymbol EX -s 50 -o out.pdf

# namespace two exams so their QR payloads never collide
assemble_PDF4exam.py front.pdf back.pdf -x 8 --idsymbol HS1 -s 50 -o klausur1.pdf
assemble_PDF4exam.py front.pdf back.pdf -x 8 --idsymbol HS2 -s 50 -o klausur2.pdf
```

Notes:
- Use it to **distinguish exams / runs** — reused ids across exams would produce
  identical QR payloads and collide.
- A new `idsymbol` generates a fresh set of PNGs in `--qr-dir`; old ones are left
  untouched (harmless).
- The `001` / `01` zero-padding is fixed in code, not part of `idsymbol`.

### Templates & `--pattern`

- **Default (no `--pattern`)** — pass exactly two templates `FRONT BACK` and `-x N`;
  the sheet `[FRONT BACK]` is replicated N times.
  ```bash
  assemble_PDF4exam.py front.pdf back.pdf -x 8 --idsymbol ID -s 50 -o out.pdf
  ```
- **`--pattern`** — the string maps letters to templates (A = 1st, B = 2nd, …) and
  *is* the per-student page sequence (`-x` is ignored). `ABBBBBBB` = 1× A then 7× B = 8 pages.
  ```bash
  assemble_PDF4exam.py A.pdf B.pdf C.pdf --pattern ABBC --idsymbol ID -s 50 -o out.pdf
  ```

### Students / Matrikel

Give **either** `-s/--students N` **or** `--matrikel FILE` (the file's line count sets S).

`--matrikel FILE` accepts:
- a plain list — one Matrikel number per line, or
- a **German CSV** (`;` separated) — the **first column** is the Matrikel number, rest ignored.

```bash
# sample_roster.csv rows look like: 90000001;Alster;Anna;;HS1;...
assemble_PDF4exam.py front.pdf back.pdf -x 8 --idsymbol ID \
  --matrikel sample_roster.csv -o out.pdf
```
The Matrikel number is printed to the right of the QR on every page of that student.
(No header-row skipping — strip any header line first.)

### Key options

| Option | Default | Meaning |
|--------|---------|---------|
| `--idsymbol` | *(required)* | QR id prefix, e.g. `ID` |
| `-o/--output` | *(required)* | final merged PDF |
| `--pattern` | — | page sequence over template letters (e.g. `ABBC`) |
| `-s/--students` | — | student count (or use `--matrikel`) |
| `-x/--replicate` | — | sheet repeats in default FRONT/BACK mode |
| `--matrikel` | — | Matrikel file (plain lines or `;`-CSV first column) |
| `--qr-dir` | `qrcodes` | where QR PNGs are read/generated (relative to cwd) |
| `--qr-x-cm` / `--qr-y-cm` | `0.95` / `0.12` | QR bottom-left from page bottom-left (cm) |
| `--qr-size-cm` | `2.0` | QR edge length (cm) |
| `--dpi` / `--border` | `300` / `4` | QR PNG resolution / quiet-zone modules |
| `--mat-font` / `--mat-size` / `--mat-gap-cm` | Helvetica / 11 / 0.3 | Matrikel text style & gap from QR |
| `--num-font` / `--num-size` | Helvetica / 12 | page-number font |
| `--num-dx` / `--num-dy` | 36 / 28 | page-number offset from bottom-right corner (pt) |
| `--no-numbers` | off | skip page numbering |
| `--keep-temp` | off | keep the working directory |

Coordinates use cpdf's convention: origin = page **bottom-left**, unit = point
(1 cm = 28.3465 pt). The default QR placement matches the HRW template
(frame left ≈ 0.95 cm, frame bottom ≈ 2.22 cm from page bottom).

### Performance

QR stamping is one `cpdf` call per page (cpdf has no per-page batch image mode),
so a build is `students × pages_per_student` invocations, each on a small
per-student booklet. A 50-student × 16-page run is a few hundred calls — expect a
couple of minutes.

---

## gen_qrcode_numbers.py

Generate the index CSV (semicolon-separated, UTF-8 **with BOM** so German-locale
spreadsheet tools detect it) and, unless `--no-qr`, the QR PNGs.

```
gen_qrcode_numbers.py STUDINUM PAGENUM IDSYMBOL [-o out.csv] [options]
```

```bash
# 50 students × 16 pages, prefix ID  ->  CSV + qrcodes/ID001-P01.png ...
gen_qrcode_numbers.py 50 16 ID -o QRcodeNumbers.csv --qr-dir qrcodes
```

CSV columns: `StudentID;PageNum;Combined[;QRcode]` where `Combined` = `ID001-P01`
and `QRcode` is the absolute path to that page's QR PNG. Use `--no-qr` for the
first three columns only. Options: `--size-cm` (2.0), `--dpi` (300), `--border`
(4), `--qr-dir` (`qrcodes`).

---

## Files in this directory

- `front.pdf` → `../../templates/HRWKlausurFP.pdf`, `back.pdf` → `HRWKlausurBP.pdf` (symlinks to the real seeds)
- `sample_roster.csv` — example roster (German `;`-CSV, Matrikel in column 1)
- `qrcodes/` — generated QR PNGs
- `QRcodeNUmbers.csv` / `.xlsx` — sample index sheet
