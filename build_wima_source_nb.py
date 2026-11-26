#!/usr/bin/env python3
"""Generate the e2x nbgrader SOURCE notebook for the WiMath SS26 exam.

Mirrors the cell contract exam_layout.py parses:
  * one `taskheader_<N>` marker cell per Aufgabe (sets task id N)
  * per subtask: a `<Slug>_<L>_Description0` question cell (locked, ungraded)
    + a `<Slug>_<L>` answer cell (grade+solution, points) — crop = <grade_id>.jpg
Points come from the Lösungsschema (bonus scaled ×0.5 → integer, task 6 = 10).

NOTE: HEADER/TASKS below are EXAM CONTENT. For a new exam, edit them from
its Aufgaben + Lösungsschema and bump TASKS_FOR to the new batch name —
the script refuses to write another batch's notebook with stale content.
Output lands at the batch's notebook path (batch.json "notebook").
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scan2e2x.batch import load_batch, pop_batch_arg  # noqa: E402

TASKS_FOR = "WiMa-SS26"   # batch these TASKS were authored for
HEADER = ("### Klausur Wirtschaftsmathematik — Sommersemester 2026\n\n"
          "Bitte tragen Sie Ihre Lösungen in die dafür vorgesehenen Antwortzellen ein.\n")

# task N -> (slug, title, [(letter, points, question)])
TASKS = {
    1: ("Sparplan", "Zins- und Rentenrechnung", [
        ("A", 6,  "Beschreiben Sie die Kapitalentwicklung während der ersten beiden Halbjahre. "
                  "Berechnen Sie das Kapital nach dem ersten Halbjahr."),
        ("B", 3,  "Stellen Sie einen Term zur Berechnung des Kapitals nach 4 Jahren auf."),
        ("C", 11, "Interpretieren Sie die gegebene Formel vollständig: Kapitalauf-/-abbau, "
                  "Rentenanteil (vor-/nachschüssig) und Rate, Zinssätze der Phasen, zeitlicher "
                  "Ablauf und Gesamtlaufzeit."),
    ]),
    2: ("LinAlg", "Lineare Algebra", [
        ("A", 11, "Entscheiden Sie für A und B, ob sie invertierbar sind (mit Begründung), "
                  "und berechnen Sie gegebenenfalls eine der Inversen."),
        ("B", 4,  "Berechnen Sie CD − DC."),
        ("C", 5,  "Erweiterte Koeffizientenmatrix (Treppenform): Untersuchen Sie die Lösbarkeit "
                  "des LGS in Abhängigkeit vom Parameter a."),
    ]),
    3: ("Marktanteile", "Verteilungsmatrix / Marktanteile", [
        ("A", 4, "Stellen Sie die entsprechende Verteilungsmatrix (VM) auf."),
        ("B", 5, "Formulieren Sie den Zusammenhang zwischen ursprünglicher und neuer Verteilung."),
        ("C", 6, "Berechnen Sie die Marktanteile der vier Unternehmen am Ende des Jahres."),
        ("D", 3, "Gibt es nach wie vor denselben Marktführer wie zu Beginn? Begründen Sie."),
    ]),
    4: ("Gewinnfunktion", "Gewinnfunktion", [
        ("A", 4, "Stellen Sie die Gewinnfunktion G(x) auf."),
        ("B", 2, "Zeigen Sie, dass x = −2 eine Nullstelle der Gewinnfunktion ist."),
        ("C", 5, "Bestimmen Sie mithilfe der Polynomdivision die weiteren Nullstellen von G(x)."),
        ("D", 3, "Bestimmen Sie die Gewinnzone."),
        ("E", 8, "Bestimmen Sie die Extremstellen der Gewinnfunktion und geben Sie an, um "
                 "welche Art von Extremum es sich handelt."),
    ]),
    5: ("Marktgleichgewicht", "Marktgleichgewicht / Renten", [
        ("A", 4, "Übertragen Sie den Graphen schematisch auf Ihr Lösungsblatt und beschriften Sie ihn."),
        ("B", 6, "Berechnen Sie das Marktgleichgewicht (MGG)."),
        ("C", 3, "Berechnen Sie die Produzentenrente PR."),
        ("D", 3, "Berechnen Sie die Konsumentenrente KR."),
        ("E", 4, "Bestimmen Sie das unbestimmte Integral ∫ x·e^{2x} dx (partielle Integration)."),
    ]),
    6: ("Bonus", "Bonus: Ungleichung & Hesse-Matrix", [
        ("A", 5, "Aufgabe 6.1) Bestimmen Sie Definitions- und Lösungsmenge der Ungleichung "
                 "|4x − 6| − (2x − 1) < 5."),
        ("B", 2, "Aufgabe 6.2 a) Berechnen Sie alle partiellen Ableitungen erster Ordnung von f."),
        ("C", 3, "Aufgabe 6.2 b) Erstellen Sie die Hesse-Matrix der Funktion f."),
    ]),
}


def md(source, gid, *, grade=False, solution=False, locked=True, points=None, task=False):
    nb = {"grade": grade, "grade_id": gid, "locked": locked,
          "schema_version": 3, "solution": solution, "task": task}
    if points is not None:
        nb["points"] = points
    return {"cell_type": "markdown", "metadata": {"nbgrader": nb},
            "source": source if isinstance(source, list) else [source]}


def build():
    cfg = load_batch(pop_batch_arg(sys.argv))
    if cfg.name != TASKS_FOR:
        raise SystemExit(
            f"TASKS in this script are for batch {TASKS_FOR!r}, but the "
            f"selected batch is {cfg.name!r} ({cfg.root}). Edit HEADER/TASKS "
            "from the new exam's Aufgaben and bump TASKS_FOR first.")
    cells = [md(HEADER, "HeaderA", locked=True)]
    for n, (slug, title, subs) in TASKS.items():
        total = sum(p for _, p, _ in subs)
        cells.append(md(f"# Task {n}  **[{total} Point(s)]**", f"taskheader_{n}"))
        cells.append(md(f"# {title}\n", f"{slug}_Header"))
        for letter, pts, q in subs:
            low = letter.lower()
            cells.append(md(f"## Aufgabe {low})  **[{pts} Point(s)]**\n\n{q}\n",
                            f"{slug}_{letter}_Description0"))
            cells.append(md(f"## Antwort {low})\n\n*Bitte hier lösen.*\n",
                            f"{slug}_{letter}", grade=True, solution=True,
                            locked=False, points=pts))
    nb = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5}
    out = cfg.notebook_path
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    build()
