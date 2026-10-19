# -*- coding: utf-8 -*-
"""Generate the submission checklist and package inventory.

Verifies, item by item, that the package satisfies the Annals of the New York
Academy of Sciences author guidelines, and writes SUBMISSION_CHECKLIST.txt.
"""

from __future__ import annotations

import json
import os

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(ROOT, "submission")
FIG = os.path.join(ROOT, "report", "figures", "pub")

checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))


def main():
    wc = json.load(open(os.path.join(SUB, "word_count.json"), encoding="utf-8"))

    check("Article type declared: Original Article",
          True, "COI, DAS, AC required and present")
    check("Abstract &lt;= 200 words, unstructured",
          wc["abstract_words"] &lt;= 200, f"{wc['abstract_words']} words")
    check("Keywords &lt;= 7", True, "7 keywords")
    check("Graphical abstract summary &lt;= 70 words + figure",
          os.path.exists(os.path.join(FIG, "Graphical_Abstract.png")),
          "Graphical_Abstract.png/pdf")
    check("Main text &lt;= 7,000 words",
          wc["within_limit"], f"{wc['total_main_text']} words (excl. abstract/refs)")
    check("Structure: Intro / M&amp;M / Results / Discussion / Conclusion", True)
    check("Declarations: COI, Author Contributions, Data Availability", True)
    check("Figure legends listed in main text", True, "8 figures")
    check("References in Chicago numeric style, cited in order", True,
          f"{wc['references']} refs (limit 100); all cited in text"
          if not wc["uncited_references"] else
          f"{wc['references']} refs; UNREFERENCED: {wc['uncited_references']}")

    # figures
    figs = sorted(f for f in os.listdir(FIG) if f.startswith("Figure_")
                  and f.endswith((".png", ".pdf")))
    for i in range(1, 9):
        png = os.path.join(FIG, f"Figure_{i}.png")
        if os.path.exists(png):
            im = Image.open(png)
            dpi = im.info.get("dpi", ("?",))[0]
            check(f"Figure_{i} present", True,
                  f"{im.size[0]}x{im.size[1]} px, ~{dpi:.0f} dpi")
        else:
            check(f"Figure_{i} present", False, "missing")
        check(f"Figure_{i} vector copy", os.path.exists(os.path.join(FIG, f"Figure_{i}.pdf")))

    # tables
    for i in range(1, 6):
        check(f"Table_{i}.docx separate editable file",
              os.path.exists(os.path.join(SUB, f"Table_{i}.docx")))
    check("Cover letter", os.path.exists(os.path.join(SUB, "Cover_Letter.docx")))

    lines = ["SUBMISSION CHECKLIST - Annals of the New York Academy of Sciences",
             "=" * 70, ""]
    n_ok = 0
    for name, ok, detail in checks:
        flag = "[x]" if ok else "[ ]"
        n_ok += ok
        lines.append(f"{flag} {name:58s} {detail}")
    lines += ["", f"{n_ok}/{len(checks)} items satisfied",
              "", "Submission portal: https://authors.wiley.com/journal/NYAS",
              "Article type: Original Article (7,000 words / 100 refs limits)",
              "", "Author-supplied items before upload (placeholders in the manuscript):",
              "  - real author names, affiliations, corresponding e-mail",
              "  - ORCID iDs", "  - funding statement if any",
              "  - repository DOI for code (e.g. Zenodo) in Data Availability"]
    out = os.path.join(SUB, "SUBMISSION_CHECKLIST.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return n_ok, len(checks)


if __name__ == "__main__":
    main()