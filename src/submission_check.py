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
    checks.append((name, None if ok is None else bool(ok), detail))


def main():
    wc = json.load(open(os.path.join(SUB, "word_count.json"), encoding="utf-8"))

    check("Article type declared: Original Article",
          None, "Manual confirmation required before submission")
    check("Abstract <= 200 words, unstructured",
          wc["abstract_words"] <= 200, f"{wc['abstract_words']} words")
    check("Keywords <= 7", None, "Manual verification required")
    check("Graphical abstract summary <= 70 words + figure",
          os.path.exists(os.path.join(FIG, "Graphical_Abstract.png")),
          "Graphical_Abstract.png/pdf")
    check("Main text <= 7,000 words",
          wc["within_limit"], f"{wc['total_main_text']} words (excl. abstract/refs)")
    check("Structure: Intro / M&M / Results / Discussion / Conclusion", None,
          "Manual review required")
    check("Declarations: COI, Author Contributions, Data Availability", None,
          "Presence can be checked; truth and author approval require manual review")
    check("Figure legends listed in main text", None, "Manual cross-check required")
    check("References in Chicago numeric style, cited in order", None,
          f"{wc['references']} refs (limit 100); uncited references: {wc['uncited_references']}. Citation style/order need manual verification.")

    # These items are deliberately not auto-passed: the repository records known
    # blockers that must be resolved before a submission-ready package can be claimed.
    source_text = open(os.path.join(ROOT, "src", "make_manuscript.py"), encoding="utf-8").read()
    check("Author placeholders replaced",
          not any(token in source_text for token in ("First A. Author", "Second B. Author", "Third C. Author")),
          "Replace author names, affiliations, and contribution statements")
    check("Historical experiments rerun with untouched test set", False,
          "Historical metrics lack validation-split metadata; corrected reruns are not included")
    check("Dataset source revisions and licenses verified",
          "not recorded" not in open(os.path.join(ROOT, "docs", "DATA_PROVENANCE.md"), encoding="utf-8").read(),
          "PlantDoc revision/license/checksums and PlantVillage revision/license still need verification")
    required_artifacts = (
        "cache/meta.json", "cache_pd/meta.json",
        "runs/mtdnet_s0/logits.pt", "runs/robustness_ms.json",
    )
    missing_artifacts = [p for p in required_artifacts if not os.path.exists(os.path.join(ROOT, p))]
    check("All statistical inputs archived", not missing_artifacts,
          "Missing: " + ", ".join(missing_artifacts) if missing_artifacts else "Required inputs found")

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
    n_manual = 0
    for name, ok, detail in checks:
        flag = "[x]" if ok is True else "[ ]" if ok is False else "[?]"
        n_ok += int(ok is True)
        n_manual += int(ok is None)
        lines.append(f"{flag} {name:58s} {detail}")
    lines += ["", f"{n_ok}/{len(checks)} items passed automatically",
              f"{n_manual} items require manual review",
              "STATUS: NOT SUBMISSION-READY while any [ ] or [?] item remains.",
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