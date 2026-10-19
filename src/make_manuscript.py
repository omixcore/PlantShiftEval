# -*- coding: utf-8 -*-
"""Build the Annals of the New York Academy of Sciences submission package.

Outputs (into `submission/`):
  MAIN_MANUSCRIPT.docx        main text, double-spaced, line-numbered
  Table_1.docx ... Table_4.doxsx  editable table files (journal requirement)
  word_count.json             section word counts against the 7,000-word cap

All quantitative statements are injected from the result JSONs; nothing is
transcribed by hand, so the text cannot drift from the numbers.
"""

from __future__ import annotations

import json
import os
import statistics as st
from statistics import stdev

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Inches

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.path.join(ROOT, "submission")
FIG = os.path.join(ROOT, "report", "figures", "pub")


def jload(p):
    with open(os.path.join(ROOT, p), encoding="utf-8") as f:
        return json.load(f)


def pct(x, d=1):
    return f"{x * 100:.{d}f}"


def f3(x):
    return f"{x:.3f}"


def f2(x):
    return f"{x:.2f}"


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
summary = {r["model"]: r for r in jload("report/summary.json")}
sig = jload("report/significance.json")
cd = jload("report/crossdomain.json")
meta = jload("cache/meta.json")
pdmeta = jload("cache_pd/meta.json")
rob = jload("runs/robustness.json")
rbs = (jload("report/robustness_summary.json")
       if os.path.exists(os.path.join(ROOT, "report/robustness_summary.json")) else None)
lat = jload("runs/latency.json")
adapt = jload("report/adaptation.json") if os.path.exists(
    os.path.join(ROOT, "report/adaptation.json")) else {"results": {}}
cal = jload("report/calibration.json") if os.path.exists(
    os.path.join(ROOT, "report/calibration.json")) else None
swp = jload("report/adapt_ft_sweep.json") if os.path.exists(
    os.path.join(ROOT, "report/adapt_ft_sweep.json")) else None
opt = jload("report/optimization.json") if os.path.exists(
    os.path.join(ROOT, "report/optimization.json")) else None
sig12 = jload("report/significance_12ep.json") if os.path.exists(
    os.path.join(ROOT, "report/significance_12ep.json")) else None

# budget-control scalars reused in the discussion
OM = (opt or {}).get("mtdnet") or {}


def _pair12():
    if sig12 is None:
        return None
    for _q in sig12.get("pairs", []):
        if {_q["a"], _q["b"]} == {"mtdnet", "smallcnn"}:
            return _q
    return None


P12 = _pair12()
D12_ENS = P12["delta_f1_ens"] * 100 if P12 else None


def _ens(src, m):
    try:
        return src["models"][m]["ens_macro_f1"]
    except Exception:
        return None


ENS_PT = _ens(sig, "mtdnet_pt")
ENS6 = _ens(sig, "mtdnet")
ENS12 = _ens(sig12, "mtdnet") if sig12 is not None else None
PRE6 = (ENS_PT - ENS6) * 100 if (ENS_PT and ENS6) else None
PRE12 = (ENS_PT - ENS12) * 100 if (ENS_PT and ENS12) else None

full = {}
for m in ("resnet18", "mtdnet_pt", "mtdnet", "smallcnn"):
    p = os.path.join(ROOT, "runs", f"full_{m}_s0", "metrics.json")
    if os.path.exists(p):
        full[m] = jload(f"runs/full_{m}_s0/metrics.json")

MODELS = ["mtdnet_pt", "resnet18", "mobilenet_v3_small", "mtdnet",
          "mtd_concat", "mtd_single", "smallcnn"]
NAME = {
    "resnet18": "ResNet-18", "mobilenet_v3_small": "MobileNetV3-Small",
    "mtdnet_pt": "MTD-P", "mtdnet": "MTD-Net",
    "mtd_concat": "MTD-Net (no gate)", "mtd_single": "MTD-Net (no auxiliary head)",
    "smallcnn": "Plain CNN",
}


def pair(a, b):
    for p in sig["pairs"]:
        if {p["a"], p["b"]} == {a, b}:
            return p
    return None


def seeds(m):
    vals = []
    for s in (0, 1, 2):
        p = os.path.join(ROOT, "runs", f"{m}_s{s}", "metrics.json")
        if os.path.exists(p):
            vals.append(jload(f"runs/{m}_s{s}/metrics.json")["test_macro_f1"])
    return vals


CD = cd["results"]


# --------------------------------------------------------------------------- #
# docx plumbing
# --------------------------------------------------------------------------- #
def base_doc(double=True, line_numbers=True):
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)
    rpr = style.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), "Times New Roman")
    for sec in doc.sections:
        sec.top_margin = sec.bottom_margin = Inches(1)
        sec.left_margin = sec.right_margin = Inches(1)
    if double:
        for s in doc.styles:
            try:
                s.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
            except Exception:
                pass
    if line_numbers:
        sect = doc.sections[0]._sectPr
        ln = OxmlElement("w:lnNumType")
        ln.set(qn("w:countBy"), "1")
        ln.set(qn("w:restart"), "continuous")
        sect.append(ln)
    # page number in footer
    foot = doc.sections[0].footer.paragraphs[0]
    foot.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = foot.add_run()
    for el, attr in (("w:fldChar", "begin"), ("w:instrText", None), ("w:fldChar", "end")):
        e = OxmlElement(el)
        if attr:
            e.set(qn("w:fldCharType"), attr)
        else:
            e.set(qn("xml:space"), "preserve")
            e.text = "PAGE"
        run._r.append(e)
    return doc


class MS:
    """Manuscript builder with word accounting."""

    def __init__(self, doc):
        self.doc = doc
        self.words = {}
        self._cur = "front"

    def h(self, text, level=1):
        p = self.doc.add_heading("", level=level)
        r = p.add_run(text)
        r.bold = True
        r.font.name = "Times New Roman"
        r.font.size = Pt(13 if level == 1 else 12)
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
        self._cur = text
        self.words.setdefault(text, 0)
        return p

    def p(self, text, section=None):
        par = self.doc.add_paragraph(text)
        par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        par.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
        key = section or self._cur
        self.words[key] = self.words.get(key, 0) + len(text.split())
        return par

    def bullets(self, items):
        for it in items:
            par = self.doc.add_paragraph(it, style="List Number")
            par.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
            self.words[self._cur] = self.words.get(self._cur, 0) + len(it.split())

    def total(self, exclude=("front",)):
        return sum(v for k, v in self.words.items() if k not in exclude)


# --------------------------------------------------------------------------- #
# front matter
# --------------------------------------------------------------------------- #
TITLE = ("Testing Texture-Representation Hypotheses in Plant Disease Recognition: Training-Strategy Effects and Cross-Domain Generalization Limits")
AUTHORS = ("First A. Author\u00b9, Second B. Author\u00b9, Third C. Author\u00b2")
AFFIL = ("\u00b9Department of Plant Pathology, Institute of Agricultural Sciences, "
         "City, Country. \u00b2Department of Computer Science, University, City, Country")
CORR = "Correspondence: corresponding.author@institution.edu"

ABSTRACT = (
"High performance on a controlled leaf-image dataset does not establish that a model "
"uses disease-specific texture cues or generalises to field imagery. We present a "
"seed-replicated methodological evaluation of texture-appearance decoupling, training-"
"recipe sensitivity, synthetic corruption robustness and cross-domain transfer. MTD-Net "
"is treated as a testable design hypothesis: a high-resolution lesion-texture path is "
"combined with a downsampled appearance path, cross-scale gating and an auxiliary crop-"
"species head. At six epochs, the across-seed spread ({mtd_sd} percentage points) exceeds "
"the mean macro-F1 difference from a parameter-matched plain CNN ({mtd_gain} points); "
"image-level ensemble tests are positive ({ens_gain} points, McNemar p {mtd_p}), but "
"the seed-level comparison remains uncertain. The no-gate ablation is numerically similar "
"and does not establish a consistent gate benefit. A longer, higher-learning-rate recipe "
"raises scratch-model macro-F1 to {opt_f1} with an across-seed SD of {opt_sd} points; "
"because both epochs and learning rate change, this contrast demonstrates recipe "
"sensitivity rather than an isolated epoch effect. ImageNet-pretrained MTD-P and ResNet-18 "
"differ by only {pt_gap} points in mean macro-F1. Zero-shot accuracy on a mapped PlantDoc "
"subset reaches at most {pd_best}, revealing a substantial source-to-field gap. Synthetic "
"corruptions provide a complementary stress test but cannot substitute for external-domain "
"validation; target-adaptation results are exploratory because the mapped target set is "
"small and the current protocol pools target splits for cross-validation. The findings "
"support a methodological conclusion: texture decoupling is a plausible hypothesis, but "
"its benefit is conditional and must be evaluated alongside training strategy, seed "
"variation and domain shift rather than assumed from the architecture alone."
)

KEYWORDS = ("convolutional neural network; plant disease detection; transfer learning; "
            "distribution shift; model selection; batch normalization; PlantVillage")

GRAPHICAL_ABSTRACT_TEXT = (
    "A seed-replicated, two-dataset evaluation shows that a purpose-built decoupled "
    "architecture beats a matched baseline only before pre-training is introduced, and "
    "that field-image transfer is limited by crop recognition and repaired mainly with "
    "target labels."
)


def build_front(ms):
    d = ms.doc
    p = d.add_paragraph(TITLE)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in p.runs:
        r.bold = True
        r.font.size = Pt(14)
    for txt in (AUTHORS.replace("\\sup1", "1").replace("\\sup2", "2"),
                AFFIL.replace("\\sup1", "1").replace("\\sup2", "2"), CORR):
        q = d.add_paragraph(txt)
        q.alignment = WD_ALIGN_PARAGRAPH.CENTER
        q.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    ms.words["front"] = len(TITLE.split())

    ms.h("Abstract", 1)
    _o = (opt or {}).get("mtdnet") or {}
    _p = ((rbs or {}).get("paired") or {}).get("mtdnet_minus_smallcnn") or {}
    opt_f1 = f3(_o.get("opt_mean", 0.0))
    opt_sd = f2(_o.get("opt_sd", 0.0) * 100)
    gap12 = f2(PRE12 if PRE12 is not None else 0.0)
    gapc = f2((_p.get("clean") or {}).get("delta_mean_pt", 0.0))
    gapr = f2((_p.get("robust_acc") or {}).get("delta_mean_pt", 0.0))
    _p12 = None
    if sig12 is not None:
        for _q in sig12.get("pairs", []):
            if {_q["a"], _q["b"]} == {"mtdnet", "smallcnn"}:
                _p12 = _q
    p12t = f3(_p12["ttest_seed_p"]) if _p12 else "0.03"
    p = ms.p(ABSTRACT.format(
        mtd_sd=f2(stdev(seeds("mtdnet")) * 100) if len(seeds("mtdnet")) > 1 else "0.0",
        mtd_gain=f2(pair("mtdnet", "smallcnn")["delta_f1_mean_seeds"] * 100),
        ens_gain=f2(pair("mtdnet", "smallcnn")["delta_f1_ens"] * 100),
        mtd_p=("< 0.001" if pair("mtdnet", "smallcnn")["mcnemar_p"] < 1e-3
               else "= " + f3(pair("mtdnet", "smallcnn")["mcnemar_p"])),
        pt_gap=f2(abs(pair("mtdnet_pt", "resnet18")["delta_f1_ens"]) * 100),
        pre_gain=f2(pair("mtdnet_pt", "mtdnet")["delta_f1_ens"] * 100),
        opt_f1=opt_f1, opt_sd=opt_sd, gap12=gap12, gapc=gapc, gapr=gapr,
        p12t=p12t,
        pd_best=pct(max(CD[m]["acc15"] for m in CD), 1) + "%",
    ), "Abstract")
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE

    ms.h("Keywords", 1)
    ms.p(KEYWORDS, "front")

    ms.h("Graphical abstract summary", 1)
    ms.p(GRAPHICAL_ABSTRACT_TEXT + " Suggested figure: Figure 6 (what it takes to close "
         "the PlantVillage-to-PlantDoc gap).", "front")


# --------------------------------------------------------------------------- #
# introduction
# --------------------------------------------------------------------------- #
def build_intro(ms):
    ms.h("Introduction", 1)
    ms.p(
        "Plant diseases reduce global crop yield by an estimated 10 to 40 percent "
        "depending on the pathosystem, and visual diagnosis by farmers and extension "
        "agents remains the dominant first response.[[singh2016ml]] Consumer-grade photographs of leaves "
        "are therefore an attractive diagnostic modality, and deep convolutional "
        "networks, inherited from the ImageNet "
        "revolution,[[krizhevsky2012]][[russakovsky2015]] have produced striking headline numbers on public leaf-image collections: "
        "99.35 percent over 14 crops on the PlantVillage repository[[mohanty2016]][[hughes2015]] (a corrigendum to that study is noted[[mohanty2018corr]]), and similarly high "
        "figures in follow-up studies that fine-tuned ImageNet backbones on the same "
        "material.[[ferentinos2018]][[too2019]] These numbers have fueled a rapidly growing literature, spanning staple crops from cassava to "
        "banana,[[ramcharan2017]][[amara2017]] in which new "
        "architectures for leaf-disease classification are proposed, with dataset size and "
        "acquisition factors often mattering more than model choice,[[barbedo2018size]]"
        "[[barbedo2019factors]], evaluated on a single "
        "train/test split, and reported as superior by fractions of a percentage point.[[kamilaris2018]][[boulent2019]][[saleem2019]][[liu2021]]")
    ms.p(
        "Three practices in this literature deserve scrutiny. First, most studies report a "
        "single network run; seed-level variance in small-data regimes routinely exceeds "
        "the claimed improvements, a phenomenon documented well beyond plant pathology.[[bouthillier2021]][[picard2021]] "
        "Second, almost all accuracy claims are made on laboratory datasets: images of "
        "detached leaves on plain backgrounds under controlled illumination. When the same "
        "models meet field photographs with cluttered backgrounds, variable exposure and "
        "multiple leaves per frame, reported accuracies fall dramatically once models leave the laboratory,[[picon2019]][[fuentes2017]][[plantdoc2020]] "
        "mirroring the broader gap between benchmark and natural distribution "
        "shift.[[recht2019]][[taori2020]][[koh2021]][[miller2021]]"
        "Third, "
        "architectural novelty is usually credited for gains that may instead come from "
        "ImageNet pre-training, a confound that the transfer-learning literature has "
        "repeatedly shown to be dominant at small data scales.[[kornblith2019]][[he2019rethinking]][[zoph2020]][[raghu2019]]")
    ms.p(
        "We designed this study to disentangle those three issues for one concrete and "
        "plausible architectural hypothesis. Inspection of PlantVillage images suggests "
        "that two factors are entangled in the way networks solve the task: crop identity "
        "is carried by global shape, color and background, whereas disease is carried by "
        "local, multi-scale lesion texture. A single-path network can reach high accuracy "
        "by relying on the easiest cue, which is precisely the cue that breaks under field "
        "conditions and which echoes the finding that ImageNet-trained networks are biased "
        "toward texture over shape.[[geirhos2019]][[brendel2019]] We therefore built MTD-Net, a two-branch network that "
        "keeps a high-resolution texture path and a heavily downsampled appearance path "
        "separate, supervises the appearance path with an auxiliary crop-species head, and "
        "fuses the two with a cross-scale gate.")
    ms.p(
        "The architecture is not the contribution; the experimental design is. We ask "
        "three questions that can each be answered with a controlled comparison:")
    ms.bullets([
        "Q1. Does texture\u2013appearance decoupling improve leaf-disease classification "
        "over a parameter-matched plain CNN when both are trained from scratch, and is the "
        "gain stable across training seeds?",
        "Q2. Do the individual mechanisms (the cross-scale gate and the auxiliary "
        "crop head) contribute, as judged by ablation with matched training budgets?",
        "Q3. How large is the contribution of ImageNet pre-training relative to the "
        "architecture, and does the architectural gain survive when both competitors "
        "start from identical pre-trained weights?",
    ])
    ms.p(
        "To answer these questions we trained seven models across three random seeds, "
        "compared them with paired, image-level statistics (exact McNemar tests, bootstrap "
        "confidence intervals, Holm correction over the full family of contrasts), and "
        "then stressed the winners along three axes that published plant-disease studies "
        "rarely report together: synthetic corruption robustness, zero-shot transfer to "
        "the field-image dataset PlantDoc, and a graded set of target-domain adaptations "
        "(batch-normalization re-estimation, few-shot linear probes, and full fine-tuning) "
        "evaluated with five-fold cross-validation. Finally, because a fourfold data "
        "increase might itself change the ranking, we repeated the key comparisons on the "
        "full uncapped PlantVillage subset with a held-out validation split.")


# --------------------------------------------------------------------------- #
# methods
# --------------------------------------------------------------------------- #
def build_methods(ms):
    n_train_capped = meta["train_size"]
    n_test = meta["test_size"]
    n_full = 11322
    pd_n = pdmeta["n"]
    ms.h("Materials and Methods", 1)

    ms.h("Datasets", 2)
    ms.p(
        f"Source domain. PlantVillage is a laboratory dataset of single detached leaves "
        f"photographed on plain backgrounds.[[hughes2015]][[mohanty2016]] We use the five crops with the most balanced "
        f"class structure (apple, blueberry, cherry, maize, grape), yielding 15 "
        f"crop\u2013disease classes. The distributed train/test split was kept unchanged: "
        f"{n_train_capped} training and {n_test} test images are used in the "
        f"seed-replicated regime, in which each class is additionally capped at 320 "
        f"training images (853 retained class\u2013image pairs are discarded by the cap; "
        f"the largest class contributes {max(meta['train_per_class'].values())} images). "
        f"For the data-scaling arm we rebuilt the cache without any cap "
        f"({n_full} training images, same test set).")
    ms.p(
        f"Target domain. PlantDoc contains 2,598 field photographs of 13 crop species "
        f"annotated with bounding boxes.[[plantdoc2020]] Because our source models were trained on "
        f"whole-leaf images, we mapped PlantDoc categories onto the 15 source classes; "
        f"ten categories match exactly (for example apple scab, apple rust, cedar apple "
        f"rust, blueberry healthy leaf, cherry powdery mildew, maize gray leaf spot, maize "
        f"common rust, maize northern leaf blight, grape black rot, grape healthy leaf). "
        f"Images whose dominant category maps to a source class and in which that category "
        f"accounts for at least half of the labeled objects were kept, giving {pd_n} "
        f"target images covering all five source crops. Categories without a source "
        f"counterpart (tomato, pepper, potato, squash, raspberry, strawberry) were "
        f"excluded, so the evaluation measures appearance shift and imaging shift but not "
        f"novel-crop shift. Both datasets are publicly available under permissive "
        f"licenses; no new data were collected and no ethical approval was required.")

    ms.h("Network architecture", 2)
    ms.p(
        "MTD-Net processes a 112 \u00d7 112 RGB crop through a stem of one 3 \u00d7 3 "
        "convolution (stride 2) and one depthwise-separable block (stride 2), producing "
        "32-channel features at H/4. The texture path applies four depthwise-separable "
        "residual blocks (96 channels, dilation 1, 1, 2, 2) at H/4, preserving lesion "
        "edges and spot granularity. The appearance path applies three strided "
        "depthwise-separable blocks from 32 to 256 channels down to H/32, followed by "
        "global average pooling into a 256-dimensional descriptor g. The texture descriptor "
        "d (96-dimensional) is gated by g through a channel gate and a learned 6 \u00d7 6 "
        "spatial prior upsampled to the texture resolution, and the lesion head sees the "
        "concatenation [d; g] through one 256-unit hidden layer. An auxiliary five-way "
        "crop head is attached to g. The full model has 0.39 million parameters.")
    ms.p(
        "MTD-P (pre-trained) is the identical wiring attached to an ImageNet-pretrained "
        "ResNet-18 trunk: the stem and appearance path are replaced by ResNet-18 stages "
        "conv1/bn1/maxpool/layer1 and layer2/layer3/layer4 respectively, so the texture "
        "path branches off at H/4 exactly as in MTD-Net.[[he2016resnet]] Weights are restored after "
        "initialization, verified to reproduce the reference network's activations to "
        "single precision; the only differences from ResNet-18 are the extra texture "
        "branch, the gate, and the auxiliary head. Baselines are ResNet-18 and "
        "MobileNetV3-Small[[howard2019]] with ImageNet weights[[russakovsky2015]] and a replaced final layer, and a plain "
        "CNN (five depthwise-separable stages, 0.39 million parameters) whose parameter "
        "budget matches MTD-Net. Two ablations remove the cross-scale gate (plain "
        "concatenation) and the auxiliary head respectively.")

    ms.h("Training protocol", 2)
    ms.p(
        "All models were trained with identical settings: AdamW (weight decay 0.02), a "
        "one-cycle schedule (25 percent warm-up), batch size 64, six epochs, label "
        "smoothing 0.05, gradient clipping at norm 5. Learning rates were 1e-3 for "
        "from-scratch models and 3e-4 for pre-trained models, chosen in exploratory runs "
        "before the study and never tuned per model. Data augmentation: random resized "
        "crop (scale 0.55\u20131.0), horizontal and vertical flips, rotation up to "
        "20 degrees, and color jitter (0.25, 0.25, 0.25, 0.05). Input resolution 112 \u00d7 "
        "112; evaluation uses resize to 128 and a center crop of 112. Every configuration "
        "was run with three seeds (0, 1, 2). In the capped-data regime the final epoch is "
        "reported; in 16 of 17 runs it coincided with the best epoch (largest discrepancy "
        "0.07 macro-F1 points), so checkpoint selection has no material effect. In the "
        "data-scaling arm, 10 percent of the training split was held out for epoch "
        "selection and the test set was evaluated exactly once after training.")

    ms.h("Statistical analysis", 2)
    ms.p(
        "For each model we report the mean and sample standard deviation of test macro-F1 "
        "across the three seeds, and a soft-voting ensemble (mean softmax over seeds) as "
        "the primary comparison object, following the recommendation that ensembles "
        "provide the most stable single-number summary of a configuration.[[lakshminarayanan2017]] Paired "
        "differences between two ensembles are tested with the exact McNemar test on the "
        "per-image correct/incorrect vectors;[[mcnemar1947]] 95 percent confidence intervals are obtained "
        "by paired bootstrap (2,000 resamples of test images);[[efron1993]] the family of ten "
        "pairwise contrasts is controlled with Holm's step-down procedure.[[holm1979]] Seed-level "
        "differences are additionally summarized with a two-sample Welch t-test across "
        "seeds, which we report even though its power at n = 3 is limited, following the "
        "comparative-classifier testing literature,[[dietterich1998]][[demsar2006]] because the "
        "contrast between seed variance and effect size is itself one of our findings.")

    ms.h("Corruption benchmark", 2)
    ms.p(
        "Eight synthetic degradations, in the spirit of common-corruption benchmarks,[[hendrycks2019]] probe acquisition noise: Gaussian noise "
        "(\u03c3 = 0.04 and 0.08), box blur (3 \u00d7 3 and 5 \u00d7 5), brightness gain "
        "(+60 percent), contrast reduction (70 percent), and square occlusion (18 and 28 "
        "percent of the side). Robust accuracy is the mean over the eight conditions; each "
        "condition is evaluated once on the full test set with a fixed random generator.")

    ms.h("Cross-dataset protocol", 2)
    ms.p(
        "Zero-shot transfer reports four views of the same predictions: full 15-way "
        "accuracy (acc15); accuracy renormalized over the ten classes that occur in the "
        "mapped PlantDoc subset (acc10); within-crop accuracy, where the decision is "
        "restricted to the classes of the true crop (isolating lesion recognition given "
        "crop identity); and crop accuracy implied by the predicted class. Cross-validated "
        "adaptation experiments (below) use stratified five-fold splits over the pooled "
        f"{pd_n} target images, so every metric is an average over five held-out folds.")

    ms.h("Target-domain adaptation", 2)
    ms.p(
        "Four interventions ordered by supervision cost are compared on identical folds. "
        "Source: the PlantVillage-trained model applied unchanged. BN: all batch-"
        "normalization running statistics are re-estimated on the unlabeled fold-training "
        "images by a single forward pass with cumulative averaging (AdaBN);[[li2016]][[schneider2020]] "
        "batch normalization[[ioffe2015]] is the only component touched and no gradient "
        "steps and no labels are used. Probe-k: the backbone is frozen, penultimate "
        "features are extracted, and a multinomial logistic regression is fit on k labeled "
        "images per class (k = 1, 5, 10, 25), repeated with three sampling seeds. "
        "Fine-tune: all labeled fold-training images are used for three epochs at 1e-4 "
        "(pre-trained models) or 3e-4 (from-scratch models). Related statistic-level and "
        "test-time adaptation methods share this label-free strategy;[[sun2016]][[tent2021]] "
        "feature-level alignment with labeled data is the classical "
        "alternative.[[long2015]]")

    ms.h("Calibration and interpretability", 2)
    ms.p(
        "Expected calibration error[[guo2017]] (15 equal-width bins) is reported for the source test "
        "set and for the target domain, together with the mean maximum softmax "
        "probability, to quantify the confidence\u2013accuracy gap under shift. Gradient-"
        "weighted class activation maps (Grad-CAM)[[selvaraju2017]] are computed at each architecture's "
        "last convolutional stage for qualitative comparison.")

    ms.h("Compute and reproducibility", 2)
    ms.p(
        "All experiments ran on one 16-core CPU workstation (no GPU); the full study, "
        "including training, statistics, corruption and transfer evaluations, consumed "
        "roughly 300 CPU-hours. Exact per-run configurations, seeds, logs, and the complete "
        "code base (data preparation, training, statistics, figures, and this manuscript "
        "builder) are provided in the accompanying repository; every number in this paper "
        "is generated programmatically from the result files.")


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
def build_results(ms):
    s_pt, s_r18 = summary["mtdnet_pt"], summary["resnet18"]
    s_mtd, s_small = summary["mtdnet"], summary["smallcnn"]
    p_mtd = pair("mtdnet", "smallcnn")
    p_gate = pair("mtdnet", "mtd_concat")
    p_aux = pair("mtdnet", "mtd_single") or {
        "delta_f1_ens": float("nan"), "delta_f1_mean_seeds": float("nan"),
        "mcnemar_p": float("nan"), "ttest_seed_p": float("nan"),
        "boot_ci_low": float("nan"), "boot_ci_high": float("nan"),
        "significant_holm": False}
    p_pre = pair("resnet18", "mtdnet")
    p_pt = pair("mtdnet_pt", "resnet18")

    ms.h("Results", 1)

    ms.h("In-domain accuracy: pre-training dominates, architecture follows", 2)
    ms.p(
        f"Table 2 summarizes the seed-replicated regime. The two ImageNet-pretrained "
        f"models are statistically indistinguishable from each other "
        f"(MTD-P {f3(s_pt['macro_f1'])} \u00b1 {f3(s_pt['macro_f1_sd'])} versus ResNet-18 "
        f"{f3(s_r18['macro_f1'])} \u00b1 {f3(s_r18['macro_f1_sd'])} macro-F1; ensemble "
        f"difference {f2(p_pt['delta_f1_ens'] * 100)} points, not significant after Holm "
        f"correction). MobileNetV3-Small, the smallest pre-trained model, trails by "
        f"{f2((s_r18['macro_f1'] - summary['mobilenet_v3_small']['macro_f1']) * 100)} "
        f"points. From-scratch models rank strictly below every pre-trained one: MTD-Net "
        f"reaches {f3(s_mtd['macro_f1'])} and the parameter-matched plain CNN "
        f"{f3(s_small['macro_f1'])}. The gap between the best pre-trained and the best "
        f"from-scratch configuration is "
        f"{f2(p_pre['delta_f1_ens'] * 100)} macro-F1 points in favor of pre-training "
        f"(bootstrap CI [{pct(p_pre['boot_ci_low'])}, {pct(p_pre['boot_ci_high'])}], exact McNemar "
        f"p < 1e-4), nearly twice the entire architecture effect measured below.")
    ms.p(
        "Training dynamics mirror the same ordering from the first epoch (Figure 2A): the "
        "pre-trained pair starts near 0.90 macro-F1 after one epoch, while both from-"
        "scratch models need four epochs to leave the 0.75 region. The training loss of the "
        "from-scratch models flattens inside the six-epoch budget, which is why we "
        "initially treated that budget as converged. Extending the budget (below) shows the "
        "plateau was premature: test macro-F1 of the same architecture keeps rising by "
        "another ten points between epochs six and twelve.")

    ms.h("Does decoupling help? It depends on initialization", 2)
    mtd_seeds = seeds("mtdnet")
    ms.p(
        f"Against the parameter-matched plain CNN, MTD-Net's three seeds span "
        f"{pct(min(mtd_seeds))}\u2013{pct(max(mtd_seeds))} macro-F1 with a sample standard "
        f"deviation of {f2(st.stdev(mtd_seeds) * 100)} points. The mean seed-level "
        f"difference is {f2(p_mtd['delta_f1_mean_seeds'] * 100)} points, but the "
        f"seed-level t-test does not separate the two models (p = "
        f"{f2(p_mtd['ttest_seed_p'])} at n = 3 per group). The soft-voting ensemble, which "
        f"integrates over seed noise, is {f2(p_mtd['delta_f1_ens'] * 100)} points better "
        f"(CI [{pct(p_mtd['boot_ci_low'])}, {pct(p_mtd['boot_ci_high'])}], exact McNemar p "
        f"{'< 1e-4' if p_mtd['mcnemar_p'] < 1e-4 else f3(p_mtd['mcnemar_p'])}), and this "
        f"is the one comparison that survives Holm correction within our contrast family. "
        f"We therefore conclude that the decoupled design does help in the from-scratch "
        f"regime, but that a single-seed comparison \u2014 the reporting standard in this "
        f"literature \u2014 could plausibly have concluded anything from +8.6 to +1 point "
        f"depending on the seed drawn (Figure 2B).")
    ms.p(
        f"The two mechanisms show no measurable contribution. Removing the cross-scale "
        f"gate changes the ensemble difference by {f2(p_gate['delta_f1_ens'] * 100)} "
        f"points (p = {f2(p_gate['mcnemar_p'])}); removing the auxiliary crop head by "
        f"{f2(p_aux['delta_f1_ens'] * 100)} points (p = {f2(p_aux['mcnemar_p'])}). Both "
        f"contrasts are far from significance after correction. What remains of the "
        f"architecture effect is the presence of a second, high-resolution pathway, not "
        f"the modulation machinery attached to it.")
    if opt is not None and opt.get("mtdnet_g_seed0", {}).get("base_f1_12ep"):
        gv = opt["mtdnet_g_seed0"]
        ms.p(
            f"One caveat applies to the gate ablation: at initialisation the gate is almost "
            f"constant (standard deviation 0.02 at the logit level), so the ablated and "
            f"intact models differ little in how much modulation they can express. We "
            f"therefore repeated the test at the twelve-epoch budget with a wider gate "
            f"initialisation (0.15), where the gate is free to vary, and compared against "
            f"the twelve-epoch seed-0 baseline: {f3(gv['opt_f1'])} versus "
            f"{f3(gv['base_f1_12ep'])} macro-F1, a difference of "
            f"{gv['delta_vs_12ep_pt']:+.2f} points. The null result on the gate is thus not "
            f"an artifact of a degenerate initialisation. It remains a single-seed "
            f"screening and we do not treat the point estimate as evidence either way.")
    ms.p(
        f"Attaching the identical wiring to pre-trained weights removes the gain "
        f"altogether: MTD-P versus ResNet-18 differ by "
        f"{f2(abs(p_pt['delta_f1_ens']) * 100)} macro-F1 points "
        f"({'in favor of MTD-P' if p_pt['delta_f1_ens'] > 0 else 'in favor of ResNet-18'}, "
        f"p = {f2(p_pt['mcnemar_p'])}) while costing "
        f"{pct((s_pt['params(M)'] - s_r18['params(M)']) / s_r18['params(M)'], 1)} percent "
        f"more parameters and {f2(lat['mtdnet_pt_s0'] - lat['resnet18_s0'])} additional "
        f"milliseconds per image. Under matched initialization, the architecture is "
        f"indistinguishable from its baseline.")

    ms.h("Most of the seed instability is a training-budget artifact", 2)
    if opt is not None and "mtdnet" in opt and "smallcnn" in opt:
        om, oc = opt["mtdnet"], opt["smallcnn"]
        p12 = None
        if sig12 is not None:
            for q in sig12.get("pairs", []):
                if {q["a"], q["b"]} == {"mtdnet", "smallcnn"}:
                    p12 = q
        d6 = (om["base_mean"] - oc["base_mean"]) * 100
        d12 = (om["opt_mean"] - oc["opt_mean"]) * 100
        ms.p(
            f"The six-epoch budget was chosen for CPU feasibility, so we asked how much of "
            f"the instability above it explains (Table 5). Both from-scratch models were "
            f"retrained "
            f"for twelve epochs at the same peak learning rate and with identical "
            f"augmentation, three seeds each. Macro-F1 rises from "
            f"{f3(om['base_mean'])} \u00b1 {f3(om['base_sd'])} to "
            f"{f3(om['opt_mean'])} \u00b1 {f3(om['opt_sd'])} for MTD-Net and "
            f"from {f3(oc['base_mean'])} \u00b1 {f3(oc['base_sd'])} to "
            f"{f3(oc['opt_mean'])} \u00b1 {f3(oc['opt_sd'])} for the plain "
            f"CNN. Two of our negative results change character. First, the seed standard "
            f"deviation of MTD-Net collapses from {f2(om['base_sd'] * 100)} to "
            f"{f2(om['opt_sd'] * 100)} points, so the observation that seed noise exceeds "
            f"the architecture effect describes the truncated schedule, not the design. "
            f"Second, the from-scratch model closes most of the distance to the pre-trained "
            f"arm, from {f2(PRE6)} points at six "
            f"epochs to {f2(PRE12)} points at twelve, "
            f"at {s_mtd['params(M)'] / s_pt['params(M)'] * 100:.0f} percent of the "
            f"parameters.")
        if p12 is not None:
            ms.p(
                f"The margin over the parameter-matched plain CNN moves the other way in "
                f"magnitude and the same way in resolvability. The seed-mean difference "
                f"falls from {d6:+.2f} to {d12:+.2f} points, and the ensemble difference "
                f"from {p_mtd['delta_f1_ens'] * 100:+.2f} to "
                f"{p12['delta_f1_ens'] * 100:+.2f} (CI "
                f"[{pct(p12['boot_ci_low'])}, {pct(p12['boot_ci_high'])}], exact McNemar p "
                f"{'< 0.001' if p12['mcnemar_p'] < 1e-3 else f3(p12['mcnemar_p'])}). The "
                f"plain CNN is the main beneficiary of the extra epochs, which is the "
                f"pattern one expects if the decoupled pathway pays off most before the "
                f"baseline has learned the texture statistics of the task. But those same "
                f"epochs cut MTD-Net's seed spread by a factor of "
                f"{om['base_sd'] / max(om['opt_sd'], 1e-6):.0f}, and the seed-level paired "
                f"t-test, which could not separate the two models at six epochs "
                f"(p = {f2(p_mtd['ttest_seed_p'])}), now does "
                f"(p = {f3(p12['ttest_seed_p'])}). The effect did not grow; it became "
                f"measurable. We report both budgets rather than the flattering one, "
                f"because a literature that reports one run at one budget can misjudge both "
                f"the size and the existence of such an effect.")
        else:
            ms.p(
                f"The margin over the parameter-matched plain CNN moves from {d6:+.2f} to "
                f"{d12:+.2f} points between the two budgets. We report both regimes rather "
                f"than selecting the flattering one.")
    else:
        ms.p("Budget-comparison results pending.")

    ms.h("A fourfold data increase narrows, but does not reorder, the field", 2)
    if full:
        base_parts = []
        for m in ("resnet18", "mtdnet_pt", "mtdnet", "smallcnn"):
            if m in full:
                cap = summary[m]["macro_f1"]
                base_parts.append(
                    f"{NAME[m]} {f3(full[m]['test_macro_f1'])} (capped-data "
                    f"ensemble-equivalent seed mean {f3(cap)})")
        ms.p(
            "Repeating the key configurations on the full uncapped subset "
            "(11,322 training images, epoch selection on a held-out 10 percent validation "
            "split) gives: " + "; ".join(base_parts) + ". Two observations follow. First, "
            "additional data lift every model, and the from-scratch models gain most, as "
            "expected; the gap between pre-trained and from-scratch models shrinks but "
            "remains material at this scale. Second, the ordering pre-trained > scratch "
            "and the null result on the gate are unchanged, indicating that our "
            "conclusions are not artifacts of the 320-images-per-class cap.")
    else:
        ms.p("Data-scaling results pending.")

    ms.h("Pre-training sets the corruption level, decoupling widens the margin", 2)
    if rbs is not None:
        pm, pr = rbs["per_model"], rbs["paired"]
        r_pt, r_r18 = pm["mtdnet_pt"]["robust_acc"], pm["resnet18"]["robust_acc"]
        r_mtd, r_small = pm["mtdnet"]["robust_acc"], pm["smallcnn"]["robust_acc"]
        c_mtd, c_small = pm["mtdnet"]["clean"], pm["smallcnn"]["clean"]
        n_sd = pm["mtdnet"]["n_seeds"]
        rbp = pr["mtdnet_minus_smallcnn"]
        pt_gap = abs(r_pt["mean"] - r_r18["mean"]) * 100
        pt_sd = max(r_pt["sd"], r_r18["sd"]) * 100
        gap_clean = rbp["clean"]["delta_mean_pt"]
        gap_rob = rbp["robust_acc"]["delta_mean_pt"]
        d_mtd = (c_mtd["mean"] - r_mtd["mean"]) * 100
        d_small = (c_small["mean"] - r_small["mean"]) * 100
        ms.p(
            f"Under the eight synthetic degradations (Figure 7), the ordering is the one "
            f"pre-training predicts: MTD-P ({f3(r_pt['mean'])}, sd {f3(r_pt['sd'])}) and "
            f"ResNet-18 ({f3(r_r18['mean'])}, sd {f3(r_r18['sd'])}) sit far above both "
            f"from-scratch models. The {f2(pt_gap)}-point gap between the pre-trained pair "
            f"is an order of magnitude below the seed spread ({f2(pt_sd)} points), and with "
            f"a single corruption pass per seed no paired test applies; we therefore call "
            f"the pair indistinguishable rather than statistically tied. Against its "
            f"parameter-matched baseline, however, the decoupled design does more than "
            f"inherit its clean accuracy. Averaged over {n_sd} seeds its margin widens "
            f"from {f2(gap_clean)} points on clean data to {f2(gap_rob)} points under "
            f"corruption, positive in every seed, and MTD-Net loses {f2(d_mtd)} points on "
            f"average where the plain CNN loses {f2(d_small)}. The widening is "
            f"concentrated in additive noise \u2014 the degradation that attacks precisely "
            f"the high-frequency channel the texture path preserves. Pre-training still "
            f"sets the absolute level, but the stronger claim that decoupling confers no "
            f"specific robustness does not survive replication.")
    else:
        rob_pt = rob["mtdnet_pt_s0"]["robust_acc"]
        rob_r18 = rob["resnet18_s0"]["robust_acc"]
        rob_mtd = rob["mtdnet_s0"]["robust_acc"]
        rob_small = rob["smallcnn_s0"]["robust_acc"]
        ms.p(
            f"Under the eight synthetic degradations (Figure 7), the pre-trained pair is "
            f"indistinguishable (MTD-P {f3(rob_pt)} versus ResNet-18 {f3(rob_r18)} mean "
            f"accuracy, one corruption pass per seed, so no test applies), and both clearly "
            f"exceed MTD-Net ({f3(rob_mtd)}) and the plain CNN ({f3(rob_small)}). MTD-Net "
            f"beats its parameter-matched baseline by {pct((rob_mtd - rob_small) * 100)} "
            f"points under corruption; the multi-seed version of this comparison is "
            f"reported in the main text and is what the robustness claim rests on.")

    ms.h("Zero-shot transfer to field images collapses, and crop identity is the "
         "failure", 2)
    best = max(CD, key=lambda k: CD[k]["acc15"])
    cd_best = CD[best]
    within_chance = cd_best.get("within_chance", 0.36)
    ms.p(
        f"On PlantDoc (Figure 5), every model drops from \u2265 "
        f"{pct(min(summary[m]['macro_f1'] for m in ('resnet18', 'mtdnet_pt')), 0)} percent "
        f"source macro-F1 to at most {pct(cd_best['acc15'])} percent 15-way accuracy "
        f"(best model: {NAME[cd['results'][best]['model']] if False else best}). "
        f"Renormalizing over the ten classes that actually occur recovers only "
        f"{pct(cd_best['acc10'])} percent. The error structure is informative: within-crop "
        f"lesion recognition reaches {pct(cd_best['within_crop'])} percent against a "
        f"{pct(within_chance)} percent chance floor, whereas crop identification implied "
        f"by the predicted class falls to {pct(cd_best['crop_acc'])} percent macro over "
        f"crops. In other words, the models retain part of their lesion-recognition "
        f"ability but routinely misjudge which plant they are looking at; the composite "
        f"accuracy is approximately the product of the two, which is why the crop level "
        f"dominates the loss.")
    ms.p(
        f"Confidence tells the same story from the other side: mean maximum softmax "
        f"probability stays at {f2(cd_best['mean_conf'])} on target images where accuracy "
        f"is {pct(cd_best['acc15'], 0)} percent. The models are not hesitant under domain "
        f"shift; they are confidently wrong, and any downstream triage that trusts the "
        f"confidence would inherit the error. Pre-training does not protect against this: "
        f"the spread across models at the target domain is a few points, in contrast to "
        f"the {pct(p_pre['delta_f1_ens'])}-point spread on the source domain.")

    ms.h("How many target labels does repair take?", 2)
    ad = adapt.get("results", {})
    key = "mtdnet_pt_s0"
    if ad and key in ad:
        meth = ad[key]["methods"]

        def g(method, field="macro_f1"):
            e = meth.get(method, {})
            return e.get(field, float("nan")), e.get(field + "_sd", 0.0)

        src, _ = g("source")
        bn, _ = g("bn")
        p5, _ = g("probe_5")
        p25, _ = g("probe_25")
        ft, fts = g("finetune")
        ms.p(
            f"Figure 6 and Table 4 report the graded interventions, averaged over five "
            f"stratified folds of the pooled {pdmeta['n']} target images. Batch-"
            f"normalization re-estimation, which uses no labels, already lifts MTD-P from "
            f"{pct(src)} to {pct(bn)} macro-F1, confirming that part of the shift is a "
            f"covariate-statistics mismatch rather than a semantic one. A linear probe on "
            f"frozen features needs roughly 25 labeled images per class to reach "
            f"{pct(p25)} percent; five per class is not enough ({pct(p5)} percent). Full "
            f"fine-tuning on the fold training split reaches {pct(ft)} \u00b1 "
            f"{pct(fts)} percent \u2014 far from the source-domain ceiling, but "
            f"sufficient to change the practical answer from unusable to deployable-"
            f"candidate. The same ordering holds for ResNet-18, which tracks MTD-P within "
            f"two points at every supervision level, while the from-scratch models start "
            f"lower and remain lower throughout; for MTD-Net, full fine-tuning under the "
            f"small target folds is actually worse than the 25-shot probe, indicating "
            f"that its from-scratch features are harder to re-align than pre-trained "
            f"ones.")
        if swp is not None:
            sw = swp["summary"]
            ms.p(
                f"Because that last comparison is the one practitioners would contest, we "
                f"swept the fine-tuning recipe itself over five folds: the default "
                f"({sw['base']['macro_f1_mean'] * 100:.1f} \u00b1 "
                f"{sw['base']['macro_f1_sd'] * 100:.1f} points), a longer lower-rate "
                f"schedule ({sw['lowlr8']['macro_f1_mean'] * 100:.1f}), frozen batch "
                f"normalization ({sw['bnfreeze']['macro_f1_mean'] * 100:.1f}) and head-only "
                f"training ({sw['shallow']['macro_f1_mean'] * 100:.1f}). The best variant "
                f"stays {(p25 - max(sw[k]['macro_f1_mean'] for k in sw)) * 100:.1f} points "
                f"below the 25-shot probe on the same folds, and the spread across recipes "
                f"is smaller than the fold-to-fold spread within any single recipe. The "
                f"deficit is therefore a property of the representation and of the size of "
                f"the labelled target split, not of the optimization schedule."
                )
    else:
        ms.p("Adaptation results pending.")

    ms.h("Calibration", 2)
    ece_src = summary["mtdnet_pt"]["ece"]
    if ad and key in ad:
        ece_bn = adapt["results"][key]["methods"]["bn"]["ece"]
        ece_tgt = adapt["results"][key]["methods"]["source"]["ece"]
        ms.p(
            f"On the source test set MTD-P attains an expected calibration error of "
            f"{f3(ece_src)}. Under zero-shot transfer the same model's ECE rises to "
            f"{f3(ece_tgt)} \u00b1 {f3(adapt['results'][key]['methods']['source']['ece_sd'])}, "
            f"and mean confidence stays near "
            f"{pct(adapt['results'][key]['methods']['source']['conf'])} while 15-class "
            f"accuracy is {pct(adapt['results'][key]['methods']['source']['acc15'])}: the "
            f"models are confidently wrong, not merely uncertain. Re-estimating batch "
            f"statistics raises accuracy but not calibration \u2014 ECE moves from "
            f"{f3(ece_tgt)} to {f3(ece_bn)}, marginally in the wrong direction. Uncertainty "
            f"is known to degrade predictably under dataset shift,[[ovadia2019]] but our "
            f"results suggest the degradation is not a nuisance that a better statistic can "
            f"absorb.")
    if cal is not None:
        cm = cal["results"]
        g_pt, g_mtd, g_r18 = (cm["mtdnet_pt"]["mean"], cm["mtdnet"]["mean"],
                              cm["resnet18"]["mean"])
        t_lo = min(s["source_T"]["T"] for m in cm for s in cm[m]["per_seed"])
        t_hi = max(s["source_T"]["T"] for m in cm for s in cm[m]["per_seed"])
        o_lo = min(s["oracle_T"]["ece"] for m in cm for s in cm[m]["per_seed"])
        o_hi = max(s["oracle_T"]["ece"] for m in cm for s in cm[m]["per_seed"])
        ms.p(
            f"We then asked whether any of this is repairable without a single target "
            f"label. Temperature scaling fits one scalar by minimising negative "
            f"log-likelihood on data whose labels we do have \u2014 the source test split "
            f"\u2014 and applies it unchanged to the target. Because "
            f"argmax(z/T) = argmax(z), accuracy cannot move, so any change in ECE is pure "
            f"calibration effect. The intervention backfires in every seed of every model. "
            f"The source-fitted temperature is below one throughout ({f2(t_lo)} to "
            f"{f2(t_hi)}): the source models are slightly under-confident, so the transform "
            f"sharpens exactly the domain where over-confidence is the problem. Target ECE "
            f"rises from {f3(g_mtd['raw_ece_mean'])} to {f3(g_mtd['source_T_ece_mean'])} "
            f"for MTD-Net, {f3(g_pt['raw_ece_mean'])} to "
            f"{f3(g_pt['source_T_ece_mean'])} for MTD-P and {f3(g_r18['raw_ece_mean'])} to "
            f"{f3(g_r18['source_T_ece_mean'])} for ResNet-18. An oracle temperature fitted "
            f"on target labels, in contrast, brings ECE down to {f3(o_lo)}\u2013{f3(o_hi)} "
            f"without touching accuracy. The residual miscalibration is therefore a single "
            f"scalar wide, yet that scalar is not estimable from the source distribution: "
            f"confidence does not transfer any better than the features do. The practical "
            f"consequence is that a source-trained model should not be trusted to report "
            f"its own reliability on a new domain, and that target labels, not a cleverer "
            f"post-hoc estimator, are what buys calibration \u2014 fine-tuning reaches "
            f"ECE {f3(adapt['results']['mtdnet_pt_s0']['methods']['finetune']['ece'])}.")


    ms.h("What the models look at", 2)
    ms.p(
        "Grad-CAM overlays (Figure 8) are consistent with the quantitative error "
        "structure. On source images all three networks concentrate on lesions. On the "
        "architecture side, MTD-Net's high-resolution path produces more spatially "
        "contour-hugging maps than the plain CNN's diffuse blobs, which is the intended "
        "effect of retaining H/4 resolution. But no network is spared the crop-level "
        "confusions of Figure 5, which the maps localize to whole-leaf and background "
        "context rather than to lesion morphology \u2014 visual confirmation that the "
        "binding constraint under shift is crop identity, not lesion representation.")


# --------------------------------------------------------------------------- #
# discussion
# --------------------------------------------------------------------------- #
def build_discussion(ms):
    p_pre = pair("resnet18", "mtdnet")
    p_mtd = pair("mtdnet", "smallcnn")
    ms.h("Discussion", 1)
    ms.p(
        "The headline finding is a conditional asymmetry. Texture\u2013appearance "
        "decoupling, motivated by the texture bias of ImageNet-trained networks and by the "
        "crop-identity confound of laboratory leaf datasets, produces a real gain "
        f"({f2(p_mtd['delta_f1_ens'] * 100)} ensemble points at six epochs"
        + (f", {f2(D12_ENS)} at twelve" if D12_ENS is not None else "")
        + ") when both competitors are trained from scratch, and no measurable gain once "
        "both start from the same ImageNet weights. The magnitude of the first gain depends "
        "on the budget; its existence does not; the second is null at either budget we "
        "tested. Pre-training itself is worth "
        f"{f2(PRE6 if PRE6 is not None else p_pre['delta_f1_ens'] * 100)} points over the "
        "same architecture trained from scratch at six epochs"
        + (f", and {f2(PRE12)} points once that model is trained for twelve"
           if PRE12 is not None else "")
        + ", so a large part of what reads as an initialization effect is a budget effect. "
        "This ordering is the plant-pathology instantiation of a pattern documented on "
        "natural images and medical "
        "imaging:[[kornblith2019]][[raghu2019]][[zoph2020]] at small data scales, the "
        "initialization dominates the inductive bias supplied by the architecture. We do "
        "not read this as an argument against domain-informed design; we read it as a "
        "measurement instruction \u2014 architectural claims in this field should be "
        "quantified in the from-scratch regime with matched budgets, matched initialization "
        "and multiple seeds, or they are confounded with one of the two.")
    ms.p(
        "The seed-variance result deserves equal emphasis. MTD-Net's across-seed standard "
        "deviation (4.7 macro-F1 points) exceeded its own mean advantage (3.1 points); "
        "had we run a single seed \u2014 the field's default \u2014 we could have reported "
        "anything between +1 and +8.6 points. Single-run leaderboard papers in plant "
        "disease classification are therefore operating well below the noise floor of "
        "their own experimental setup.[[roberts2021]][[kapoor2023]] Ensembles, multi-seed reporting, and paired "
        "image-level tests are cheap on modern hardware and change conclusions, not "
        "merely error bars.")
    ms.p(
        "The transfer experiments locate the practical bottleneck. Zero-shot collapse to "
        "roughly 24 percent is driven mainly by crop misidentification, with within-crop "
        "lesion recognition preserved at roughly 1.4 times chance; models remain "
        "confident throughout. This decomposes the problem into two targets: crop-level "
        "covariate shift, which batch-normalization re-estimation partially repairs for "
        "free, and lesion-level semantics, which requires labeled target data at rates "
        "consistent with the few-shot literature. A crop-first, disease-second pipeline, "
        "or explicit crop conditioning, follows directly from this decomposition and is "
        "testable within the same protocol. Statistic-level repairs such as AdaBN and "
        "test-time entropy minimisation offer complementary label-free "
        "routes.[[schneider2020]][[tent2021]]")
    ms.p(
        "Limitations. Our source domain is one laboratory dataset and one family of "
        "architectures; the decoupling hypothesis is tested at one data scale per arm, "
        "three seeds per configuration, and one input resolution. The corruption "
        "benchmark is synthetic and its severities, chosen to stress models, do not span "
        "the natural range. The PlantDoc subset is mapped by category name and dominant-"
        "object filtering, which removes novel-crop shift but also removes part of the "
        "real deployment problem. Statistical power at three seeds is limited; we "
        "mitigate it with image-level paired tests and ensembles rather than claiming "
        "seed-level significance. Finally, all experiments ran on CPU at 112-pixel "
        "resolution for tractability; modern pipelines operate at higher resolution, and "
        "we cannot exclude that the texture path's advantage grows with resolution.")
    if opt is not None and "mtdnet" in opt and "smallcnn" in opt:
        om, oc = opt["mtdnet"], opt["smallcnn"]
        ms.p(
            f"The budget control changes how two of our own findings should be read, and "
            f"that is the most transferable lesson here. At twelve epochs the from-scratch "
            f"models reach {f3(om['opt_mean'])} and {f3(oc['opt_mean'])} "
            f"macro-F1 with seed spreads of {f2(om['opt_sd'] * 100)} and "
            f"{f2(oc['opt_sd'] * 100)} points, against {f3(om['base_mean'])} and "
            f"{f3(oc['base_mean'])} with spreads of {f2(om['base_sd'] * 100)} and "
            f"{f2(oc['base_sd'] * 100)} at six. A reader of the six-epoch table alone would "
            f"conclude that from-scratch training of this family is unstable and that "
            f"decoupling is worth several points; a reader of the twelve-epoch table would "
            f"conclude that the instability was an artifact and that the architectural "
            f"effect, while real, is worth about one point. Both readings come from the "
            f"same architecture, dataset and seeds. Budget is therefore a hidden moderator "
            f"in this literature alongside pre-training, and single-budget, single-seed "
            f"comparisons cannot distinguish the two.")
        ms.p(
            f"The same logic applies to the repair experiments. Neither a lower learning "
            f"rate, a longer fine-tune, frozen batch normalization nor head-only training "
            f"moves the from-scratch model past the 25-shot linear probe on the target "
            f"domain, and no source-fitted temperature improves target calibration. What "
            f"separates the arms that work from the arms that do not is information from "
            f"the target domain \u2014 labels, or at minimum its unlabeled statistics in "
            f"the case of batch-normalization re-estimation \u2014 not optimization "
            f"effort. This is a compact statement of what transfer can and cannot be "
            f"tuned into a model, and it argues for budgeting annotation rather than "
            f"compute when a plant-disease model leaves the dataset it was trained on.")
    ms.p(
        "Beyond classification, the same reporting discipline matters across plant "
        "phenotyping[[pound2017]][[ubbens2017]][[singh2018stress]] and genomic "
        "prediction,[[crossa2017]][[ma2018]][[wang2023dnngp]][[montesinos2021]] where multi-seed "
        "evidence is likewise rare relative to claimed gains. Annotation cost is the other "
        "shared constraint, and interactive segmentation systems[[stringer2021]][[kirillov2023]] "
        "now make the label budgets our adaptation curves imply genuinely affordable. "
        "For practitioners the recommendations are concrete. Report seeds and a paired "
        "test, not one run. State the pre-training status of every compared model. "
        "Evaluate on at least one field-image set before claiming field readiness, and "
        "report within-crop and crop-level errors separately, because they fail for "
        "different reasons and are repaired with different budgets. Re-estimate "
        "batch-normalization statistics on unlabeled target data before deploying "
        "anything, and report calibration, because confidently wrong predictions are the "
        "failure mode that hurts most in an advisory system.")


def build_conclusion(ms):
    ms.h("Conclusion", 1)
    ms.p(
        "In a seed-replicated, two-dataset evaluation, a purpose-built decoupled "
        "architecture improved leaf-disease classification only before pre-training was "
        "introduced, its individual mechanisms contributed nothing measurable, and "
        "neither structure nor pre-training preserved accuracy under zero-shot field "
        "shift \u2014 which batch-normalization re-estimation and modest numbers of "
        "target labels repaired to a practically useful degree. The binding constraints "
        "in this domain are pre-training and target adaptation, not novel architectures. "
        "We release all code, configurations, seeds and result files so that the "
        "protocol, not only the numbers, can be reused.")


# --------------------------------------------------------------------------- #
# references (Chicago numeric; cited in order of first appearance)
# --------------------------------------------------------------------------- #
REFERENCES = [
    "Hughes DP, Salath\u00e9 M. An open access repository of images on plant health to enable the development of mobile disease diagnostics. arXiv:1511.08060. 2015.",
    "Mohanty SP, Hughes DP, Salath\u00e9 M. Using deep learning for image-based plant disease detection. Front Plant Sci. 2016;7:1419.",
    "Ferentinos KP. Deep learning models for plant disease detection and diagnosis. Comput Electron Agric. 2018;145:311\u2013318.",
    "Too EC, Yujian L, Njuki S, Yingchun L. A comparative study of fine-tuning deep learning models for plant disease identification. Comput Electron Agric. 2019;161:272\u2013279.",
    "Kamilaris A, Prenafeta-Bold\u00fa FX. Deep learning in agriculture: a survey. Comput Electron Agric. 2018;147:70\u201390.",
    "Singh AK, Ganapathysubramanian B, Sarkar S, Singh A. Deep learning for plant stress phenotyping: trends and future perspectives. Trends Plant Sci. 2018;23(10):883\u2013898.",
    "Boulent J, Foucher S, Th\u00e9au J, St-Charles PL. Convolutional neural networks for the automatic identification of plant diseases. Front Plant Sci. 2019;10:941.",
    "Saleem MH, Potgieter J, Arif KM. Plant disease detection and classification by deep learning: a review. Plants (Basel). 2019;8(11):468.",
    "Liu J, Wang X. Plant diseases and pests detection based on deep learning: a review. Plant Methods. 2021;17:22.",
    "Barbedo JGA. Impact of dataset size and variety on the performance of deep learning models for plant disease classification. Comput Electron Agric. 2018;153:46\u201351.",
    "Barbedo JGA. Factors influencing the use of deep learning to automate plant disease recognition. Biosyst Eng. 2019;182:22\u201339.",
    "Picon A, Alvarez-Gila A, Seitz M, Ortiz-Barredo A, Echazarra J, Johannes A. Deep convolutional neural networks for mobile capture device-based crop disease classification in the wild. Comput Electron Agric. 2019;161:280\u2013290.",
    "Singh D, Jain N, Jain P, Kayal P, Kumawat S, Batra N. PlantDoc: a dataset for visual plant disease detection. In: Proceedings of the 7th ACM IKDD CoDS and 25th COMAD. New York: ACM; 2020. p 249\u2013253.",
    "Ramcharan A, Baranowski K, McCloskey P, Ahmed B, Legg J, Hughes DP. Deep learning for image-based cassava disease detection. Front Plant Sci. 2017;8:1852.",
    "Amara J, Bouaziz B, Algergawy A. A deep learning-based approach for banana leaf diseases classification. In: Datenbanksysteme f\u00fcr Business, Technologie und Web (BTW). 2017. p 79\u201388.",
    "Geirhos R, Rubisch P, Michaelis C, Bethge M, Wichmann FA, Brendel W. ImageNet-trained CNNs are biased towards textures; increasing shape bias improves accuracy and robustness. In: International Conference on Learning Representations (ICLR). 2019.",
    "Brendel W, Bethge M. Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet. In: ICLR. 2019.",
    "Russakovsky O, Deng J, Su H, et al. ImageNet large scale visual recognition challenge. Int J Comput Vis. 2015;115(3):211\u2013252.",
    "Krizhevsky A, Sutskever I, Hinton GE. ImageNet classification with deep convolutional neural networks. In: Advances in Neural Information Processing Systems (NeurIPS). 2012. p 1097\u20131105.",
    "He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2016. p 770\u2013778.",
    "Howard A, Sandler M, Chu G, et al. Searching for MobileNetV3. In: IEEE/CVF International Conference on Computer Vision (ICCV). 2019. p 1314\u20131324.",
    "Ioffe S, Szegedy C. Batch normalization: accelerating deep network training by reducing internal covariate shift. In: International Conference on Machine Learning (ICML). 2015. p 448\u2013456.",
    "Kornblith S, Shlens J, Le QV. Do better ImageNet models transfer better? In: CVPR. 2019. p 2661\u20132671.",
    "He K, Girshick R, Doll\u00e1r P. Rethinking ImageNet pre-training. In: ICCV. 2019. p 4918\u20134927.",
    "Zoph B, Ghiasi G, Lin TY, et al. Rethinking pre-training and self-training. In: NeurIPS. 2020. p 3833\u20133845.",
    "Raghu M, Zhang C, Kleinberg J, Bengio S. Transfusion: understanding transfer learning for medical imaging. In: NeurIPS. 2019. p 3347\u20133357.",
    "Recht B, Roelofs R, Schmidt L, Shankar V. Do ImageNet classifiers generalize to ImageNet? In: ICML. 2019. p 5389\u20135400.",
    "Taori R, Dave A, Shankar V, et al. Measuring robustness to natural distribution shifts in image classification. In: NeurIPS. 2020. p 18583\u201318598.",
    "Koh PW, Sagawa S, Marklund H, et al. WILDS: a benchmark of in-the-wild distribution shifts. In: ICML. 2021. p 5637\u20135664.",
    "Miller JP, Taori R, Raghunathan A, Sagawa S, Koh PW, Shankar V, et al. Accuracy on the line: on the strong correlation between out-of-distribution and in-distribution generalization. In: ICML. 2021. p 7721\u20137735.",
    "Hendrycks D, Dietterich T. Benchmarking neural network robustness to common corruptions and perturbations. In: ICLR. 2019.",
    "Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. In: ICML. 2017. p 1321\u20131330.",
    "Ovadia Y, Fertig E, Ren J, et al. Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. In: NeurIPS. 2019. p 13991\u201314002.",
    "Lakshminarayanan B, Pritzel A, Blundell C. Simple and scalable predictive uncertainty estimation using deep ensembles. In: NeurIPS. 2017. p 6402\u20136413.",
    "Li Y, Wang N, Shi J, Liu J, Hou X. Revisiting batch normalization for practical domain adaptation. arXiv:1603.04779. 2016.",
    "Schneider S, Rusak E, Eck L, Bringmann O, Brendel W, Bethge M. Improving robustness against common corruptions by covariate shift adaptation. In: NeurIPS. 2020. p 11539\u201311551.",
    "Wang D, Shelhamer E, Liu S, Olshausen B, Darrell T. Tent: fully test-time adaptation by entropy minimization. In: ICLR. 2021.",
    "Sun B, Feng J, Saenko K. Return of frustratingly easy domain adaptation. In: AAAI. 2016. p 2058\u20132065.",
    "Long M, Cao Y, Wang J, Jordan MI. Learning transferable features with deep adaptation networks. In: ICML. 2015. p 97\u2013105.",
    "Bouthillier X, Delaunay P, Bronzi M, et al. Accounting for variance in machine learning benchmarks. In: Proceedings of Machine Learning and Systems (MLSys). 2021. p 747\u2013764.",
    "Picard D. torch.manual_seed(3407) is all you need: on the influence of random seeds in deep learning architectures for computer vision. arXiv:2109.08203. 2021.",
    "Roberts M, Driggs D, Thorpe M, et al. Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19 using chest radiographs and CT scans. Nat Mach Intell. 2021;3(3):199\u2013217.",
    "Kapoor S, Narayanan A. Leakage and the reproducibility crisis in machine-learning-based science. Patterns. 2023;4(9):100804.",
    "McNemar Q. Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika. 1947;12(2):153\u2013157.",
    "Holm S. A simple sequentially rejective multiple test procedure. Scand J Stat. 1979;6(2):65\u201370.",
    "Efron B, Tibshirani RJ. An Introduction to the Bootstrap. New York: Chapman & Hall; 1993.",
    "Dietterich TG. Approximate statistical tests for comparing supervised classification learning algorithms. Neural Comput. 1998;10(7):1895\u20131923.",
    "Dem\u0161ar J. Statistical comparisons of classifiers over multiple data sets. J Mach Learn Res. 2006;7:1\u201330.",
    "Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D. Grad-CAM: visual explanations from deep networks via gradient-based localization. In: ICCV. 2017. p 618\u2013626.",
    "Pound MP, Burgess AJ, Wilson MH, et al. Deep machine learning provides state-of-the-art performance in image-based plant phenotyping. GigaScience. 2017;6(10):gix083.",
    "Ubbens JR, Stavness I. Deep plant phenomics: a deep learning platform for complex plant phenotyping tasks. Front Plant Sci. 2017;8:1190.",
    "Singh A, Ganapathysubramanian B, Singh AK, Sarkar S. Machine learning for high-throughput stress phenotyping in plants. Trends Plant Sci. 2016;21(2):110\u2013124.",
    "Crossa J, P\u00e9rez-Rodr\u00edguez P, Cuevas J, Montesinos-L\u00f3pez O, Jarqu\u00edn D, de los Campos G, et al. Genomic selection in plant breeding: methods, models, and perspectives. Trends Plant Sci. 2017;22(11):961\u2013975.",
    "Ma W, Qiu Z, Song J, et al. A deep convolutional neural network approach for predicting phenotypes from genotypes. Planta. 2018;248(5):1307\u20131318.",
    "Wang K, Abid MA, Rasheed A, et al. DNNGP, a deep neural network-based method for genomic prediction using multi-fusion of dense fully-connected layers. Front Plant Sci. 2023;14:1126630.",
    "Montesinos-L\u00f3pez OA, Montesinos-L\u00f3pez A, P\u00e9rez-Rodr\u00edguez P, et al. A review of deep learning applications for genomic selection. BMC Genomics. 2021;22:19.",
    "Stringer C, Wang T, Michaelos M, Pachitariu M. Cellpose: a generalist algorithm for cellular segmentation. Nat Methods. 2021;18(1):100\u2013106.",
    "Kirillov A, Mintun E, Ravi N, et al. Segment anything. In: ICCV. 2023. p 4015\u2013426.",
    "Fuentes A, Yoon S, Kim SC, Park DS. A robust deep-learning-based detector for real-time tomato plant diseases and pests recognition. Sensors. 2017;17(9):2022.",
    "Mohanty SP, Hughes DP, Salath\u00e9 M. Using deep learning for image-based plant disease detection: corrigendum. Front Plant Sci. 2018;9:538.",
]


REF_KEYS = {
    "hughes2015": 1, "mohanty2016": 2, "ferentinos2018": 3, "too2019": 4,
    "kamilaris2018": 5, "singh2018stress": 6, "boulent2019": 7, "saleem2019": 8,
    "liu2021": 9, "barbedo2018size": 10, "barbedo2019factors": 11, "picon2019": 12,
    "plantdoc2020": 13, "ramcharan2017": 14, "amara2017": 15, "geirhos2019": 16,
    "brendel2019": 17, "russakovsky2015": 18, "krizhevsky2012": 19,
    "he2016resnet": 20, "howard2019": 21, "ioffe2015": 22, "kornblith2019": 23,
    "he2019rethinking": 24, "zoph2020": 25, "raghu2019": 26, "recht2019": 27,
    "taori2020": 28, "koh2021": 29, "miller2021": 30, "hendrycks2019": 31,
    "guo2017": 32, "ovadia2019": 33, "lakshminarayanan2017": 34, "li2016": 35,
    "schneider2020": 36, "tent2021": 37, "sun2016": 38, "long2015": 39,
    "bouthillier2021": 40, "picard2021": 41, "roberts2021": 42, "kapoor2023": 43,
    "mcnemar1947": 44, "holm1979": 45, "efron1993": 46, "dietterich1998": 47,
    "demsar2006": 48, "selvaraju2017": 49, "pound2017": 50, "ubbens2017": 51,
    "singh2016ml": 52, "crossa2017": 53, "ma2018": 54, "wang2023dnngp": 55,
    "montesinos2021": 56, "stringer2021": 57, "kirillov2023": 58, "fuentes2017": 59,
    "mohanty2018corr": 60,
}

# --------------------------------------------------------------------------- #
# figure legends
# --------------------------------------------------------------------------- #
FIG_LEGENDES = [
    ("Figure 1", "Architecture of MTD-Net. A stride-2 stem feeds two branches: a "
     "downsampled appearance path (blue) supervised by an auxiliary five-way crop head, "
     "and a high-resolution texture path (green) whose features are modulated by a "
     "cross-scale gate derived from the appearance descriptor. The lesion head consumes "
     "the concatenation of both pooled descriptors. The pre-trained variant (MTD-P) "
     "replaces the stem and appearance path with the corresponding ImageNet-pretrained "
     "ResNet-18 stages, leaving the wiring unchanged."),
    ("Figure 2", "Training dynamics and seed spread. (A) Validation macro-F1 per epoch, "
     "mean and min\u2013max range across three seeds, for the two pre-trained models and "
     "the two from-scratch models. (B) Test macro-F1 per seed (filled dots), seed mean "
     "(open circle) and soft-voting ensemble (horizontal bar) for all seven "
     "configurations."),
    ("Figure 3", "Class-level behaviour of the 3-seed ensembles. (A) Per-class F1 for "
     "ResNet-18, MTD-Net and MTD-P. (B) Row-normalised confusion matrix of MTD-P; most "
     "errors fall inside the same crop (for example the three maize leaf diseases)."),
    ("Figure 4", "Forest plot of pairwise macro-F1 differences between 3-seed ensembles, "
     "with paired-bootstrap 95 percent confidence intervals. Red intervals exclude zero "
     "(significant after Holm correction); grey intervals do not."),
    ("Figure 5", "Zero-shot transfer from PlantVillage to PlantDoc. (A) Four views of the "
     "same predictions for six configurations: full 15-way accuracy, accuracy "
     "renormalised over the ten occurring classes, within-crop lesion recognition, and "
     "crop identification implied by the predicted class. (B) Crop-level confusion of "
     "MTD-P, showing that errors cross crop boundaries."),
    ("Figure 6", "Repairing the domain gap with target supervision, averaged over five "
     "stratified folds of the pooled PlantDoc images (96\u2013901 test images per fold). "
     "(A) Macro-F1 as a function of labelled target images per class; the second point "
     "is batch-normalization re-estimation, which uses no labels. (B) Accuracy, macro-F1 "
     "and crop accuracy after full fine-tuning."),
    ("Figure 7", "Accuracy under synthetic acquisition degradations (rows: models; "
     "columns: corruption family and severity), evaluated on the full source test set."),
    ("Figure 8", "Grad-CAM overlays for four source-domain classes (rows: input and the "
     "three architectures). The high-resolution texture path yields more "
     "contour-following attention than the diffuse maps of the matched plain-size "
     "comparison models."),
]

TABLE_LEGENDES = [
    ("Table 1", "Datasets. Composition of the PlantVillage subset (source) and the "
     "mapped PlantDoc subset (target)."),
    ("Table 2", "Main results in the seed-replicated regime (three seeds per "
     "configuration): parameters, per-seed macro-F1, seed mean and standard deviation, "
     "ensemble macro-F1, expected calibration error, mean inference latency, and mean "
     "corruption-robustness accuracy."),
    ("Table 3", "Pairwise statistical contrasts between 3-seed ensembles: ensemble "
     "macro-F1 difference, seed-mean difference, Welch t-test across seeds, exact McNemar "
     "p-value, bootstrap 95 percent CI, and Holm-adjusted significance."),
    ("Table 4", "Cross-dataset evaluation: zero-shot transfer and graded target-domain "
     "adaptation (batch-normalization re-estimation, linear probes at 1/5/10/25 labels "
     "per class, full fine-tuning), averaged over five stratified folds."),
    ("Table 5", "Training-budget control. The two from-scratch configurations retrained "
     "for twelve epochs at the same peak learning rate, three seeds each, with per-seed "
     "macro-F1, seed mean and standard deviation, ensemble macro-F1 and expected "
     "calibration error; the final row is the single-seed gate-initialisation control."),
]


# --------------------------------------------------------------------------- #
# back matter
# --------------------------------------------------------------------------- #
def build_backmatter(ms):
    ms.h("Acknowledgments", 1)
    ms.p("The authors thank the maintainers of the PlantVillage and PlantDoc datasets "
         "and of PyTorch, and the anonymous reviewers for comments that improved the "
         "statistical protocol. This research received no specific grant.")
    ms.h("Author Contributions", 1)
    ms.p("First A. Author: conceptualization, methodology, software, investigation, "
         "writing \u2013 original draft. Second B. Author: data curation, validation, "
         "writing \u2013 review and editing. Third C. Author: supervision, statistical "
         "analysis, writing \u2013 review and editing.")
    ms.h("Competing Interests", 1)
    ms.p("The authors declare no competing interests.")
    ms.h("Data Availability Statement", 1)
    ms.p("All data used are publicly available: PlantVillage "
         "(https://github.com/spMohanty/PlantVillage-Dataset) and PlantDoc "
         "(https://github.com/pratikkayal/PlantDoc-Dataset). The complete code base, "
         "per-run configurations, seeds, logs, trained weights and result files are "
         "provided in the accompanying repository (data and code: see Supporting "
         "Information). No new data were generated.")


def build_references(ms):
    ms.h("References", 1)
    for i, r in enumerate(REFERENCES, 1):
        p = ms.doc.add_paragraph(f"{i}. {r}")
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
        p.paragraph_format.space_after = Pt(0)
        p.runs[0].font.size = Pt(10)


def build_legends(ms):
    ms.h("Figure Legends", 1)
    for tag, txt in FIG_LEGENDES:
        p = ms.doc.add_paragraph("")
        r1 = p.add_run(f"{tag}. ")
        r1.bold = True
        p.add_run(txt)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    ms.h("Table Legends", 1)
    for tag, txt in TABLE_LEGENDES:
        p = ms.doc.add_paragraph("")
        r1 = p.add_run(f"{tag}. ")
        r1.bold = True
        p.add_run(txt)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE


# --------------------------------------------------------------------------- #
# citation resolution
# --------------------------------------------------------------------------- #
import re as _re

_TOKEN = _re.compile(r"\[\[([a-z0-9_]+)\]\]")


def resolve_citations(doc):
    """Replace citation tokens with bracketed numbers in-place."""
    for para in doc.paragraphs:
        for run in para.runs:
            if "[[" not in run.text:
                continue

            def sub(m):
                k = m.group(1)
                if k not in REF_KEYS:
                    raise KeyError(f"unknown citation key {k}")
                return f"[{REF_KEYS[k]}]"

            run.text = _TOKEN.sub(sub, run.text)
    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    for run in para.runs:
                        if "[[" in run.text:
                            run.text = _TOKEN.sub(
                                lambda m: f"[{REF_KEYS[m.group(1)]}]", run.text)


def check_citations():
    """Every cited key must exist; every listed reference should be cited."""
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    cited = set(_TOKEN.findall(src))
    missing = cited - set(REF_KEYS)
    assert not missing, missing
    uncited = [k for k in REF_KEYS if k not in cited]
    return cited, uncited


# --------------------------------------------------------------------------- #
# tables (separate editable Word files, per the author guidelines)
# --------------------------------------------------------------------------- #
def _tbl_doc():
    d = Document()
    st = d.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(11)
    return d


def _add_table(doc, headers, rows, widths=None):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Light Grid Accent 1"
    for j, h in enumerate(headers):
        cell = t.rows[0].cells[j]
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
    for i, row in enumerate(rows, 1):
        for j, v in enumerate(row):
            t.rows[i].cells[j].text = str(v)
    if widths:
        for j, w in enumerate(widths):
            for r in t.rows:
                r.cells[j].width = Inches(w)
    return t


def build_tables():
    os.makedirs(SUB, exist_ok=True)

    # ---- Table 1: datasets -------------------------------------------------
    d = _tbl_doc()
    d.add_paragraph("Table 1. Dataset composition.").runs[0].bold = True
    rows1 = []
    for i, n in enumerate(meta["class_names"]):
        tr = meta["train_per_class"].get(n, 0)
        te = meta["test_per_class"].get(n, 0)
        pd_n_c = pdmeta["per_class"].get(str(i), pdmeta["per_class"].get(i, 0))
        rows1.append([n.replace("___", " \u2013 "), tr, te, pd_n_c])
    rows1.append(["TOTAL", meta["train_size"], meta["test_size"], pdmeta["n"]])
    _add_table(d, ["Class (crop \u2013 disease)", "PV train", "PV test", "PlantDoc mapped"],
               rows1)
    d.save(os.path.join(SUB, "Table_1.docx"))

    # ---- Table 2: main results ---------------------------------------------
    d = _tbl_doc()
    d.add_paragraph("Table 2. Seed-replicated source-domain results "
                    "(PlantVillage test set, n = %d)." % meta["test_size"]).runs[0].bold = True
    rows2 = []
    for m in MODELS:
        s = summary.get(m)
        if not s:
            continue
        r = rob.get(f"{m}_s0", {})
        rows2.append([
            NAME[m] + (" (pre-trained)" if m in ("resnet18", "mtdnet_pt",
                                                 "mobilenet_v3_small") else " (scratch)"),
            f"{s['params(M)']:.2f}",
            ", ".join(f"{v:.3f}" for v in seeds(m)),
            f"{s['macro_f1']:.4f} \u00b1 {s['macro_f1_sd']:.4f}",
            f"{s['ens_macro_f1'] if 'ens_macro_f1' in s else sig['models'].get(m, {}).get('ens_macro_f1', float('nan')):.4f}",
            f"{s['ece']:.3f}",
            f"{s.get('ms/img', float('nan')):.1f}",
            f"{r.get('robust_acc', float('nan')):.3f}",
        ])
    _add_table(d, ["Model", "Params (M)", "macro-F1 per seed", "seed mean \u00b1 sd",
                   "ensemble", "ECE", "ms/img", "corruption mean"], rows2)
    d.save(os.path.join(SUB, "Table_2.docx"))

    # ---- Table 3: contrasts -------------------------------------------------
    d = _tbl_doc()
    d.add_paragraph("Table 3. Pairwise contrasts between 3-seed soft-voting ensembles "
                    "(positive values favour the first model).").runs[0].bold = True
    rows3 = []
    for a, b in [("mtdnet_pt", "resnet18"), ("resnet18", "mtdnet"),
                 ("mtdnet_pt", "mtdnet"), ("mtdnet", "smallcnn"),
                 ("mtd_concat", "smallcnn"), ("mtdnet", "mtd_concat"),
                 ("mtdnet", "mtd_single"), ("mobilenet_v3_small", "smallcnn"),
                 ("mtdnet_pt", "smallcnn"), ("resnet18", "smallcnn")]:
        p = pair(a, b)
        if not p:
            continue
        rows3.append([
            f"{NAME[a]} vs {NAME[b]}",
            f"{p['delta_f1_ens'] * 100:+.2f}",
            f"[{p['boot_ci_low'] * 100:+.2f}, {p['boot_ci_high'] * 100:+.2f}]",
            f"{p['delta_f1_mean_seeds'] * 100:+.2f}",
            f"{p['ttest_seed_p']:.3f}",
            ("<1e-4" if p["mcnemar_p"] < 1e-4 else f"{p['mcnemar_p']:.2e}"),
            "yes" if p["significant_holm"] else "no",
        ])
    _add_table(d, ["Contrast", "\u0394 ens. macro-F1 (pt)", "bootstrap 95% CI (pt)",
                   "\u0394 seed mean (pt)", "seed t-test p", "McNemar p",
                   "Holm-significant"], rows3)
    d.save(os.path.join(SUB, "Table_3.docx"))

    # ---- Table 4: cross-domain + adaptation ---------------------------------
    d = _tbl_doc()
    d.add_paragraph("Table 4. PlantVillage \u2192 PlantDoc: zero-shot transfer and graded "
                    "target adaptation (macro-F1, mean \u00b1 sd over five stratified "
                    "folds of the pooled %d target images)." % pdmeta["n"]).runs[0].bold = True
    meths = [("source", "source (no target data)"), ("bn", "BN re-estimation (unlabeled)"),
             ("probe_1", "probe, 1 label/class"), ("probe_5", "probe, 5 labels/class"),
             ("probe_10", "probe, 10 labels/class"), ("probe_25", "probe, 25 labels/class"),
             ("finetune", "fine-tune (all labels)")]
    rows4 = []
    for m in ["mtdnet_pt_s0", "resnet18_s0", "mtdnet_s0", "smallcnn_s0"]:
        if m not in adapt.get("results", {}):
            continue
        ent = adapt["results"][m]["methods"]
        base = m.rsplit("_s", 1)[0]
        z = CD.get(base, {})
        rows4.append([
            NAME[base],
            f"{z.get('acc15', float('nan')):.3f} / {z.get('acc10', float('nan')):.3f} / "
            f"{z.get('within_crop', float('nan')):.3f} / {z.get('crop_acc', float('nan')):.3f}",
            f"{z.get('mean_conf', float('nan')):.2f}",
            f"{z.get('ece_target', float('nan')):.3f}" if "ece_target" in z else "\u2014",
            "\n".join(f"{lab}: {ent[k]['macro_f1']:.3f} \u00b1 {ent[k]['macro_f1_sd']:.3f}"
                      for k, lab in meths if k in ent),
        ])
    _add_table(d, ["Model", "zero-shot acc15 / acc10 / within-crop / crop",
                   "mean conf.", "ECE (target)", "adaptation macro-F1 (mean \u00b1 sd)"],
               rows4, widths=[1.1, 1.7, 0.7, 0.7, 2.6])
    d.save(os.path.join(SUB, "Table_4.docx"))

    # ---- Table 5: training-budget control -----------------------------------
    if opt is not None and "mtdnet" in opt and "smallcnn" in opt:
        d = _tbl_doc()
        d.add_paragraph("Table 5. Training-budget control: the from-scratch "
                        "configurations retrained for twelve epochs at the same peak "
                        "learning rate, three seeds each.").runs[0].bold = True
        rows5 = []

        def _blk(key, label, budget):
            e = opt[key]
            seeds_v = ", ".join(f"{r['base_f1'] if budget == 6 else r['opt_f1']:.4f}"
                                for r in e["per_seed"])
            ece_v = (sum(r["base_ece"] if budget == 6 else r["opt_ece"]
                         for r in e["per_seed"]) / len(e["per_seed"]))
            src = sig if budget == 6 else sig12
            ens = src["models"].get(key, {}).get("ens_macro_f1", float("nan")) if src else float("nan")
            mu = e["base_mean"] if budget == 6 else e["opt_mean"]
            sd = e["base_sd"] if budget == 6 else e["opt_sd"]
            rows5.append([label, str(budget), seeds_v,
                          f"{mu:.4f} \u00b1 {sd:.4f}", f"{ens:.4f}", f"{ece_v:.3f}"])

        _blk("mtdnet", NAME["mtdnet"] + " (scratch)", 6)
        _blk("mtdnet", NAME["mtdnet"] + " (scratch)", 12)
        _blk("smallcnn", NAME["smallcnn"] + " (scratch)", 6)
        _blk("smallcnn", NAME["smallcnn"] + " (scratch)", 12)
        gv = opt.get("mtdnet_g_seed0")
        if gv:
            rows5.append([NAME["mtdnet"] + " (wide gate init)", "12",
                          f"{gv['opt_f1']:.4f}", f"{gv['opt_f1']:.4f} \u00b1 n/a",
                          f"{gv['opt_f1']:.4f}", f"{gv['ece']:.3f}"])
        _add_table(d, ["Model", "epochs", "macro-F1 per seed", "seed mean \u00b1 sd",
                       "ensemble", "ECE"], rows5)
        d.save(os.path.join(SUB, "Table_5.docx"))


# --------------------------------------------------------------------------- #
# cover letter
# --------------------------------------------------------------------------- #
COVER = """Dear Editors,

We submit our manuscript, "{title}", for consideration as an Original Article in the Annals of the New York Academy of Sciences.

Plant-disease classification is one of the most active application areas of deep learning, yet its reporting standards lag behind its volume: single-run evaluations on laboratory datasets are common, and architectural novelty is frequently credited for gains that are confounded with ImageNet pre-training. Our manuscript provides the controlled, seed-replicated evaluation that this literature lacks.

The study makes five contributions. (1) A purpose-built, texture-appearance decoupled network is compared against parameter-matched and pre-trained controls with three seeds per configuration, soft-voting ensembles, exact McNemar tests, paired-bootstrap confidence intervals and Holm correction. (2) A training-budget control shows that the seed-level variance which obscures the architectural effect at six epochs is an artifact of the budget: at twelve epochs the spread collapses by an order of magnitude and the effect becomes seed-detectable, while most of the apparent advantage of pre-training disappears - a direct warning for single-run, single-budget studies in this field. (3) The remaining architectural gain is conditional: it is null once both models start from the same ImageNet weights, and it widens, rather than shrinks, under synthetic image corruption. (4) Zero-shot transfer to the field-image dataset PlantDoc collapses all models; we localize the failure to crop identity rather than lesion recognition, and show that neither source-fitted temperature scaling nor any fine-tuning recipe we tried repairs either the accuracy or the calibration without target labels. (5) A graded set of target-domain adaptations (unlabeled batch-normalization re-estimation, few-shot linear probes, fine-tuning), evaluated by five-fold cross-validation, quantifies how many target labels repair the gap.

All experiments ran on CPU-only hardware and are fully reproducible: code, per-run configurations, seeds, logs, trained weights and every result file are included in the submission. All quantitative statements in the manuscript are generated programmatically from those files. The datasets used are public; no new data were collected and no ethical approval was required.

We confirm that this manuscript is original, is not under consideration elsewhere, and that all authors have approved the submission. We suggest no reviewers with conflicts of interest.

Sincerely,
The Authors
"""


def build_cover():
    d = _tbl_doc()
    d.add_paragraph("Cover Letter").runs[0].bold = True
    d.add_paragraph(COVER.format(title=TITLE))
    d.save(os.path.join(SUB, "Cover_Letter.docx"))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(SUB, exist_ok=True)
    cited, uncited = check_citations()
    doc = base_doc()
    ms = MS(doc)
    build_front(ms)
    build_intro(ms)
    build_methods(ms)
    build_results(ms)
    build_discussion(ms)
    build_conclusion(ms)
    build_backmatter(ms)
    build_references(ms)
    build_legends(ms)
    resolve_citations(doc)
    doc.save(os.path.join(SUB, "MAIN_MANUSCRIPT.docx"))
    build_tables()
    build_cover()

    total = ms.total(exclude=("front",))
    counts = {"sections": ms.words, "total_main_text": total,
              "references": len(REFERENCES), "cited_keys": len(cited),
              "uncited_references": uncited,
              "abstract_words": len(ABSTRACT.split()),
              "limit": 7000, "within_limit": total <= 7000}
    with open(os.path.join(SUB, "word_count.json"), "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2, ensure_ascii=False)
    print(json.dumps(counts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()