"""Integrate observed frozen-DINO baseline and target bootstrap results into the manuscript."""
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def replace_or_insert_section(text, heading, block, before_heading):
    start=text.find(heading)
    if start>=0:
        candidates=[x for x in (text.find("\n### ",start+len(heading)),text.find("\n## ",start+len(heading))) if x>=0]
        end=min(candidates) if candidates else len(text)
        return text[:start]+block.rstrip()+"\n\n"+text[end:].lstrip("\n")
    anchor=text.find(before_heading)
    if anchor<0: raise ValueError("Could not find insertion anchor: "+before_heading)
    return text[:anchor]+block.rstrip()+"\n\n"+text[anchor:]

def main():
    baseline_path=ROOT/"report/additional_baselines/dinov2_s14.json"
    stats_path=ROOT/"report/corrected_summary/target_bootstrap.json"
    if not baseline_path.exists() or not stats_path.exists():
        raise SystemExit("DINOv2 and target-bootstrap outputs are required before manuscript update.")
    b=load(baseline_path); s=load(stats_path)
    if b.get("status")!="completed": raise SystemExit("DINOv2 baseline did not complete.")
    dino=b["target_test"]; src=b["source_test"]
    dino_record=s["models"].get("DINOv2-S/14 frozen linear probe")
    if not dino_record: raise SystemExit("Target bootstrap did not include DINOv2.")
    ens=dino_record["five_seed_probability_ensemble"]
    aci=ens["accuracy_image_bootstrap_ci95"]; fci=ens["macro_f1_supported_stratified_bootstrap_ci95"]
    supports=list(s["class_support"].values()); support_note=f"Observed target class support ranged from {min(supports)} to {max(supports)} images; the smallest represented class has only {min(supports)} images."
    comps={r["baseline"]:r for r in s["paired_dinov2_comparisons"]}
    def pctci(pair,key):
        lo,hi=pair[key]
        return f"{lo:+.4f} to {hi:+.4f}"
    p_mtd=comps.get("MTD-P")
    p_res=comps.get("ResNet-18")
    sentence=(f"A frozen DINOv2-S/14 linear probe, with regularization selected on source validation only, achieved source-test macro-F1 {src['macro_f1_15']:.4f} and zero-shot accuracy {dino['accuracy']:.4f} on the same 92-image mapped PlantDoc holdout "
              f"(ordinary image-bootstrap 95% CI {aci[0]:.4f}–{aci[1]:.4f}; supported-class macro-F1 {dino['macro_f1_supported']:.4f}, class-stratified 95% CI {fci[0]:.4f}–{fci[1]:.4f}).")
    manuscript=ROOT/"manuscript/PlantShiftEval_corrected_manuscript.md"
    text=manuscript.read_text(encoding="utf-8")
    # Repair a legacy truncated paragraph before adding newly measured results.
    uncertainty_heading="### Conditional target image uncertainty"
    discussion_heading="\\n## Discussion"
    uncertainty_start=text.find(uncertainty_heading)
    discussion_start=text.find(discussion_heading, uncertainty_start+len(uncertainty_heading)) if uncertainty_start>=0 else -1
    if uncertainty_start<0 or discussion_start<0:
        raise ValueError("Target uncertainty section or Discussion heading is missing")
    uncertainty_block=("### Conditional target image uncertainty\\n\\n"
      "The five-seed MTD-P probability ensemble correctly classified 31/92 target-test images (accuracy 0.3370; image-level Wilson 95% interval 0.2486 to 0.4383). "
      "The five-seed ResNet-18 probability ensemble correctly classified 30/92 target-test images (accuracy 0.3261; image-level Wilson 95% interval 0.2389 to 0.4272). "
      "These image-level intervals do not account for possible dependence among images from the same plant, leaf, or acquisition session. "
      "The target holdout contains 92 images across ten represented classes, with class support ranging from 3 to 12 images. "
      "Per-class precision, recall, F1 and class-stratified bootstrap intervals are provided in report/corrected_summary/target_per_class_metrics.csv.")
    text=text[:uncertainty_start]+uncertainty_block+"\\n\\n"+text[discussion_start+1:]
    abs_start=text.find("## Abstract")
    intro_start=text.find("## Introduction")
    if abs_start<0 or intro_start<0: raise ValueError("Manuscript abstract/introduction headings missing")
    abstract=text[abs_start:intro_start]
    if "A frozen DINOv2-S/14 linear probe" not in abstract:
        abstract=abstract.rstrip()+" "+sentence+"\n\n"
        text=text[:abs_start]+abstract+text[intro_start:]
    result_block=("### Frozen DINOv2-S/14 baseline\n\n"
      f"The official DINOv2-S/14 backbone was kept frozen. A multinomial logistic-regression linear probe was selected from the declared C grid using only the source validation partition (selected C = {b['classifier']['selected_C']:g}) and then refit on all permitted source-cache rows. No target labels or target-test scores were used for model selection. The baseline reached source-test accuracy {src['accuracy']:.4f} and 15-class macro-F1 {src['macro_f1_15']:.4f}.\n\n"
      f"On the same mapped target holdout, DINOv2 achieved accuracy {dino['accuracy']:.4f} (ordinary image-bootstrap 95% CI {aci[0]:.4f}–{aci[1]:.4f}) and supported-class macro-F1 {dino['macro_f1_supported']:.4f} (class-stratified 95% CI {fci[0]:.4f}–{fci[1]:.4f}). The supported-class metric averages the ten classes present in the target holdout; 15-class macro-F1 is also retained in the machine-readable output. The confidence intervals condition on the selected images and do not account for unknown acquisition dependence. {support_note}\n\n"
      "Paired comparisons use the five-seed probability ensemble for each existing model and the identical target-image order. They are summarized in report/corrected_summary/target_bootstrap.json; the raw class-wise metrics and intervals are in report/corrected_summary/target_per_class_metrics.csv.\n")
    text=replace_or_insert_section(text,"### Frozen DINOv2-S/14 baseline",result_block,"### Conditional target image uncertainty")
    if "The frozen DINOv2-S/14 baseline adds a direct representation-level comparison" not in text:
        para=("The frozen DINOv2-S/14 baseline adds a direct representation-level comparison under the same mapped target holdout. Its classifier hyperparameter was chosen using source validation only, and target labels were reserved for final scoring. "
              f"Against the five-seed MTD-P probability ensemble, the paired accuracy difference (DINOv2 minus MTD-P) was {p_mtd['accuracy_difference']:+.4f} with a class-stratified bootstrap interval {pctci(p_mtd,'accuracy_difference_stratified_ci95')} and Holm-adjusted exact McNemar p = {p_mtd['mcnemar_holm_adjusted_p']:.4g}. "
              f"Against the ResNet-18 ensemble, the difference was {p_res['accuracy_difference']:+.4f} with interval {pctci(p_res,'accuracy_difference_stratified_ci95')} and Holm-adjusted p = {p_res['mcnemar_holm_adjusted_p']:.4g}. "
              f"These paired comparisons apply to the selected 92 images and do not establish equivalence or generalization to the full PlantDoc population. {support_note} Frozen foundation features are a direct comparator, not a novel contribution of this study.")
        anchor="The closest 2026 literature"
        pos=text.find(anchor)
        if pos<0: raise ValueError("Cannot locate domain-generalization discussion anchor")
        start=text.rfind("\n\n",0,pos)+2
        text=text[:start]+para+"\n\n"+text[start:]
    conclusion_start=text.find("## Conclusion")
    refs_start=text.find("## Figure legends")
    if conclusion_start<0 or refs_start<0: raise ValueError("Conclusion or figure legends missing")
    conclusion=text[conclusion_start:refs_start]
    if "The frozen DINOv2-S/14 baseline" not in conclusion:
        conclusion=conclusion.rstrip()+" The frozen DINOv2-S/14 baseline and paired target-image analysis place the candidate architecture in direct context with a strong self-supervised representation, while keeping conclusions conditional on the mapped 92-image holdout.\n\n"
        text=text[:conclusion_start]+conclusion+text[refs_start:]
    manuscript.write_text(text,encoding="utf-8")

    readme=ROOT/"README.md"
    rtext=readme.read_text(encoding="utf-8")
    drow=f"| DINOv2-S/14 frozen probe | {src['macro_f1_15']:.4f} (single fit) | not evaluated | {dino['accuracy']:.4f} |"
    anchor="| ResNet-18 | 0.8742 ± 0.0151 | 0.7437 | 0.3109 |"
    if drow not in rtext:
        if anchor not in rtext: raise ValueError("README core result table anchor missing")
        rtext=rtext.replace(anchor,anchor+"\n"+drow)
    evidence_line="- [Frozen DINOv2 baseline and target bootstrap report](report/additional_baselines/dinov2_s14.json), [paired target comparisons](report/corrected_summary/target_bootstrap.json), and [per-class metrics](report/corrected_summary/target_per_class_metrics.csv)."
    if evidence_line not in rtext:
        rtext=rtext.replace("- [Full evidence audit]",evidence_line+"\n- [Full evidence audit]")
    readme.write_text(rtext,encoding="utf-8")

    positioning=ROOT/"docs/RESEARCH_POSITIONING.md"
    pos_text=positioning.read_text(encoding="utf-8")
    heading="## Frozen foundation-model baseline"
    block=(heading+"\n\n"
      f"The frozen DINOv2-S/14 source-only linear probe achieved source macro-F1 {src['macro_f1_15']:.4f} and target accuracy {dino['accuracy']:.4f} on the same 92-image mapped target holdout. The target accuracy interval is {aci[0]:.4f}–{aci[1]:.4f}; supported-class macro-F1 is {dino['macro_f1_supported']:.4f} with class-stratified interval {fci[0]:.4f}–{fci[1]:.4f}. Regularization was selected on source validation, not on target labels. The paired comparisons and McNemar/Holm results are in report/corrected_summary/target_bootstrap.json. This baseline narrows the novelty claim to controlled architecture/optimization contrasts; it does not turn published DINOv2 results from different label spaces into direct comparisons.\n\n")
    if heading in pos_text:
        st=pos_text.index(heading); nxt=pos_text.find("\n## ",st+len(heading)); en=nxt if nxt>=0 else len(pos_text)
        pos_text=pos_text[:st]+block.rstrip()+"\n\n"+pos_text[en:].lstrip("\n")
    else:
        anchor="## Narrative and boundaries"
        pos=pos_text.find(anchor)
        if pos<0: raise ValueError("Research positioning insertion anchor missing")
        pos_text=pos_text[:pos]+block+pos_text[pos:]
    positioning.write_text(pos_text,encoding="utf-8")
    print(json.dumps({"manuscript_updated":True,"readme_updated":True,"positioning_updated":True,
      "dino_target_accuracy":dino["accuracy"],"dino_source_macro_f1":src["macro_f1_15"],
      "mtdp_accuracy_difference":p_mtd["accuracy_difference"],"resnet_accuracy_difference":p_res["accuracy_difference"]},indent=2))

if __name__=="__main__": main()
