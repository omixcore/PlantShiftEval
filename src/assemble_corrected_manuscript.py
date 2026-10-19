"""Assemble a journal-neutral corrected manuscript only from a complete study."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    folder=ROOT/'report/corrected_summary'
    status=json.loads((folder/'completeness.json').read_text())
    if status['status']!='complete' or status['completed_training_and_evaluation']!=65:
        raise SystemExit('Final manuscript blocked: the 65-run study is incomplete.')
    agg=json.loads((folder/'aggregates.json').read_text())
    comp=json.loads((folder/'paired_comparisons.json').read_text())
    def row(model,ep=6,lr='1e-3'):
        return next(r for r in agg if (r['model'],r['epochs'],r['lr'])==(model,ep,lr))
    m=row('mtdnet');c=row('smallcnn');p=row('mtdnet_pt',lr='3e-4');r=row('resnet18',lr='3e-4')
    primary=next(x for x in comp['family'] if x['name']=='primary_mtd_vs_cnn')
    delta=primary['mean_difference']*100;lo,hi=[v*100 for v in primary['ci95_student_t']]
    abstract=(
        'Texture-oriented architectures for plant disease recognition require evaluation that separates configuration effects from optimization and domain shift. '
        'We examined MTD-Net using a corrected, auditable protocol with known-leaf and image-overlap screening, training-internal validation, and 65 fixed training configurations over five seeds. '
        f'On the cleaned 15-class source test, six-epoch MTD-Net and a parameter-matched CNN achieved mean macro-F1 values of {m["source_macro_f1"]["mean"]:.4f} and {c["source_macro_f1"]["mean"]:.4f}, respectively. '
        f'The paired difference was {delta:+.2f} percentage points (seed-level 95% interval {lo:+.2f} to {hi:+.2f}). '
        f'Pretrained MTD-P and ResNet-18 achieved {p["source_macro_f1"]["mean"]:.4f} and {r["source_macro_f1"]["mean"]:.4f}. '
        'Epoch-by-learning-rate controls and component ablations characterize protocol dependence without establishing semantic disentanglement. '
        f'Mean zero-shot accuracies on 92 official target-test images were {p["target_accuracy"]["mean"]:.3f} and {r["target_accuracy"]["mean"]:.3f} for the pretrained pair. '
        'Corruption, calibration, image-only adaptation and supervised adaptation are reported separately. '
        'Limited leaf metadata, uneven source support and a small target holdout constrain interpretation. The evidence supports configuration-specific conclusions rather than universal architectural superiority.'
    )
    base=(ROOT/'manuscript/INTRODUCTION_METHODS.md').read_text()
    title,body=base.split('\n',1)
    body=body[body.index('## Introduction'):]
    methods,refs=body.split('## References',1)
    raw_results=(ROOT/'manuscript/corrected_results.md').read_text()
    results='\n'+raw_results[raw_results.index('| Model |'):raw_results.index('## Interpretation boundaries')]
    labels={'mtd_appearance':'Appearance only','mtd_concat':'No gate','mtd_single':'No auxiliary task','mtdnet_pt':'MTD-P','mtdnet':'MTD-Net','resnet18':'ResNet-18','smallcnn':'Plain CNN'}
    for key,value in labels.items():results=results.replace('| '+key+' |','| '+value+' |')
    results+='All cells summarize five training seeds. Source macro-F1 and target accuracy are different outcomes; uncertainty is reported separately below.\n\n'
    results+='### Model capacity\n\n| Configuration | Trainable parameters |\n|---|---:|\n'
    for model in ('mtdnet','smallcnn','mtd_concat','mtd_single','mtd_appearance','mtdnet_pt','resnet18'):
        rr=row(model,lr='3e-4' if model in ('mtdnet_pt','resnet18') else '1e-3')
        results+=f"| {labels[model]} | {rr['params']:,} |\n"
    contrasts=['### Paired configuration contrasts','']
    for x in comp['family']:
        l,h=x['ci95_student_t']
        names={'primary_mtd_vs_cnn':'MTD-Net minus plain CNN','gate':'MTD-Net minus no gate','auxiliary_task':'MTD-Net minus no auxiliary task','texture_branch_with_capacity_change':'MTD-Net minus appearance only','pretrained_increment':'MTD-P minus ResNet-18'}
        contrasts.append(f"- {names[x['name']]}: Δ macro-F1 {100*x['mean_difference']:+.2f} points; seed-level 95% interval {100*l:+.2f} to {100*h:+.2f}; Holm-adjusted p = {x['p_holm']:.4g}; n = {x['n']} paired seeds.")
    contrasts+=['','Intervals describe training-seed variability conditional on the frozen split. They do not establish practical equivalence or uncertainty over future acquisition settings.','']
    numerical=['### Training strategy contrasts','']
    for model,label in [('mtdnet','MTD-Net'),('smallcnn','the plain CNN')]:
        a=row(model,6,'1e-3');b=row(model,12,'1e-3');c2=row(model,6,'2e-3');d=row(model,12,'2e-3')
        interaction=next(x for x in comp['factorial_interactions'] if x['model']==model)
        l,h=interaction['ci95_student_t']
        numerical.append(f"For {label}, mean source macro-F1 changed from {a['source_macro_f1']['mean']:.4f} to {b['source_macro_f1']['mean']:.4f} between six and twelve epochs at peak LR 1e-3, and from {c2['source_macro_f1']['mean']:.4f} to {d['source_macro_f1']['mean']:.4f} at peak LR 2e-3. The epoch-by-LR difference-of-differences was {100*interaction['mean_difference']:+.2f} points (seed-level 95% interval {100*l:+.2f} to {100*h:+.2f}). These contrasts apply to the declared OneCycle schedules and do not identify a general optimal recipe.")
        numerical.append('')
    numerical+=['### Corruption and independent target transfer','',
        '| Configuration | Source clean accuracy | Mean corruption accuracy | Official target accuracy |',
        '|---|---:|---:|---:|']
    for rr,label in [(m,'MTD-Net'),(c,'Plain CNN'),(p,'MTD-P'),(r,'ResNet-18')]:
        numerical.append(f"| {label} | {rr['source_accuracy']['mean']:.4f} | {rr['corruption_accuracy']['mean']:.4f} | {rr['target_accuracy']['mean']:.4f} |")
    numerical+=['','These are means over five training seeds. Corruption accuracy averages eight declared conditions; target accuracy uses the same 92 independently held-out mapped target images for each model. The two outcomes must not be substituted for one another.','',
        '### Confidence under target shift','',
        '| Configuration | Raw target ECE | Target ECE after source fitted temperature |',
        '|---|---:|---:|']
    for rr,label in [(m,'MTD-Net'),(c,'Plain CNN'),(p,'MTD-P'),(r,'ResNet-18')]:
        numerical.append(f"| {label} | {rr['target_ece']['mean']:.4f} | {rr['target_calibrated_ece']['mean']:.4f} |")
    numerical+=['','Temperature fitting used source-validation labels only. ECE changes describe this source-fitted calibrator on the selected target test; no target-label oracle was fitted.','',
        '### Target image and label access','',
        '| Configuration | Zero shot accuracy | Image only AdaBN accuracy | Supervised fine tuning accuracy |',
        '|---|---:|---:|---:|']
    for rr,label in [(p,'MTD-P'),(r,'ResNet-18')]:
        numerical.append(f"| {label} | {rr['target_accuracy']['mean']:.4f} | {rr['adabn_accuracy']['mean']:.4f} | {rr['supervised_ft_accuracy']['mean']:.4f} |")
    numerical+=['','All methods use the same held-out target test and five source seeds. The adaptation columns have additional target-domain information and are not zero-shot results. Complete per-seed values and sample SDs accompany the tables in the machine-readable evidence package.','']
    numerical='\n'.join(numerical)
    uncertainty=json.loads((folder/'target_ensemble_uncertainty.json').read_text())
    numerical+='\n### Conditional target image uncertainty\n\n'
    for model,label in [('mtdnet_pt','MTD-P'),('resnet18','ResNet-18')]:
        u=next(x for x in uncertainty if x['model']==model and x['method']=='zero_shot')
        l,h=u['wilson_ci95']
        numerical+=f"The five-seed {label} probability ensemble correctly classified {u['correct']}/{u['n_images']} target-test images (accuracy {u['accuracy']:.4f}; image-level Wilson interval {l:.4f} to {h:.4f}). "
    numerical+='These ensemble estimates are distinct from mean per-seed accuracy. The approximate intervals condition on the selected images and assume independent image outcomes; unknown acquisition dependencies and population selection remain outside their scope. Per-class supports and recalls are archived.\n\n'
    numerical+='### Observed configuration effects\n\nAt the primary six-epoch budget, the gate and auxiliary-task removals have numerically higher mean macro-F1 than the full candidate, while branch removal has a lower mean. None of the five declared architecture contrasts establishes a benefit after Holm adjustment. This is limited evidence, not a demonstration of equivalence or proof that those components can never help. The higher-budget scratch cells improve both models descriptively; their effect sizes are larger than the primary six-epoch architecture difference.\n\n'
    numerical+=f"Source-fitted temperature scaling increases mean target ECE for all four core configurations. For MTD-P it changes from {p['target_ece']['mean']:.4f} to {p['target_calibrated_ece']['mean']:.4f}; for ResNet-18, from {r['target_ece']['mean']:.4f} to {r['target_calibrated_ece']['mean']:.4f}. AdaBN yields modest descriptive accuracy increases, while supervised target fine-tuning reaches {p['supervised_ft_accuracy']['mean']:.4f} and {r['supervised_ft_accuracy']['mean']:.4f}. The latter result depends on target labels and does not repair zero-shot transfer. No target-method superiority test is asserted from these descriptive means.\n\n"
    discussion='''## Discussion

The central methodological result is the set of conditional contrasts rather than a declaration that one architecture is generally best. The source comparison evaluates a whole candidate configuration, including its optimization and auxiliary supervision. Component ablations narrow the interpretation, but removing a branch changes capacity, and neither the branch name nor a predictive gain establishes a uniquely texture-based mechanism. Direct feature interventions and additional matched controls would be required for that stronger claim.

The factorial controls address an important ambiguity in the historical study: increasing training duration while also increasing peak learning rate does not isolate an epoch effect. The corrected design reports both factors and their interaction under common split assignments and seed identifiers. These schedules still sample a limited hyperparameter space and are not a comprehensive tuning contest. Any favorable contrast is specific to these declared budgets and should not be generalized to all well-tuned baselines.

The pretrained comparison estimates whether the candidate additions help when both configurations have access to pretrained ResNet-18 features. It does not isolate pretraining itself because the scratch and pretrained architectures differ. Parameter counts and data access accompany scores so that an apparent benefit is not conflated with an identical-capacity or identical-information comparison.

Source performance, synthetic stress tests and target transfer answer different questions. Corruptions modify source images while largely preserving the source labeling and acquisition context. The target experiment changes image composition, collection conditions and taxonomy support simultaneously; it does not identify a single causal reason for errors. Temperature scaling fitted on source validation addresses source confidence and need not remain calibrated under target shift. Numerical changes in target ECE must therefore be read alongside target accuracy and the explicit fitting split.

AdaBN and supervised fine-tuning introduce different information. The former accesses target-training pixels without labels; the latter uses labels and a separate target-validation set. Their gains or losses are conditional on that access and must not be presented as zero-shot generalization. Using official target-test membership after documented filtering provides a clearer held-out estimate than the historical pooled cross-validation, but only 92 mapped test images remain. Seed replication does not remove this small-test-set limitation.

Data curation also changes the population being measured. Known source leaf overlap was substantial, and removing it reduces some class supports severely. Missing biological group IDs remain unresolved for part of both source partitions. Difference-hash screening can remove similar independent images and can miss duplicates with larger transformations. The corrected study is thus more explicit about available evidence, without certifying complete biological independence or a representative field population. Historical and corrected scores cannot quantify the impact of checkpoint-selection leakage alone because the training composition also changed.

The reusable contribution is the traceable evaluation record: fixed inputs, class mapping, explicit target access, validation-selected checkpoints, seed-level contrasts and reconstructable figures. This record supports independent scrutiny of the conditional effects and negative findings. Further acquisition-level and deployment validation would be needed to assess practical use.

## Conclusion

This study evaluates texture-oriented configuration choices under explicit optimization and domain-access controls. The reported estimates, intervals and failure boundaries define where the current evidence applies. Additional datasets, better biological grouping metadata and independently collected target images are needed before making broader claims about disease-specific texture mechanisms or practical field reliability.

## Figure legends

**Figure 1. Corrected source-domain configuration comparisons.** Individual points show five training seeds; the central mark and error bars show the mean and sample standard deviation. Scratch models use six epochs and peak LR 1e-3; pretrained models use six epochs and peak LR 3e-4. Comparisons are interpreted within the declared initialization stratum.

**Figure 2. Training-recipe factorial.** Source-test macro-F1 for MTD-Net and the plain CNN at 6/12 epochs and peak LR 1e-3/2e-3. Points show individual seeds and lines join cell means. The design separates the sampled factors; it does not establish an exhaustive optimum.

**Figure 3. Component contrasts.** Source-test macro-F1 for the candidate model, gate removal, disabled auxiliary task and appearance-only ablation. Points show seeds and error bars show sample SD. Appearance-only removal changes parameter count and cannot by itself identify a texture-specific mechanism.

**Figure 4. Corruption and domain-transfer limits.** Mean accuracy over eight fixed synthetic source corruptions and accuracy on the mapped official target holdout. These are distinct evaluations. Source and target images differ in more than one acquisition factor.

**Figure 5. Target information conditions.** Accuracy on the same held-out target test after zero-shot transfer, target-training-image AdaBN or supervised target-training fine-tuning. Individual points represent five source training seeds. Fine-tuning checkpoints are chosen using only target validation; target-test labels are used for final evaluation only.
'''
    text=title+'\n\n**Evidence status: complete corrected numerical study; author-reviewed submission materials pending.**\n\n## Abstract\n\n'+abstract+'\n\n'+methods+'\n## Results\n'+results+'\n'+'\n'.join(contrasts)+numerical+discussion+'\n## References'+refs
    (ROOT/'manuscript/PlantShiftEval_corrected_manuscript.md').write_text(text)
    print('Assembled manuscript; abstract words:',len(abstract.split()))

if __name__=='__main__':main()
