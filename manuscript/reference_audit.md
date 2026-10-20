# PlantShiftEval bibliography and claim-support audit

**Audit date:** 20 October 2026  
**Manuscript:** `manuscript/PlantShiftEval_corrected_manuscript.md`  
**Reference count:** 51  
**Scope:** Bibliographic identity and metadata, followed by an assessment of whether each source supports the specific claim for which it is cited.

## Summary of corrections

- Corrected author-name order in reference 4 and standardized the first author's initials in reference 17.
- Corrected the PlantWild paper title, DOI and pages in reference 19.
- Replaced the arXiv-only PlantSeg citation with the 2026 *Scientific Data* version of record in reference 20.
- Added the missing author to reference 21 and expanded the author list for reference 23.
- Added direct plant-specific texture evidence: the TC-MRSN maize texture–color dual-branch study at reference 10 and the citrus intensity-invariant texture study at reference 51.
- Corrected the ICCV DOI in reference 33: the verified DOI ends in `00951`, not `00946`.
- Added official proceedings links and page ranges for references 36, 47 and 48; corrected the first-posted year of arXiv:1708.07120 to 2017.
- Confirmed the identity of the 2025–2026 domain-generalization, self-supervised and foundation-model studies. Their results are described as protocol-specific rather than ranked directly against the present 92-image target holdout.

## Reference-by-reference metadata and claim-support table

| No. | Reference | Metadata check | Relevance / limitation | Source |
|---:|---|---|---|---|
| 1 | Mohanty et al. (2016), plant disease detection | Verified | PlantVillage benchmark context. | [Official/source record](https://doi.org/10.3389/fpls.2016.01419) |
| 2 | Hughes & Salathé (2015), PlantVillage repository | Verified; arXiv data-release paper | Dataset provenance, not a peer-reviewed performance comparison. | [Official/source record](https://arxiv.org/abs/1511.08060) |
| 3 | Ferentinos (2018), CNN disease detection | Verified | Publisher metadata matches; broad plant-disease baseline. | [Official/source record](https://doi.org/10.1016/j.compag.2018.01.009) |
| 4 | Too et al. (2019), fine-tuning comparison | Corrected author order | Publisher lists Too, Li Yujian, Njuki and Liu Yingchun; standardized to Li Y / Liu Y. | [Official/source record](https://doi.org/10.1016/j.compag.2018.03.032) |
| 5 | PlantDoc dataset | Verified | ACM title and DOI confirmed; external field-oriented dataset. | [Official/source record](https://doi.org/10.1145/3371158.3371196) |
| 6 | Picon et al. (2019), mobile capture in the wild | Verified | Publisher metadata confirmed; supports acquisition variability. | [Official/source record](https://doi.org/10.1016/j.compag.2018.04.002) |
| 7 | Barbedo (2018), dataset size and variety | Verified | Publisher metadata confirmed; supports dataset-composition concerns. | [Official/source record](https://doi.org/10.1016/j.compag.2018.08.013) |
| 8 | Ramcharan et al. (2017), cassava disease | Verified | Frontiers record confirms authors, year, volume and article number. | [Official/source record](https://doi.org/10.3389/fpls.2017.01852) |
| 9 | Fuentes et al. (2017), tomato diseases and pests | Verified | Publisher metadata confirmed; broad plant-vision context only. | [Official/source record](https://doi.org/10.3390/s17092022) |
| 10 | Wang et al. (2024), TC-MRSN texture–color dual branch | Added; direct texture-branch evidence | Publisher confirms volume 221, article 108915; explicitly combines maize-lesion texture and RGB features. | [Official/source record](https://doi.org/10.1016/j.compag.2024.108915) |
| 11 | Barbedo (2016), challenges in visible-range disease ID | Verified | Publisher confirms volume, pages and DOI; supports background, illumination and symptom ambiguity. | [Official/source record](https://doi.org/10.1016/j.biosystemseng.2016.01.017) |
| 12 | Boulent et al. (2019), CNN review | Verified | Frontiers/PubMed metadata confirmed. | [Official/source record](https://doi.org/10.3389/fpls.2019.00941) |
| 13 | Geirhos et al. (2019), texture versus shape bias | Verified | General-vision evidence; motivates a test but does not prove plant-disease texture behavior. | [Official/source record](https://arxiv.org/abs/1811.12231) |
| 14 | Torralba & Efros (2011), dataset bias | Verified | IEEE/CVPR metadata confirmed; general dataset-bias evidence. | [Official/source record](https://doi.org/10.1109/CVPR.2011.5995347) |
| 15 | Beery et al. (2018), Terra Incognita | Verified | ECCV/CVF record confirmed; animal-camera-trap evidence used only as a general domain-shift analogy. | [Official/source record](https://openaccess.thecvf.com/content_ECCV_2018/html/Beery_Recognition_in_Terra_ECCV_2018_paper.html) |
| 16 | Wu et al. (2023), plant disease UDA | Verified | PubMed/publisher metadata confirmed. | [Official/source record](https://doi.org/10.34133/plantphenomics.0038) |
| 17 | Yang et al. (2024), cross-domain few-shot learning | Corrected first-author initials | Publisher lists Sen Yang; citation standardized to Yang S. | [Official/source record](https://doi.org/10.3389/fpls.2024.1434222) |
| 18 | Tunio et al. (2024), transformer and Wasserstein adaptation | Verified | Publisher title, authors, volume, article number and DOI confirmed. | [Official/source record](https://doi.org/10.1016/j.compag.2024.109574) |
| 19 | PlantWild benchmark paper | Corrected title, DOI and pages | Official title omits “Plant” before “Disease”; ACM MM 2024, pp. 1593–1601, DOI added. | [Official/source record](https://doi.org/10.1145/3664647.3680599) |
| 20 | PlantSeg dataset | Updated to version of record | Replaced 2024 arXiv-only citation with 2026 Scientific Data 13:205, DOI 10.1038/s41597-025-06513-4. | [Official/source record](https://doi.org/10.1038/s41597-025-06513-4) |
| 21 | Zhan et al. (2025), plant disease domain generalization | Added missing author | Publisher lists Yinglong Wang as fourth author; this work supports DG benchmarks, not foundation-model claims. | [Official/source record](https://doi.org/10.1016/j.engappai.2025.111168) |
| 22 | Rezaei et al. (2024), few-shot plant disease recognition | Verified | Publisher confirms title, five authors, volume 219 and article 108812. | [Official/source record](https://doi.org/10.1016/j.compag.2024.108812) |
| 23 | Chai et al. (2024), unseen disease SSL | Expanded author list | Publisher metadata confirms Chai, Lee, Tay, Bonnet and Joly. | [Official/source record](https://doi.org/10.1016/j.neucom.2024.128608) |
| 24 | Huan et al. (2025), unified SSL | Verified | MDPI metadata confirmed. | [Official/source record](https://doi.org/10.3390/electronics14173410) |
| 25 | Yilma et al. (2025), ASCL | Verified | Publisher metadata confirms title, authors, volume 25 and article 103922. | [Official/source record](https://doi.org/10.1016/j.rineng.2025.103922) |
| 26 | Espejo-Garcia et al. (2025), agricultural foundation models | Verified | Publisher metadata confirmed; includes DINOv2, LoRA and distillation. | [Official/source record](https://doi.org/10.1016/j.compag.2025.110900) |
| 27 | PlantCLR (2026) | Standardized author initials | Nature/Springer record confirms authors, date, volume 16, article 10550 and DOI. | [Official/source record](https://doi.org/10.1038/s41598-026-45684-x) |
| 28 | Nguyen et al. (2026), frozen foundation features | Verified; online record | Publisher confirms article 102531. Its 27-class PlantDoc evaluation is not directly comparable with this paper's 10-class/92-image holdout. | [Official/source record](https://doi.org/10.1016/j.atech.2026.102531) |
| 29 | Xiang et al. (2026), cross-domain reliability gap | Verified; closest comparator | Frontiers/PubMed confirms metadata. Directly evaluates PlantVillage→PlantDoc, calibration, AdaBN and frozen backbones. | [Official/source record](https://doi.org/10.3389/fpls.2026.1826962) |
| 30 | El Karch et al. (2026), frozen multi-backbone fusion | Verified; direct contemporary comparator | Frontiers/PubMed confirms metadata. PlantWild/PlantDoc scores use different protocols and label spaces. | [Official/source record](https://doi.org/10.3389/fpls.2026.1860665) |
| 31 | He et al. (2016), ResNet | Verified | CVF/IEEE confirms title, authors, CVPR pages and DOI. | [Official/source record](https://doi.org/10.1109/CVPR.2016.90) |
| 32 | Dosovitskiy et al. (2021), Vision Transformer | Verified; arXiv access record | Title and authors match the ICLR paper. | [Official/source record](https://arxiv.org/abs/2010.11929) |
| 33 | Caron et al. (2021), DINO | Corrected DOI | Official IEEE/ICCV DOI ends in 00951, not 00946; pages 9650–9660 confirmed. | [Official/source record](https://doi.org/10.1109/ICCV48922.2021.00951) |
| 34 | Oquab et al. (2024), DINOv2 | Verified; TMLR publication, arXiv full text | Publication metadata and accessible preprint identity match. | [Official/source record](https://arxiv.org/abs/2304.07193) |
| 35 | Chen et al. (2020), SimCLR | Verified | PMLR confirms ICML volume 119, pages 1597–1607. | [Official/source record](https://proceedings.mlr.press/v119/chen20j.html) |
| 36 | Grill et al. (2020), BYOL | Official proceedings link added | NeurIPS metadata confirms volume 33 and pages 21271–21284. | [Official/source record](https://proceedings.neurips.cc/paper_files/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html) |
| 37 | He et al. (2020), MoCo | Verified | CVF/IEEE confirms CVPR 2020 pages and DOI. | [Official/source record](https://doi.org/10.1109/CVPR42600.2020.00975) |
| 38 | Gulrajani & Lopez-Paz (2021), DomainBed | Verified; arXiv access record | ICLR 2021 title and authors confirmed; supports fair DG protocol discussion. | [Official/source record](https://arxiv.org/abs/2007.01434) |
| 39 | Koh et al. (2021), WILDS | Verified | PMLR confirms volume 139 and pages 5637–5664. | [Official/source record](https://proceedings.mlr.press/v139/koh21a.html) |
| 40 | Zhou et al. (2023), domain-generalization survey | Verified | IEEE/PubMed confirms authors, volume 45, pages 4396–4415 and DOI. | [Official/source record](https://doi.org/10.1109/TPAMI.2022.3195549) |
| 41 | Ganin et al. (2016), DANN | Verified | JMLR confirms volume, article number and pages. | [Official/source record](https://jmlr.org/papers/v17/15-239.html) |
| 42 | Sun & Saenko (2016), Deep CORAL | Verified | Proceedings/DOI metadata confirms ECCV Workshops pages 443–450. | [Official/source record](https://doi.org/10.1007/978-3-319-49409-8_35) |
| 43 | Li et al. (2016), AdaBN | Verified; arXiv method paper | arXiv metadata confirmed; supports unlabeled target-statistics adaptation. | [Official/source record](https://arxiv.org/abs/1603.04779) |
| 44 | Hendrycks & Dietterich (2019), common corruptions | Verified; ICLR 2019 | Title and authors confirmed; motivates synthetic stress tests only. | [Official/source record](https://arxiv.org/abs/1903.12261) |
| 45 | Schneider et al. (2020), covariate-shift adaptation | Verified | NeurIPS proceedings metadata confirmed. | [Official/source record](https://proceedings.neurips.cc/paper_files/paper/2020/hash/85690f81aadc1749175c187784afc9ee-Abstract.html) |
| 46 | Guo et al. (2017), calibration | Verified | PMLR confirms ICML volume 70 and pages 1321–1330. | [Official/source record](https://proceedings.mlr.press/v70/guo17a.html) |
| 47 | Ovadia et al. (2019), uncertainty under shift | Added page range and official link | NeurIPS record confirms pages 13991–14002. | [Official/source record](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html) |
| 48 | Lakshminarayanan et al. (2017), deep ensembles | Added page range and official link | NeurIPS record confirms pages 6402–6413. | [Official/source record](https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) |
| 49 | Loshchilov & Hutter (2019), AdamW | Verified; arXiv access record | ICLR paper title and authors match. | [Official/source record](https://arxiv.org/abs/1711.05101) |
| 50 | Smith & Topin, Super-Convergence | Corrected preprint year | arXiv:1708.07120 was first posted in 2017; it motivates a learning-rate factor but does not determine the best plant-disease recipe. | [Official/source record](https://arxiv.org/abs/1708.07120) |
| 51 | Gómez-Flores et al. (2019), intensity-invariant texture analysis | Added; direct texture evidence | Publisher confirms authors, volume 162, pages 825–835 and DOI; also demonstrates illumination sensitivity of texture descriptors. | [Official/source record](https://doi.org/10.1016/j.compag.2019.05.032) |

## Claim-support review

### Texture branch

Plant-specific texture plausibility is directly supported by reference 10 (a texture–color dual-branch maize model) and reference 51 (intensity-invariant texture analysis for citrus Huanglongbing). References 13–15 provide broader vision evidence about texture/shape preference, dataset bias and unfamiliar environments.

These sources do **not** establish that MTD-Net's high-resolution branch represents texture exclusively, disentangles texture from shape/color, or identifies a disease-specific biological mechanism. The present ablation changes model capacity, and the experiments do not directly intervene on texture while preserving other visual properties. The manuscript therefore describes a *texture-oriented configuration* and reports configuration effects rather than claiming semantic disentanglement.

### Training-strategy effects

References 49–50 provide background on AdamW and large-learning-rate schedules. They do not prove that twelve epochs at peak LR 2e-3 is the best recipe for this dataset. The evidence for the observed recipe effect is the corrected 65-configuration experiment, including paired seeds and the epoch-by-learning-rate factorial. The manuscript reports the observed effect from its own results and does not present the tested schedule as a universal optimum.

### Cross-domain generalization and foundation-model comparisons

Reference 29 is the closest contemporary comparator: it already evaluates PlantVillage-to-PlantDoc transfer, calibration, selective prediction, target-aware adaptation and frozen representations under parent-image-aware partitions. Reference 28 reports frozen DINOv2 feature evaluation on PlantDoc, and reference 30 studies frozen multi-backbone fusion on PlantWild and PlantDoc. The manuscript must not claim novelty for the existence of the laboratory-to-field gap, frozen foundation features, calibration under shift or multi-backbone fusion.

The present study differs in its specific question and protocol: paired analysis of MTD-Net architecture and optimization choices across 65 fixed configurations, using a 15-class source subset and a 10-class mapped target holdout of 92 images. The label space, target sample, split design, model families and adaptation conditions differ across papers, so the reported accuracies are not directly comparable. The small target holdout limits precision and breadth; claims are restricted to the selected mapped subset.

### General-vision sources versus plant-specific evidence

References 13–15 and 31–50 are methodological or general-vision sources. They are appropriate for explaining texture bias, dataset bias, residual/Transformer architectures, self-supervision, domain adaptation, corruption testing, calibration and optimization. They should not be presented as direct evidence that MTD-Net improves plant disease recognition. Plant-specific empirical support comes from references 1–12 and 16–30, with references 10 and 51 specifically grounding the texture-cue discussion.

### Editorial checks

- All 51 numbered references appear in the manuscript's in-text citation set.
- Metadata changes are based on publisher, official proceedings, PubMed or original preprint records.
- Recent 2025–2026 papers are included as contemporary context, with publication type and experimental protocol kept distinct.
- This audit checks identity, metadata and claim fit; it does not substitute literature results for the study's metrics or constitute a journal-specific formatting proof.
