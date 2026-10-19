# Six task completion record

The fixed study completed 65/65 training-and-evaluation pairs. All 67 jobs in workflow run 39265931897 succeeded. The training revision is `bd7cd1d94ff2591fa58dac5eb69f0e66225b8f72`; later commits assemble and audit those observations.

| Task | Completed evidence | Remaining scientific boundary |
|---|---|---|
| Data provenance and splits | Five pinned Parquet SHA-256 checks; all 14,238 source image bytes matched original Git blobs; leaf/identity/similarity screens; frozen source and target splits | Missing biological identifiers prevent complete independence certification |
| Core comparisons | MTD-Net, matched CNN, MTD-P and pretrained ResNet-18; five seeds each; checkpoint and held-out output hashes verified | Small seed count and one source split |
| Training strategy | All 40 model × epoch × LR × seed cells; cell estimates and paired difference-of-differences | Only two durations and two LRs under OneCycle |
| Component ablations | Gate removal, no auxiliary task, branch removal; five seeds each and paired intervals | Branch removal changes capacity; no causal texture proof |
| Domain evaluation | All 65 zero-shot and eight-condition corruption evaluations; both pretrained models have AdaBN and supervised FT; source-only temperature fit; target ensemble uncertainty | Only 92 mapped official target test images |
| Manuscript and figures | Complete numerical manuscript, five reproducible figures with plotting CSVs, editable document export and all public raw prediction arrays | Author declarations and venue preparation remain |

The complete evidence audit passes: one source-input fingerprint set, one target-test split fingerprint set and one training source revision, with no output hash or protocol disagreements across 65 pairs. Public array metric regeneration also passes for all 65 runs. Local CPU repeats are separate diagnostics and are not extra study seeds.

The cleaned source split contains 2,922 optimizer-training images, 333 validation images and 2,916 official test images. Filtering removes 5,820 training rows in known leaf/identity components overlapping test and 12 further similarity candidates before per-class capping. Class support is uneven; one retained source-cache class has only 16 images. Biological leaf IDs remain missing for 1,604 retained source-training-cache images and 857 source-test images. Image-identity fallback and similarity screening cannot certify full biological or acquisition independence.

The mapped target pool is 946 images after filtering, with 683 target-training, 170 validation and 92 official test images; one duplicate train row is unused. Results cover the selected mapped subset, not the whole PlantDoc population. Five seeds quantify training variability conditional on these splits. The supplementary ensemble Wilson image intervals do not capture acquisition dependence or dataset-selection uncertainty. Removing a branch changes parameter count and does not isolate a texture mechanism. Historical and corrected scores use different training populations and cannot isolate the impact of the old checkpoint-selection problem.
