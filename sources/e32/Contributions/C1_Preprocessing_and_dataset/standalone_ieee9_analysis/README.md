# New-cohort preprocessing and representation-fusion evaluation

Direct-leakage guards stop before training if an acquisition/window is reused,
saved raw/numeric waveform hashes repeat, or a load factor triple (rounded to
12 decimals) occurs in more than one family. These guards supplement, not
replace, family-disjoint splits and training-only scaler fitting. They do NOT
certify independence of similar load regimes. Exact descriptor collisions are
reported separately as possible projection/identifiability issues, not silently
removed or mislabeled as proof that raw data were reused.

Status: original campaign externally executed, PARTIAL AND STOPPED on GaussianNB
probability validation. Numerical correction/recovery PREPARED, NOT EXECUTED.
Experiment C1-E32, original scope decision C1-D87. The researcher executes the
recovery command externally; publication does not start training or edit a manuscript.

## Numerical correction and scoped recovery, 26 October 2026

Retained PC2 job97 (RAW12, clean-trained Yeo-Johnson/GaussianNB, none, fold1)
passes on clean measurements but native probability row sums exceed 1e-10 on
4/296 AWGN20 and 34/296 mixed10 held-out windows. Values remain finite and in
[0,1]. The pinned library computes exp(joint - logsumexp(joint)); subtracting
two very large log quantities can lose the small normalizing correction.
Compute the same Gaussian posterior using shifted softmax directly on finite
joint log likelihoods instead. This is not clipping/repairing raw probabilities,
changing the 1e-10 guard, regularizing variances, tuning a scaler or retraining.
It cannot recover precision already lost inside joint likelihood calculation,
prove calibration, or remove Yeo-Johnson out-of-distribution sensitivity.

Recovery first verifies the stable formula on this exact retained PC2 model,
requiring original strict validation and unchanged aligned argmax decisions
on all three views. Failure stops before any fit. Other classifiers retain the
same frozen native prediction path. A distinct zero-fit numerical task uses the
existing worker and transport; no infrastructure qualification or installation.

Keep all ten original preprocessing returns and four full learning returns
(644 fits, including eight inference-ineligible logistic convergence warnings).
Only 201 missing learning units are rebuilt: PC1 9,267 and PC2 9,019 new fits,
18,286 total. All original outputs remain immutable. The failed NB unit has
97 fitted models but no complete fit_records/receipt. Do not pretend these are
verified reusable records. Preserve them, explicitly authorize one repetition
of that unit, and account for 19,027 cumulative attempts for 18,930 unique final
fits. No retry loop exists. Each retained return keeps its original assignment;
the final collector validates both assignments rather than rebinding markers.

Externally execute the prepared Resume-C1E32DatasetAnalysis-PC1.ps1 with both
ConfirmRecovery18286Fits and Acknowledge97PriorPartialFits. New outputs use a
fresh sibling recovery_numeric_v1 folder. No whole-campaign rerun is permitted.

## Scope and input

The question is: **How do preprocessing and representation-fusion choices affect
the performance and noise robustness of transmission-line fault classification?**
The frozen input is the 1,839-window candidate view: 156 dedicated healthy cases
and 153 cases of each of eleven fault types, grouped into 39 load families.
View SHA256: `61c43740c1c589634b0922070a6cfab2e559c09915e5fb2e4a766c39ac247aee`.
The analysis rehashes/readbacks saved raw arrays and re-extracts the same six
measurement channels over the half-open 200-ms useful interval. The common
50-ms settled prefix is excluded; no clearing signal or true fault distance
is an input. This is analysis of existing simulations, not new native simulation.

Five family-disjoint outer folds are fixed before scores. A family, including
its healthy-only enrichment and every derived corruption, belongs to one fold.
Folds contain 8/8/8/8/7 families, not equal numbers of windows. Primary scores
give each family-by-class cell equal total weight; pooled scores and the common
healthy/fault anchor-support view are secondary. See docs/statistical_protocol.md.

## Exact computational ceiling

| New fits | Scope |
|---:|---|
| 15,600 | 26 conditions x 5 preprocessing settings x 3 base representations x 8 learners x 5 folds |
| 1,800 | Five non-default train-only scalers, 3 representations, 8 learners, clean/AWGN20/mixed10, 5 folds |
| 560 | Skewness-only and larger descriptor bundles, 4 representations, 4 learners, 7 conditions, 5 folds |
| 360 | Current-only/voltage-only controls, 6 representations, 2 learners, 3 conditions, none/median, 5 folds |
| 180 | Healthy anchor-only training support control |
| 180 | Nominal spatial-density training control |
| 240 | Two extra corruption realizations on AWGN20 and mixed10 |
| 10 | Training-prior and majority dummy baselines |
| **18,930** | **No reuse of historical fitted models; PC1 9,466 / PC2 9,464** |

Eight learners are logistic regression, random forest, naive Bayes, k-nearest
neighbors, decision tree, support vector machine, gradient boosting and a
descriptor MLP. The MLP is NOT a temporal deep baseline. Hyperparameters are
fixed in the pinned upstream specification; this is not model optimization.

Conditions are separate clean plus AWGN, low-frequency, impulsive, gain+AWGN
and mixed gain/additive mechanisms at 10/20/40/60/120 dB additive SNR. Five
settings are none, median, Savitzky-Golay and the two frozen wavelet variants.
For gain mechanisms, additive SNR is not total error SNR; both are recorded.
Filters are offline window operators, not a proven causal relay implementation.

The main representations are RMS6, RAW12 (mean/std descriptors, NOT raw temporal
samples) and their early concatenation MULTI18. Fixed 50/50 probability fusion
uses the two constituent models. A separate true-injected-SNR weighting rule
is explicitly an ORACLE diagnostic, not an operational or learned fusion result.
All main decisions use aligned-probability argmax, including SVM; native
predictions and disagreements are saved separately for transparency.

## One external launch on PC1

Keep the existing qualified PC2 worker open. Do not reinstall its kit or change
SSH/Prefect. PC2 tasks use the dedicated H: workspace. PC1 uses the existing
scientific environment. PDF conversion uses the already available document
runtime on PC1 only; nothing is installed into either scientific environment.

From the canonical repository location, execute this single physical line:

```powershell
& 'F:\ResearchRepositories\ieee9bus_research_canonical\Contributions\C1_Preprocessing_and_dataset\standalone_ieee9_analysis\scripts\Run-C1E32DatasetAnalysis-PC1.ps1' -Stage all -ConfirmNewCohortAnalysis
```

The same wrapper accepts `-Stage check` without the confirmation switch for a
non-mutating source/input/fold/runtime check. `-Stage prepare` re-extracts data
and prepares capsules without fitting or submitting. `all` requires a fresh
output directory and is NOT an automatic resume of `prepare` or a failed run.
Do not repeat `all` to recover a failure. Preserve logs and reconcile the exact
state first; no evidence deletion, overwrite or automatic scientific retry.

Default external locations, configurable wrapper parameters:

- Dataset: `G:\IEEE9_Datasets\C1_E31\reduced_v1`.
- PC1 output: `F:\Backup\C1_E32\campaign_v1` (not the more constrained G: drive).
- PC2 workspace: `H:\r07node_e32\workspace`.
- Existing task/result transport: `F:\Backup\R07_Dispatch`.

At least 80 GiB free is required on PC1 before preparation; the later stage
requires 35 GiB. PC2 H: requires 40 GiB headroom and its control drive C: 3 GiB.
The guards stop rather than delete old data. They are headroom checks, not
guarantees against other applications filling a disk.

The automatic sequence is raw readback -> 10 shared preprocessing chunks ->
verified two-host return barrier -> 205 distinct learning units -> verified
two-host return barrier -> scoring/inference -> SVG/PDF export. No lot-by-lot
researcher approval is requested. Existing infrastructure performs task delivery
and result return. Progress records show counts and provisional execution ETA
for both PCs; inference/render time is excluded from that provisional ETA.
No measured speedup is claimed without a comparable serial baseline.

## Outputs and Overleaf

`dataset_analysis/` holds acquisition-support, healthy diversity and fold
descriptions. `verified_diagnostics/` retains signal/noise/TPI tensors.
`final_analysis/` contains complete OOF predictions, scores and all numerical
confusion matrices, per-class errors, signed robustness drops, support/density
controls, 728 paired contrasts with conditional intervals/Holm bookkeeping,
convergence exclusions, calibration slices and compute/model-size summaries.

`final_analysis/figures/` contains SVGs and an HTML index; `figures/pdf/` has a
vector PDF for EVERY SVG. The embedded Arial font and conversion versions/hashes
are recorded in `pdf_export_receipt.json` and `pdf_manifest.csv`. Conversion
rejects unsupported shapes, clipped text and undersized publication fonts.
The `publication_` files are seven-inch two-column candidates with fonts at
least 8.4 pt at that width. Use `figure*`/`width=\textwidth`, or redesign if a
single-column layout is desired; shrinking to one column halves the font size.
Large review atlases are not insertion-ready manuscript panels. The generated
`overleaf_include_examples.tex` lists commented inclusion examples. Final visual
review, panel selection, concise captions and scientific wording remain joint
publication work; successful conversion is not full figure acceptance.

The 82-item checklist is carried forward conservatively. Computational
completion does not automatically validate all rows, alter the original
manuscript, answer all 27 reviewer items or close the paper. Temporal DL,
learned/adaptive fusion, a fair sequence/descriptor granularity comparison and
external/dense acquisition remain conditional, not hidden additions to this run.

## Verification and provenance

`provenance/release.json` pins this source set, reused scientific/infrastructure
sources and the prospective scope. Exact task archives, configurations, family
identities, runtime versions, model hashes and return barriers are preserved.
Tests are synthetic/stubbed and include a full mocked learning unit, raw signal
operators, budget/fold/weight guards, missing/corrupt return rejection and PDF
layout QA. They do not establish scientific accuracy, physical validity or a
measured runtime for the real cohort.

Unknown fit/integrity exceptions stop execution. Convergence warnings remain
visible; affected pooled results and dependent fusions are excluded from formal
comparisons without refitting. Preserve E16's fourteen voltage failures,
ABC/ABCG ambiguity, static ideal-source/constant-impedance and zero-sequence
limits. Different hashes are not independent networks; balanced counts do not
remove healthy-only load-support confounding. The historical 93.47% grouped
interpolation pilot is not a new-cohort result or external-generalization claim.
