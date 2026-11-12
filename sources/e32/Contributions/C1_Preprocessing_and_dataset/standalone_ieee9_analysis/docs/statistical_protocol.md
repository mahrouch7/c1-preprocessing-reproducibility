# Prospective statistical protocol for the new load-family cohort

Additional prefit direct-reuse audit: acquisition/window IDs must be unique;
saved raw and numeric-window hashes cannot repeat; load factor triples rounded
to 12 decimals cannot cross family boundaries. A collision stops preparation
before any fit and requires scientific regrouping/reconciliation, never a guard
bypass. Exact clean-descriptor collisions are separately tabulated; equality of
a lossy representation is not necessarily identical raw data. Synthetic tests
verify that extreme held-out values cannot influence a fitted training scaler.

Experiment C1-E32; decision C1-D87. Frozen before any new-cohort fit or score
inspection. This is a prospective implementation specification, not a registered
preregistration. Historical experiments have informed method development.

## Question, hypotheses and reversal criteria

Question: how do preprocessing and representation-fusion choices change fault
classification and synthetic-noise robustness under a fixed, family-held-out
evaluation? The candidate empirical contribution is the controlled interaction
and its limits, not inventing denoising, fusion, grouped validation or a larger
grid. Literature overlap must still inform the final novelty assessment.

Primary paired endpoints are signed Macro-F1 differences: early concatenation
and fixed simple late fusion versus EACH RMS/mean-std constituent; denoisers
versus no denoising, using deployment evaluation at severe 10-dB corruption.
Claims of benefit require the direction/magnitude, conditional interval,
multiplicity bookkeeping, numerical eligibility and support controls to agree.
Harm, discordance, small effects or inconclusive comparisons weaken the claim.
Inconclusive is not equivalent. No universal fusion/denoiser superiority is
assumed, no winning test cell is used to prescribe a deployed method, and no
oracle knowledge may be called deployable.

## Cohort and observational unit

1,839 useful windows come from 39 designed load families: 156 dedicated healthy
cases, 153 per fault class, 12 labels total. Common anchors share load support
between healthy and faults; three additional healthy-only points per family
increase healthy diversity but introduce an explicit support confound. Nominal
fault sampling is spatially denser (39 settings per class versus three in each
other family). The load families are designed synthetic groups, not random
independent networks, and neighboring load points can remain highly similar.

The six measured channels are currents/voltages at the fixed observation point.
No distance, fault position/line ID, family ID or acquisition metadata enters
predictors. The 50-ms common settled prefix is excluded. One fixed half-open
200-ms useful window (8,000 samples at 40 kHz) is retained; initial detection
does not require clearing. Dataset re-extraction checks saved identities,
finite shapes, timestamps, channels, useful-window hash and feature parity.
Receipt-derived electrical screens are reported as such, not independently
rerun physical-model qualification or proof of field validity.

## Split, train/test isolation and estimands

Sort 39 family identities, shuffle with PCG64 seed 20261026 and assign round
robin to five folds (8/8/8/8/7 families). Every family member, both healthy
enrichment and all derived noisy/filter variants, follows its family. Each
train/test view must contain all 12 labels. Splitting occurs before learned
scaling; scalers are fitted only on each training partition. Hyperparameters,
conditions, features, noise streams and test populations are fixed, not tuned
using OOF results. SVM probability calibration is training-only internal work.

Primary evaluation uses weight 1/n(family,class) on each test window, so each
of 39 x 12 cells contributes total weight one. This is a designed-family/class
estimand, NOT an estimate of field class prevalence. Training is ordinary
empirical training, not this evaluation reweighting. Secondary outputs include
unweighted pooled metrics and a common-anchor healthy evaluation view. The
extra healthy false-alarm rate is reported separately. Global standardization
for descriptive PCA/correlation is NOT used by trained models.

Two training sensitivities keep the same held-out test cases: remove only the
extra healthy training points; reduce nominal training-fault density to three
distinct lines selected deterministically at positions nearest 0.5 (first,
middle and last line identities). Their metadata is retained. These controls
measure sensitivity but cannot fully remove all designed-support confounding
or disentangle every load-by-spatial effect in a reduced nonfactorial dataset.

## Exposure, representations and fairness

Clean-trained deployment evaluation tests clean and each corrupted condition;
matched evaluation trains/tests on the same specified corruption distribution
with family disjointness. Clean matched and clean deployment can share the
same fit/predictions; they are not independent evidence. Fresh noisy training
fits do not constitute a deployment strategy that knows unknown test SNR.

Noise uses common deterministic per-case/channel streams across all filters
and representations. AWGN, low-frequency, impulses, gain+AWGN and mixed
gain/additive are evaluated at 10/20/40/60/120 dB additive SNR; clean is separate,
not a finite-SNR value. Total error SNR, mean/RMS/peak error and measured additive
SNR are recorded. Multiplicative gain can dominate total error at high additive
SNR. The bounded repeat slice uses two extra streams at AWGN20/mixed10; these
are not additional independent acquisitions or full training-seed replication.

RMS6 and RAW12 (six mean/std pairs, not temporal waveforms) concatenate into
MULTI18. Same measurements/time windows/folds are used for constituents, early
fusion and 50/50 aligned-probability late fusion. All main label decisions use
argmax of aligned probabilities, including SVM; native decisions are retained.
An injected-SNR-dependent 0.7/0.3 RMS weighting at <=20 dB, else 0.3/0.7, is an
ORACLE ablation only. Representation fusion is not independent sensor acquisition.
Separate I/V controls use current-only, voltage-only and combined channels.

All eight learners use fixed source-pinned hyperparameters. Six scaler choices
are evaluated on a declared bounded subgrid, not all noisy/preprocessed cells.
Skewness alone and the larger skewness/kurtosis/crest-factor bundle are distinct
controls. The descriptor extension uses four learners/seven conditions; I/V,
support, nominal-density and noise-repeat studies use two learners/none+median
on bounded conditions. See README fit table and scope.json, not an implied
unrestricted full Cartesian product. No cross-paper numerical ranking follows.

## Scores, uncertainty and multiplicity

Macro-F1 is primary. Balanced accuracy and multiclass MCC are secondary;
accuracy, weighted F1, per-class precision/recall/F1, healthy false alarms,
missed faults, Brier and log loss (epsilon 1e-15) are descriptive complements.
Pooled OOF predictions must contain each window exactly once across all five
outer folds. Class support, numeric matrices and selected calibration bins are
retained. Clean-minus-corrupted changes remain signed, including improvements.

Predeclare 728 paired contrasts in comparison_plan.json. They cover both
constituent comparisons, denoiser effects, clean scaler contrasts, extra
descriptors, I/V and training-support/density sensitivities. 10,000 resamples
of entire family confusion contributions give percentile 95% intervals.
50,000 whole-family swaps of paired predictions give a two-sided Monte Carlo
randomization statistic with plus-one correction. Stable contrast-derived RNG
seeds are recorded. The code also reports 99% Monte Carlo precision bounds and
flags Holm decisions whose bounds straddle 0.05. These are marginal precision
diagnostics, not simultaneous global certainty about all decisions.

Holm adjustment is separate within the predeclared contrast families. There is
no across-all-endpoints familywise claim. Metric intervals are pointwise and
conditional on the particular trained outer models and designed cohort. They
do not retrain in the bootstrap, estimate full training-procedure/seed
uncertainty, or prove external-population generalization. Shared overlapping
training folds and purposive, similar load families limit exchangeability and
independence assumptions. Treat p-values as model-conditional comparative
evidence, not definitive causal or field-population significance. Report effect
sizes and controls even if p-values look favorable; collapse/narrow claims if
support sensitivity, numerical instability or family dependence undermines them.

## Exclusions, signal diagnostics and publication boundary

Convergence warnings are retained, never automatically refitted. A warned fit
excludes its pooled five-fold case and dependent fusions from inference. The
excluded planned comparison remains in Holm bookkeeping with p=1 and explicit
EXCLUDED status, not as a successfully tested null. Unknown errors/integrity
failures stop the campaign without retry. All attempts, warnings and exclusions
remain in provenance; do not present attempted fits as all converged.

Primary TPI uses DWT D3 at nominal 2.5-5 kHz. DWT D1 and FFT band diagnostics,
clean/noisy/error energies, near-zero flags and ratios above one are retained.
Energy retention is not preserved diagnostic information. TPI/F1 associations
are exploratory condition-dependent descriptions with no independent p-value
or causal denoiser-selection claim. Offline filters are not demonstrated causal
real-time protection and a 200-ms window is not validated relay latency.

Automated outputs include curves/radars/confusions, paired effect atlases,
scaler matrices, dataset/diversity/support descriptions and SVG plus vector PDF
assets. Selected class/probability/error diagnostics are tabulated; not every
planned checklist interpretation is automatically written or verified. Final
old/new claim mapping, captions, original table/figure replacement, reviewer
responses and physical limitations require joint manuscript review. The
conditional temporal DL/learned fusion/granularity/external-data items are not
silently executed here. Preserve fourteen historical E16 voltage failures,
ABC/ABCG ambiguity, ideal-source/constant-Z/zero-sequence limitations and the
historical-only 93.47% grouped-interpolation pilot caveat.

## Numerical implementation amendment, 26 October 2026

The GaussianNB posterior arithmetic is amended after a retained-model numerical
failure, not chosen by test classification scores. Use shifted softmax on the
same finite joint log likelihoods for every remaining GaussianNB prediction.
Keep parameters, training rows, train-only transforms, fixed folds, corruptions,
comparisons and the original strict probability guard unchanged. Other learners
retain the frozen native calculation. This prevents cancellation in the final
log-normalizer subtraction; it does not guarantee accurate likelihoods under
extreme extrapolation, calibration or improved classification. Preserve poor
results and report the numerical amendment and its limits transparently.

Before any recovery fit, the existing PC2 saved model must pass the stable
calculation on clean, AWGN20 and mixed10 with no aligned-argmax changes. Keep
the original failure evidence and original probabilities as negative evidence.
The four complete same-campaign returns are reused with their genuine source
assignments and original exclusions. Ninety-seven unreceipted partial NB models
are preserved but not silently reused; externally acknowledged repetition is
accounted separately from the unique 18,930-fit analysis grid. The recovery
adds 18,286 new fits and 97 duplicate attempts cumulatively, not new comparisons.
