# C1-E33 protocol v1 — CURRENT

## Question, hypothesis and falsification

Question: do the manuscript's processing/noise conclusions remain informative
when classification uses learned local temporal patterns instead of compact
summary descriptors? Hypothesis: representation choice may change clean scores,
corruption sensitivity and the clean-trained/matched gap. There is no assumed
direction and no target score. Strong contrary or negative results must be kept.
One baseline cannot establish superiority over every deep-model family.

This experiment is a reviewer-driven empirical extension, not a new architecture
contribution, formal preregistration or external/field validation.

## Immutable cohort and information set

Exactly the current E32 1,839 windows / 39 acquisition families. One simulation
per window; 156 dedicated healthy windows and 1,683 fault windows, eleven fault
labels plus healthy, with the existing class order. Six inputs: three B7--B5
currents and three phase-to-ground voltages. The half-open interval [50,250) ms
contains 8,000 samples/channel at 40 kHz. No prefault/clearing samples, location,
load factors, family IDs or simulation IDs enter the neural predictor.

Use the identical five family folds, verified both against metadata file hashes
and the original PCG64 seed20261026 rule. Validate class coverage, numeric/raw
identity uniqueness, exact-load grouping and all nested family separations.
Direct-identity checks do not prove independence of similar simulated templates.

## Architecture fixed before scientific execution

Wang, Yan and Oates's published temporal residual baseline motivates three
residual blocks, three temporal convolutions per block, shortcut connections,
global average pooling and a softmax classifier. This is an **explicitly adapted**
multichannel, dilated ResNet, not a byte-faithful implementation of that paper.
Widths are 32/64/128; odd kernels 9/5/3; block dilations 1/8/64. Each convolution
is followed by batch normalization; ReLU follows the first two convolutions and
the residual sum. Projection shortcuts have 1×1 convolution plus normalization.
There are nine main-path convolutions and three projection convolutions.

The model has 279,692 trainable parameters. Effective main-path receptive field
before pooling is 1 + (8+4+2)×(1+8+64) = 1,023 samples = 25.575 ms, about 1.53
nominal60-Hz cycles. Global pooling then aggregates the complete 200-ms window.
No resampling, striding, local pooling, spectrogram or hand-engineered descriptor
is applied. Full recorded bandwidth is retained at input; this is not proof that
the network uses all bandwidth or captures arbitrary long-range temporal order.

The dilations address physical-scale mismatch at40kHz without silently removing
high-frequency corruption. No kernel/width/dilation search uses outer outcomes.

Reference: [Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline](https://arxiv.org/html/1611.06455v4).
InceptionTime is a credible multi-scale alternative, not an experimentally rejected
competitor: [original paper](https://arxiv.org/abs/1909.04939).

## Training and nested stopping

For each outer fold, choose ceil(20% of available training families) as internal
validation families using PCG64(seed20261029 + outer_fold). This split is the
same for all model seeds and corruption conditions. Both inner and outer splits
must contain all12 classes. No window-level random split is allowed.

Fit a single mean and population standard deviation per channel over the inner
training windows and their time samples. No individual-window centering, RMS
division or amplitude normalization. Train with unweighted cross entropy to
match the unweighted E32 fitting population; scoring weights do not become
training weights. AdamW lr0.001, weight decay0.0001, batch16, gradient-norm clip5,
at most150epochs, patience20. Select the earliest epoch improving weighted inner
Macro-F1 by more than0.0001. No learning-rate or architecture search. Train once
for stopping, then reinitialize and refit on **all outer-training families** for
the selected number of epochs. Refit channel normalization on those training
families only. This avoids discarding validation families relative to E32 fits.

Three reported initialization seeds42/43/44. Phase-specific initialization/order
seeds are derived from seed, outer fold and phase, not the corruption condition.
The same selected configuration is used in every evaluated condition. All seeds
are reported separately; averaging seed probabilities is not the main estimator.
BatchNorm running statistics are never updated during validation/test inference.

The stopping validation and three-seed design give this comparator a different
training procedure from the fixed tabular models. Describe it accurately; do
not call the comparison equal hyperparameter-search budgets or optimal rankings.

## Corruption and exposure

Port the exact E32 repeat0 noise identities: historical E25 AWGN and E27 physical
rate adapters, with window ID suffix `__C1E32_repeat0`. Float64 corruption occurs
before a documented float32 NCT cache conversion. The neural model uses float32;
descriptor calculations historically used float64. Neither is described as a
different random realization. Regression hashes cover all26 clean/noise cases;
preflight compares three real windows'18 core descriptors in every condition
against the hash-pinned saved E32 arrays. No new noise augmentation is introduced.

`paper`: clean plus five mechanisms at10/20/40/60/120dB; `pilot`: clean, AWGN20,
mixed10; `clean`: clean only. The trained clean model is reused on every evaluated
condition for clean-trained deployment. A separate model trained on each noisy
condition is evaluated on its matching condition. The clean result is stored
once. No denoiser, learned fusion or external dataset is added in E33v1.

## Scoring, reporting and inference boundary

Argmax of class-aligned probabilities, no threshold tuning. Pool the five held-
out folds first, then compute the E32 family/class-standardized confusion matrix
with each family/class cell total weight one. Macro-F1 is primary; BA, MCC, class
recalls, healthy false-alarm/missed-fault rates, Brier and log loss are retained.
Do not substitute mean fold F1. Raw probabilities and identities permit subsequent
paired family analysis. Per-initialization SD is not a confidence interval.

Descriptive comparisons cover all24 fixed E32 unfiltered core pipelines. Their
convergence-inference exclusions remain visible. No automatic winning-pipeline
selection or new significance statement is produced. Any added bootstrap/test
family must be declared before computing it; it does not silently enter the
original728-contrast E32 family. No epoch or architecture is chosen from test data.

## Reproducibility, failures and cost

Source/config Git must be committed and clean. Every run pins source hashes,
Git identity, code tree, package versions, device and numerical options. Strict
deterministic algorithms, TF32/AMP disabled, DataLoader workers0. Determinism is
not a guarantee of bitwise equality across different software or hardware.

Checkpoint atomically after each epoch: model, optimizer, full loss history,
stopping state, Python/Torch/CUDA RNG and shuffle-generator state. An interrupted
epoch is repeated from the last completed checkpoint. No automatic numerical
retry, batch shrink, CPU/GPU fallback or exception suppression. Resume requires
the same run identity; finished jobs are hash-verified and not retrained.

Budgets: clean15jobs/30fits; pilot45jobs/90fits; paper390jobs/780fits. Model-side
elapsed prediction includes batching/scaling/transfer and has no warmup exclusion;
it is not a relay latency or an optimized deployment benchmark. Training/refit
time and selected epochs are saved separately. Production CPU use requires
explicit acknowledgement. Sleeping/hibernating/shutting down stops progress;
no system power settings are altered by this project.
