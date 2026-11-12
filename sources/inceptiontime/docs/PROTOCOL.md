# C1-E43 — CURRENT protocol frozen before held-out evaluation

Question: do observations from descriptor models and ResNet persist with a
multiscale waveform model? This is an empirical comparator, not a new algorithm.
Author authorizes preparation/versioning and private Kaggle launch on 7 November
2026. No electrical simulation, external validation or new significance test.

## Architecture

Independent PyTorch implementation inspired by InceptionTime:
https://doi.org/10.1007/s10618-020-00710-y and
https://github.com/hfawaz/InceptionTime . Six complete input channels x 8000.
Six modules, each with a 16-channel 1x1 projection, parallel convolutions with
kernels 41/21/11 and 32 filters per branch; fourth branch: stride-one max pool
3 on the original module input, then 1x1 projection. Concatenate, batch norm,
ReLU. Projection residuals after modules 3 and 6. Dilations 1/1/1/8/8/8.
Global average pooling, twelve logits; no downsampling, dropout or attention.
Batch norm epsilon 1e-5, PyTorch momentum 0.1. Kaiming normal fan-out initialization.

276012 parameters, convolutional local support 1081 samples / 27.025 ms before
pooling (evaluation-mode normalization), versus E33 279692 / 1023 / 25.575 ms.
Capacity/support are broadly comparable; equal epochs do not imply equal cost.
This is not an isolated causal test of multiscale processing. Odd kernels,
dilations, bottleneck, initialization and the E33 optimization protocol are
declared adaptations. The original five-model InceptionTime ensemble is NOT
reproduced. Seeds 42/43/44 are separately reported models, never a hidden ensemble.

## Comparison contract unchanged from ResNet

Exact frozen float32 NCT tensors from the private E33 dataset: clean, AWGN20,
mixed10; corruption was computed in float64 before conversion. No regeneration
on Linux. Same 1839 acquisitions, 39 families, twelve labels and five fixed folds.
Inner family validation: 20% rounded up, PCG64 20261029+fold. Channel normalization
uses training rows/time only, preserving between-window amplitude differences.

Same AdamW lr 1e-3, decay 1e-4, unweighted cross entropy, batch 16/eval 32,
gradient clip 5, max 150 epochs, patience 20, min validation Macro-F1 improvement
1e-4. Reinitialize and refit scaler on all outer-training families for the
selected epoch count. Same seed derivation and shuffled batch policy as E33.
Deterministic float32, no AMP/TF32, one GPU. No test-based changes to these choices.

45 jobs / 90 fit procedures: 3 training conditions x 5 folds x 3 seeds, each with
inner selection and outer refit. Clean-trained jobs also predict both corrupted
test conditions: 75 fold/seed/test-condition prediction sets in total. Metrics
use the same family/class weighting and pooled OOF predictions. Report all seeds,
mean/sample SD (not CI), healthy false alarms, missed faults, ABC/ABCG errors and
probability quality. No automatic amendment of the 728 E32 contrasts.

## Operation and decision rule

New private notebook; existing E33 private input dataset is read-only. A code-only
Git bundle is embedded in the notebook. Commit and SHA-256 checks bind code,
configuration and the original handoff. Temporary copies wrap the same input
bytes in an E43 manifest; no E33 source/result is overwritten or uploaded again.
Same isolated Python 3.12.12, torch 2.8.0 cu126, numpy 2.3.5, scipy 1.18.1,
scikit-learn 1.9.1. Up to nine compute hours, with five-minute export headroom.
Checkpoints after each epoch; resume archive after each job and graceful exit.
Explicit future resume must attach E43 previous output and match all identities.
Another Kaggle session is not started automatically. Preserve failures.

An advantage limited to one seed/condition, increased healthy false alarms or
contradictory effects weakens a broad improvement claim; retain negative outcomes.
Near-coincident ABC/ABCG inputs remain an acquisition limitation. No score is
anticipated. Generalization beyond this simulated network is unestablished.
The manuscript stays unchanged until verified results are discussed and approved.
