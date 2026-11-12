# Required data and saved artifacts

The cohort contains 1839 windows, 39 acquisition/load families and 12 labels.
metadata.json carries window IDs, family IDs, folds and labels; preserve all IDs,
row order, channel order and weights implied by the recorded evaluation procedure.
Clean windows contain 8000 samples at 40 kHz on six ordered channels (200 ms).

## Saved-score reproduction (compare)

1. Campaign metadata.json (pinned SHA256).
2. E32 final_analysis/pooled_prediction_index.json and every .npz probability file
   referenced by its original relative paths. The five-case comparator uses a subset
   but the original tree should remain intact. Do not select only winning models.
3. Both neural run trees: run_manifest.json; all jobs/fold/seed completed receipts;
   predictions and every referenced non-.pt artifact, including training histories.
   Checkpoint tensors are not loaded by this comparison; keep them separately for
   any future inference/retraining work. All three seeds must be retained.
4. Each corresponding retrieval_audit.json, at its pinned SHA256.
5. The pinned ResNet summary reference CSV (included under reference/).

## Raw-waveform/preflight/training reference

The campaign root additionally needs scope.json, fold_design.json and base/clean.npy
at their pinned identities. Shared preprocessing/feature cache manifests and all
referenced arrays are required by original neural preflight/parity checks. The
simulation source dataset and its views/paper_reduced/view_manifest.json are a
separate dependency of the historical E32 distributed training path. No generated
or synthetic substitute is used when an artifact is absent.

## Frozen ML table export

analysis_receipt.json and metrics.csv, per_class_metrics.csv,
paired_family_inference.csv, pooled_prediction_index.json. The export checks each
artifact against the original receipt. This action does not recompute bootstrap,
permutation tests, noise or classifier fits, and does not export superseded TPI tables.

Data roles and known hashes are machine-readable in data_requirements.json. Some
requirements refer to whole manifest-linked subtrees: an existence check alone does
not establish completeness. The scientific comparator performs its original checks
when the author explicitly runs it. Original C1 data and figures use CC BY 4.0; see LICENSING.md.
Public hosting and archive URLs remain to be established. Keep large data/model payloads outside source Git.
