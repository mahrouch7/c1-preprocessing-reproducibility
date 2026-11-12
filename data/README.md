# Companion data archives

`archives.json` identifies two companion ZIPs by filename, byte count and SHA256:

1. `C1_predictions_metadata.zip`: saved ML/neural predictions, probabilities,
   cohort metadata, training/run receipts and reference numerical tables.
2. `C1_clean_windows_features.zip`: clean six-channel windows and frozen feature
   arrays needed by the preserved data/feature contract.

The source repository does not contain these large archives. Public download URLs
have not yet been assigned; `null` in the catalogue means no public location is
being claimed. Replace it only with the verified archive location. Keep both ZIPs
together outside the source repository and supply that directory to Reproduce.ps1.
The restorer verifies full ZIP hashes and all declared extracted member hashes.

Original C1 data and figures are licensed under CC BY 4.0. Retain the licence,
credit the C1 study authors, identify the source and indicate changes as required
by its terms. Please cite the associated article once its reference is available.
Neither the electrical simulator nor trained checkpoint tensors are in these ZIPs.
