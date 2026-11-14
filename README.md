# C1: preprocessing and power-system fault classification

This repository provides the code and saved-artifact reproduction tools for the
C1 study. The study compares descriptor-based ML pipelines with adapted ResNet-1D
and InceptionTime waveform models on 1,839 simulated acquisitions from a modified
nine-bus network. Training and evaluation separate acquisition/load families.

The practical reproduction path uses the saved held-out predictions. It rebuilds
17 comparative-analysis CSV tables without training a model. A second step draws
the five processing/result charts numbered 3–7 in the fifteen-page manuscript.
Those charts use frozen plotting data; drawing them does not repeat every raw
diagnostic calculation. Figures 1–2 (network/protocol and acquisition) are excluded
from generation. The plotting bundle retains seven historical references for
integrity checks and visual comparison, including the two excluded figures.

## Contents

- `sources/e32/`: frozen amended ML analysis and required upstream code.
- `sources/resnet/` and `sources/inceptiontime/`: architectures, configurations,
  tests and analysis tools. All original seeds are retained.
- `research_record/tools/`: isolated restoration, pinned environments and plotting
  tools needed for the reproduction path. Internal discussion logs are absent.
- `reference/resnet_seeds_verified.csv`: pinned comparator reference.
- `data/archives.json`: the two external data archives, expected hashes and sizes.
- `provenance/assembly_manifest.json`: SHA256 inventory of this source release.

The historical diagnostic collection, local path configuration, credentials,
environments, checkpoints and manuscript drafts are not included. Large data
archives are distributed separately; do not commit them to source Git.

## Data archives: public Kaggle version 1

The two companion archives are publicly released in
[Kaggle dataset version 1](https://www.kaggle.com/datasets/mahrouch/c1-reproduction-archives-v1/versions/1)
(`mahrouch/c1-reproduction-archives-v1`, version **1**). The dataset and this
GitHub repository are public. The link identifies the versioned dataset page;
downloads are available through Kaggle's web interface or API rather than a
stable per-file URL stored in this repository.

| Original ZIP filename | Filename on Kaggle |
| --- | --- |
| `C1_predictions_metadata.zip` | `C1_predictions_metadata.zip.bin` |
| `C1_clean_windows_features.zip` | `C1_clean_windows_features.zip.bin` |

The `.bin` suffix is for transport only: the files contain the original ZIP
bytes. Download **version 1**, verify each size and SHA256 against
[`data/archives.json`](data/archives.json), and copy or rename the two files to
the original ZIP filenames above without extracting or recompressing them.
Keep the ZIPs together outside this repository and pass that folder to
`Reproduce.ps1 -ArchiveRoot`. The Kaggle README provides a PowerShell copy/hash
example; the restorer also verifies ZIP and extracted-member hashes.

The catalogue records the public dataset location and filename mapping. Its
`download_url` fields remain `null` because no stable per-file URL is asserted;
the versioned Kaggle page is the authoritative public access point. Publication
does not alter the archive bytes or their recorded hashes.

## Reproduce the saved results

The validated platform is **Windows x64 with CPython 3.12**. A GPU is not needed
for this saved-prediction analysis. Other platforms have not been validated by
this release. Obtain the two archives described in `data/README.md`, preserve
their filenames, and place them together in a folder outside this repository.

From this repository, run the following with your actual paths:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Reproduce.ps1 `
  -BootstrapPython "<path-to-python.exe>" `
  -ArchiveRoot "<folder-containing-both-data-ZIPs>" `
  -OutputRoot "<new-output-parent-outside-this-repository>"
```

This explicit reproduction command installs separate pinned numerical/graphics
environments, verifies/restores the archives and checks all 17 reference tables
byte-for-byte before plotting. It downloads dependencies from PyPI. Existing
environments and scientific inputs are preserved. Add `-TablesOnly` to omit plots.
The output receipts distinguish numerical identity from visual approval of PDFs.
Inspect newly generated PDFs before using them in a manuscript.

The folder assembler itself does none of this computation or installation.
The upstream layout was validated before assembly; runtime validation of this
newly assembled copy remains a separate author-run step.

## Direct tools and original experiments

`Run-C1.ps1` exposes the saved-prediction comparison and plan-only neural commands.
For direct use, copy `paths.example.json` to ignored `paths.local.json`, edit the
roots and use the environment matching the frozen experiment versions. For the
archive-based route above, `Reproduce.ps1` creates a relative local configuration
if absent; restoration supplies the actual data paths in its isolated workspace.

Original experiment files can retain historical storage roots as provenance.
Runtime adapters replace the configured roots; never run the historical
`sources/inceptiontime/Analyze-Results.ps1` launcher. That file is retained solely
because the frozen comparator hashes it. Use the root launchers instead.

This is not a turnkey regeneration of the electrical simulator or the complete
historical distributed ML campaign. No model checkpoints are bundled. Neural
training sources are supplied, but the validated reproduction claim concerns
saved predictions, numerical tables and the specified plotting inputs.

## Licence and citation

Original C1 software and accompanying software documentation use **MIT**
(`LICENSE`). Original C1 data, saved predictions, numerical results and scientific
figures use **CC BY 4.0** (`LICENSE-DATA-CC-BY-4.0.txt`). See `LICENSING.md` and
`THIRD_PARTY_NOTICES.md` for scope, exceptions and external dependencies.

**Please cite the associated C1 article when using this code, these data or these
figures in research.** The final article reference will be added to `CITATION.cff`
after acceptance. The active citation currently identifies the software and
public repository; no article DOI or acceptance date has been invented. This
scholarly request adds no restriction to the standard licence terms.

## Integrity and release status

Run `python verify_release.py` to check the assembly inventory without importing
scientific packages. Configure ignored local paths and keep reproduction outputs
outside the source tree. The source manifest preserves frozen file bytes.

The code is publicly hosted at
https://github.com/Mahrouch7/c1-preprocessing-reproducibility. The companion
archives are publicly hosted in the versioned Kaggle dataset linked above. The
local assembler itself does not publish or modify either remote resource.
