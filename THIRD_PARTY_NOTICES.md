# Third-party notices and scientific antecedents

Original C1 code is offered under MIT. Retain all existing copyright and licence
notices when copying a component. This document does not override component
licences or grant rights in external software.

## External dependencies

Python and the scientific/graphics libraries listed in the environment files
retain their own licences. In particular, NumPy, SciPy, scikit-learn, PyTorch,
PyWavelets, Matplotlib, ReportLab, pypdf and PDFium-related bindings are external
projects, not C1-authored software. They are installed separately; the source
package does not relicense their distributions or any installed environment.
Preserve their notices if redistributing wheels or binaries separately.

## Scientific antecedents

- The waveform ResNet is adapted from the time-series baseline of Wang et al.,
  "Time Series Classification from Scratch with Deep Neural Networks: A Strong
  Baseline" (2017). Its architecture reference is not a licence for upstream code.
- The adapted InceptionTime comparator is documented in the frozen E43 protocol
  as an independent PyTorch implementation inspired by Fawaz et al.,
  "InceptionTime: Finding AlexNet for time series classification" (2020),
  https://doi.org/10.1007/s10618-020-00710-y . The original project is
  https://github.com/hfawaz/InceptionTime . This package does not claim authorship
  of the InceptionTime method or apply MIT to that upstream repository.
- MATPOWER case9 is the topology/data reference for the modified nine-bus model.
  MATPOWER remains a separate project: https://matpower.org/ . Any redistributed
  MATPOWER file must retain its own applicable notice and licence.

Architectural or bibliographic credit does not replace an upstream licence when
source code has actually been copied. Historical diagnostic/reference material
and journal assets require their own provenance review before public inclusion.
No exhaustive clearance of those internal reference collections is asserted here.
