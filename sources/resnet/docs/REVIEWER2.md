# Skeptical review — CURRENT, before scientific results

1. **A weak strawman comparator — high risk.** A recognized residual family,
   adequate physical receptive field, train-only stopping and all-initialization
   reporting are used. The dilation/width adaptation is transparent. No claim of
   an optimal or exhaustive deep-model comparison. Inspect training-only loss,
   gradients and stopping behavior; any test-driven redesign must be disclosed.
2. **Leakage — critical.** Preserve original acquisition/family folds; keep
   corruptions of an event together; fit scalers only on appropriate training
   families. No random slicing of the8000samples into nominally independent data.
3. **Different information — high.** Exactly the same six channels and200ms
   window, but temporal information differs deliberately from descriptors. No
   privileged location, prefault context, extra observation point or class label.
4. **Different optimization — medium/high.** Classical models are fixed; the
   network selects stopping epochs internally. Report this rather than asserting
   identical search budgets. Do not conclude each family is optimally represented.
5. **Dependence and small effective sample — high.**1839windows are not1839
   independent operating regimes.39designed families and template dependence
   remain; three model seeds do not create new acquisition families.
6. **Multiple comparisons — high.** All24 fixed baselines are retained in a
   descriptive table. No post-hoc best-adversary or best-seed significance claim.
   Formal paired inference must declare its comparison family and MC precision.
7. **Physical validity — high.** Same static ideal-source/sequence assumptions,
   ABC/ABCG ambiguity and healthy support constraints as E32. E33 cannot repair
   the physical model or close R2-C01 by adding a neural network.
8. **Deployment — high if overclaimed.** Complete onset-aligned200ms window,
   noncausal context, no relay timing, calibration or streaming validation. CPU
   timing is not a validated real-time protection benchmark.
9. **Results status.** Before complete runs, only software/data qualification may
   be reported. R2-C02 is IMPLEMENTATION_IN_PROGRESS, not CLOSED. Preserve harms,
   failures and inconclusive outcomes alongside favorable outcomes.
