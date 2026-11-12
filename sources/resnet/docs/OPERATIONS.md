# Operations and recovery

All paths below are relative to this repository. Source data are external,
read-only. `cache/` and `outputs/` are ignored by Git and are not public datasets.
Keep backups of scientific outputs; Git tracks the code and research record,
not the large model/data artifacts themselves.

## Background execution

`Start-Experiment.ps1` starts the selected run as a hidden background PowerShell
process and writes its PID/launch metadata under `outputs/launches/`. It redirects
stdout and stderr to separate logs. The runner does not register a scheduled
task, change sleep settings or monitor itself. It exits on an integrity/numerical
failure; it does not automatically restart an uncertain computation.

```powershell
.\Start-Experiment.ps1 -Profile pilot -RunName pilot_v1 -Device cpu -AllowCPU
```

After checking that a prior background process stopped, use the same launcher
with `-Resume`. It never deletes another process's lock. Do not start two
coordinators for the same run. Optional foreground `-Fold`, `-Seed`, `-Condition`
filters support bounded execution; they do not redefine the run's expected scope.

## Status without a monitoring automation

```powershell
Get-Content .\outputs\launches\pilot_v1_*.stdout.log -Tail 20
Get-Content .\outputs\launches\pilot_v1_*.stderr.log -Tail 20
Get-ChildItem .\outputs\pilot_v1\jobs -Filter completed.json -Recurse | Measure-Object
```

Logs contain training and **inner-validation** values. Outer scores are saved
only after each final refit. A coordinator's successful exit means its selected
jobs finished; only `Analyze` checks that all expected jobs exist.

## Crash locks and partial files

Normal interruption releases the active lock, preserving checkpoints and a
failure receipt. Hard termination/power loss may leave a lock. Read its PID and
check process identity before unlocking. Do not kill an unrelated recycled PID.

```powershell
.\.venv\Scripts\python.exe -m c1resnet.cli unlock --path .\outputs\pilot_v1\coordinator.lock --confirmed-process-stopped
```

Do the same for a job's `running.lock` only if needed. A cache `.npy` without its
receipt or a prediction without its matching JSON is deliberately not trusted.
Inspect and preserve such artifacts before an explicit repair; there is no
automatic broad delete, overwrite or inference from mere file existence.

Resume restores phase-specific RNG/shuffle/optimizer states. Synthetic tests
interrupt immediately after checkpoint publication and compare resumed weights
and history bitwise with an uninterrupted run. This proves the tested software
path, not robustness across different devices.

## Outputs

`outputs/<run>/run_manifest.json`: immutable identity and numerical environment.
`noise_parity.json`: real E32 feature/noise alignment audit, zero model fits.
`jobs/<condition>/foldXX/seedYY/`: split, identity, selection/refit scaler/history/
checkpoint/summary, outer-test probabilities, failure records and `completed.json`.
`outputs/analysis_*/`: OOF archives, per-seed metrics, initialization summaries,
descriptive comparisons to E32 and completeness receipt. Fractions are not percent.

For progress-only aggregation use CLI `analyze --allow-partial`; it explicitly
marks partial data NOT_PUBLICATION_READY and omits incomplete five-fold scores.
Never insert partial pilot values or synthetic smoke values into the manuscript.
