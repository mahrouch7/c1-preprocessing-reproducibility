[CmdletBinding()]
param([switch]$ConfirmRecovery18286Fits,[switch]$Acknowledge97PriorPartialFits,[string]$Output='F:\Backup\C1_E32\recovery_numeric_v1',[switch]$CheckOnly)
$ErrorActionPreference='Stop'
$python='G:\venvs\C1_R07\Scripts\python.exe'
if(-not(Test-Path -LiteralPath $python -PathType Leaf)){throw 'Existing qualified Python missing; no installation'}
$entry=Join-Path $PSScriptRoot 'run_c1e32_recovery.py'
if(-not(Test-Path -LiteralPath $entry -PathType Leaf)){throw 'Canonical recovery bootstrap missing; no submission'}
if($CheckOnly){
    if($ConfirmRecovery18286Fits -or $Acknowledge97PriorPartialFits){throw 'CheckOnly cannot authorize scientific execution'}
    & $python -B -u $entry --help
    if($LASTEXITCODE -ne 0){throw "Recovery entry-point check failed: $LASTEXITCODE"}
    Write-Host '[ENTRY POINT CHECK PASS; NO FIT, API CONTACT OR TASK SUBMISSION]'
    return
}
if(-not $ConfirmRecovery18286Fits -or -not $Acknowledge97PriorPartialFits){throw 'Explicit recovery budget and prior partial-attempt acknowledgement required; no submission'}
Write-Host '[SCOPE] Preserve 644 verified fits; recover 201 missing units / 18286 fits across PC1 and PC2.'
Write-Host '[ATTEMPTS] Preserve 97 unreceipted failed-unit fits; explicitly repeat that unit once; 19027 total attempted fits, 18930 unique final fits.'
Write-Host '[GATE] Saved-model numerical verification on PC2 before any fit; original tolerance and model parameters unchanged.'
& $python -B -u $entry --output $Output --confirm-new-fits 18286 --acknowledge-prior-partial-fits 97
if($LASTEXITCODE -ne 0){throw "Recovery stopped (exit $LASTEXITCODE). Preserve all old/new folders; no automatic retry."}
