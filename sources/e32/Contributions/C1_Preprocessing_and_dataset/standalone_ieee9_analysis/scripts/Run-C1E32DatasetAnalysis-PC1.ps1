[CmdletBinding()]
param(
    [ValidateSet('check','prepare','all')][string]$Stage='check',
    [string]$Dataset='G:\IEEE9_Datasets\C1_E31\reduced_v1',
    [string]$Output='F:\Backup\C1_E32\campaign_v1',
    [string]$PythonExecutable='G:\venvs\C1_R07\Scripts\python.exe',
    [string]$PC2Workspace='H:\r07node_e32\workspace',
    [string]$GraphicsPython='C:\Users\Maher369\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe',
    [string]$GraphicsFont='C:\Windows\Fonts\arial.ttf',
    [switch]$ConfirmNewCohortAnalysis
)
$ErrorActionPreference='Stop'
$package=Split-Path -Parent $PSScriptRoot
if(-not(Test-Path -LiteralPath $PythonExecutable -PathType Leaf)){throw 'Pinned research Python missing; no automatic installation'}
if($Stage -eq 'all' -and -not $ConfirmNewCohortAnalysis){throw 'External -ConfirmNewCohortAnalysis is required; no work submitted'}
if($Stage -ne 'all' -and $ConfirmNewCohortAnalysis){throw 'Confirmation is only valid with -Stage all'}
Write-Host '[SCOPE] NEW 1839-window/39-family cohort; no old fitted models, no native simulations, no installations.'
Write-Host '[CONTROL] Fixed family folds, healthy-enrichment and nominal-density controls; external single campaign authorization.'
Write-Host '[FAILURE] Preserve all evidence; no automatic fit retry, output overwrite or deletion.'
$arguments=@('-B','-u',(Join-Path $PSScriptRoot 'run_c1e32_pipeline.py'),'--stage',$Stage,'--dataset',$Dataset,'--output',$Output,'--pc2-workspace',$PC2Workspace,'--graphics-python',$GraphicsPython,'--graphics-font',$GraphicsFont)
if($Stage -eq 'all'){$arguments+=@('--confirm-fits','18930')}
& $PythonExecutable @arguments
if($LASTEXITCODE -ne 0){throw "Scientific pipeline stopped (exit $LASTEXITCODE). Preserve output; do not rerun all or remove old folders."}
