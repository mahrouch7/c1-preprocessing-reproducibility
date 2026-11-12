[CmdletBinding()]
param(
    [string]$Python = '',
    [string]$OutputRoot = '',
    [switch]$SkipPlots,
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$taskRoot = $PSScriptRoot
$policyFile = Join-Path $taskRoot 'configs\comparative_analysis.json'
$policy = Get-Content -LiteralPath $policyFile -Raw | ConvertFrom-Json
function Resolve-TaskPath([string]$Relative) { return [IO.Path]::GetFullPath((Join-Path $taskRoot $Relative)) }
function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    $prior = $ErrorActionPreference
    try { $ErrorActionPreference = 'Continue'; & $Executable @Arguments; $code = $LASTEXITCODE }
    finally { $ErrorActionPreference = $prior }
    if ($code -ne 0) { throw ('Analysis step failed with exit code ' + $code) }
}
if (-not $Python) { $Python = Resolve-TaskPath $policy.python_relative }
if (-not $OutputRoot) { $OutputRoot = Resolve-TaskPath $policy.output_root_relative }
$outputParent = [IO.Path]::GetFullPath($OutputRoot)
foreach ($relative in $policy.protected_roots_relative) {
    $protected = (Resolve-TaskPath $relative).TrimEnd('\','/')
    if ($outputParent -eq $protected -or $outputParent.StartsWith($protected + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Analysis output must remain outside protected experiment inputs.'
    }
}
$graphics = Resolve-TaskPath $policy.graphics_deps_relative
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Compatible scientific Python not found; use -Python.' }
foreach ($name in @('InceptionTime','ResNet')) {
    $entry = $policy.neural.$name
    if (-not (Test-Path -LiteralPath (Resolve-TaskPath $entry.run_relative) -PathType Container)) { throw ('Missing imported run: ' + $name) }
    $auditPath = Resolve-TaskPath $entry.audit_relative
    if ((Get-FileHash -LiteralPath $auditPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.audit_sha256) { throw ('Retrieval audit changed: ' + $name) }
}
Write-Host 'Scope: pooled held-out metrics; two neural models x three seeds and 24 fixed ML pipelines; five common evaluation cases.'
Write-Host 'No training, simulations, new noise, significance tests, winning-seed selection or manuscript modification.'
Write-Host ('Output parent: ' + $OutputRoot)
if (-not $SkipPlots -and -not (Test-Path -LiteralPath (Join-Path $graphics 'matplotlib') -PathType Container)) {
    throw 'Graphics dependencies unavailable. Use -SkipPlots to run the scientific analysis alone.'
}
if ($CheckOnly) { Write-Host 'CHECK ONLY: paths and audit fingerprints verified; no scientific metrics computed.'; exit 0 }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$runName = 'comparison_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0,8)
$destination = Join-Path $OutputRoot $runName
$logFile = Join-Path $OutputRoot ($runName + '.log')
Start-Transcript -LiteralPath $logFile | Out-Null
try {
    Invoke-Checked $Python @((Join-Path $taskRoot 'tools\analyze_results.py'),'--policy',$policyFile,'--output',$destination)
    if (-not $SkipPlots) {
        # A separate process keeps graphics-only NumPy out of the scientific analysis.
        Invoke-Checked $Python @((Join-Path $taskRoot 'tools\plot_comparison.py'),'--analysis',$destination,'--graphics-deps',$graphics)
    }
    Write-Host ('Analysis finished: ' + $destination)
    Write-Host ('French reading report: ' + (Join-Path $destination 'analyse_fr.md'))
    Write-Host ('Receipt: ' + (Join-Path $destination 'analysis_receipt.json'))
    if (-not $SkipPlots) { Write-Host 'PDF/SVG/PNG exported. Actual PDF visual inspection remains pending before manuscript integration.' }
} finally { Stop-Transcript | Out-Null }
