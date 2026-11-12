[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$BootstrapPython,
    [Parameter(Mandatory=$true)][string]$ArchiveRoot,
    [Parameter(Mandatory=$true)][string]$OutputRoot,
    [switch]$TablesOnly
)
$ErrorActionPreference='Stop'
if (-not (Test-Path -LiteralPath $BootstrapPython -PathType Leaf)) { throw 'Supply CPython 3.12 Windows x64.' }
$tools=Join-Path $PSScriptRoot 'research_record\tools'
$local=Join-Path $PSScriptRoot 'paths.local.json'
if (-not (Test-Path -LiteralPath $local)) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'paths.example.json') -Destination $local
}
foreach ($entry in @(@('C1_clean_environment','test_clean_environment.py'),@('C1_paper_figures','test_paper_figures.py'))) {
    if ($TablesOnly -and $entry[0] -eq 'C1_paper_figures') { continue }
    & $BootstrapPython -I -B -m unittest discover -s (Join-Path $tools $entry[0]) -p $entry[1] -v
    if ($LASTEXITCODE -ne 0) { throw 'Reproduction helper tests failed before installation.' }
}
$arguments=@('-I','-B',(Join-Path $tools 'C1_clean_environment\clean_environment.py'),
    '--package-root',$PSScriptRoot,'--archive-root',$ArchiveRoot,'--output-root',$OutputRoot)
if (-not $TablesOnly) { $arguments+='--paper-figures' }
Write-Host '[SCOPE] Explicit reproduction: pinned dependency installation, isolated restoration, 17 tables and optional charts 3--7.'
Write-Host '[SCOPE] No model training, model inference or publication.'
& $BootstrapPython @arguments
if ($LASTEXITCODE -ne 0) { throw 'Reproduction failed. Preserve the output and inspect fresh_environment_receipt.json and logs.' }
