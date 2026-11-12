[CmdletBinding()]
param(
    [ValidateSet('inspect','compare','export-ml','plan-resnet','plan-inception','plot-comparison')][string]$Action='inspect',
    [string]$PythonExecutable='python',
    [string]$Config='',
    [switch]$VerifyHashes
)
$ErrorActionPreference='Stop'
if (-not $Config) { $Config=Join-Path $PSScriptRoot 'paths.local.json' }
$arguments=@((Join-Path $PSScriptRoot 'run.py'),'--action',$Action,'--config',$Config)
if ($VerifyHashes) { $arguments+='--verify-hashes' }
& $PythonExecutable @arguments
if ($LASTEXITCODE -ne 0) { throw 'Action failed; inspect missing inputs, identity checks or environment requirements above.' }
