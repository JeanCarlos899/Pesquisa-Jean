param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $Root ".venv"

Push-Location $Root
try {
    if (-not (Test-Path -LiteralPath $Venv)) {
        & $Python -m venv .venv
    }
    $Pip = Join-Path $Venv "Scripts\python.exe"
    & $Pip -m pip install --upgrade pip
    & $Pip -m pip install -r requirements-local.txt
    Write-Output "Ambiente pronto. Rode: .\run.ps1 check"
}
finally {
    Pop-Location
}
