param(
    [ValidateSet("check", "status", "download-data", "kfold", "gpu", "clean-cache")]
    [string]$Step = "check",
    [switch]$Force,
    [string]$Config = "configs/local_3060.json",
    [string]$Output = "results/kfold_box144_yolo11s",
    [int]$Folds = 5,
    [int]$Epochs = 0,
    [int]$Patience = 0,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Ambiente .venv nao encontrado. Crie com Python 3.12 e instale requirements-local.txt."
}

Push-Location $Root
try {
    if ($Step -eq "gpu") {
        nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv -l 2
        return
    }
    if ($Step -eq "clean-cache") {
        Remove-Item -LiteralPath ".\__pycache__", ".\src\cervical_cell_detection\__pycache__" -Recurse -Force -ErrorAction SilentlyContinue
        Write-Output "Caches Python removidos."
        return
    }

    $env:PYTHONPATH = Join-Path $Root "src"
    $argsList = @("-m", "cervical_cell_detection")
    switch ($Step) {
        "kfold" {
            $argsList += @("kfold", "--folds", $Folds)
            if ($Epochs -gt 0) { $argsList += @("--epochs", $Epochs) }
            if ($Patience -gt 0) { $argsList += @("--patience", $Patience) }
            $argsList += @("--output", $Output)
            if ($DryRun) { $argsList += "--dry-run" }
        }
        default { $argsList += $Step }
    }
    $argsList += @("--config", $Config)
    if ($Force) { $argsList += "--force" }
    & $Python @argsList
    if ($LASTEXITCODE -ne 0) {
        throw "Etapa '$Step' falhou com codigo $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
