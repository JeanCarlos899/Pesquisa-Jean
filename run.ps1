param(
    [ValidateSet("check", "status", "download-data", "prepare", "train", "predict-val", "predict-test", "eval-val", "eval-test", "finalize", "bbox-sweep", "model-compare", "model-s", "model-m", "all", "materials", "materials-s", "materials-m", "package", "gpu", "clean-cache")]
    [string]$Step = "check",
    [switch]$Force,
    [string]$Config = "configs/local_3060.json",
    [string]$Output = "results/article_materials",
    [string[]]$Models = @("s"),
    [switch]$Test
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
        "predict-val" { $argsList += @("predict", "--split", "val") }
        "predict-test" { $argsList += @("predict", "--split", "test") }
        "eval-val" { $argsList += @("eval", "--split", "val") }
        "eval-test" { $argsList += @("eval", "--split", "test") }
        "model-s" { $argsList += @("model-compare", "--models", "s") }
        "model-m" { $argsList += @("model-compare", "--models", "m") }
        "materials-s" { $argsList += @("materials", "--model", "s") }
        "materials-m" { $argsList += @("materials", "--model", "m") }
        "model-compare" {
            $argsList += @("model-compare", "--models")
            foreach ($Model in $Models) {
                $argsList += $Model
            }
        }
        default { $argsList += $Step }
    }
    $argsList += @("--config", $Config)
    if ($Force) { $argsList += "--force" }
    if ($Step -eq "materials" -or $Step -eq "materials-s" -or $Step -eq "materials-m" -or $Step -eq "all") {
        $argsList += @("--output", $Output)
    }
    if ($Test -and ($Step -eq "model-compare" -or $Step -eq "model-s" -or $Step -eq "model-m")) {
        $argsList += "--test"
    }
    & $Python @argsList
    if ($LASTEXITCODE -ne 0) {
        throw "Etapa '$Step' falhou com codigo $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
