param(
    [Parameter(Mandatory = $true)] [string[]] $Images,
    [string] $Mission = 'Inspect images for visible conditions relevant to the assigned task.',
    [string] $UeId = 'ue01',
    [string] $YoloCheckpoint = '',
    [switch] $AllowHistoricalThreeClass
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$clipCheckpoint = Join-Path $projectRoot 'models\priority_classifiers\clip'
$release = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'config\parameter_release_v2.json') -Raw | ConvertFrom-Json
$supportedLevels = @($release.spi.supported_levels)
$fourClassDifference = @(Compare-Object -ReferenceObject @('Low', 'Medium', 'High', 'Critical') -DifferenceObject $supportedLevels)
if (-not $AllowHistoricalThreeClass -and ($supportedLevels.Count -ne 4 -or $fourClassDifference.Count -ne 0)) {
    throw 'No four-class SPI release is selected. The historical three-class fallback cannot satisfy this run. Use -AllowHistoricalThreeClass only for an explicitly historical comparison.'
}
$modelPath = Join-Path $projectRoot $release.spi.model_path
foreach ($requiredPath in @($pythonExe, $clipCheckpoint, $modelPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) { throw "Required file or directory not found: $requiredPath" }
}
$resolvedImages = @($Images | ForEach-Object { (Resolve-Path -LiteralPath $_).Path })
$runRoot = Join-Path $PSScriptRoot 'results\user_runs'
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
$runId = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$outputPath = Join-Path $runRoot ("parameters-" + $runId + '.json')
$kbPath = Join-Path $runRoot ("kb-" + $UeId + '.json')
$cliArgs = @('-m', 'parameters.parameter_cli', '--images') + $resolvedImages + @(
    '--mission', $Mission, '--ue-id', $UeId, '--clip-checkpoint', $clipCheckpoint,
    '--model', $modelPath, '--kb', $kbPath, '--output', $outputPath
)
if ($YoloCheckpoint) {
    $resolvedYolo = (Resolve-Path -LiteralPath $YoloCheckpoint).Path
    $cliArgs += @('--yolo-checkpoint', $resolvedYolo)
} else {
    Write-Host 'YOLO weights not specified: scene fields will be unavailable.'
}
if (-not $AllowHistoricalThreeClass) { $cliArgs += '--require-four-classes' }
Write-Host "SPI model: $($release.spi.model_status), supported levels: $($release.spi.supported_levels -join ', ')"
Write-Host "Results: $outputPath"
Write-Host "UE knowledge base: $kbPath"
Push-Location $projectRoot
try {
    & $pythonExe @cliArgs
    if ($LASTEXITCODE -ne 0) { throw "Parameter extraction failed (exit $LASTEXITCODE)" }
} finally { Pop-Location }
