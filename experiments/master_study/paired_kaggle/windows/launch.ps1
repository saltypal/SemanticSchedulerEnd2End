param([switch]$Submit)
$ErrorActionPreference = "Stop"
Write-Output "Windows account target:satyapaladugu; source:$PSScriptRoot"
if (-not $Submit) {
    Write-Output "Preview only. Attach private mini/shared inputs and update dataset_sources before -Submit."
    exit 0
}
$meta = Get-Content -LiteralPath (Join-Path $PSScriptRoot "kernel-metadata.json") -Raw | ConvertFrom-Json
if ($meta.dataset_sources.Count -eq 0) { throw "Attach private inputs and record their dataset_sources IDs first." }
if (-not $meta.id.StartsWith("satyapaladugu/")) { throw "Wrong account metadata." }
kaggle kernels push -p $PSScriptRoot --accelerator NvidiaTeslaT4
if ($LASTEXITCODE -ne 0) { throw "Kaggle submission failed; inspect CLI output." }
