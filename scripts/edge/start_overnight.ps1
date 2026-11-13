# Start the overnight orchestrator detached (hidden, survives closing the terminal). Never writes to C:.
# Usage: powershell -File scripts\edge\start_overnight.ps1 [-Only S4,S5] [-Parallel 6] [-DryRun]
param([string[]]$Only = @(), [int]$Parallel = 0, [switch]$DryRun, [string]$Python = "python")
$Scratch = "D:\Bunker\BaseCamp\_claude_scratch"
$Out = Join-Path $Scratch "overnight"
$Tmp = Join-Path $Scratch "tmp"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
New-Item -ItemType Directory -Force $Out, $Tmp | Out-Null
$env:TMP = $Tmp; $env:TEMP = $Tmp; $env:TMPDIR = $Tmp
$env:PYTHONUNBUFFERED = "1"; $env:PYTHONPYCACHEPREFIX = Join-Path $Tmp "pycache"; $env:MPLCONFIGDIR = Join-Path $Tmp "mpl"
$args_ = "-u -m scripts.edge.run_overnight"
if ($Only.Count) { $args_ += " --only " + ($Only -join " ") }
if ($Parallel -gt 0) { $args_ += " --parallel $Parallel" }
if ($DryRun) { $args_ += " --dry-run" }
$log = Join-Path $Out "overnight.log"
$p = Start-Process -FilePath "cmd.exe" -WindowStyle Hidden -WorkingDirectory $Repo -PassThru `
    -ArgumentList "/s /c `"$Python $args_ >> `"$log`" 2>&1`""
"started cmd pid $($p.Id); log $log; status $(Join-Path $Out 'STATUS.md')"
