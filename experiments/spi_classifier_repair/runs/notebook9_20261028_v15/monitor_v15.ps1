$ErrorActionPreference = 'Stop'
$Kernel = 'satyapaladugu/betterscheduler-spi-20261027finalc'
$Version = '15'
$RunRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$MonitorRoot = Join-Path $RunRoot 'monitor_v15'
$ArchivePath = Join-Path $MonitorRoot 'run_recovery.zip'
$Verifier = Join-Path $RunRoot 'verify_recovery_v15.py'
$MonitorDeadline = [DateTime]::UtcNow.AddMinutes(45)
New-Item -ItemType Directory -Path $MonitorRoot -Force | Out-Null

function Invoke-ProgramBounded([string]$Executable, [string[]]$Arguments, [int]$TimeoutSeconds, [string]$Tag) {
    $start = [System.Diagnostics.ProcessStartInfo]::new()
    $start.FileName = $Executable
    $start.UseShellExecute = $false
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    foreach ($argument in $Arguments) { [void]$start.ArgumentList.Add($argument) }
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $start
    if (-not $process.Start()) { throw "Could not start Kaggle CLI for $Tag" }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $finished = $process.WaitForExit($TimeoutSeconds * 1000)
    if (-not $finished) {
        try { $process.Kill($true) } catch { }
        $process.WaitForExit()
    }
    $stdout = $stdoutTask.GetAwaiter().GetResult()
    $stderr = $stderrTask.GetAwaiter().GetResult()
    [pscustomobject]@{
        ExitCode = if ($finished) { $process.ExitCode } else { 124 }
        TimedOut = -not $finished
        Stdout = $stdout
        Stderr = $stderr
    }
}

function Invoke-KaggleBounded([string[]]$Arguments, [int]$TimeoutSeconds, [string]$Tag) {
    Invoke-ProgramBounded (Get-Command kaggle -ErrorAction Stop).Source $Arguments $TimeoutSeconds $Tag
}

function Save-Text([string]$Path, [string]$Value) {
    [System.IO.File]::WriteAllText($Path, $Value, [System.Text.UTF8Encoding]::new($false))
}

$logSnapshots = Join-Path $MonitorRoot 'log_snapshots'
New-Item -ItemType Directory -Path $logSnapshots -Force | Out-Null
$lastStatus = 'unknown'
while ([DateTime]::UtcNow -lt $MonitorDeadline) {
    $stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
    $statusResult = Invoke-KaggleBounded @('kernels', 'status', "$Kernel/$Version") 45 'status'
    $combinedStatus = ($statusResult.Stdout + "`n" + $statusResult.Stderr).Trim()
    $match = [regex]::Match($combinedStatus, 'KernelWorkerStatus\.([A-Z_]+)')
    $status = if ($match.Success) { $match.Groups[1].Value } else { 'UNKNOWN' }
    $statusRecord = [ordered]@{
        timestamp_utc = [DateTime]::UtcNow.ToString('o')
        version = [int]$Version
        status = $status
        exit_code = $statusResult.ExitCode
        timed_out = $statusResult.TimedOut
        output = $combinedStatus
    }
    Add-Content -LiteralPath (Join-Path $MonitorRoot 'status_changes.jsonl') -Value (ConvertTo-Json $statusRecord -Compress) -Encoding utf8
    if ($status -ne $lastStatus) {
        Save-Text (Join-Path $MonitorRoot 'latest_status.txt') (ConvertTo-Json $statusRecord -Depth 5)
        $lastStatus = $status
    }

    $logs = Invoke-KaggleBounded @('kernels', 'logs', "$Kernel/$Version") 45 'logs'
    $logText = $logs.Stdout
    if ($logs.Stderr) { $logText += "`n[stderr]`n" + $logs.Stderr }
    if ($logText.Trim()) {
        Save-Text (Join-Path $logSnapshots ("kaggle_logs_$stamp.txt")) $logText
        Save-Text (Join-Path $MonitorRoot 'kaggle_logs_latest.txt') $logText
    }
    Save-Text (Join-Path $MonitorRoot 'monitor_heartbeat.txt') ("$stamp status=$status log_exit=$($logs.ExitCode) log_timeout=$($logs.TimedOut)`n")

    if ($status -in @('COMPLETE', 'ERROR', 'CANCELLED')) {
        $priorityRoot = Join-Path $MonitorRoot 'decision_artifacts'
        New-Item -ItemType Directory -Path $priorityRoot -Force | Out-Null
        $priorityPattern = '^(validation\.json|status\.json|selection_history\.json|refit_history\.json|predictions\.json|backbone_identity\.json|cell_status\.json|runtime\.json|cell_[0-9]+\.log|executed\.ipynb|retrieval_checksums\.json)$'
        $priority = Invoke-KaggleBounded @('kernels', 'output', '-p', $priorityRoot, '--file-pattern', $priorityPattern, "$Kernel/$Version") 120 'decision artifacts retrieval'
        Save-Text (Join-Path $MonitorRoot 'decision_artifacts_stdout.txt') $priority.Stdout
        Save-Text (Join-Path $MonitorRoot 'decision_artifacts_stderr.txt') $priority.Stderr
        $downloaded = @(Get-ChildItem -LiteralPath $priorityRoot -File -Recurse -ErrorAction SilentlyContinue)
        $summary = [ordered]@{status=$status;download_exit=$priority.ExitCode;download_timed_out=$priority.TimedOut;files=@($downloaded | ForEach-Object { [ordered]@{path=$_.FullName;bytes=$_.Length;sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()} })}
        foreach ($file in $downloaded | Where-Object { $_.Name -eq 'validation.json' }) {
            try {
                $validation = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json
                if ($validation.student) { $summary.validation = $validation.student }
            } catch { $summary.validation_parse_error = $_.Exception.Message }
        }
        foreach ($file in $downloaded | Where-Object { $_.Name -eq 'status.json' }) {
            try { $summary.candidate_status = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json } catch { }
        }
        Save-Text (Join-Path $MonitorRoot 'decision_artifacts_summary.json') (ConvertTo-Json $summary -Depth 12)
        Start-Sleep -Seconds 15

        $download = Invoke-KaggleBounded @('kernels', 'output', '-p', $MonitorRoot, '--file-pattern', '^run_recovery\.zip$', "$Kernel/$Version") 120 'terminal recovery ZIP retrieval'
        Save-Text (Join-Path $MonitorRoot 'recovery_download_stdout.txt') $download.Stdout
        Save-Text (Join-Path $MonitorRoot 'recovery_download_stderr.txt') $download.Stderr
        if ($download.ExitCode -eq 0 -and (Test-Path -LiteralPath $ArchivePath)) {
            $verify = Invoke-ProgramBounded (Get-Command python -ErrorAction Stop).Source @($Verifier) 120 'recovery verification'
            Save-Text (Join-Path $MonitorRoot 'recovery_verify_stdout.txt') $verify.Stdout
            Save-Text (Join-Path $MonitorRoot 'recovery_verify_stderr.txt') $verify.Stderr
            Save-Text (Join-Path $MonitorRoot 'recovery_verify_exit.txt') ([string]$verify.ExitCode)
        } else {
            Save-Text (Join-Path $MonitorRoot 'recovery_unavailable.txt') ("single terminal retrieval failed or did not produce the exact archive; exit=$($download.ExitCode) timeout=$($download.TimedOut)`n$($download.Stdout)`n$($download.Stderr)")
        }
        break
    }
    Start-Sleep -Seconds 30
}
if ([DateTime]::UtcNow -ge $MonitorDeadline) {
    Save-Text (Join-Path $MonitorRoot 'monitor_deadline.txt') '45-minute monitor limit reached; remote job was not cancelled.'
}


