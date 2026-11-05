$ErrorActionPreference = 'Stop'
$runRoot = 'D:\Bunker\BaseCamp\WirelessCommunication\SemanticCommunication\SemanticSchedulerEnd2End\experiments\parameter_finalization\runs\notebook9_20261028'
$monitorRoot = Join-Path $runRoot 'monitor_v13'
$retrieveRoot = Join-Path $runRoot 'submission_v13_features\retrieved_version13'
$kernel = 'satyapaladugu/betterscheduler-spi-20261027finalc/13'
$kaggleExe = 'C:\Users\satya\AppData\Roaming\Python\Python314\Scripts\kaggle.exe'
New-Item -ItemType Directory -Path $monitorRoot -Force | Out-Null
New-Item -ItemType Directory -Path $retrieveRoot -Force | Out-Null
$historyPath = Join-Path $monitorRoot 'status_changes.jsonl'
$latestLogPath = Join-Path $monitorRoot 'kaggle_logs_v13_latest.txt'
$lastStatus = ''
$lastLogHash = ''
$started = [DateTime]::UtcNow
$deadline = $started.AddHours(3)
function Invoke-KaggleBounded([string[]]$arguments, [int]$timeoutSeconds) {
    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $kaggleExe
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8
    $psi.Arguments = [string]::Join(' ', @($arguments | ForEach-Object { '"' + $_.Replace('"','\"') + '"' }))
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $psi
    [void]$process.Start()
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    if (-not $process.WaitForExit($timeoutSeconds * 1000)) {
        try { $process.Kill() } catch {}
        return [pscustomobject]@{ timedOut = $true; exitCode = $null; stdout = $stdoutTask.Result; stderr = $stderrTask.Result }
    }
    $process.WaitForExit()
    return [pscustomobject]@{ timedOut = $false; exitCode = $process.ExitCode; stdout = $stdoutTask.Result; stderr = $stderrTask.Result }
}
function Save-TextUtf8([string]$path, [string]$value) {
    [System.IO.File]::WriteAllText($path, $value, [System.Text.UTF8Encoding]::new($false))
}
while ([DateTime]::UtcNow -lt $deadline) {
    $statusResult = Invoke-KaggleBounded @('kernels','status',$kernel) 30
    $statusText = ($statusResult.stdout + $statusResult.stderr).Trim()
    $match = [regex]::Match($statusText, 'KernelWorkerStatus\.([A-Za-z]+)')
    $state = if ($match.Success) { $match.Groups[1].Value } else { 'UNKNOWN' }
    if ($state -ne $lastStatus) {
        $event = [ordered]@{ timestamp_utc = [DateTime]::UtcNow.ToString('o'); status = $state; status_output = $statusText; timed_out = $statusResult.timedOut }
        Add-Content -LiteralPath $historyPath -Value ($event | ConvertTo-Json -Compress -Depth 4) -Encoding utf8
        $lastStatus = $state
    }
    $logsResult = Invoke-KaggleBounded @('kernels','logs',$kernel) 45
    $logsText = $logsResult.stdout + $logsResult.stderr
    if ($logsText -match 'stream_name|START |papermill|FileNotFoundError|RuntimeError|Traceback|END ') {
        $hasher = [Security.Cryptography.SHA256]::Create(); $logHash = [BitConverter]::ToString($hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($logsText))).Replace('-','').ToLowerInvariant(); $hasher.Dispose()
        if ($logHash -ne $lastLogHash) {
            Save-TextUtf8 $latestLogPath $logsText
            $snapshot = Join-Path $monitorRoot ('kaggle_logs_v13_' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') + '.txt')
            Save-TextUtf8 $snapshot $logsText
            $lastLogHash = $logHash
        }
    }
    if ($state -in @('COMPLETE','ERROR','CANCELLED','FAILED')) {
        $pull = Invoke-KaggleBounded @('kernels','output',$kernel,'-p',$retrieveRoot,'-o') 600
        Save-TextUtf8 (Join-Path $monitorRoot 'output_retrieval_stdout.txt') $pull.stdout
        Save-TextUtf8 (Join-Path $monitorRoot 'output_retrieval_stderr.txt') $pull.stderr
        $manifestResults = @()
        $checksumManifests = Get-ChildItem -LiteralPath $retrieveRoot -Recurse -File -Filter 'retrieval_checksums.json' -ErrorAction SilentlyContinue
        foreach ($manifestFile in $checksumManifests) {
            $manifest = Get-Content -LiteralPath $manifestFile.FullName -Raw | ConvertFrom-Json
            foreach ($entry in $manifest.PSObject.Properties) {
                $target = Join-Path $manifestFile.DirectoryName $entry.Name
                if (-not (Test-Path -LiteralPath $target -PathType Leaf)) { $result = 'missing' }
                else {
                    $actual = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
                    $result = if ($actual -eq $entry.Value) { 'match' } else { 'mismatch' }
                }
                $manifestResults += [ordered]@{ path = $entry.Name; result = $result; expected_sha256 = $entry.Value }
            }
        }
        $files = @(Get-ChildItem -LiteralPath $retrieveRoot -Recurse -File -ErrorAction SilentlyContinue)
        $recovery = @($files | Where-Object { $_.Name -eq 'run_recovery.zip' })
        $terminal = [ordered]@{ terminal_utc = [DateTime]::UtcNow.ToString('o'); status = $state; log_path = $latestLogPath; log_sha256 = $lastLogHash; retrieval_timed_out = $pull.timedOut; retrieval_exit_code = $pull.exitCode; retrieved_file_count = $files.Count; checksum_results = $manifestResults; recovery_zip_paths = @($recovery | ForEach-Object { $_.FullName }); monitor_deadline_utc = $deadline.ToString('o') }
        $terminal | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $monitorRoot 'terminal_receipt.json') -Encoding utf8
        exit 0
    }
    Start-Sleep -Seconds 45
}
$timeout = [ordered]@{ deadline_utc = $deadline.ToString('o'); stopped_utc = [DateTime]::UtcNow.ToString('o'); last_status = $lastStatus; note = 'Local monitor deadline reached; remote run was not cancelled.' }
$timeout | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $monitorRoot 'monitor_timeout.json') -Encoding utf8




