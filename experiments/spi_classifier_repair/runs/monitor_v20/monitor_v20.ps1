$ErrorActionPreference = 'Stop'

$kernel = 'satyapaladugu/betterscheduler-spi-20261027finalc/20'
$root = $PSScriptRoot
$started = [DateTime]::UtcNow
$deadline = $started.AddMinutes(60)
$history = Join-Path $root 'status_history.jsonl'
$logsDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$lastLogHash = ''

function Invoke-KaggleCapture([string[]]$Arguments) {
    $output = & kaggle @Arguments 2>&1
    $exitCode = $LASTEXITCODE
    return [pscustomobject]@{
        exit_code = $exitCode
        text = (($output | ForEach-Object { "$_" }) -join "`n")
    }
}

while ([DateTime]::UtcNow -lt $deadline) {
    $now = [DateTime]::UtcNow
    $stamp = $now.ToString('yyyyMMddTHHmmssZ')
    $statusResult = Invoke-KaggleCapture @('kernels', 'status', $kernel)
    $logResult = Invoke-KaggleCapture @('kernels', 'logs', $kernel)
    $statusText = $statusResult.text
    $logText = $logResult.text
    $logBody = (($logText -split "`n" | Where-Object { $_ -notmatch '^Warning: Looks like you''re using an outdated `kaggle` version' }) -join "`n").Trim()
    $terminal = $null
    if ($statusText -match 'KernelWorkerStatus\.COMPLETE') { $terminal = 'COMPLETE' }
    elseif ($statusText -match 'KernelWorkerStatus\.ERROR') { $terminal = 'ERROR' }
    elseif ($statusText -match 'KernelWorkerStatus\.CANCEL') { $terminal = 'CANCELLED' }

    $logHash = ''
    $logChanged = $false
    if ($logBody) {
        $sha = [System.Security.Cryptography.SHA256]::Create()
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($logBody)
        $logHash = [Convert]::ToHexString($sha.ComputeHash($bytes)).ToLowerInvariant()
        if ($logHash -ne $lastLogHash) {
            Set-Content -LiteralPath (Join-Path $logsDir "logs_$stamp.txt") -Value $logBody -Encoding UTF8
            Set-Content -LiteralPath (Join-Path $root 'logs_latest.txt') -Value $logBody -Encoding UTF8
            $lastLogHash = $logHash
            $logChanged = $true
        }
    }
    $entry = [ordered]@{
        observed_utc = $now.ToString('o')
        kernel = $kernel
        version = 20
        status_exit_code = $statusResult.exit_code
        status_text = $statusText
        status_query_error = ($statusResult.exit_code -ne 0)
        logs_exit_code = $logResult.exit_code
        logs_sha256 = $logHash
        logs_bytes = [System.Text.Encoding]::UTF8.GetByteCount($logBody)
        terminal = $terminal
    }
    Add-Content -LiteralPath $history -Value ($entry | ConvertTo-Json -Compress)
    if ($terminal) {
        $terminalReceipt = [ordered]@{
            kernel = $kernel
            version = 20
            terminal = $terminal
            terminal_observed_utc = $now.ToString('o')
            status_exit_code = $statusResult.exit_code
            status_text = $statusText
            logs_exit_code = $logResult.exit_code
            logs_sha256 = $logHash
            monitor_started_utc = $started.ToString('o')
            monitor_deadline_utc = $deadline.ToString('o')
        }
        $terminalReceipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $root 'terminal_receipt.json') -Encoding UTF8
        Write-Output "TERMINAL $terminal $($now.ToString('o'))"
        Write-Output $statusText
        if ($logBody) { Write-Output "LOGS_SNAPSHOT $logHash $($logBody.Length)" }
        exit 0
    }
    if ($statusResult.exit_code -ne 0 -or $statusText -notmatch 'KernelWorkerStatus\.RUNNING') {
        Write-Output "STATUS $($now.ToString('o')) exit=$($statusResult.exit_code) $statusText"
    }
    if ($logChanged) {
        Write-Output "LOGS_UPDATED $($now.ToString('o')) $logHash $($logBody.Length)"
    }
    Start-Sleep -Seconds 60
}

$timeoutReceipt = [ordered]@{
    kernel = $kernel
    version = 20
    terminal = 'MONITOR_TIME_CAP'
    terminal_observed_utc = [DateTime]::UtcNow.ToString('o')
    monitor_started_utc = $started.ToString('o')
    monitor_deadline_utc = $deadline.ToString('o')
}
$timeoutReceipt | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $root 'terminal_receipt.json') -Encoding UTF8
Write-Output "MONITOR_TIME_CAP $([DateTime]::UtcNow.ToString('o'))"
