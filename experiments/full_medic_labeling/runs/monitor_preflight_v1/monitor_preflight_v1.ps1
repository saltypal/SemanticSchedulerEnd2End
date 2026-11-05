$ErrorActionPreference = 'Stop'
$kernel = 'satyapaladugu/medic-full-spi-labeling/1'
$root = Join-Path (Get-Location) 'experiments/full_medic_labeling/runs/monitor_preflight_v1'
$history = Join-Path $root 'status_history.jsonl'
$lastLogHash = ''
$start = [DateTimeOffset]::UtcNow
$cap = [TimeSpan]::FromSeconds(1800)
while (([DateTimeOffset]::UtcNow - $start) -lt $cap) {
  $now = [DateTimeOffset]::UtcNow
  $statusText = (& kaggle kernels status $kernel 2>&1 | Out-String).Trim()
  $statusCode = $LASTEXITCODE
  $state = if ($statusText -match 'KernelWorkerStatus\.([A-Z_]+)') { $Matches[1] } else { 'UNKNOWN' }
  $logText = (& kaggle kernels logs $kernel 2>&1 | Out-String)
  $logCode = $LASTEXITCODE
  $sha = [Security.Cryptography.SHA256]::Create(); $logHash = ([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($logText)))).Replace("-","").ToLowerInvariant(); $sha.Dispose()
  $record = [ordered]@{ observed_utc=$now.ToString('o'); status=$state; status_exit=$statusCode; log_exit=$logCode; log_sha256=$logHash; log_bytes=[Text.Encoding]::UTF8.GetByteCount($logText) }
  Add-Content -LiteralPath $history -Value ($record | ConvertTo-Json -Compress)
  if ($logHash -ne $lastLogHash -and $logText.Trim()) {
    $stamp = $now.ToString('yyyyMMddTHHmmssZ')
    [IO.File]::WriteAllText((Join-Path $root "logs_$stamp.txt"), $logText, [Text.UTF8Encoding]::new($false))
    $lastLogHash = $logHash
    Write-Output "[$($now.ToString('o'))] status=$state log_bytes=$($record.log_bytes) log_snapshot=logs_$stamp.txt"
  } else {
    Write-Output "[$($now.ToString('o'))] status=$state log_bytes=$($record.log_bytes)"
  }
  if ($state -in @('COMPLETE','ERROR','CANCELLED','CANCELED','FAILED')) { break }
  Start-Sleep -Seconds 60
}
@{ observed_utc=[DateTimeOffset]::UtcNow.ToString('o'); final_status=if ($state -in @('COMPLETE','ERROR','CANCELLED','CANCELED','FAILED')) {$state} else {'TIME_CAP'}; run_elapsed_seconds=[math]::Round(([DateTimeOffset]::UtcNow-$start).TotalSeconds,1); kernel=$kernel } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $root 'monitor_terminal_receipt.json') -Encoding utf8

