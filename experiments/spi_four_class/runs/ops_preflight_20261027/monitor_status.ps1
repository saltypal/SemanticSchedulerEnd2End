param([int]$IntervalSeconds=35,[int]$MaximumMinutes=145)
$ErrorActionPreference='Continue'
$kernel='satyapaladugu/betterscheduler-spi-four-class-evidence-pipeline'
$log=Join-Path $PSScriptRoot 'monitor_status_changes.jsonl'
$deadline=[DateTime]::UtcNow.AddMinutes($MaximumMinutes)
$previous=$null
while([DateTime]::UtcNow -lt $deadline){
  $stamp=[DateTime]::UtcNow.ToString('o')
  $lines=& kaggle kernels status $kernel 2>&1
  $raw=($lines -join "`n")
  $m=[regex]::Match($raw,'KernelWorkerStatus\.([A-Z_]+)')
  $state=if($m.Success){$m.Groups[1].Value}else{'UNKNOWN'}
  if($state -ne $previous){
    $row=[ordered]@{utc=$stamp;kernel=$kernel;state=$state;detail=($raw -replace 'Warning: Looks like.*?\r?\n','').Trim()}
    $json=$row|ConvertTo-Json -Compress
    Add-Content -LiteralPath $log -Value $json -Encoding utf8
    Write-Output $json
    $previous=$state
  }
  if($state -in @('COMPLETE','ERROR','CANCELLED','CANCELED','FAILED')){exit 0}
  Start-Sleep -Seconds $IntervalSeconds
}
$row=[ordered]@{utc=[DateTime]::UtcNow.ToString('o');kernel=$kernel;state='MONITOR_TIMEOUT';detail='bounded monitor deadline reached; no job cancellation attempted'}
$json=$row|ConvertTo-Json -Compress;Add-Content -LiteralPath $log -Value $json -Encoding utf8;Write-Output $json;exit 2
