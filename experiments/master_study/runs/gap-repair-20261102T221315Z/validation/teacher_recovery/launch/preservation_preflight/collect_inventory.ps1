param(
    [Parameter(Mandatory=$true)][string]$Base,
    [Parameter(Mandatory=$true)][ValidateSet('wsl_v12','windows_v1')][string]$Target
)
function Parse-Page([string]$Path) {
    $raw=Get-Content -Raw -LiteralPath $Path
    $token=[regex]::Match($raw,'Next Page Token = ([^\r\n]+)').Groups[1].Value
    $start=$raw.IndexOf('[')
    if($start -lt 0){throw "No JSON array in $Path"}
    $items=@($raw.Substring($start) | ConvertFrom-Json)
    return @{Items=$items;Token=$token}
}
if($Target -eq 'wsl_v12'){
    $kernel='satyapaladuguplus/semantic-decoder-scheduler-20261030/12'
    $prefix='wsl_v12'
    $mode='wsl'
}else{
    $kernel='satyapaladugu/semantic-decoder-scheduler-rl/1'
    $prefix='windows_v1'
    $mode='windows'
}
$pages=New-Object System.Collections.Generic.List[object]
$all=New-Object System.Collections.Generic.List[object]
for($n=1;$n -le 2;$n++){
    $path=Join-Path $Base ($prefix+'_files200.stdout.txt')
    if($n -eq 2){$path=Join-Path $Base ($prefix+'_page2.stdout.txt')}
    $parsed=Parse-Page $path
    foreach($item in $parsed.Items){$all.Add($item)}
    $pages.Add(@{number=$n;items=$parsed.Items.Count;source=$path})
    $token=$parsed.Token
}
$page=3
while($token){
    if($page -gt 20){throw "Refusing to follow more than 20 pages for $Target"}
    $outPath=Join-Path $Base ($prefix+'_page'+$page+'.stdout.txt')
    $errPath=Join-Path $Base ($prefix+'_page'+$page+'.stderr.log')
    $exitPath=Join-Path $Base ($prefix+'_page'+$page+'.exit_code.txt')
    $cmdPath=Join-Path $Base ($prefix+'_page'+$page+'.command.txt')
    $command="kaggle kernels files $kernel --page-size 200 --page-token $token --format json"
    Set-Content -LiteralPath $cmdPath -Value $command
    if($mode -eq 'wsl'){
        wsl.exe -d Ubuntu-26.04 -u Satya -- /home/Satya/.venvs/semantic-kaggle/bin/kaggle kernels files $kernel --page-size 200 --page-token $token --format json > $outPath 2> $errPath
    }else{
        & 'C:\Users\satya\AppData\Roaming\Python\Python314\Scripts\kaggle.exe' kernels files $kernel --page-size 200 --page-token $token --format json > $outPath 2> $errPath
    }
    $rc=$LASTEXITCODE
    Set-Content -LiteralPath $exitPath -Value $rc
    if($rc -ne 0){throw "Kaggle inventory page $page failed for $Target; see $errPath"}
    $parsed=Parse-Page $outPath
    foreach($item in $parsed.Items){$all.Add($item)}
    $pages.Add(@{number=$page;items=$parsed.Items.Count;source=$outPath})
    $token=$parsed.Token
    $page++
}
$duplicateNames=@($all | Group-Object name | Where-Object Count -gt 1 | ForEach-Object Name)
$total=($all | Measure-Object -Property size -Sum).Sum
$pagesOut=@($pages.ToArray())
$filesOut=@($all.ToArray())
$inventory=[ordered]@{
  target=$Target
  kernel=$kernel
  pages=$pagesOut
  total_files=$filesOut.Count
  total_bytes=$total
  duplicate_names=$duplicateNames
  files=$filesOut
}
ConvertTo-Json -InputObject $inventory -Depth 8 | Set-Content -LiteralPath (Join-Path $Base ($prefix+'_inventory.json'))
$selected=@($filesOut | Where-Object { $_.name -match '(^|/)(model\.json|resume\.pt|.*checkpoint.*|.*actor.*|.*manifest.*|.*config.*|.*status\.json|.*history.*|.*optimizer.*|.*rng.*)$' })
$selected | Select-Object name,size | Sort-Object name | Export-Csv -NoTypeInformation -LiteralPath (Join-Path $Base ($prefix+'_selected_models.csv'))
[pscustomobject]@{target=$Target;pages=$pages.Count;files=$all.Count;total_bytes=$total;selected_model_files=$selected.Count;inventory=($prefix+'_inventory.json')} | Format-List


