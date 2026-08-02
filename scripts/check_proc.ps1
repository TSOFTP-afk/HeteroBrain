$t1 = (Get-Process -Id 7500 -ErrorAction SilentlyContinue).CPU
Start-Sleep -Seconds 5
$p2 = Get-Process -Id 7500 -ErrorAction SilentlyContinue
$t2 = $p2.CPU
"CPU t1=" + [math]::Round($t1,1) + "s  t2=" + [math]::Round($t2,1) + "s  delta=" + [math]::Round($t2-$t1,1) + "s (5s wall)"
$th = $p2.Threads | Sort-Object -Property TotalProcessorTime -Descending | Select-Object -First 3 Id, ThreadState, WaitReason, @{n='CPU_s';e={[math]::Round($_.TotalProcessorTime.TotalSeconds,2)}}
$th | Format-Table
