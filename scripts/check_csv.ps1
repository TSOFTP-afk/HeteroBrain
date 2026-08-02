$f = 'F:\thetrueai\run_curriculum_n3f_scratch_20k.csv'
$data = Import-Csv $f | Where-Object { $_.step -match '^\d+$' }
"rows=" + $data.Count
$cols = @('da','ach','ne','ht5','gaba','oxy','pleasure','arousal','dominance','temp_delta','empathy')
function Seg($rows, $label) {
    $out = "[$label] n=" + $rows.Count
    foreach ($c in $cols) {
        $vals = $rows | ForEach-Object { [double]$_.$c }
        $m = ($vals | Measure-Object -Average -Minimum -Maximum)
        $std = 0.0
        $mean = $m.Average
        if ($vals.Count -gt 1) {
            $sq = ($vals | ForEach-Object { ($_ - $mean) * ($_ - $mean) } | Measure-Object -Sum).Sum
            $std = [math]::Sqrt($sq / ($vals.Count - 1))
        }
        $out += "`n  " + $c + " mean=" + [math]::Round($mean,3) + " std=" + [math]::Round($std,3) + " min=" + [math]::Round($m.Minimum,3) + " max=" + [math]::Round($m.Maximum,3)
    }
    return $out
}
Seg ($data | Where-Object { [int]$_.step -lt 5000 }) "0-5K"
Seg ($data | Where-Object { [int]$_.step -ge 5000 -and [int]$_.step -lt 10000 }) "5-10K"
Seg ($data | Where-Object { [int]$_.step -ge 10000 -and [int]$_.step -lt 15000 }) "10-15K"
Seg ($data | Where-Object { [int]$_.step -ge 15000 }) "15-20K"
