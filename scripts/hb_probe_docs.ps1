# Probe what's inside docs/ (3.4GB) and models/ (656MB) before migration.
$ErrorActionPreference = "Stop"
$ws = Split-Path -Parent $PSScriptRoot

Write-Host "===== docs/ top-level entries =====" -ForegroundColor Cyan
Get-ChildItem -LiteralPath (Join-Path $ws "docs") -Force | Select-Object Mode, Name, @{N="SizeMB";E={if ($_.PSIsContainer) { "" } else { [math]::Round($_.Length/1MB,2) }}}

Write-Host ""
Write-Host "===== docs/ subdirs (recursive size) =====" -ForegroundColor Cyan
Get-ChildItem -LiteralPath (Join-Path $ws "docs") -Force -Directory | ForEach-Object {
    $sz = (Get-ChildItem -LiteralPath $_.FullName -Recurse -Force -File -ErrorAction SilentlyContinue |
           Measure-Object -Property Length -Sum).Sum
    [PSCustomObject]@{ Name = $_.Name; SizeMB = [math]::Round($sz/1MB,2) }
} | Sort-Object SizeMB -Descending

Write-Host ""
Write-Host "===== Top 15 largest files in workspace =====" -ForegroundColor Cyan
Get-ChildItem -LiteralPath $ws -Recurse -Force -File -ErrorAction SilentlyContinue |
    Sort-Object Length -Descending | Select-Object -First 15 |
    ForEach-Object {
        $rel = $_.FullName.Substring($ws.Length).TrimStart('\')
        [PSCustomObject]@{
            Path   = $rel
            SizeMB = [math]::Round($_.Length/1MB,2)
        }
    }

Write-Host ""
Write-Host "===== .gitignore content =====" -ForegroundColor Cyan
$gi = Join-Path $ws ".gitignore"
if (Test-Path -LiteralPath $gi) {
    Get-Content -LiteralPath $gi
} else {
    Write-Host "[MISS] no .gitignore"
}

Write-Host ""
Write-Host "===== git tracked files count =====" -ForegroundColor Cyan
Push-Location -LiteralPath $ws
try {
    $tracked = & git ls-files 2>$null
    Write-Host ("Tracked files: " + $tracked.Count)
    $trackedSize = ($tracked | ForEach-Object {
        $p = Join-Path $ws $_
        if (Test-Path -LiteralPath $p) { (Get-Item -LiteralPath $p).Length } else { 0 }
    } | Measure-Object -Sum).Sum
    Write-Host ("Tracked total size: " + [math]::Round($trackedSize/1MB,2) + " MB")
} finally {
    Pop-Location
}
