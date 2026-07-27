# Inventory workspace before migration to F:\thetrueai
# Uses $PSScriptRoot to avoid any non-ASCII char in this script.
$ErrorActionPreference = "Stop"

# Script is at <workspace>\scripts\hb_inventory.ps1, so workspace = parent of scripts dir
$src = Split-Path -Parent $PSScriptRoot
Write-Host ("Workspace: " + $src) -ForegroundColor Cyan

Write-Host ""
Write-Host "===== Top-level entries =====" -ForegroundColor Cyan
Get-ChildItem -LiteralPath $src -Force | Select-Object Mode, Name, @{N="SizeMB";E={if ($_.PSIsContainer) { "" } else { [math]::Round($_.Length/1MB,2) }}}

Write-Host ""
Write-Host "===== Dir sizes (recursive) =====" -ForegroundColor Cyan
Get-ChildItem -LiteralPath $src -Force -Directory | ForEach-Object {
    $sz = (Get-ChildItem -LiteralPath $_.FullName -Recurse -Force -File -ErrorAction SilentlyContinue |
           Measure-Object -Property Length -Sum).Sum
    [PSCustomObject]@{
        Name   = $_.Name
        SizeMB = [math]::Round($sz/1MB, 2)
    }
} | Sort-Object SizeMB -Descending

Write-Host ""
Write-Host "===== Total file count =====" -ForegroundColor Cyan
$all = Get-ChildItem -LiteralPath $src -Recurse -Force -File -ErrorAction SilentlyContinue
Write-Host ("Files: " + $all.Count)
$totalSize = ($all | Measure-Object -Property Length -Sum).Sum
Write-Host ("Total size: " + [math]::Round($totalSize/1MB,2) + " MB")

Write-Host ""
Write-Host "===== .git presence =====" -ForegroundColor Cyan
$gitDir = Join-Path $src ".git"
if (Test-Path -LiteralPath $gitDir) {
    $gitSize = (Get-ChildItem -LiteralPath $gitDir -Recurse -Force -File -ErrorAction SilentlyContinue |
                Measure-Object -Property Length -Sum).Sum
    Write-Host ("[OK] .git exists, size: " + [math]::Round($gitSize/1MB,2) + " MB")
} else {
    Write-Host "[MISS] no .git directory"
}
