# Migrate project from f:\项目\THE TRUE AI to F:\thetrueai via robocopy.
# Source path derived from $PSScriptRoot (script is at <workspace>\scripts\).
# All script content in ASCII.
$ErrorActionPreference = "Stop"

$src = Split-Path -Parent $PSScriptRoot
$dst = "F:\thetrueai"

Write-Host ("Source: " + $src) -ForegroundColor Cyan
Write-Host ("Dest:   " + $dst) -ForegroundColor Cyan

# Step 1: Create dest dir if missing
if (-not (Test-Path -LiteralPath $dst)) {
    New-Item -ItemType Directory -Path $dst -Force | Out-Null
    Write-Host "[OK] created $dst" -ForegroundColor Green
} else {
    Write-Host "[WARN] $dst already exists" -ForegroundColor Yellow
}

# Step 2: robocopy
# /E  include empty subdirs
# /COPY:DAT  data+attrib+timestamps (no ACL)
# /R:1 /W:1  retry once, wait 1s
# /MT:8  8 threads
# /NFL /NDL  no file/dir list (less noise)
# /NP  no progress percent
# /TEE  output to console AND log
$logFile = "F:\thetrueai_migration.log"

Write-Host ""
Write-Host "===== robocopy started =====" -ForegroundColor Yellow
$rc = Start-Process -FilePath "robocopy.exe" `
    -ArgumentList `
        "`"$src`"", `
        "`"$dst`"", `
        "/E", "/COPY:DAT", "/R:1", "/W:1", "/MT:8", `
        "/NFL", "/NDL", "/NP", "/TEE", `
        "/LOG+:$logFile" `
    -NoNewWindow -Wait -PassThru

Write-Host ("robocopy exit code: " + $rc.ExitCode) -ForegroundColor Cyan
# robocopy exit codes < 8 are success
if ($rc.ExitCode -ge 8) {
    throw "robocopy failed with exit code $($rc.ExitCode)"
} else {
    Write-Host "[OK] robocopy completed successfully" -ForegroundColor Green
}

# Step 3: Verify
Write-Host ""
Write-Host "===== Verification =====" -ForegroundColor Cyan

$srcFiles = Get-ChildItem -LiteralPath $src -Recurse -Force -File -ErrorAction SilentlyContinue
$dstFiles = Get-ChildItem -LiteralPath $dst -Recurse -Force -File -ErrorAction SilentlyContinue

$srcCount = $srcFiles.Count
$dstCount = $dstFiles.Count
$srcSize  = ($srcFiles | Measure-Object -Property Length -Sum).Sum
$dstSize  = ($dstFiles | Measure-Object -Property Length -Sum).Sum

Write-Host ("Source files: " + $srcCount + ", size: " + [math]::Round($srcSize/1MB,2) + " MB")
Write-Host ("Dest   files: " + $dstCount + ", size: " + [math]::Round($dstSize/1MB,2) + " MB")
Write-Host ("File count diff: " + ($srcCount - $dstCount))
Write-Host ("Size diff (MB): " + [math]::Round(($srcSize - $dstSize)/1MB,4))

# Step 4: Check key files
Write-Host ""
Write-Host "===== Key files check =====" -ForegroundColor Cyan
$keyFiles = @(
    "PROJECT_MEMORY.md",
    "README.md",
    ".gitignore",
    "CMakeLists.txt",
    "LICENSE",
    "llama.cpp-master.zip",
    "docs\migration-to-F-thetrueai.md",
    "docs\roadmap.md",
    "models\MiniCPM5-1B-Q4_K_M.gguf",
    "scripts\hb_run_test.ps1",
    "scripts\hb_build_cli.ps1",
    "scripts\hb_extract_template.ps1",
    "scripts\hb_inventory.ps1",
    "scripts\hb_probe_docs.ps1"
)

$missing = 0
foreach ($kf in $keyFiles) {
    $p = Join-Path $dst $kf
    if (Test-Path -LiteralPath $p) {
        $sz = (Get-Item -LiteralPath $p).Length
        Write-Host ("  [OK] " + $kf + " (" + $sz + " bytes)") -ForegroundColor Green
    } else {
        Write-Host ("  [MISS] " + $kf) -ForegroundColor Red
        $missing++
    }
}

if ($missing -gt 0) {
    Write-Host ""
    Write-Host "[FAIL] $missing key file(s) missing" -ForegroundColor Red
} else {
    Write-Host ""
    Write-Host "[OK] All key files present" -ForegroundColor Green
}

# Step 5: gguf size check
$gguf = Join-Path $dst "models\MiniCPM5-1B-Q4_K_M.gguf"
if (Test-Path -LiteralPath $gguf) {
    $ggufSize = (Get-Item -LiteralPath $gguf).Length
    $expected = 688065920
    if ($ggufSize -eq $expected) {
        Write-Host ("[OK] GGUF size matches: " + $ggufSize + " bytes") -ForegroundColor Green
    } else {
        Write-Host ("[WARN] GGUF size mismatch: got " + $ggufSize + ", expected " + $expected) -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "===== DONE =====" -ForegroundColor Yellow
Write-Host ("Log file: " + $logFile)
