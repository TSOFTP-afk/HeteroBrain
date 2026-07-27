# Rebuild llama.cpp with LLAMA_BUILD_TOOLS=ON to get llama-cli.
# llama-cli supports --chat-template-file, which simple-chat does not.
# All paths in ASCII to avoid PS 5.x encoding bugs.
$ErrorActionPreference = "Stop"

$vsShell = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1"
$cmakeDir = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
$env:PATH = "$cmakeDir;$env:PATH"

Write-Host "=== Step 1: Load VS DevShell (amd64) ===" -ForegroundColor Cyan
& $vsShell -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
if ($LASTEXITCODE -ne 0) { throw "vsdevshell load failed" }

$src   = "F:\hb_llama"
$build = "F:\hb_build"

# Force re-configure by removing cache and build.ninja (keep dir for incremental obj)
Write-Host ""
Write-Host "=== Step 2: Force re-configure (remove cache + build.ninja) ===" -ForegroundColor Cyan
if (Test-Path (Join-Path $build "CMakeCache.txt")) {
    Remove-Item (Join-Path $build "CMakeCache.txt") -Force
}
if (Test-Path (Join-Path $build "build.ninja")) {
    Remove-Item (Join-Path $build "build.ninja") -Force
}
# Remove CMakeFiles to avoid stale cache
$cmakeFiles = Join-Path $build "CMakeFiles"
if (Test-Path $cmakeFiles) {
    Remove-Item $cmakeFiles -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "=== Step 3: cmake configure (TOOLS=ON, EXAMPLES=ON, SERVER=OFF) ===" -ForegroundColor Cyan
& cmake -S $src -B $build `
    -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DGGML_CUDA=ON `
    -DGGML_CUDA_ARCH=86 `
    -DLLAMA_CURL=OFF `
    -DGGML_RPC=OFF `
    -DBUILD_SHARED_LIBS=OFF `
    -DLLAMA_BUILD_COMMON=ON `
    -DLLAMA_BUILD_TOOLS=ON `
    -DLLAMA_BUILD_EXAMPLES=ON `
    -DLLAMA_BUILD_TESTS=OFF `
    -DLLAMA_BUILD_SERVER=OFF `
    -DLLAMA_BUILD_UI=OFF `
    -DLLAMA_USE_PREBUILT_UI=OFF
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

Write-Host ""
Write-Host "=== Step 4: Verify llama-cli target in build.ninja ===" -ForegroundColor Cyan
$ninja = Get-Content (Join-Path $build "build.ninja") -Raw
$cliMatches = [regex]::Matches($ninja, 'build [^:]*llama-cli[^:]*: ')
if ($cliMatches.Count -gt 0) {
    Write-Host "[OK] llama-cli target found ($($cliMatches.Count) rules)" -ForegroundColor Green
    $cliMatches | Select-Object -First 3 | ForEach-Object { Write-Host "  $($_.Value)" }
} else {
    throw "llama-cli target NOT found in build.ninja after configure"
}

Write-Host ""
Write-Host "=== Step 5: ninja llama-cli (this may take 5-15 min) ===" -ForegroundColor Cyan
& ninja -C $build llama-cli
if ($LASTEXITCODE -ne 0) { throw "ninja llama-cli failed" }

Write-Host ""
Write-Host "=== Step 6: Verify output ===" -ForegroundColor Cyan
$exe = Join-Path $build "bin\llama-cli.exe"
if (Test-Path $exe) {
    $f = Get-Item $exe
    Write-Host ("[OK] " + $f.FullName + " (" + [math]::Round($f.Length/1MB,2) + " MB)") -ForegroundColor Green
} else {
    throw "llama-cli.exe not found at $exe"
}

Write-Host ""
Write-Host "=== DONE ===" -ForegroundColor Yellow
