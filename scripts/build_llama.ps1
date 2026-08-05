# Build llama.cpp with CUDA for vita (RTX 3060 sm_86)
# Loads VS 2022 x64 dev shell, runs cmake + ninja, outputs llama-cli.exe
$ErrorActionPreference = "Stop"

$vsShell = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1"
$cmakeDir = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
$env:PATH = "$cmakeDir;$env:PATH"

Write-Host "=== Step 1: Load VS DevShell (amd64) ===" -ForegroundColor Cyan
& $vsShell -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
if ($LASTEXITCODE -ne 0) { throw "vsdevshell load failed" }

Write-Host ""
Write-Host "=== Step 2: Verify tools ===" -ForegroundColor Cyan
foreach ($t in @("cmake", "ninja", "nvcc", "cl")) {
    $p = Get-Command $t -ErrorAction SilentlyContinue
    if ($p) { Write-Host "  ${t}: $($p.Source)" -ForegroundColor Green }
    else { Write-Host "  ${t}: NOT FOUND" -ForegroundColor Red; throw "missing $t" }
}

Write-Host ""
Write-Host "=== Step 3: CMake configure (CUDA + sm_86) ===" -ForegroundColor Cyan
$src = "F:\hb_llama"
$build = "F:\hb_build"
$origin = "f:\项目\THE TRUE AI\third_party\llama.cpp"

# Verify origin exists
if (-not (Test-Path "$origin\CMakeLists.txt")) {
    throw "origin not found: $origin\CMakeLists.txt"
}

# Copy llama.cpp source to ASCII path using .NET API
# (avoids Chinese path encoding issues with robocopy/cmake/nvcc/msvc
#  which silently corrupt paths containing CJK chars in PS 5.x)
if (-not (Test-Path "$src\CMakeLists.txt")) {
    Write-Host "  Copying llama.cpp source to F:\hb_llama (ASCII path)..."
    if (Test-Path $src) { Remove-Item -Recurse -Force $src }
    New-Item -ItemType Directory -Path $src -Force | Out-Null

    # .NET recursive copy, skipping build/ and .git/ dirs
    $excludeDirs = @("build", ".git")
    function Copy-DirRecursive($srcPath, $dstPath, $exclude) {
        New-Item -ItemType Directory -Path $dstPath -Force | Out-Null
        foreach ($item in [System.IO.Directory]::EnumerateFileSystemEntries($srcPath, "*", [System.IO.SearchOption]::TopDirectoryOnly)) {
            $name = [System.IO.Path]::GetFileName($item)
            $dstItem = [System.IO.Path]::Combine($dstPath, $name)
            if ([System.IO.Directory]::Exists($item)) {
                if ($exclude -notcontains $name) {
                    Copy-DirRecursive $item $dstItem $exclude
                }
            } else {
                [System.IO.File]::Copy($item, $dstItem, $true)
            }
        }
    }
    Copy-DirRecursive $origin $src $excludeDirs
    $fileCount = (Get-ChildItem $src -Recurse -File).Count
    Write-Host "  Copy done: $fileCount files"
}

# Reuse existing build dir if configured (skip reconfigure)
if (-not (Test-Path "$build\build.ninja")) {
    if (Test-Path $build) { Remove-Item -Recurse -Force $build }
    New-Item -ItemType Directory -Path $build -Force | Out-Null
    & cmake -S $src -B $build `
        -G Ninja `
        -DCMAKE_BUILD_TYPE=Release `
        -DGGML_CUDA=ON `
        -DGGML_CUDA_ARCH=86 `
        -DLLAMA_CURL=OFF `
        -DGGML_RPC=OFF `
        -DBUILD_SHARED_LIBS=OFF `
        -DLLAMA_BUILD_EXAMPLES=ON `
        -DLLAMA_BUILD_TESTS=OFF `
        -DLLAMA_BUILD_SERVER=OFF
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }
} else {
    Write-Host "  build.ninja exists, skipping configure" -ForegroundColor Yellow
}

Set-Location $build

Write-Host ""
Write-Host "=== Step 4: Build llama-simple-chat (5-10 min) ===" -ForegroundColor Cyan
& cmake --build . --target llama-simple-chat --config Release -j
if ($LASTEXITCODE -ne 0) { throw "build failed" }

Write-Host ""
Write-Host "=== Step 5: Verify artifact ===" -ForegroundColor Cyan
$cli = "$build\bin\llama-simple-chat.exe"
if (Test-Path $cli) {
    Write-Host "OK: $cli" -ForegroundColor Green
    Write-Host "Size: $([math]::Round((Get-Item $cli).Length/1MB,1)) MB"
    & $cli --version
} else {
    Write-Host "WARN: llama-simple-chat.exe not at expected path, listing bin/:" -ForegroundColor Yellow
    Get-ChildItem "$build\bin\*.exe" -ErrorAction SilentlyContinue | Select-Object Name, Length
}

Write-Host ""
Write-Host "=== DONE ===" -ForegroundColor Green
