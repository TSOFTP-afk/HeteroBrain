# 克隆 llama.cpp 国内镜像
$ErrorActionPreference = "Continue"
$env:GIT_TERMINAL_PROMPT = "0"
$env:GIT_LFS_SKIP_SMUDGE = "1"

if (Test-Path "third_party/llama.cpp") {
    Remove-Item -Recurse -Force "third_party/llama.cpp"
}
if (-not (Test-Path "third_party")) {
    New-Item -ItemType Directory -Path "third_party" | Out-Null
}

$mirrors = @(
    "https://kkgithub.com/ggml-org/llama.cpp.git",
    "https://gitclone.com/github.com/ggml-org/llama.cpp.git",
    "https://gh-proxy.com/https://github.com/ggml-org/llama.cpp.git",
    "https://mirror.ghproxy.com/https://github.com/ggml-org/llama.cpp.git"
)

foreach ($m in $mirrors) {
    Write-Host "=== 尝试: $m ===" -ForegroundColor Cyan
    if (Test-Path "third_party/llama.cpp") {
        Remove-Item -Recurse -Force "third_party/llama.cpp"
    }
    & git clone --depth 1 $m third_party/llama.cpp 2>&1 | ForEach-Object { Write-Host $_ }
    if (Test-Path "third_party/llama.cpp/CMakeLists.txt") {
        $size = (Get-ChildItem -Recurse third_party/llama.cpp | Measure-Object -Property Length -Sum).Sum
        Write-Host "OK: $m ($([math]::Round($size/1MB,1)) MB)" -ForegroundColor Green
        exit 0
    }
}

Write-Host "所有镜像均失败" -ForegroundColor Red
exit 1
