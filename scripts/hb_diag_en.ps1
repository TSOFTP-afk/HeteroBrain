# Diagnose MiniCPM5-1B output garbled issue.
# Test 1: list built exe files.
# Test 2: run with ASCII English prompt to isolate whether issue is
#         Chinese-specific or model-wide.
$ErrorActionPreference = "Stop"

Write-Host "===== Binaries in F:\hb_build\bin =====" -ForegroundColor Cyan
Get-ChildItem "F:\hb_build\bin" -Filter "*.exe" |
    Select-Object Name, @{N="SizeMB";E={[math]::Round($_.Length/1MB,2)}}

$inputFile  = "F:\hb_input_en.txt"
$outputFile = "F:\hb_output_en.txt"
$errorFile  = "F:\hb_error_en.txt"

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($inputFile,
    "Hello, please introduce yourself in 3 sentences.`n", $utf8NoBom)

if (Test-Path $outputFile) { Remove-Item $outputFile -Force }
if (Test-Path $errorFile)  { Remove-Item $errorFile -Force }

$exe   = "F:\hb_build\bin\llama-simple-chat.exe"
$model = "F:\hb_models\MiniCPM5-1B-Q4_K_M.gguf"

Write-Host ""
Write-Host "[RUN] English prompt test ..." -ForegroundColor Yellow
$proc = Start-Process -FilePath $exe `
    -ArgumentList "-m",$model,"-ngl","99","-c","2048" `
    -NoNewWindow `
    -RedirectStandardInput  $inputFile `
    -RedirectStandardOutput $outputFile `
    -RedirectStandardError  $errorFile `
    -PassThru

$proc | Wait-Process -Timeout 60 -ErrorAction SilentlyContinue
if (-not $proc.HasExited) {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

Write-Host ""
Write-Host "===== English output (raw bytes -> UTF-8) =====" -ForegroundColor Cyan
if (Test-Path $outputFile) {
    $b = [System.IO.File]::ReadAllBytes($outputFile)
    Write-Host ("Total bytes: " + $b.Length)
    [System.Text.Encoding]::UTF8.GetString($b)
}

Write-Host ""
Write-Host "===== English stderr (first 40 lines) =====" -ForegroundColor Cyan
if (Test-Path $errorFile) {
    (Get-Content $errorFile -Encoding UTF8) | Select-Object -First 40
}
