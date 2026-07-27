# Auto test for llama-simple-chat with MiniCPM5-1B-Q4_K_M
# All paths and script content in ASCII to avoid PS 5.x encoding bugs.
# Prompt is built from Unicode codepoints so the .ps1 has zero non-ASCII chars.
$ErrorActionPreference = "Stop"

$inputFile  = "F:\hb_input.txt"
$outputFile = "F:\hb_output.txt"
$errorFile  = "F:\hb_error.txt"

# Build prompt string from codepoints (avoids any non-ASCII char in this script)
$prompt = ([char]0x4F60).ToString() + ([char]0x597D).ToString() + "," + `
          ([char]0x8BF7).ToString() + ([char]0x7528).ToString() + `
          ([char]0x4E2D).ToString() + ([char]0x6587).ToString() + `
          ([char]0x81EA).ToString() + ([char]0x6211).ToString() + `
          ([char]0x4ECB).ToString() + ([char]0x7ECD).ToString() + "`n"

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($inputFile, $prompt, $utf8NoBom)
Write-Host "[OK] input file created: $inputFile"

if (Test-Path $outputFile) { Remove-Item $outputFile -Force }
if (Test-Path $errorFile)  { Remove-Item $errorFile -Force }

$exe   = "F:\hb_build\bin\llama-simple-chat.exe"
$model = "F:\hb_models\MiniCPM5-1B-Q4_K_M.gguf"

Write-Host "[RUN] starting llama-simple-chat -ngl 99 -c 2048 ..."
$proc = Start-Process -FilePath $exe `
    -ArgumentList "-m",$model,"-ngl","99","-c","2048" `
    -NoNewWindow `
    -RedirectStandardInput  $inputFile `
    -RedirectStandardOutput $outputFile `
    -RedirectStandardError  $errorFile `
    -PassThru

Write-Host "[WAIT] waiting up to 60s for inference (PID=$($proc.Id))..."
$proc | Wait-Process -Timeout 60 -ErrorAction SilentlyContinue

if (-not $proc.HasExited) {
    Write-Host "[KILL] process still running, killing..."
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

Write-Host ""
Write-Host "===== STDERR (load log) =====" -ForegroundColor Cyan
if (Test-Path $errorFile) {
    Get-Content $errorFile -Encoding UTF8 | Select-Object -First 80
}

Write-Host ""
Write-Host "===== STDOUT (model reply) =====" -ForegroundColor Cyan
if (Test-Path $outputFile) {
    Get-Content $outputFile -Encoding UTF8
}

Write-Host ""
Write-Host "[DONE] proc exited: $($proc.HasExited), exit code: $($proc.ExitCode)" -ForegroundColor Yellow
