# Binary-search the GGUF file for "chat_template" and other metadata keys.
# Also list llama.cpp examples/main to plan llama-cli rebuild.
$ErrorActionPreference = "Stop"

$gguf = "F:\hb_models\MiniCPM5-1B-Q4_K_M.gguf"
$bytes = [System.IO.File]::ReadAllBytes($gguf)
Write-Host ("GGUF size: " + $bytes.Length + " bytes") -ForegroundColor Cyan

# Convert to ASCII string (lossy, but enough for keyword search)
$ascii = [System.Text.Encoding]::ASCII.GetString($bytes)

$keys = @(
    "chat_template",
    "tokenizer.chat_template",
    "minicpm5",
    "MiniCPM5",
    "general.architecture",
    "tokenizer.ggml.pre",
    "bos_token_id",
    "eos_token_id"
)

Write-Host ""
Write-Host "===== Metadata key search =====" -ForegroundColor Cyan
foreach ($k in $keys) {
    $idx = $ascii.IndexOf($k)
    if ($idx -ge 0) {
        $snippet = $ascii.Substring([math]::Max(0,$idx-2), [math]::Min(80, $ascii.Length-$idx))
        $clean = ($snippet -replace "[\x00-\x1F\x7F]", ".").Substring(0, [math]::Min(80, $snippet.Length))
        Write-Host ("[FOUND] '{0}' at offset {1}" -f $k, $idx) -ForegroundColor Green
        Write-Host ("         snippet: " + $clean)
    } else {
        Write-Host ("[MISS]  '{0}'" -f $k) -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "===== llama.cpp examples/main dir =====" -ForegroundColor Cyan
if (Test-Path "F:\hb_llama\examples\main") {
    Get-ChildItem "F:\hb_llama\examples\main" | Select-Object Name, Length
} else {
    Write-Host "examples/main not found"
}

Write-Host ""
Write-Host "===== llama.cpp examples dir =====" -ForegroundColor Cyan
Get-ChildItem "F:\hb_llama\examples" -Directory | Select-Object Name
